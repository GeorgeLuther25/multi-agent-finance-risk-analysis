"""
RAG utilities for storing and retrieving 10-K/10-Q financial documents.
Provides document ingestion, vector storage, and retrieval capabilities.
"""

import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter


from utils.config import get_llm, get_embeddings


class FundamentalRAG:
    """
    RAG system for fundamental analysis using 10-K/10-Q documents.
    """
    
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = get_embeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.collection_name = "fundamental_filings_Ollama" if os.getenv("OLLAMA_MODEL") else "fundamental_filings"
        
        # Initialize or load existing vector store with dimension compatibility check
        try:
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
            # Test if embeddings are compatible by trying a simple similarity search
            test_embedding = self.embeddings.embed_query("test")
            print(f"🔍 Testing ChromaDB compatibility with {len(test_embedding)}-dimensional embeddings")
            # Try a simple query to check if dimensions match
            self.vectorstore.similarity_search("test", k=1)
            print("✅ ChromaDB collection is compatible")
        except Exception as e:
            if "dimension" in str(e).lower() or "expecting embedding with dimension" in str(e):
                print(f"🔄 Embedding dimension mismatch detected: {str(e)}")
                print("🔄 Recreating vector store with new embedding dimensions...")
                # Remove existing collection and create new one
                import shutil
                if os.path.exists(persist_directory):
                    shutil.rmtree(persist_directory)
                os.makedirs(persist_directory, exist_ok=True)
                self.vectorstore = Chroma(
                    persist_directory=persist_directory,
                    embedding_function=self.embeddings
                )
                print("✅ Created new ChromaDB collection")
            else:
                raise e
        
        # Ensure data directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
    def ingest_document(self, ticker: str, filing_type: str, filing_year: int, filing_start_month: int,
                        filing_end_month: int, document_path: str) -> bool:
        """
        Ingest a 10-K or 10-Q document into the vector store.
        
        Args:
            ticker: Stock ticker symbol
            filing_type: "10-K" or "10-Q"
            document_path: Path to the PDF document
            
        Returns:
            bool: Success status
        """
        try:
            # Load PDF document
            loader = PyPDFLoader(document_path)
            pages = loader.load()
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(pages)
            
            # Add metadata to chunks
            for chunk in chunks:
                chunk.metadata.update({
                    "ticker": ticker.upper(),
                    "filing_type": filing_type,
                    "filing_year": filing_year,
                    "filing_start_month": filing_start_month,
                    "filing_end_month": filing_end_month,
                    "source": document_path,
                    "ingestion_date": datetime.now().isoformat()
                })
            
            # Add to vector store
            self.vectorstore.add_documents(chunks)
            
            print(f"Successfully ingested {len(chunks)} chunks from ticker:{ticker}, filing_type:{filing_type}, " +
                  f"filing_year:{filing_year} from month {filing_start_month} to month {filing_end_month}")
            return True
            
        except Exception as e:
            print(f"Error ingesting document for {ticker}: {e}")
            return False
    
    def retrieve_relevant_chunks(
        self,
        ticker: str,
        query: str,
        filing_type: Optional[str] = None,
        from_year: Optional[int] = datetime.now().year,
        from_month: Optional[int] = 1,
        to_year: Optional[int] = datetime.now().year,
        to_month: Optional[int] = 12,
        k: int = 5
    ) -> List[Document]:
        """
        Retrieve relevant document chunks for a given query within a date range.

        Args:
            ticker: Stock ticker symbol
            query: Search query
            filing_type: Optional filing type filter ("10-K" or "10-Q")
            from_year: Start year (inclusive)
            from_month: Start month (inclusive)  
            to_year: End year (inclusive)
            to_month: End month (inclusive)
            k: Number of chunks to retrieve
            
        Returns:
            List of relevant document chunks
        """
        try:
            # Build base filter
            filter_conditions: list[Any] = [{"ticker": ticker.upper()}]
            
            if filing_type:
                filter_conditions.append({"filing_type": filing_type})
            
            # Add date range filters
            if from_year is not None:
                # Documents that end after the start period
                filter_conditions.append({
                    "$or": [
                        {"filing_year": {"$gt": from_year}},
                        {
                            "$and": [
                                {"filing_year": from_year},
                                {"filing_end_month": {"$gte": from_month}}
                            ]
                        }
                    ]
                })
            
            if to_year is not None:
                # Documents that start before the end period
                filter_conditions.append({
                    "$or": [
                        {"filing_year": {"$lt": to_year}},
                        {
                            "$and": [
                                {"filing_year": to_year},
                                {"filing_start_month": {"$lte": to_month}}
                            ]
                        }
                    ]
                })
            
            # Combine all conditions with AND
            if len(filter_conditions) > 1:
                filter_dict = {"$and": filter_conditions}
            else:
                filter_dict = filter_conditions[0]
            
            # Search for relevant chunks
            results = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filter_dict
            )
            
            return results
            
        except Exception as e:
            print(f"Error retrieving chunks for {ticker}: {e}")
            return []
    
    def get_available_filings(self, ticker: str) -> List[Dict[str, Any]]:
        """
        Get list of available filings for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            List of filing information dictionaries
        """
        try:
            # Query the collection to get all documents for this ticker
            results = self.vectorstore.get(
                where={"ticker": ticker.upper()}
            )
            
            # Extract unique filings
            filings = {}
            for metadata in results.get("metadatas", []):
                key = f"{metadata.get('ticker')}_{metadata.get('filing_type')}"
                if key not in filings:
                    filings[key] = {
                        "ticker": metadata.get("ticker"),
                        "filing_type": metadata.get("filing_type"),
                        "source": metadata.get("source"),
                        "ingestion_date": metadata.get("ingestion_date")
                    }
            
            return list(filings.values())
            
        except Exception as e:
            print(f"Error getting available filings for {ticker}: {e}")
            return []

# Sample 10-K/10-Q document URLs (these would normally be downloaded from SEC EDGAR)
SAMPLE_DOCUMENTS = {
    "AAPL": {
        "10-K": "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm",
        "ticker": "AAPL",
        "company": "Apple Inc."
    },
    "MSFT": {
        "10-K": "https://www.sec.gov/Archives/edgar/data/789019/000156459023003001/msft-10k_20230630.htm",
        "ticker": "MSFT", 
        "company": "Microsoft Corporation"
    },
    "GOOGL": {
        "10-K": "https://www.sec.gov/Archives/edgar/data/1652044/000165204423000016/goog-20221231.htm",
        "ticker": "GOOGL",
        "company": "Alphabet Inc."
    }
}


def initialize_sample_data(rag_system: FundamentalRAG) -> None:
    """
    Initialize the RAG system with sample 10-K documents.
    Note: In a production system, these would be actual PDF files.
    For now, we'll create sample text documents.
    """
    
    sample_data_dir = Path("./data/sample_filings")
    sample_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample 10-K content for demonstration
    sample_10k_content = """
    UNITED STATES
    SECURITIES AND EXCHANGE COMMISSION
    Washington, D.C. 20549
    
    FORM 10-K
    
    ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d) OF THE SECURITIES EXCHANGE ACT OF 1934
    
    For the fiscal year ended September 30, 2023
    
    Commission File Number: 001-36743
    
    Apple Inc.
    (Exact name of registrant as specified in its charter)
    
    BUSINESS
    
    Company Background
    
    The Company designs, manufactures and markets smartphones, personal computers, tablets, wearables and accessories, and sells a variety of related services. The Company's fiscal year is the 52- or 53-week period that ends on the last Saturday of September.
    
    Products
    
    iPhone
    iPhone® is the Company's line of smartphones based on its iOS operating system. In October 2022, the Company introduced iPhone 14, iPhone 14 Plus, iPhone 14 Pro and iPhone 14 Pro Max. iPhone includes Siri®, the Company's voice-activated digital assistant.
    
    Mac
    Mac® is the Company's line of personal computers based on its macOS® operating system. The Mac product line includes laptops MacBook Air® and MacBook Pro®, as well as desktops iMac®, Mac mini®, Mac Studio® and Mac Pro®.
    
    iPad
    iPad® is the Company's line of multipurpose tablets based on its iPadOS® operating system. The iPad product line includes iPad, iPad mini®, iPad Air® and iPad Pro®.
    
    FINANCIAL PERFORMANCE
    
    Net Sales
    Total net sales were $383.3 billion for 2023, compared to $394.3 billion for 2022. The year-over-year decrease in total net sales was primarily due to lower sales of iPhone and Mac, partially offset by higher sales of Services.
    
    Gross Margin
    Gross margin percentage was 44.1% for 2023, compared to 43.3% for 2022. The year-over-year increase in gross margin percentage was primarily due to a different mix of products and services sold.
    
    Operating Income
    Operating income was $114.3 billion for 2023, compared to $119.4 billion for 2022. The year-over-year decrease in operating income was primarily due to lower gross profit, partially offset by lower operating expenses.
    
    RISK FACTORS
    
    The Company's business, reputation, results of operations, financial condition and stock price can be affected by a number of factors, whether currently known or unknown, including those described below. When any one or more of these risks materialize from time to time, the Company's business, reputation, results of operations, financial condition and stock price can be materially and adversely affected.
    
    Global and regional economic conditions could materially adversely affect the Company.
    
    The Company's operations and performance depend significantly on global and regional economic conditions and adverse economic conditions can reduce demand for the Company's products and services.
    """
    
    # Save sample content as text files (in production, these would be PDFs)
    for ticker, doc_info in SAMPLE_DOCUMENTS.items():
        file_path = sample_data_dir / f"{ticker}_10K_sample.txt"
        
        # Customize content for each company
        customized_content = sample_10k_content.replace("Apple Inc.", doc_info["company"])
        customized_content = customized_content.replace("AAPL", ticker)
        
        with open(file_path, "w") as f:
            f.write(customized_content)
        
        # Convert text to Document object and ingest
        doc = Document(
            page_content=customized_content,
            metadata={
                "ticker": ticker,
                "filing_type": "10-K",
                "source": str(file_path),
                "company": doc_info["company"],
                "ingestion_date": datetime.now().isoformat()
            }
        )
        
        # Split and add to vector store
        chunks = rag_system.text_splitter.split_documents([doc])
        for chunk in chunks:
            chunk.metadata.update(doc.metadata)
        
        rag_system.vectorstore.add_documents(chunks)
        print(f"Initialized sample data for {ticker} ({doc_info['company']})")


def validate_file_path(file_path: str) -> bool:
    """Validate that the file exists and is a PDF."""
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        return False

    if not file_path.lower().endswith('.pdf'):
        print(f"⚠️  Warning: File is not a PDF: {file_path}")
        print("   The system is designed for PDF documents but will attempt to process.")

    return True


def validate_ticker(ticker: str) -> bool:
    """Validate ticker symbol format."""
    if not ticker or len(ticker) < 1 or len(ticker) > 5:
        print(f"❌ Error: Invalid ticker symbol: {ticker}")
        print("   Ticker should be 1-5 characters long (e.g., AAPL, MSFT)")
        return False
    return True


def validate_filing_type(filing_type: str) -> bool:
    """Validate filing type."""
    valid_types = ["10K", "10Q", "8K", "20F"]
    if filing_type not in valid_types:
        print(f"❌ Error: Invalid filing type: {filing_type}")
        print(f"   Valid types are: {', '.join(valid_types)}")
        return False
    return True


def ingest_single_document(
    rag_system: FundamentalRAG,
    ticker: str,
    filing_type: str,
    filing_year: int,
    filing_start_month: int,
    filing_end_month: int,
    document_path: str
) -> bool:
    """Ingest a single document."""
    print("\n📄 Ingesting document:")
    print(f"   Ticker: {ticker}")
    print(f"   Filing Type: {filing_type}")
    print(f"   Path: {document_path}")

    # Validate inputs
    if not validate_ticker(ticker):
        return False

    if not validate_filing_type(filing_type):
        return False

    if not validate_file_path(document_path):
        return False

    # Perform ingestion
    try:
        success = rag_system.ingest_document(ticker.upper(), filing_type, filing_year, filing_start_month,
                                             filing_end_month, document_path)
        if success:
            print(f"✅ Successfully ingested {ticker} {filing_type}")
            return True
        else:
            print(f"❌ Failed to ingest {ticker} {filing_type}")
            return False

    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        return False


def parse_filename_metadata(filename: str) -> dict[str, Any] | None:
    """
    Parse metadata from filename using common naming conventions.

    Expected formats:
    - AAPL_10K_2023.pdf
    - MSFT-10Q-Q1-2023.pdf
    - ticker_filing_year.pdf
    """
    basename = os.path.basename(filename).replace('.pdf', '').replace('.PDF', '')

    # Try different parsing patterns
    parts = basename.replace('-', '_').split('_')

    if len(parts) >= 2:
        ticker = parts[0].upper()
        filing_type = parts[1].upper()
        filing_freq = parts[2].upper()
        filing_start_month = parts[3]
        filing_end_month = parts[4]
        filing_year = parts[5]

        return {
            'ticker': ticker,
            'filing_type': filing_type,
            'filing_year': int(filing_year),
            'filing_start_month': int(filing_start_month),
            'filing_end_month': int(filing_end_month)
        }

    return None


def batch_ingest_documents(rag_system: FundamentalRAG, directory: str) -> int:
    """Ingest all PDF documents from a directory."""
    if not os.path.exists(directory):
        print(f"❌ Error: Directory not found: {directory}")
        return 0

    # Find all PDF files
    pdf_patterns = [
        os.path.join(directory, "*.pdf"),
        os.path.join(directory, "*.PDF")
        # os.path.join(directory, "**", "*.pdf"),
        # os.path.join(directory, "**", "*.PDF")
    ]

    pdf_files = []
    for pattern in pdf_patterns:
        pdf_files.extend(glob.glob(pattern, recursive=True))

    if not pdf_files:
        print(f"❌ No PDF files found in directory: {directory}")
        return 0

    print(f"\n📁 Found {len(pdf_files)} PDF files to process:")
    for pdf_file in pdf_files:
        print(f"   - {os.path.basename(pdf_file)}")

    successful_ingestions = 0

    for pdf_file in pdf_files:
        print("\n" + "="*60)

        # Try to parse metadata from filename
        metadata = parse_filename_metadata(pdf_file)

        if metadata:
            success = ingest_single_document(
                rag_system,
                metadata['ticker'],
                metadata['filing_type'],
                metadata['filing_year'],
                metadata['filing_start_month'],
                metadata['filing_end_month'],
                pdf_file
            )
            if success:
                successful_ingestions += 1
        else:
            print(f"⚠️  Could not parse metadata from filename: {os.path.basename(pdf_file)}")
            print("   Please use format: TICKER_FILING-TYPE_YEAR.pdf (e.g., AAPL_10K_2023.pdf)")
            print("   Skipping this file.")

    return successful_ingestions
