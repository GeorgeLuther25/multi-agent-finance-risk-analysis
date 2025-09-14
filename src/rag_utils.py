"""
RAG utilities for storing and retrieving 10-K/10-Q financial documents.
Provides document ingestion, vector storage, and retrieval capabilities.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnableConfig

from .config import get_llm


class FundamentalRAG:
    """
    RAG system for fundamental analysis using 10-K/10-Q documents.
    """
    
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Initialize or load existing vector store
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        
        # Ensure data directory exists
        os.makedirs(persist_directory, exist_ok=True)
        
    def ingest_document(self, ticker: str, filing_type: str, document_path: str) -> bool:
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
                    "source": document_path,
                    "ingestion_date": datetime.now().isoformat()
                })
            
            # Add to vector store
            self.vectorstore.add_documents(chunks)
            
            print(f"Successfully ingested {len(chunks)} chunks from {ticker} {filing_type}")
            return True
            
        except Exception as e:
            print(f"Error ingesting document for {ticker}: {e}")
            return False
    
    def retrieve_relevant_chunks(
        self,
        ticker: str,
        query: str,
        filing_type: Optional[str] = None,
        k: int = 5
    ) -> List[Document]:
        """
        Retrieve relevant document chunks for a given query.
        
        Args:
            ticker: Stock ticker symbol
            query: Search query
            filing_type: Optional filing type filter ("10-K" or "10-Q")
            k: Number of chunks to retrieve
            
        Returns:
            List of relevant document chunks
        """
        try:
            # Build metadata filter
            filter_dict = {"ticker": ticker.upper()}
            if filing_type:
                filter_dict["filing_type"] = filing_type
            
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


def query_fundamental_data(
    rag_system: FundamentalRAG,
    ticker: str,
    query: str,
    config: RunnableConfig
) -> str:
    """
    Query fundamental data using RAG and LLM for analysis.
    
    Args:
        rag_system: The RAG system instance
        ticker: Stock ticker symbol
        query: Natural language query
        config: Runnable config for LLM
        
    Returns:
        LLM analysis response
    """
    
    # Retrieve relevant chunks
    relevant_chunks = rag_system.retrieve_relevant_chunks(ticker, query, k=5)
    
    if not relevant_chunks:
        return f"No relevant information found for {ticker} in the knowledge base."
    
    # Combine chunk content
    context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])
    
    # Create LLM prompt
    system_prompt = """As a fundamental financial equity analyst your primary responsibility is to analyze the most recent 10K report provided for a company. You have access to a powerful tool that can help you extract relevant information from the 10K. Your analysis should be based solely on the information that you retrieve using this tool. You can interact with this tool using natural language queries. The tool will understand your requests and return relevant text snippets and data points from the 10K document. Keep checking if you have answered the users' question to avoid looping.

Based on the 10-K document excerpts provided below, please provide a comprehensive analysis addressing the user's query. Focus on factual information from the documents and provide clear, actionable insights.

DOCUMENT EXCERPTS:
{context}

USER QUERY: {query}

Please provide a detailed analysis based solely on the information available in the document excerpts above."""
    
    user_prompt = system_prompt.format(context=context, query=query)
    
    # Get LLM response
    llm = get_llm()
    response = llm.invoke([{"role": "user", "content": user_prompt}], config=config)
    
    return response.content if hasattr(response, 'content') else str(response)
