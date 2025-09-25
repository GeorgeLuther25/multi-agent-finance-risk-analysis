import os
from typing import Any, List, Optional
from langchain_core.language_models import BaseLLM, BaseChatModel
from langchain_core.outputs import LLMResult
from langchain_core.messages import BaseMessage, AIMessage

# Import actual LLM providers
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None

try:
    from langchain_community.llms import Ollama
except ImportError:
    Ollama = None

try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None

# Import embedding providers
try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    OpenAIEmbeddings = None

try:
    # from langchain_community.embeddings import OllamaEmbeddings
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    OllamaEmbeddings = None

def get_llm(temperature: float = 0.1, model_provider: str = "auto"):
    """
    Get an actual LLM instance based on available API keys and preferences.
    
    Args:
        temperature: Controls randomness in responses (0.0 = deterministic, 1.0 = creative)
        model_provider: Preferred provider ("openai", "anthropic", "google", "ollama", "auto")
                       Can also be set via MODEL_PROVIDER environment variable
    
    Returns:
        An actual LLM instance for the sentiment agent and other components
    """
    
    # Check for model provider from environment variable
    env_provider = os.getenv("MODEL_PROVIDER", "").lower()
    if env_provider and model_provider == "auto":
        model_provider = env_provider
        print(f"📝 Model provider set from environment: {model_provider}")
    
    # Priority order for auto-detection
    if model_provider == "auto":
        # Check for OpenAI API key
        if os.getenv("OPENAI_API_KEY") and ChatOpenAI:
            print("🤖 Using OpenAI GPT models")
            return ChatOpenAI(
                # model="gpt-3.5-turbo",  # Cost-effective option
                model="gpt-4o",
                temperature=temperature,
                max_tokens=1000
            )
        
        # Check for Anthropic API key
        elif os.getenv("ANTHROPIC_API_KEY") and ChatAnthropic:
            print("🤖 Using Anthropic Claude models")
            return ChatAnthropic(
                model="claude-3-haiku-20240307",  # Fast and cost-effective
                temperature=temperature,
                max_tokens=1000
            )
        
        # Check for Google API key
        elif os.getenv("GOOGLE_API_KEY") and ChatGoogleGenerativeAI:
            print("🤖 Using Google Gemini models")
            return ChatGoogleGenerativeAI(
                model="gemini-pro",
                temperature=temperature,
                max_output_tokens=1000
            )
        
        # Check for local Ollama installation
        elif ChatOllama:
            try:
                print("🤖 Using local Ollama models")
                # Try to connect to Ollama with a default model
                ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")  # Default lightweight model
                llm = ChatOllama(
                    model=ollama_model,
                    temperature=temperature,
                    base_url="http://localhost:11434"  # Default Ollama URL
                )
                # Test the connection by trying a simple generation
                # llm.invoke("test")
                return llm
            except Exception as e:
                print(f"❌ Failed to connect to Ollama: {e}")
                print("   Make sure Ollama is running: ollama serve")
                print(f"   And model is available: ollama pull {os.getenv('OLLAMA_MODEL', 'llama3.2:3b')}")
                pass
    
    # Specific provider selection
    elif model_provider == "openai" and os.getenv("OPENAI_API_KEY") and ChatOpenAI:
        print("🤖 Using OpenAI GPT models")
        return ChatOpenAI(
            model="gpt-4o-mini",  # Latest cost-effective model
            temperature=temperature,
            max_tokens=1000
        )
    
    elif model_provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY") and ChatAnthropic:
        print("🤖 Using Anthropic Claude models")
        return ChatAnthropic(
            model="claude-3-haiku-20240307",
            temperature=temperature,
            max_tokens=1000
        )
    
    elif model_provider == "google" and os.getenv("GOOGLE_API_KEY") and ChatGoogleGenerativeAI:
        print("🤖 Using Google Gemini models")
        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=temperature,
            max_output_tokens=1000
        )
    
    elif model_provider == "ollama" and ChatOllama:
        print("🤖 Using local Ollama models")
        try:
            ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
            print(f"   Model: {ollama_model}")
            llm = ChatOllama(
                model=ollama_model,
                temperature=temperature,
                base_url="http://localhost:11434"
            )
            # Test the connection by trying a simple generation
            llm.invoke("test")
            return llm
        except Exception as e:
            print(f"❌ Failed to connect to Ollama: {e}")
            print("   Make sure Ollama is running: ollama serve")
            ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
            print(f"   And model is available: ollama pull {ollama_model}")
            # Fall through to mock LLM
    
    # Fall back to enhanced mock for development/testing
    print("⚠️  No API keys found - using mock LLM for testing")
    print("   To use actual agents, set one of:")
    print("   • OPENAI_API_KEY for OpenAI GPT models")
    print("   • ANTHROPIC_API_KEY for Anthropic Claude models")
    print("   • GOOGLE_API_KEY for Google Gemini models")
    print("   • Install Ollama locally for free models")
    
    return MockLLM(temperature=temperature)


def get_embeddings(model_provider: str = "auto"):
    """
    Get an embeddings instance based on available API keys and preferences.
    
    Args:
        model_provider: Preferred provider ("openai", "ollama", "auto")
                       Can also be set via MODEL_PROVIDER environment variable
    
    Returns:
        An embeddings instance for RAG functionality
    """
    
    # Check for model provider from environment variable
    env_provider = os.getenv("MODEL_PROVIDER", "").lower()
    if env_provider and model_provider == "auto":
        model_provider = env_provider
    
    # Priority order for auto-detection
    if model_provider == "auto":
        # Check for OpenAI API key first
        if os.getenv("OPENAI_API_KEY") and OpenAIEmbeddings:
            print("🔍 Using OpenAI embeddings")
            return OpenAIEmbeddings()
        
        # Check for local Ollama installation
        elif OllamaEmbeddings:
            try:
                print("🔍 Using local Ollama embeddings")
                # Use a lightweight embedding model
                ollama_embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
                print(f"   Embedding Model: {ollama_embed_model}")
                embeddings = OllamaEmbeddings(
                    model=ollama_embed_model,
                    base_url="http://localhost:11434"
                )
                # Test the connection by trying a simple embedding
                embeddings.embed_query("test")
                return embeddings
            except Exception as e:
                print(f"❌ Failed to connect to Ollama embeddings: {e}")
                print("   Make sure Ollama is running: ollama serve")
                embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
                print(f"   And embedding model is available: ollama pull {embed_model}")
                pass
    
    # Specific provider selection
    elif model_provider == "openai" and os.getenv("OPENAI_API_KEY") and OpenAIEmbeddings:
        print("🔍 Using OpenAI embeddings")
        return OpenAIEmbeddings()
    
    elif model_provider == "ollama" and OllamaEmbeddings:
        print("🔍 Using local Ollama embeddings")
        try:
            ollama_embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
            print(f"   Embedding Model: {ollama_embed_model}")
            embeddings = OllamaEmbeddings(
                model=ollama_embed_model,
                base_url="http://localhost:11434"
            )
            # Test the connection by trying a simple embedding
            embeddings.embed_query("test")
            return embeddings
        except Exception as e:
            print(f"❌ Failed to connect to Ollama embeddings: {e}")
            print("   Make sure Ollama is running: ollama serve")
            embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
            print(f"   And embedding model is available: ollama pull {embed_model}")
            # Fall through to mock embeddings
    
    # Fall back to mock embeddings for development/testing
    print("⚠️  No embeddings provider found - RAG functionality will be limited")
    print("   To use RAG features, set:")
    print("   • OPENAI_API_KEY for OpenAI embeddings")
    print("   • Install Ollama with: ollama pull nomic-embed-text")
    
    return MockEmbeddings()


class MockEmbeddings:
    """Simple mock embeddings for free demonstration - no API calls needed"""
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return dummy embeddings for documents"""
        # Simple hash-based pseudo-embeddings
        import hashlib
        embeddings = []
        for text in texts:
            # Create a simple hash-based embedding
            hash_obj = hashlib.md5(text.encode())
            hash_bytes = hash_obj.digest()
            # Convert to normalized float vector
            embedding = [float(b) / 255.0 for b in hash_bytes[:16]]  # 16-dim vector
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Return dummy embedding for query"""
        return self.embed_documents([text])[0]


class MockLLM(BaseChatModel):
    """Enhanced mock LLM for free demonstration - supports both LLM and Chat interfaces"""
    
    temperature: float = 0.0
    
    @property
    def _llm_type(self) -> str:
        return "mock_chat"
    
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, 
                  run_manager: Optional[Any] = None, **kwargs: Any) -> Any:
        """Generate chat completions for messages"""
        from langchain_core.outputs import ChatResult, ChatGeneration
        
        # Simple responses for the agent prompts
        last_message = messages[-1] if messages else None
        if not last_message:
            response_content = "No message provided"
        else:
            message_content = last_message.content if hasattr(last_message, 'content') else str(last_message)
            message_str = str(message_content)
            
            if "Data Agent" in message_str:
                response_content = "Data fetched successfully"
            elif "Risk Agent" in message_str:
                response_content = "Risk metrics computed"
            elif "Writer Agent" in message_str:
                response_content = "Report generated"
            elif "fundamental" in message_str.lower() or "10-k" in message_str.lower():
                response_content = "Fundamental analysis completed using mock tools"
            elif "sentiment" in message_str.lower():
                response_content = "Sentiment analysis completed"
            elif "valuation" in message_str.lower():
                response_content = "Valuation analysis completed"
            else:
                response_content = "Processing complete"
        
        # Create proper ChatGeneration
        generation = ChatGeneration(message=AIMessage(content=response_content))
        return ChatResult(generations=[generation])
    
    # Legacy LLM interface support
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        """Legacy LLM interface"""
        if "Data Agent" in prompt:
            return "Data fetched successfully"
        elif "Risk Agent" in prompt:
            return "Risk metrics computed"
        elif "Writer Agent" in prompt:
            return "Report generated"
        else:
            return "Processing complete"
