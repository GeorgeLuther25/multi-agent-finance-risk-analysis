import os
from typing import Any, List, Optional
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import LLMResult

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

# Prefer chat interface for Ollama when using message-based prompts
try:
    from langchain_community.chat_models import ChatOllama
except ImportError:
    ChatOllama = None

def get_llm(temperature: float = 0.1, model_provider: str = "auto"):
    """
    Get an actual LLM instance based on available API keys and preferences.
    
    Args:
        temperature: Controls randomness in responses (0.0 = deterministic, 1.0 = creative)
        model_provider: Preferred provider ("openai", "anthropic", "google", "ollama", "auto")
    
    Returns:
        An actual LLM instance for the sentiment agent and other components
    """
    
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

        # Local Qwen (via Ollama)
        elif (ChatOllama or Ollama):
            try:
                if ChatOllama:
                    print("🤖 Using local Qwen (chat) via Ollama")
                    return ChatOllama(
                        model="qwen:4b",
                        temperature=temperature,
                        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                    )
                # Fallback to completion interface
                print("🤖 Using local Qwen (completion) via Ollama")
                return Ollama(
                    model="qwen:4b",
                    temperature=temperature,
                    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                )
            except Exception as e:
                print("⚠️ Exception occurred using Qwen:", e)
        
        # # Check for local Ollama installation
        # elif Ollama:
        #     try:
        #         print("🤖 Using local Ollama models")
        #         return Ollama(
        #             model="llama2",  # Default model, can be changed
        #             temperature=temperature
        #         )
        #     except Exception:
        #         print("Exception occured in Using local Ollama models")
        #         pass
    
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
    
    # elif model_provider == "ollama" and Ollama:
    #     print("🤖 Using local Ollama models")
    #     try:
    #         return Ollama(
    #             model="llama2",
    #             temperature=temperature
    #         )
    #     except Exception:
    #         print("Exception occured in Using local Ollama models")
    #         pass
    
    # Fall back to enhanced mock for development/testing
    print("⚠️  No API keys found - using mock LLM for testing")
    print("   To use actual agents, set one of:")
    print("   • OPENAI_API_KEY for OpenAI GPT models")
    print("   • ANTHROPIC_API_KEY for Anthropic Claude models") 
    print("   • GOOGLE_API_KEY for Google Gemini models")
    print("   • Install Ollama locally for free models")
    
    return MockLLM(temperature=temperature)


class MockLLM(BaseLLM):
    """Simple mock LLM for free demonstration - no API calls needed"""
    
    temperature: float = 0.0
    
    @property
    def _llm_type(self) -> str:
        return "mock"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        # Simple responses for the agent prompts
        if "Data Agent" in prompt:
            return "Data fetched successfully"
        elif "Risk Agent" in prompt:
            return "Risk metrics computed"
        elif "Writer Agent" in prompt:
            return "Report generated"
        else:
            return "Processing complete"
    
    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None, **kwargs: Any) -> LLMResult:
        responses = [self._call(prompt, stop, **kwargs) for prompt in prompts]
        return LLMResult(generations=[[{"text": response}] for response in responses])