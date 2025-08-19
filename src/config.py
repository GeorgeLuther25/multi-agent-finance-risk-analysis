import os
from langchain_community.llms import HuggingFaceHub
from langchain_core.language_models import BaseLLM
from langchain_core.outputs import LLMResult
from typing import Any, List, Optional

# Use HuggingFace free tier (requires HF_TOKEN environment variable)
# Or fall back to a simple mock LLM for demonstration

def get_llm(temperature: float = 0.0):
    # Try to use HuggingFace free tier if token is available
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        return HuggingFaceHub(
            repo_id="microsoft/DialoGPT-medium",  # Free model
            huggingfacehub_api_token=hf_token,
            temperature=temperature
        )
    
    # Fall back to mock LLM for free demonstration
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