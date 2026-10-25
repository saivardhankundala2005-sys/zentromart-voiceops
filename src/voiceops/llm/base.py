"""Base abstraction for LLM providers."""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

class BaseLLM(ABC):
    """Abstract interface for all LLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> str:
        """Generate response text from prompt and optional system instruction."""
        pass
