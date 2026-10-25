"""Google Gemini implementation of BaseLLM using google-genai."""
import json
import time
from typing import Optional, Dict, Any
from google import genai
from google.genai import types
from voiceops.config import (
    GEMINI_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TIMEOUT
)
from voiceops.llm.base import BaseLLM

class GeminiProvider(BaseLLM):
    """Client for Google Gemini models via the google-genai SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ):
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not configured.")
        self.model_name = model_name or LLM_MODEL
        self.default_temperature = temperature if temperature is not None else LLM_TEMPERATURE
        self.default_max_tokens = max_tokens if max_tokens is not None else LLM_MAX_TOKENS
        self.timeout = timeout or LLM_TIMEOUT
        self.client = genai.Client(api_key=self.api_key)

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> str:
        """Generate content from Gemini with retry and JSON-mode support."""
        temp = temperature if temperature is not None else self.default_temperature
        tokens = max_tokens if max_tokens is not None else self.default_max_tokens

        config_kwargs = {
            "temperature": temp,
            "max_output_tokens": tokens,
        }
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"

        config = types.GenerateContentConfig(**config_kwargs)

        # Supported fallback models if specific model hits daily limits
        fallback_candidates = [self.model_name, "gemini-3.5-flash-lite", "gemini-3.5-flash"]
        models_to_try = []
        for m in fallback_candidates:
            if m not in models_to_try:
                models_to_try.append(m)

        max_attempts = 4
        last_error = None

        for current_model in models_to_try:
            for attempt in range(max_attempts):
                try:
                    response = self.client.models.generate_content(
                        model=current_model,
                        contents=prompt,
                        config=config,
                    )
                    if response and response.text:
                        return response.text.strip()
                    return ""
                except Exception as e:
                    last_error = e
                    err_msg = str(e)
                    if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                        import re
                        delay_match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_msg) or re.search(r"retryDelay':\s*'(\d+)s", err_msg)
                        sleep_sec = 25
                        if delay_match:
                            sleep_sec = max(int(float(delay_match.group(1))) + 2, 5)
                        
                        # If daily limit exceeded (sleep > 60s), switch to fallback model immediately
                        if sleep_sec > 60:
                            print(f"[Gemini Quota] Model {current_model} daily quota reached ({sleep_sec}s delay). Switching to fallback model...")
                            break
                        
                        print(f"[Gemini 429 Rate Limit] Model {current_model} pausing {sleep_sec}s before retry {attempt+1}/{max_attempts}...")
                        time.sleep(sleep_sec)
                    else:
                        time.sleep(2 * (attempt + 1))

        raise RuntimeError(f"Gemini generation failed across models {models_to_try}: {last_error}")

# Global singleton factory
_LLM_INSTANCE: Optional[BaseLLM] = None

def get_llm_provider() -> BaseLLM:
    global _LLM_INSTANCE
    if _LLM_INSTANCE is None:
        _LLM_INSTANCE = GeminiProvider()
    return _LLM_INSTANCE
