"""
Custom LLM router supporting multiple providers without LangChain dependencies.
Supports OpenAI, Claude (Anthropic), Gemini, and xAI with unified interface.
"""

import os
import asyncio
import json
from typing import Dict, Any, Optional, AsyncGenerator, Generator, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class LLMResponse:
    """Standardized response format for all LLM providers."""
    output: str
    tokens: int
    provider: str
    model: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response from the LLM."""
        pass
    
    @abstractmethod
    async def generate_async(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response asynchronously."""
        pass
    
    @abstractmethod
    def stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Stream response from the LLM."""
        pass
    
    @abstractmethod
    async def stream_async(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Stream response asynchronously."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider."""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        super().__init__(api_key, model)
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
            self.async_client = openai.AsyncOpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package is required for OpenAI provider")
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        
        return LLMResponse(
            output=response.choices[0].message.content,
            tokens=response.usage.total_tokens,
            provider="openai",
            model=self.model,
            finish_reason=response.choices[0].finish_reason,
            usage=response.usage.__dict__
        )
    
    async def generate_async(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response asynchronously."""
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        
        return LLMResponse(
            output=response.choices[0].message.content,
            tokens=response.usage.total_tokens,
            provider="openai",
            model=self.model,
            finish_reason=response.choices[0].finish_reason,
            usage=response.usage.__dict__
        )
    
    def stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Stream response from OpenAI."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **kwargs
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    
    async def stream_async(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Stream response asynchronously."""
        stream = await self.async_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **kwargs
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content


class ClaudeProvider(BaseLLMProvider):
    """Anthropic Claude API provider."""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        super().__init__(api_key, model)
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            self.async_client = anthropic.AsyncAnthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic package is required for Claude provider")
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using Claude API."""
        max_tokens = kwargs.pop('max_tokens', 1000)
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        
        return LLMResponse(
            output=response.content[0].text,
            tokens=response.usage.input_tokens + response.usage.output_tokens,
            provider="claude",
            model=self.model,
            finish_reason=response.stop_reason,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }
        )
    
    async def generate_async(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response asynchronously."""
        max_tokens = kwargs.pop('max_tokens', 1000)
        
        response = await self.async_client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        
        return LLMResponse(
            output=response.content[0].text,
            tokens=response.usage.input_tokens + response.usage.output_tokens,
            provider="claude",
            model=self.model,
            finish_reason=response.stop_reason,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }
        )
    
    def stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Stream response from Claude."""
        max_tokens = kwargs.pop('max_tokens', 1000)
        
        with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        ) as stream:
            for text in stream.text_stream:
                yield text
    
    async def stream_async(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Stream response asynchronously."""
        max_tokens = kwargs.pop('max_tokens', 1000)
        
        async with self.async_client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        ) as stream:
            async for text in stream.text_stream:
                yield text


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API provider."""
    
    def __init__(self, api_key: str, model: str = "gemini-pro"):
        super().__init__(api_key, model)
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model_instance = genai.GenerativeModel(model)
            self.genai = genai
        except ImportError:
            raise ImportError("google-generativeai package is required for Gemini provider")
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using Gemini API."""
        response = self.model_instance.generate_content(prompt, **kwargs)
        
        # Estimate token count (Gemini doesn't always provide exact counts)
        token_count = len(prompt.split()) + len(response.text.split()) if response.text else 0
        
        return LLMResponse(
            output=response.text,
            tokens=token_count,
            provider="gemini",
            model=self.model,
            finish_reason=getattr(response, 'finish_reason', None),
            usage={"estimated_tokens": token_count}
        )
    
    async def generate_async(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response asynchronously."""
        # Gemini doesn't have native async support, so we use asyncio
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, 
            lambda: self.model_instance.generate_content(prompt, **kwargs)
        )
        
        token_count = len(prompt.split()) + len(response.text.split()) if response.text else 0
        
        return LLMResponse(
            output=response.text,
            tokens=token_count,
            provider="gemini",
            model=self.model,
            finish_reason=getattr(response, 'finish_reason', None),
            usage={"estimated_tokens": token_count}
        )
    
    def stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Stream response from Gemini."""
        response = self.model_instance.generate_content(prompt, stream=True, **kwargs)
        for chunk in response:
            if chunk.text:
                yield chunk.text
    
    async def stream_async(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Stream response asynchronously."""
        # Gemini doesn't have native async streaming, so we simulate it
        loop = asyncio.get_event_loop()
        
        def _generate_stream():
            response = self.model_instance.generate_content(prompt, stream=True, **kwargs)
            return list(response)
        
        chunks = await loop.run_in_executor(None, _generate_stream)
        for chunk in chunks:
            if chunk.text:
                yield chunk.text


class XAIProvider(BaseLLMProvider):
    """xAI (Grok) API provider - Future-compatible implementation."""
    
    def __init__(self, api_key: str, model: str = "grok-1"):
        super().__init__(api_key, model)
        # xAI API is compatible with OpenAI format
        try:
            import openai
            self.client = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.x.ai/v1"  # xAI endpoint
            )
            self.async_client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.x.ai/v1"
            )
        except ImportError:
            raise ImportError("openai package is required for xAI provider")
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using xAI API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        
        return LLMResponse(
            output=response.choices[0].message.content,
            tokens=response.usage.total_tokens,
            provider="xai",
            model=self.model,
            finish_reason=response.choices[0].finish_reason,
            usage=response.usage.__dict__
        )
    
    async def generate_async(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response asynchronously."""
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        
        return LLMResponse(
            output=response.choices[0].message.content,
            tokens=response.usage.total_tokens,
            provider="xai",
            model=self.model,
            finish_reason=response.choices[0].finish_reason,
            usage=response.usage.__dict__
        )
    
    def stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Stream response from xAI."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **kwargs
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    
    async def stream_async(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Stream response asynchronously."""
        stream = await self.async_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **kwargs
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content


class LLMRouter:
    """Main router class for managing multiple LLM providers."""
    
    PROVIDER_CLASSES = {
        "openai": OpenAIProvider,
        "claude": ClaudeProvider,
        "gemini": GeminiProvider,
        "xai": XAIProvider
    }
    
    def __init__(self, provider: str = None, model: str = None, api_key: str = None, max_tokens: int = None):
        """
        Initialize LLM router.
        
        Args:
            provider: LLM provider name (openai, claude, gemini, xai)
            model: Model name to use
            api_key: API key (if not provided, will try to get from environment)
            max_tokens: Maximum tokens for generation (if not provided, will try to get from environment)
        """
        # Get configuration from environment if not provided
        self.provider_name = provider or os.getenv("LLM_PROVIDER", "openai")
        
        # Default models for each provider
        default_models = {
            "openai": "gpt-3.5-turbo",
            "claude": "claude-3-sonnet-20240229",
            "gemini": "gemini-pro",
            "xai": "grok-1"
        }
        
        self.model = model or os.getenv("LLM_MODEL") or default_models.get(self.provider_name)
        
        # Get max_tokens from environment if not provided
        self.max_tokens = max_tokens or int(os.getenv("MAX_TOKENS", "1000"))
        
        # Get streaming preference from environment
        self.use_streaming = os.getenv("USE_STREAMING", "false").lower() in ("true", "1", "yes", "on")
        
        # Get API key from environment if not provided
        if not api_key:
            key_env_vars = {
                "openai": "OPENAI_API_KEY",
                "claude": "ANTHROPIC_API_KEY",
                "gemini": "GOOGLE_API_KEY",
                "xai": "XAI_API_KEY"
            }
            api_key = os.getenv(key_env_vars.get(self.provider_name))
        
        if not api_key:
            raise ValueError(f"API key not provided for {self.provider_name}")
        
        # Initialize provider
        if self.provider_name not in self.PROVIDER_CLASSES:
            raise ValueError(f"Unsupported provider: {self.provider_name}")
        
        provider_class = self.PROVIDER_CLASSES[self.provider_name]
        self.provider = provider_class(api_key, self.model)
        self.api_key = api_key
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using the configured provider."""
        # Set max_tokens from environment if not provided in kwargs
        if 'max_tokens' not in kwargs:
            kwargs['max_tokens'] = self.max_tokens
        return self.provider.generate(prompt, **kwargs)
    
    async def generate_async(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response asynchronously."""
        # Set max_tokens from environment if not provided in kwargs
        if 'max_tokens' not in kwargs:
            kwargs['max_tokens'] = self.max_tokens
        return await self.provider.generate_async(prompt, **kwargs)
    
    def stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """Stream response from the LLM."""
        # Set max_tokens from environment if not provided in kwargs
        if 'max_tokens' not in kwargs:
            kwargs['max_tokens'] = self.max_tokens
        return self.provider.stream(prompt, **kwargs)
    
    async def stream_async(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Stream response asynchronously."""
        # Set max_tokens from environment if not provided in kwargs
        if 'max_tokens' not in kwargs:
            kwargs['max_tokens'] = self.max_tokens
        async for chunk in self.provider.stream_async(prompt, **kwargs):
            yield chunk
    
    def get_provider_info(self) -> Dict[str, str]:
        """Get information about the current provider."""
        return {
            "provider": self.provider_name,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "streaming_enabled": self.use_streaming,
            "available_providers": list(self.PROVIDER_CLASSES.keys())
        }
    
    @classmethod
    def create_from_env(cls) -> 'LLMRouter':
        """Create router using environment variables."""
        return cls()
    
    @classmethod
    def get_available_providers(cls) -> Dict[str, Any]:
        """Get information about all available providers."""
        return {
            "openai": {
                "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
                "env_var": "OPENAI_API_KEY",
                "required_package": "openai"
            },
            "claude": {
                "models": ["claude-3-sonnet-20240229", "claude-3-opus-20240229", "claude-3-haiku-20240307"],
                "env_var": "ANTHROPIC_API_KEY",
                "required_package": "anthropic"
            },
            "gemini": {
                "models": ["gemini-pro", "gemini-pro-vision"],
                "env_var": "GOOGLE_API_KEY",
                "required_package": "google-generativeai"
            },
            "xai": {
                "models": ["grok-1"],
                "env_var": "XAI_API_KEY",
                "required_package": "openai"
            }
        }