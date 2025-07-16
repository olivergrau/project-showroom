from typing import Any

from openai import OpenAI


class OpenAIService:
    """Simple wrapper around the OpenAI client used by the agents."""

    def __init__(self, api_key: str, base_url: str = "https://openai.vocareum.com/v1") -> None:
        """Create a client for communicating with the OpenAI API."""
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(self, **kwargs: Any) -> Any:
        """Call the chat completion endpoint and return the API response."""
        return self.client.chat.completions.create(**kwargs)

    def embed(self, **kwargs: Any) -> Any:
        """Call the embeddings endpoint and return the API response."""
        return self.client.embeddings.create(**kwargs)
