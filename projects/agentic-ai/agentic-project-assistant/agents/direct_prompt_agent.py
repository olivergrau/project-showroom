from __future__ import annotations

from .base import BasicAgent, MODEL
from .openai_service import OpenAIService


class DirectPromptAgent(BasicAgent):
    """Agent that directly forwards a prompt to the OpenAI API."""

    def __init__(self, openai_service: OpenAIService) -> None:
        super().__init__(openai_service)

    def respond(self, prompt: str) -> str:
        response = self.openai_service.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()
