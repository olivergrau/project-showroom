from __future__ import annotations

from .base import BasicAgent, MODEL
from .openai_service import OpenAIService


class AugmentedPromptAgent(BasicAgent):
    """Agent that injects a persona before forwarding prompts."""

    def __init__(self, openai_service: OpenAIService, persona: str) -> None:
        super().__init__(openai_service)
        self.persona = persona

    def respond(self, input_text: str) -> str:
        response = self.openai_service.chat(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"Forget all previous context. You are {self.persona}.",
                },
                {"role": "user", "content": input_text},
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip()
