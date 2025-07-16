from __future__ import annotations

from .base import BasicAgent, MODEL
from .openai_service import OpenAIService


class KnowledgeAugmentedPromptAgent(BasicAgent):
    """Agent that prepends domain knowledge to each prompt."""

    def __init__(self, openai_service: OpenAIService, persona: str, knowledge: str) -> None:
        super().__init__(openai_service)
        self.persona = persona
        self.knowledge = knowledge

    def respond(self, input_text: str) -> str:
        response = self.openai_service.chat(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"""
Forget all previous context.
You are {self.persona}, a knowledge-based assistant.
Use only the following knowledge to answer, do not use your own knowledge:
KNOWLEDGE: {self.knowledge} KNOWLEDGE END
Answer the prompt based on this knowledge, not your own.""",
                },
                {"role": "user", "content": input_text},
            ],
            temperature=0,
        )
        return response.choices[0].message.content
