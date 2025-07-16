from __future__ import annotations

from typing import List

from .base import BasicAgent, MODEL
from .openai_service import OpenAIService


class ActionPlanningAgent(BasicAgent):
    """Agent that extracts an ordered action plan from a prompt."""

    def __init__(self, openai_service: OpenAIService, knowledge: str) -> None:
        super().__init__(openai_service)
        self.knowledge = knowledge

    def extract_steps_from_prompt(self, prompt: str) -> List[str]:
        response = self.openai_service.chat(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"""
                                You are an action planning agent tasked with creating a clear, ordered sequence of instructions for completing a product development task.

Your output must:
– Be a numbered list (e.g., 1., 2., 3.)
– Follow this strict order:
   1. Product Manager → 2. Program Manager → 3. Development Engineer
– Each numbered step must be a **single sentence**, beginning with: 'You as the [role] should...'

Only use these roles:
• Product Manager: Defines user stories from the product spec.
• Program Manager: Extracts features from user stories.
• Development Engineer: Translates features into development tasks.

Do not create role headers. Do not split instructions across lines.
Each item in the list must be a full, standalone directive.

Use this knowledge: {self.knowledge}
""",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        response_text = response.choices[0].message.content.strip()
        steps = [step.strip() for step in response_text.split("\n") if step.strip() and not step.startswith("Step")]
        return steps

    # For interface compliance
    def respond(self, prompt: str):
        return self.extract_steps_from_prompt(prompt)
