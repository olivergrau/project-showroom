from __future__ import annotations

from typing import Any

from utils.logging_config import logger

from .base import BasicAgent, MODEL
from .openai_service import OpenAIService


class EvaluationAgent(BasicAgent):
    """Agent that evaluates responses from another worker agent."""

    def __init__(
        self,
        openai_service: OpenAIService,
        persona: str,
        evaluation_criteria: str,
        worker_agent: BasicAgent,
        max_interactions: int = 10,
    ) -> None:
        super().__init__(openai_service)
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent
        self.max_interactions = max_interactions

    def evaluate_once(self, prompt: str) -> tuple[str, str]:
        logger.info("Worker agent generating response")
        response_from_worker = self.worker_agent.respond(prompt)

        logger.info("Evaluator agent judging response")
        response = self.openai_service.chat(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are {self.persona}, an impartial evaluation agent.\n"
                        f"You must decide if the following answer meets the evaluation criterion:\n"
                        f"CRITERION: {self.evaluation_criteria}\n"
                        f"Your response must be strictly formatted:\n"
                        f"First line: 'Yes' or 'No'\n"
                        f"Second line: A one-sentence explanation why.\n"
                        f"Do not suggest improvements. Do not change or reinterpret the criterion."
                    ),
                },
                {"role": "user", "content": f"Answer: {response_from_worker}"},
            ],
            temperature=0,
        )
        evaluation = response.choices[0].message.content.strip()
        return response_from_worker, evaluation

    def iterate(self, initial_prompt: str) -> dict[str, Any]:
        prompt_to_evaluate = initial_prompt
        for i in range(self.max_interactions):
            logger.info("--- Interaction %d ---", i + 1)
            worker_response, evaluation = self.evaluate_once(prompt_to_evaluate)

            if evaluation.lower().startswith("yes"):
                logger.info("Final solution accepted")
                break

            logger.info("Generating instructions to improve answer")
            instruction_prompt = (
                f"Provide instructions to fix an answer based on these reasons why it is incorrect: {evaluation}"
            )
            response = self.openai_service.chat(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are {self.persona}, a helpful assistant that generates concise correction instructions.\n"
                            f"Based on the evaluation feedback, your job is to guide the worker on how to fix the answer.\n"
                            f"{instruction_prompt}\n"
                            f"Do not change the prompt or invent new constraints. Just explain how to better meet the criterion."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"The original answer was judged incorrect.\n"
                            f"Evaluation feedback: {evaluation}\n"
                            f"Write clear instructions for improving the answer to meet the criterion."
                        ),
                    },
                ],
                temperature=0,
            )
            instructions = response.choices[0].message.content.strip()
            prompt_to_evaluate = (
                f"The original prompt was: {initial_prompt}\n"
                f"The response to that prompt was: {worker_response}\n"
                f"It has been evaluated as incorrect.\n"
                f"Make only these corrections, do not alter content validity: {instructions}"
            )

        return {
            "final_response": worker_response,
            "evaluation": evaluation,
            "iterations": i + 1,
        }

    def evaluate(self, initial_prompt: str) -> dict[str, Any]:
        return self.iterate(initial_prompt)

    # Interface compliance
    def respond(self, prompt: str):
        return self.evaluate(prompt)
