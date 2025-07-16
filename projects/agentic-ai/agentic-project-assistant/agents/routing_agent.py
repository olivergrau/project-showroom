from __future__ import annotations

from typing import Any, List

import numpy as np

from utils.logging_config import logger

from .base import BasicAgent, MODEL, Route
from .openai_service import OpenAIService


class RoutingAgent(BasicAgent):
    """Select the best agent for a prompt based on description embeddings."""

    def __init__(self, openai_service: OpenAIService, agents: List[Route]) -> None:
        super().__init__(openai_service)
        self.agents: List[Route] = []
        for agent in agents:
            if isinstance(agent, dict):
                route = Route(
                    name=agent["name"],
                    description=agent["description"],
                    func=agent["func"],
                )
            else:
                route = agent
            route.embedding = self.get_embedding(route.description)
            self.agents.append(route)

    def get_embedding(self, text: str) -> List[float]:
        response = self.openai_service.embed(
            model="text-embedding-3-large", input=text, encoding_format="float"
        )
        return response.data[0].embedding

    def route(self, user_input: str) -> Any:
        input_emb = self.get_embedding(user_input)
        best_agent: Route | None = None
        best_score = -1.0

        for agent in self.agents:
            agent_emb = np.array(agent.embedding)
            if agent_emb.size == 0:
                logger.warning("Warning: Agent '%s' has no embedding. Skipping.", agent.name)
                continue

            similarity = np.dot(input_emb, agent_emb) / (
                np.linalg.norm(input_emb) * np.linalg.norm(agent_emb)
            )
            logger.debug(similarity)

            if similarity > best_score:
                best_score = similarity
                best_agent = agent

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        logger.info("[Router] Best agent: %s (score=%.3f)", best_agent.name, best_score)
        return best_agent.func(user_input)

    # For interface compliance
    def respond(self, prompt: str):
        return self.route(prompt)
