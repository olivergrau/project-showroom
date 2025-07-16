from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List

from .openai_service import OpenAIService


class OpenAIModel(str, Enum):
    GPT_35_TURBO = "gpt-3.5-turbo"
    GPT_41 = "gpt-4.1"
    GPT_41_MINI = "gpt-4.1-mini"
    GPT_41_NANO = "gpt-4.1-nano"


MODEL = OpenAIModel.GPT_35_TURBO


class BasicAgent(ABC):
    """Base interface for all agents."""

    def __init__(self, openai_service: OpenAIService) -> None:
        self.openai_service = openai_service

    @abstractmethod
    def respond(self, prompt: str):
        """Return the agent's response to ``prompt``."""
        raise NotImplementedError


@dataclass
class Route:
    """Represents a routing option for ``RoutingAgent``."""

    name: str
    description: str
    func: Callable[[str], str]
    embedding: List[float] = field(default_factory=list)

