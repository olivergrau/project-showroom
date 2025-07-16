import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.knowledge_augmented_prompt_agent import KnowledgeAugmentedPromptAgent
from agents.routing_agent import RoutingAgent
from agents.openai_service import OpenAIService
from config import load_openai_api_key, load_openai_base_url


def main() -> None:
    """Run a ``RoutingAgent`` demonstration."""
    openai_api_key = load_openai_api_key()
    openai_base_url = load_openai_base_url()
    openai_service = OpenAIService(
        api_key=openai_api_key, base_url=openai_base_url
    )

    persona = "You are a college professor"
    knowledge = "You know everything about Texas"
    texas_agent = KnowledgeAugmentedPromptAgent(
        openai_service=openai_service, persona=persona, knowledge=knowledge
    )

    knowledge = "You know everything about Europe"
    europe_agent = KnowledgeAugmentedPromptAgent(
        openai_service=openai_service, persona=persona, knowledge=knowledge
    )

    persona = "You are a college math professor"
    knowledge = (
        "You know everything about math, you take prompts with numbers, extract math formulas, and show the answer without explanation"
    )
    math_agent = KnowledgeAugmentedPromptAgent(
        openai_service=openai_service, persona=persona, knowledge=knowledge
    )

    routing_agent = RoutingAgent(
        openai_service=openai_service,
        agents=[
            {
                "name": "texas agent",
                "description": "Answer a question about Texas",
                "func": lambda x: texas_agent.respond(x),
            },
            {
                "name": "europe agent",
                "description": "Answer a question about Europe",
                "func": lambda x: europe_agent.respond(x),
            },
            {
                "name": "math agent",
                "description": "When a prompt contains numbers, respond with a math formula",
                "func": lambda x: math_agent.respond(x),
            },
        ],
    )

    prompts = [
        "Tell me about the history of Rome, Texas",
        "Tell me about the history of Rome, Italy",
        "One story takes 2 days, and there are 20 stories",
    ]
    for prompt in prompts:
        response = routing_agent.route(prompt)
        print(f"Prompt: {prompt}\nResponse: {response}\n")
    print("✅ Routing completed successfully.")


if __name__ == "__main__":
    main()
