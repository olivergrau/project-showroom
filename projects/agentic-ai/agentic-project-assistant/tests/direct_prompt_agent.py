# Test script for DirectPromptAgent class
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from agents.direct_prompt_agent import DirectPromptAgent
from agents.openai_service import OpenAIService
from config import load_openai_api_key, load_openai_base_url


def main() -> None:
    """Run a ``DirectPromptAgent`` example."""
    openai_api_key = load_openai_api_key()
    openai_base_url = load_openai_base_url()
    openai_service = OpenAIService(
        api_key=openai_api_key, base_url=openai_base_url
    )

    prompt = "What is the Capital of France?"

    direct_agent = DirectPromptAgent(openai_service=openai_service)
    direct_agent_response = direct_agent.respond(prompt)
    print(direct_agent_response)

    print(
        "DirectPromptAgent uses the OpenAI API to generate responses based on the provided prompt. The response is generated using the specified model and the agent's knowledge base, which includes general knowledge up to its last training cut-off date."
    )


if __name__ == "__main__":
    main()
