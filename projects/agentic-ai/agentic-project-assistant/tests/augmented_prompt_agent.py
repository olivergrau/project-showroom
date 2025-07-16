import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the AugmentedPromptAgent class
from agents.augmented_prompt_agent import AugmentedPromptAgent
from agents.openai_service import OpenAIService
from config import load_openai_api_key, load_openai_base_url


def main() -> None:
    """Run a simple ``AugmentedPromptAgent`` example."""
    openai_api_key = load_openai_api_key()
    openai_base_url = load_openai_base_url()
    openai_service = OpenAIService(
        api_key=openai_api_key, base_url=openai_base_url
    )

    prompt = "What is the capital of France?"
    persona = (
        "You are a college professor; your answers always start with: 'Dear students,'"
    )

    agent = AugmentedPromptAgent(openai_service=openai_service, persona=persona)

    augmented_agent_response = agent.respond(prompt)
    print(augmented_agent_response)

    # The agent's persona leads to a formal answer such as
    # "Dear students, the capital of France is Paris."
    print(
        "The agent's response is expected to be a formal answer, such as 'Dear students, the capital of France is Paris.'"
    )
    print(
        "This is because the agent's persona is set to that of a college professor, which influences"
    )
    print("the style and tone of the response.")


if __name__ == "__main__":
    main()

