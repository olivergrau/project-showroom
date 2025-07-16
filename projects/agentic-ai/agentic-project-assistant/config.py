import os
from dotenv import load_dotenv


def load_openai_api_key() -> str:
    """Return the OpenAI API key from environment variables or a .env file."""
    load_dotenv()
    return os.getenv("OPENAI_API_KEY")


def load_openai_base_url() -> str:
    """Return the OpenAI base URL, defaulting to Vocareum's endpoint."""
    load_dotenv()
    return os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1")
