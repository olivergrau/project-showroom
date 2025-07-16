import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.rag_knowledge_prompt_agent import RAGKnowledgePromptAgent
from utils.text_chunker import TextChunker
from utils.embedding_store import EmbeddingStore
from agents.openai_service import OpenAIService
from config import load_openai_api_key, load_openai_base_url


def main() -> None:
    """Demonstrate ``RAGKnowledgePromptAgent`` usage."""
    openai_api_key = load_openai_api_key()
    openai_base_url = load_openai_base_url()
    openai_service = OpenAIService(
        api_key=openai_api_key, base_url=openai_base_url
    )

    chunk_size = 1000  # Define the size of each chunk
    persona = "You are a college professor, your answer always starts with: Dear students,"

    chunker = TextChunker(chunk_size=chunk_size)
    store = EmbeddingStore("test_rag")
    RAG_knowledge_prompt_agent = RAGKnowledgePromptAgent(
        openai_service=openai_service,
        persona=persona,
        chunker=chunker,
        store=store,
    )

    knowledge_text = """
In the historic city of Boston, Clara, a marine biologist and science communicator, began each morning analyzing sonar data to track whale migration patterns along the Atlantic coast.
She spent her afternoons in a university lab, researching CRISPR-based gene editing to restore coral reefs damaged by ocean acidification and warming.
Clara was the daughter of Ukrainian immigrants—Olena and Mykola—who fled their homeland in the late 1980s after the Chernobyl disaster brought instability and fear to their quiet life near Kyiv.

Her father, Mykola, had been a radio engineer at a local observatory, skilled in repairing Soviet-era radio telescopes and radar systems that tracked both weather patterns and cosmic noise.
He often told Clara stories about jury-rigging radio antennas during snowstorms and helping amateur astronomers decode signals from distant pulsars.
Her mother, Olena, was a physics teacher with a hidden love for poetry and dissident literature. In the evenings, she would read from both Ukrainian folklore and banned Western science fiction.
They survived harsh winters, electricity blackouts, and the collapse of the Soviet economy, but always prioritized education and storytelling in their home.
Clara’s childhood was shaped by tales of how her parents shared soldering irons with neighbors, built makeshift telescopes, and taught physics to students with no textbooks but endless curiosity.

Inspired by their resilience and thirst for knowledge, Clara created a podcast called **"Crosscurrents"**, a show that explored the intersection of science, culture, and ethics.
Each week, she interviewed researchers, engineers, artists, and activists—from marine ecologists and AI ethicists to digital archivists preserving endangered languages.
Topics ranged from brain-computer interfaces, neuroplasticity, and climate migration to LLM prompt engineering, decentralized identity, and indigenous knowledge systems.
In one popular episode, she explored how retrieval-augmented generation (RAG) could help scientific researchers find niche studies buried in decades-old journals.
In another, she interviewed a Ukrainian linguist about preserving dialects lost during the Soviet era, drawing parallels to language loss in marine mammal populations.

Clara also used her technical skills to build Python-based dashboards that visualized ocean temperature anomalies and biodiversity loss, often collaborating with her best friend Amir, a data engineer working on smart city infrastructure.
Together, they discussed smart grids, blockchain for sustainability, quantum encryption, and misinformation detection in synthetic media.
At a dockside café near Boston Harbor, they often debated the ethical implications of generative AI, autonomous weapons, and the carbon footprint of LLM training runs.

In quieter moments, Clara translated traditional Ukrainian embroidery patterns into generative AI art, donating proceeds to digital archives preserving Eastern European culture.
She contributed to open-source projects involving semantic search, vector databases, and multimodal embeddings—often experimenting with few-shot learning and graph-based retrieval techniques to improve her podcast's episode discovery engine.

One night, while sharing homemade borscht, Clara told Amir how her grandparents once used Morse code to transmit encrypted weather updates through the Carpathian Mountains during World War II.
The story sparked a conversation about ancient navigation, space weather interference with submarine cables, and the neuroscience behind why humans create myths to understand uncertainty.

To Clara, knowledge was a living system—retrieved from the past, generated in the present, and evolving toward the future.
Her life and work were testaments to the power of connecting across disciplines, borders, and generations—exactly the kind of story that RAG models were born to find.
"""

    RAG_knowledge_prompt_agent.chunk_text(knowledge_text)
    print("Knowledge text chunked and embeddings calculated.")

    RAG_knowledge_prompt_agent.calculate_embeddings()
    print("Embeddings calculated for knowledge text chunks.")

    prompt = "What is the podcast that Clara hosts about?"
    print(f"Prompt: {prompt}")

    rag_knowledge_agent_response = RAG_knowledge_prompt_agent.find_prompt_in_knowledge(prompt)

    print("Response from RAGKnowledgePromptAgent:")
    print(rag_knowledge_agent_response)


if __name__ == "__main__":
    main()
