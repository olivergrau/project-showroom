from __future__ import annotations

import uuid
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd

from .base import BasicAgent, MODEL
from .openai_service import OpenAIService
from utils.logging_config import logger
from utils.text_chunker import TextChunker
from utils.embedding_store import EmbeddingStore


class RAGKnowledgePromptAgent(BasicAgent):
    """Agent that uses RAG to respond to prompts based on retrieved knowledge."""

    def __init__(
        self,
        openai_service: OpenAIService,
        persona: str,
        chunk_size: int = 2000,
        chunk_overlap: int = 100,
        chunker: TextChunker | None = None,
        store: EmbeddingStore | None = None,
    ) -> None:
        super().__init__(openai_service)
        self.persona = persona
        self.chunker = chunker or TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.store = store or EmbeddingStore(filename)

    def get_embedding(self, text: str) -> List[float]:
        response = self.openai_service.embed(
            model="text-embedding-3-large", input=text, encoding_format="float"
        )
        return response.data[0].embedding

    def calculate_similarity(self, vector_one: List[float], vector_two: List[float]) -> float:
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def chunk_text(self, text: str) -> List[dict]:
        chunks = self.chunker.chunk(text)
        self.store.save_chunks(chunks)
        return chunks

    def calculate_embeddings(self) -> pd.DataFrame:
        df = self.store.load_chunks()
        df["embeddings"] = df["text"].apply(self.get_embedding)
        self.store.save_embeddings(df)
        return df

    def find_prompt_in_knowledge(self, prompt: str) -> str:
        prompt_embedding = self.get_embedding(prompt)
        df = self.store.load_embeddings()
        df["similarity"] = df["embeddings"].apply(
            lambda emb: self.calculate_similarity(prompt_embedding, emb)
        )
        best_chunk = df.loc[df["similarity"].idxmax(), "text"]
        response = self.openai_service.chat(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"Forget previous context. You are {self.persona}, a knowledge-based assistant.",
                },
                {
                    "role": "user",
                    "content": f"Answer based only on this information: {best_chunk}. Prompt: {prompt}",
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    # For interface compliance
    def respond(self, prompt: str) -> str:
        return self.find_prompt_in_knowledge(prompt)
