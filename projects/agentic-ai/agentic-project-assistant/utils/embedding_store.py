import pandas as pd
import numpy as np
from typing import List, Dict


class EmbeddingStore:
    """Persist and load text chunks and their embeddings using CSV files."""

    def __init__(self, name: str):
        self.chunks_file = f"chunks-{name}.csv"
        self.embeddings_file = f"embeddings-{name}.csv"

    def save_chunks(self, chunks: List[Dict]) -> None:
        df = pd.DataFrame([{"text": c["text"], "chunk_size": c["chunk_size"]} for c in chunks])
        df.to_csv(self.chunks_file, encoding="utf-8", index=False)

    def load_chunks(self) -> pd.DataFrame:
        return pd.read_csv(self.chunks_file, encoding="utf-8")

    def save_embeddings(self, df: pd.DataFrame) -> None:
        df.to_csv(self.embeddings_file, encoding="utf-8", index=False)

    def load_embeddings(self) -> pd.DataFrame:
        df = pd.read_csv(self.embeddings_file, encoding="utf-8")
        df["embeddings"] = df["embeddings"].apply(lambda x: np.array(eval(x)))
        return df
