import os
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizerBase


def load_news_dataset(
    path: str,
    sample_n: int = 50,
    seed: int = 42,
    min_summary_chars: int = 30,
    min_headline_tokens: int | None = None,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> Dataset:
    """
    Load and preprocess the News Category Dataset for headline generation.

    Expected raw fields:
      - headline
      - short_description

    Returns a Hugging Face Dataset with columns:
      - summary: input text
      - headline: reference headline

    Args:
        path:
            File path to dataset (JSON/JSONL/CSV) or Hugging Face dataset ID.
        sample_n:
            Number of examples to sample for a fast, stable benchmark set.
        seed:
            Random seed for deterministic sampling.
        min_summary_chars:
            Filter out very short summaries.
        min_headline_tokens:
            Optional filter: keep only samples whose reference headline has
            at least this many tokens (tokenized with `tokenizer`).
            If None, no headline-length filtering is applied.
        tokenizer:
            Tokenizer used to compute headline token length.
            Required if min_headline_tokens is not None.
    """

    # -----------------------------
    # Load dataset
    # -----------------------------
    if os.path.isdir(path):
        candidates = [
            os.path.join(path, "News_Category_Dataset_v3.json"),
            os.path.join(path, "news_category_dataset.json"),
            os.path.join(path, "News_Category_Dataset.json"),
        ]
        path = next((p for p in candidates if os.path.exists(p)), path)

    if os.path.exists(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in [".json", ".jsonl"]:
            df = pd.read_json(path, lines=True)
        elif ext == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    else:
        ds = load_dataset(path)
        split = "train" if "train" in ds else list(ds.keys())[0]
        df = ds[split].to_pandas()

    # -----------------------------
    # Schema normalization
    # -----------------------------
    required_cols = {"headline", "short_description"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}. "
            f"Available columns: {sorted(df.columns)}"
        )

    df["headline"] = df["headline"].astype(str).str.strip()
    df["short_description"] = df["short_description"].astype(str).str.strip()

    # -----------------------------
    # Basic filtering
    # -----------------------------
    df = df[
        (df["headline"].str.len() > 0)
        & (df["short_description"].str.len() >= min_summary_chars)
    ]

    # -----------------------------
    # Optional headline-length filtering (token-based)
    # -----------------------------
    if min_headline_tokens is not None:
        if tokenizer is None:
            raise ValueError(
                "tokenizer must be provided when min_headline_tokens is set"
            )

        def headline_len_ok(text: str) -> bool:
            return len(tokenizer(text, add_special_tokens=False).input_ids) >= min_headline_tokens

        df = df[df["headline"].apply(headline_len_ok)]

    # -----------------------------
    # Final columns
    # -----------------------------
    out = df.rename(columns={"short_description": "summary"})[
        ["summary", "headline"]
    ].copy()

    # -----------------------------
    # Deterministic sampling
    # -----------------------------
    if sample_n is not None and len(out) > sample_n:
        out = out.sample(n=sample_n, random_state=seed).reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)

    return Dataset.from_pandas(out, preserve_index=False)
