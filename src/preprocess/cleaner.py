"""Text preprocessing for Reddit fashion posts.

Steps:
1. Combine title + selftext.
2. Remove URLs, markdown, HTML, control characters.
3. Demojize emoji to text tokens.
4. Filter to English posts (langdetect).
5. Lemmatize with spaCy, drop stopwords / punctuation / short tokens.
6. Save cleaned text alongside original metadata.

Usage:
    python -m src.preprocess.cleaner \
        --input data/raw/posts.parquet \
        --output data/interim/clean.parquet
"""
from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from tqdm import tqdm

from src.config import INTERIM_DIR, RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")  # [text](url) -> text
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTISPACE_RE = re.compile(r"\s+")
_NONALPHA_RE = re.compile(r"[^a-zA-Z\s#]")  # keep hashtags

# Custom fashion-domain stopwords on top of spaCy defaults.
EXTRA_STOPWORDS = {
    "outfit", "look", "style", "wear", "wearing", "buy", "bought", "thanks",
    "post", "advice", "thought", "question", "help", "guys", "girl", "lol",
    "edit", "update", "deleted", "removed", "amp",
}


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean_raw_text(text: str) -> str:
    """Remove URLs, markdown links, HTML, demojize emoji."""
    if not isinstance(text, str):
        return ""
    import emoji

    text = _MD_LINK_RE.sub(r"\1", text)
    text = _URL_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = emoji.demojize(text, delimiters=(" :", ": "))
    text = text.replace("\n", " ").replace("\r", " ")
    text = _MULTISPACE_RE.sub(" ", text).strip()
    return text


def detect_language_safe(text: str) -> str:
    from langdetect import DetectorFactory, LangDetectException, detect
    DetectorFactory.seed = 0
    if not text or len(text) < 20:
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def load_spacy():
    import spacy
    try:
        return spacy.load("en_core_web_sm", disable=["ner", "parser"])
    except OSError as e:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' not found. Install with:\n"
            "    python -m spacy download en_core_web_sm"
        ) from e


def lemmatize_batch(texts: Iterable[str], nlp, batch_size: int = 64) -> list[str]:
    """Lemmatize, lowercase, drop stopwords/punct/short tokens. Keep hashtags."""
    out: list[str] = []
    for doc in nlp.pipe(texts, batch_size=batch_size):
        tokens: list[str] = []
        for tok in doc:
            if tok.is_space or tok.is_punct or tok.like_num:
                continue
            text = tok.text.lower()
            if text.startswith("#") and len(text) > 2:
                tokens.append(text)
                continue
            lemma = tok.lemma_.lower().strip()
            if not lemma or len(lemma) < 3:
                continue
            if tok.is_stop or lemma in EXTRA_STOPWORDS:
                continue
            if not lemma.isalpha():
                continue
            tokens.append(lemma)
        out.append(" ".join(tokens))
    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def preprocess(df: pd.DataFrame, min_tokens: int = 5) -> pd.DataFrame:
    log.info("Combining title + selftext for %d rows", len(df))
    df = df.copy()
    df["raw_text"] = (df["title"].fillna("") + ". " + df["selftext"].fillna("")).str.strip()

    log.info("Cleaning raw text")
    df["clean_text"] = [clean_raw_text(t) for t in tqdm(df["raw_text"], desc="clean")]

    log.info("Detecting language")
    df["lang"] = [detect_language_safe(t) for t in tqdm(df["clean_text"], desc="langdetect")]
    before = len(df)
    df = df[df["lang"] == "en"].reset_index(drop=True)
    log.info("Kept %d / %d English posts", len(df), before)

    log.info("Lemmatizing with spaCy")
    nlp = load_spacy()
    df["tokens"] = lemmatize_batch(df["clean_text"].tolist(), nlp)
    df["n_tokens"] = df["tokens"].str.split().str.len().fillna(0).astype(int)

    before = len(df)
    df = df[df["n_tokens"] >= min_tokens].reset_index(drop=True)
    log.info("Kept %d / %d posts with >= %d tokens", len(df), before, min_tokens)

    df["date"] = pd.to_datetime(df["created_utc"]).dt.tz_convert("UTC").dt.tz_localize(None)
    df["week"] = df["date"].dt.to_period("W").dt.start_time
    df["month"] = df["date"].dt.to_period("M").dt.start_time
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Clean and lemmatize Reddit posts.")
    parser.add_argument("--input", type=Path, default=RAW_DIR / "posts.parquet")
    parser.add_argument("--output", type=Path, default=INTERIM_DIR / "clean.parquet")
    parser.add_argument("--min-tokens", type=int, default=5)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_parquet(args.input)
    log.info("Loaded %d rows from %s", len(df), args.input)

    cleaned = preprocess(df, min_tokens=args.min_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(args.output, index=False)
    log.info("Saved %d cleaned rows to %s", len(cleaned), args.output)


if __name__ == "__main__":
    main()
