"""LDA topic modeling with gensim.

- Trains LDA models for a range of topic counts.
- Picks the best by c_v coherence.
- Saves: trained model, dictionary, topic-keywords table, doc-topic assignments.
- Generates pyLDAvis HTML visualization.

Usage:
    python -m src.topics.run_lda --input data/interim/clean.parquet --topics 8 12 16 20
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import pandas as pd

from src.config import FIGURES_DIR, INTERIM_DIR, MODELS_DIR, PROCESSED_DIR, RANDOM_STATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def build_corpus(token_strings: list[str], no_below: int = 5, no_above: float = 0.5):
    from gensim.corpora import Dictionary

    tokenized = [t.split() for t in token_strings]
    dictionary = Dictionary(tokenized)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized]
    return tokenized, corpus, dictionary


def train_lda(corpus, dictionary, num_topics: int, passes: int = 10):
    from gensim.models import LdaMulticore

    return LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        random_state=RANDOM_STATE,
        workers=2,
        chunksize=2000,
    )


def coherence(model, tokenized, dictionary) -> float:
    from gensim.models import CoherenceModel

    cm = CoherenceModel(
        model=model, texts=tokenized, dictionary=dictionary, coherence="c_v"
    )
    return float(cm.get_coherence())


def topic_keywords(model, topn: int = 15) -> pd.DataFrame:
    rows = []
    for tid in range(model.num_topics):
        terms = model.show_topic(tid, topn=topn)
        rows.append({
            "topic_id": tid,
            "keywords": ", ".join(w for w, _ in terms),
            "weights": [float(p) for _, p in terms],
        })
    return pd.DataFrame(rows)


def assign_topics(model, corpus) -> pd.DataFrame:
    """Return dominant topic + probability per document."""
    rows = []
    for bow in corpus:
        dist = model.get_document_topics(bow, minimum_probability=0.0)
        if dist:
            tid, p = max(dist, key=lambda x: x[1])
        else:
            tid, p = -1, 0.0
        rows.append({"topic_id": int(tid), "topic_prob": float(p)})
    return pd.DataFrame(rows)


def save_pyldavis(model, corpus, dictionary, out_html: Path) -> None:
    try:
        import pyLDAvis
        import pyLDAvis.gensim_models as gensimvis
    except ImportError:
        log.warning("pyLDAvis not installed; skipping visualization.")
        return
    vis = gensimvis.prepare(model, corpus, dictionary)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    pyLDAvis.save_html(vis, str(out_html))
    log.info("Saved pyLDAvis to %s", out_html)


# ---------------------------------------------------------------------------
def transform_with_existing(model_path: Path, df: pd.DataFrame, output_prefix: str) -> None:
    """Load saved {model, dictionary}, assign topics for the given docs, save."""
    log.info("Transform-only mode: loading %s", model_path)
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    dictionary = bundle["dictionary"]

    tokenized = [t.split() for t in df["tokens"].tolist()]
    corpus = [dictionary.doc2bow(doc) for doc in tokenized]

    assignments = assign_topics(model, corpus)
    out_df = pd.concat([df.reset_index(drop=True), assignments], axis=1)
    out_path = PROCESSED_DIR / f"{output_prefix}_doc_topics.parquet"
    out_df.to_parquet(out_path, index=False)
    log.info("Saved doc-topic assignments to %s (%d rows)", out_path, len(out_df))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INTERIM_DIR / "clean.parquet")
    parser.add_argument(
        "--topics", type=int, nargs="+", default=[8, 12, 16, 20],
        help="Candidate numbers of topics to evaluate.",
    )
    parser.add_argument("--passes", type=int, default=10)
    parser.add_argument("--output-prefix", type=str, default="lda")
    parser.add_argument("--transform-only", type=Path, default=None,
                        help="Skip training. Use saved model at this path to "
                             "assign topics for the input documents.")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    log.info("Loaded %d documents", len(df))

    if args.transform_only:
        transform_with_existing(args.transform_only, df, args.output_prefix)
        return

    tokenized, corpus, dictionary = build_corpus(df["tokens"].tolist())
    log.info("Vocabulary size after filtering: %d", len(dictionary))

    results = []
    best = None
    for k in args.topics:
        log.info("Training LDA k=%d", k)
        model = train_lda(corpus, dictionary, num_topics=k, passes=args.passes)
        score = coherence(model, tokenized, dictionary)
        log.info("  coherence (c_v) = %.4f", score)
        results.append({"num_topics": k, "coherence_cv": score})
        if best is None or score > best["score"]:
            best = {"k": k, "score": score, "model": model}

    coh_df = pd.DataFrame(results).sort_values("num_topics")
    coh_df.to_csv(PROCESSED_DIR / f"{args.output_prefix}_coherence.csv", index=False)
    log.info("Coherence by k:\n%s", coh_df.to_string(index=False))

    log.info("Best k = %d (coherence = %.4f)", best["k"], best["score"])
    model = best["model"]

    # Save model + dict
    model_path = MODELS_DIR / f"{args.output_prefix}_k{best['k']}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "dictionary": dictionary}, f)
    log.info("Saved model to %s", model_path)

    # Topic keywords
    kw = topic_keywords(model)
    kw.to_csv(PROCESSED_DIR / f"{args.output_prefix}_topics.csv", index=False)

    # Doc assignments
    assignments = assign_topics(model, corpus)
    out_df = pd.concat([df.reset_index(drop=True), assignments], axis=1)
    out_path = PROCESSED_DIR / f"{args.output_prefix}_doc_topics.parquet"
    out_df.to_parquet(out_path, index=False)
    log.info("Saved doc-topic assignments to %s", out_path)

    # Visualization
    save_pyldavis(
        model, corpus, dictionary,
        FIGURES_DIR / f"{args.output_prefix}_k{best['k']}.html",
    )

    # Summary
    summary = {
        "best_k": best["k"],
        "best_coherence_cv": best["score"],
        "vocabulary_size": len(dictionary),
        "n_documents": len(df),
    }
    (PROCESSED_DIR / f"{args.output_prefix}_summary.json").write_text(
        json.dumps(summary, indent=2)
    )


if __name__ == "__main__":
    main()
