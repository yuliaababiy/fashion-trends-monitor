"""Configuration constants for the fashion trend prediction project."""
from __future__ import annotations

from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
MODELS_DIR = PROJECT_ROOT / "models"

for _d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Reddit collection defaults (legacy, kept for backward compat) ---
DEFAULT_SUBREDDITS = [
    "femalefashionadvice",
    "malefashionadvice",
    "streetwear",
    "frugalmalefashion",
    "OUTFITS",
]

# --- Fashion trend keywords for Google Trends / News collectors ---
FASHION_KEYWORDS = [
    # Aesthetics & subcultures
    "y2k fashion", "quiet luxury", "cottagecore", "dark academia",
    "coastal grandmother", "mob wife aesthetic", "balletcore", "gorpcore",
    "preppy style", "old money aesthetic", "clean girl aesthetic",
    "indie sleaze", "blokecore", "twee fashion",
    # Garments & trends
    "oversized blazer", "cargo pants", "slip dress", "wide leg jeans",
    "platform shoes", "leather jacket",
]

# Time window for historical analysis (years back from today)
HISTORY_YEARS = 3

# --- Topic modeling ---
LDA_NUM_TOPICS_RANGE = (5, 30)   # search range for LDA num_topics
BERTOPIC_MIN_TOPIC_SIZE = 30
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Forecasting ---
TIME_FREQ = "W"  # weekly aggregation
FORECAST_HORIZON = 8  # weeks ahead
TEST_SIZE = 12       # last 12 weeks for testing

# --- Random seed ---
RANDOM_STATE = 42
