"""Entry point for Streamlit Community Cloud.

Streamlit Cloud looks for ``streamlit_app.py`` by default. We just
forward to the real dashboard module.
"""
from pathlib import Path

import streamlit as st

# Execute the main dashboard module.
exec(Path(__file__).parent.joinpath("app.py").read_text(encoding="utf-8"))
