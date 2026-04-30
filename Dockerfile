FROM python:3.12-slim

WORKDIR /app

# Minimal deps for the bot (no torch/spacy/gensim needed for polling).
COPY requirements-bot.txt ./
RUN pip install --no-cache-dir -r requirements-bot.txt

# Copy the bot source and (read-only) data the commands need.
COPY src/ ./src/
COPY reports/ ./reports/
COPY data/raw/ ./data/raw/

ENV PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8

CMD ["python", "-m", "src.alerts.telegram", "--poll-forever"]
