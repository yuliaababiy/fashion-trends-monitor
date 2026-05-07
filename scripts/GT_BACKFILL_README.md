# Google Trends backfill (run later via VPN)

The Google Trends public endpoint is currently rate-limited from this IP
(`pytrends` returns `RetryError(... HTTPSConnectionPool(host='trends.google.com'))`
after exhausting retries). All other parts of the pipeline work without
it — internal topic timeseries are already the primary ground truth.

## When you have a VPN / different IP, run:

```powershell
.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "."

# extra fashion keywords for richer trend coverage
python scripts/backfill_google_trends.py `
  --timeframe "today 5-y" `
  --keywords "streetwear" "gorpcore" "barbiecore" "coquette" `
              "coastal grandmother" "office siren" "mob wife" "tomato girl"
```

The collector (`src/collect/google_trends.py`) already:

- fetches **one keyword at a time** (single-keyword payloads are far
  less likely to be 429-blocked),
- retries each keyword up to 5× with exponential backoff (15s → 240s),
- sleeps 8s between successful fetches,
- merges new rows into the existing `data/raw/google_trends.parquet`
  without losing what's already cached.

After it finishes, re-run the downstream pipeline so the new keywords
flow through the dashboards & backtest:

```powershell
python -m src.eval.metrics
python -m src.eval.lead_time --horizon 8
python -m src.eval.case_studies
```
