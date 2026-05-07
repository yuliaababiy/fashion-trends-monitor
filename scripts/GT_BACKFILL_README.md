# Google Trends backfill

Google Trends банить **по IP**, не по акаунту. З поточної IP `pytrends`
повертає `RetryError(... HTTPSConnectionPool(host='trends.google.com'))`
після всіх ретраїв.

Решта пайплайну від цього не страждає — внутрішні `topic_timeseries` є
основним джерелом ground truth у бектесті.

## Варіант 1 — через VPN

Просто підніми VPN на іншу країну і запусти:

```powershell
.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "."

python scripts/backfill_google_trends.py `
  --timeframe "today 5-y" `
  --keywords "streetwear" "gorpcore" "barbiecore" "coquette" `
              "coastal grandmother" "office siren" "mob wife" "tomato girl"
```

## Варіант 2 — через HTTP/SOCKS проксі

Скрипт уміє ротувати кілька проксі (передаєш `--proxy` стільки разів,
скільки треба, pytrends чергує їх між запитами):

```powershell
python scripts/backfill_google_trends.py `
  --timeframe "today 5-y" `
  --proxy "https://user:pass@proxy1.example.com:8080" `
  --proxy "https://user:pass@proxy2.example.com:8080" `
  --keywords "streetwear" "gorpcore" "barbiecore"
```

Що працює та що ні:

- Residential / mobile проксі (Bright Data, Smartproxy, Oxylabs,
  IPRoyal, або просто мобільний хотспот) — Google зазвичай пропускає.
- Datacenter проксі (DigitalOcean, AWS, Hetzner) — у 90% випадків
  теж видають 429: ці підмережі давно у блок-листі Google.
- Безкоштовні публічні проксі — переважно вже забанені скрейперами
  до тебе.
- Формат: `http://...`, `https://...`, або `socks5://...`. Якщо
  проксі з логіном — `https://user:pass@host:port`.

Якщо один з проксі сам забанений, у логах побачиш `429` саме на ньому;
pytrends просто перейде на наступний у списку.

## Що collector уже робить, аби не дратувати API

- запитує по **одному** ключу за раз (single-keyword payloads майже не
  ловлять 429),
- 5 ретраїв з експоненційним backoff: 15s → 30s → 60s → 120s → 240s,
- 8 секунд паузи між успішними запитами,
- зливає нові дані з існуючим `data/raw/google_trends.parquet` без
  втрати того, що вже є в кеші.

## Після успішного збору — переган метрик

```powershell
python -m src.eval.metrics
python -m src.eval.lead_time --horizon 8
python -m src.eval.case_studies
```
