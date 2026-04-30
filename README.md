# Fashion Trend Monitor

Автоматизована система моніторингу модних трендів у відкритих джерелах
для курсової роботи: *«Прогнозування модних трендів у соціальних
мережах із використанням тематичного моделювання, часових рядів та
алгоритмів штучного інтелекту»*.

> Збір → очищення → тематичне моделювання (LDA) → прогнозування (Naive,
> MA, ARIMA/SARIMA, Prophet, XGBoost, LSTM) → ранжування трендів →
> Telegram-сповіщення → Streamlit-дашборд.

## Архітектура

```
┌──────────────────────────────────────────────────────────────────┐
│  GitHub Actions                                                  │
│   ├─ daily.yml      07:00 UTC  → src.monitor.daily               │
│   ├─ weekly.yml     Sun 20:00  → src.monitor.daily --retrain-lda │
│   └─ bot.yml        */5 min    → src.alerts.telegram --poll-once │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────┐
│  Pipeline                                                         │
│   collect (Guardian + NewsAPI + Reddit JSON + Mastodon)           │
│        ↓ (incremental, dedup by id)                               │
│   preprocess (cleaner: langdetect + spaCy lemmatization)          │
│        ↓                                                           │
│   topics  (LDA — daily transform / weekly retrain)                │
│        ↓                                                           │
│   timeseries (weekly per-topic counts)                            │
│        ↓                                                           │
│   forecast (7 models, MAE/RMSE/MAPE/sMAPE)                        │
│        ↓                                                           │
│   emerging (momentum + forecast %, status: Rising/Stable/Decline) │
│        ↓                                                           │
│   diff vs yesterday → Telegram alert                              │
└──────────────────────────────────────────────────────────────────┘
        │
        ▼
   Streamlit Cloud  (4-tab dashboard, app.py)
```

## Стек

Python 3.12 · pandas · pyarrow · spaCy 3.8 + en_core_web_sm ·
gensim 4.4 (LDA) · statsmodels (ARIMA/SARIMA) · prophet · xgboost ·
torch (LSTM) · scikit-learn · plotly · streamlit · pytrends.

## Джерела даних

| Джерело        | Тип                              | Авторизація    |
|----------------|----------------------------------|----------------|
| The Guardian   | статті розділу `fashion`         | API key (free) |
| News API       | новини з ~150 тис. джерел        | API key (free, 30 днів) |
| Reddit         | публічні JSON-ендпоінти          | без auth       |
| Mastodon       | public hashtag timelines         | без auth       |
| Google Trends  | weekly interest, 5 років (опц.)  | без auth       |

## Структура

```
src/
├── collect/        guardian.py, newsapi.py, reddit_json.py,
│                    mastodon.py, google_trends.py, combine.py
├── preprocess/     cleaner.py
├── topics/         run_lda.py, build_timeseries.py
├── forecast/       models.py, metrics.py, run_all.py, run_trends.py
├── analysis/       emerging.py
├── monitor/        daily.py
├── alerts/         telegram.py
├── config.py
└── run_pipeline.py
data/{raw,interim,processed}/   parquet артефакти
models/lda_k*.pkl               збережені LDA-моделі
reports/metrics/                MAE/RMSE/sMAPE + emerging_*.csv
reports/state/                  bot offset, mute list, yesterday snapshot
.github/workflows/              daily.yml, weekly.yml, bot.yml
app.py · streamlit_app.py       Streamlit-дашборд
```

## Локальний запуск

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 1) Скопіюйте .env.example → .env, заповніть ключі.
# 2) Перший повний прогін:
$env:PYTHONIOENCODING = 'utf-8'
python -m src.run_pipeline                   # повний прогін
# або тільки моніторинг (швидше, інкрементально):
python -m src.monitor.daily --no-alerts

# 3) Дашборд:
streamlit run app.py
```

## Telegram-бот

Команди:

| Команда       | Опис                                        |
|---------------|---------------------------------------------|
| `/start`      | привітання + список команд                  |
| `/trends`     | топ Rising-теми                             |
| `/topic <id>` | деталі по темі                              |
| `/status`     | свіжість парquet + дата останнього прогнозу |
| `/run`        | запустити `daily.yml` зараз                 |
| `/retrain`    | запустити `weekly.yml`                      |
| `/mute`/`/unmute` | вимкнути/увімкнути сповіщення для чату  |

Доступ обмежено білим списком `TELEGRAM_ALLOWED_CHATS` (chat-id'и з
`.env`/Secrets).

## GitHub Secrets

Налаштувати у *Settings → Secrets and variables → Actions*:

| Назва                    | Значення                                            |
|--------------------------|-----------------------------------------------------|
| `GUARDIAN_API_KEY`       | ключ з open-platform.theguardian.com                |
| `NEWSAPI_KEY`            | ключ з newsapi.org                                  |
| `TELEGRAM_TOKEN`         | токен з @BotFather                                  |
| `TELEGRAM_ALLOWED_CHATS` | `id1,id2` — кому дозволено писати боту              |
| `GH_PAT`                 | Personal Access Token зі скоупом `workflow`         |

## Streamlit Cloud

1. Створити app на share.streamlit.io з цього репо, файл `streamlit_app.py`.
2. Дашборд читає parquet/csv безпосередньо з гілки — після кожного
   `git push` від workflow дашборд оновлюється автоматично.
