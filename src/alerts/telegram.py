"""Telegram alerter + interactive bot.

Two modes
---------
1. **Send mode** — used by ``src.monitor.daily`` to push alerts to
   the configured chats::

       python -m src.alerts.telegram --send "your message"

2. **Bot polling mode** — handles incoming commands. Designed to be
   invoked from a short-lived GitHub Actions cron (e.g. every 5 min)
   that processes the Telegram update queue and exits::

       python -m src.alerts.telegram --poll-once

Configuration
-------------
The token and the comma-separated list of allowed chat-ids are read
from environment variables (or ``.env``):

* ``TELEGRAM_TOKEN``
* ``TELEGRAM_ALLOWED_CHATS``       (e.g. "312042781,475975048")
* ``TELEGRAM_OFFSET_FILE``         (path to persist update offset)

The bot supports the following commands::

    /start        — show greeting + command list
    /help         — same as /start
    /status       — last run summary, rows, freshness
    /trends       — top 5 Rising topics
    /topic <id>   — details + keyword list for a topic id
    /run          — trigger the daily workflow (needs ``GH_PAT``)
    /retrain      — trigger the weekly retrain workflow (needs ``GH_PAT``)
    /mute         — disable alerts for the calling chat
    /unmute       — re-enable alerts
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from src.config import METRICS_DIR, PROCESSED_DIR, PROJECT_ROOT, RAW_DIR

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("telegram")

API = "https://api.telegram.org/bot{token}/{method}"
STATE_DIR = PROJECT_ROOT / "reports" / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
OFFSET_FILE = STATE_DIR / "telegram_offset.json"
MUTED_FILE = STATE_DIR / "telegram_muted.json"


# ---------------------------------------------------------------------------
def _load_env() -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _token() -> str:
    _load_env()
    t = os.getenv("TELEGRAM_TOKEN")
    if not t:
        raise RuntimeError("TELEGRAM_TOKEN is not set.")
    return t


def _allowed_chats() -> list[int]:
    _load_env()
    raw = os.getenv("TELEGRAM_ALLOWED_CHATS", "")
    return [int(x) for x in raw.replace(" ", "").split(",") if x]


# ---------------------------------------------------------------------------
def _api_call(method: str, **payload) -> dict:
    url = API.format(token=_token(), method=method)
    r = requests.post(url, json=payload, timeout=20)
    if r.status_code != 200:
        log.warning("Telegram %s -> %s: %s", method, r.status_code, r.text[:200])
    r.raise_for_status()
    return r.json()


def send_message(
    chat_id: int,
    text: str,
    parse_mode: str = "Markdown",
    reply_markup: dict | None = None,
) -> dict:
    payload = dict(
        chat_id=chat_id, text=text,
        parse_mode=parse_mode,
        disable_web_page_preview=True,
    )
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup
    return _api_call("sendMessage", **payload)


def _dashboard_url() -> str | None:
    _load_env()
    return os.getenv("DASHBOARD_URL") or None


def _main_menu_kb(chat_id: int | None = None) -> dict:
    muted = _load_muted()
    is_muted = chat_id in muted if chat_id is not None else False
    rows = [
        [
            {"text": "🚀 Тренди", "callback_data": "menu:trends"},
            {"text": "📊 Статус", "callback_data": "menu:status"},
        ],
        [
            {"text": "▶️ Збір даних", "callback_data": "menu:run"},
            {"text": "🔁 Перенавчання", "callback_data": "menu:retrain"},
        ],
    ]
    last = []
    url = _dashboard_url()
    if url:
        last.append({"text": "🌐 Дашборд", "url": url})
    last.append(
        {"text": "🔔 Увімкнути" if is_muted else "🔕 Вимкнути",
         "callback_data": "menu:unmute" if is_muted else "menu:mute"}
    )
    rows.append(last)
    return {"inline_keyboard": rows}


def _topics_kb(topic_ids: list[int]) -> dict:
    """Inline keyboard with a button per topic id (3 per row)."""
    rows: list[list[dict]] = []
    row: list[dict] = []
    for tid in topic_ids:
        row.append({"text": f"#{tid}", "callback_data": f"topic:{tid}"})
        if len(row) == 3:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    rows.append([{"text": "⬅️ Меню", "callback_data": "menu:help"}])
    return {"inline_keyboard": rows}


def _back_kb() -> dict:
    return {"inline_keyboard": [[{"text": "⬅️ Меню", "callback_data": "menu:help"}]]}


def send_alert_to_all(text: str) -> None:
    muted = _load_muted()
    for cid in _allowed_chats():
        if cid in muted:
            log.info("chat %s is muted — skipping", cid)
            continue
        try:
            send_message(cid, text)
            log.info("Alert delivered to %s", cid)
        except Exception as e:
            log.warning("send to %s failed: %s", cid, e)


# ---------------------------------------------------------------------------
def _load_offset() -> int:
    if OFFSET_FILE.exists():
        try:
            return int(json.loads(OFFSET_FILE.read_text())["offset"])
        except Exception:
            return 0
    return 0


def _save_offset(offset: int) -> None:
    OFFSET_FILE.write_text(json.dumps({"offset": offset}))


def _load_muted() -> set[int]:
    if MUTED_FILE.exists():
        try:
            return set(int(x) for x in json.loads(MUTED_FILE.read_text()))
        except Exception:
            return set()
    return set()


def _save_muted(s: set[int]) -> None:
    MUTED_FILE.write_text(json.dumps(sorted(s)))


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------
HELP_TEXT = (
    "*👗 Моніторинг модних трендів*\n\n"
    "Оберіть дію кнопкою нижче ⬇️"
)


STATUS_UA = {"Rising": "зростає", "Declining": "спадає", "Stable": "стабільна"}


def _fmt_topic_row(r: pd.Series) -> str:
    kw = (r.get("keywords") or "")[:80]
    mp = r.get("momentum_pct")
    fp = r.get("forecast_pct")
    parts = [f"#{int(r['topic_id'])} _{kw}_"]
    if pd.notna(mp):
        parts.append(f"моментум {mp:+.0f}%")
    if pd.notna(fp):
        parts.append(f"прогноз {fp:+.0f}%")
    parts.append(f"_{STATUS_UA.get(r.get('status',''), r.get('status',''))}_")
    return "  • " + " · ".join(parts)


def cmd_help(chat_id: int, _args: str):
    return {"text": HELP_TEXT, "reply_markup": _main_menu_kb(chat_id)}


def cmd_trends(chat_id: int, _args: str):
    p = METRICS_DIR / "emerging_topics.csv"
    if not p.exists():
        return "Поки немає файлу `emerging_topics.csv` — пайплайн ще не запускався."
    df = pd.read_csv(p)
    rising = df[df["status"] == "Rising"].head(5)
    if rising.empty:
        sub = df.head(5)
        text = "Зараз немає тем, що зростають. Топ-5 за прогнозом:\n" + "\n".join(
            _fmt_topic_row(r) for _, r in sub.iterrows()
        )
    else:
        sub = rising
        text = "🚀 *Топ тем, що зростають*\n" + "\n".join(
            _fmt_topic_row(r) for _, r in sub.iterrows()
        )
    text += "\n\nНатисніть на номер теми, щоб побачити деталі ↓"
    ids = [int(x) for x in sub["topic_id"].tolist()]
    return {"text": text, "reply_markup": _topics_kb(ids)}


def cmd_topic(chat_id: int, args: str):
    args = args.strip()
    p = METRICS_DIR / "emerging_topics.csv"
    if not p.exists():
        return "Немає даних."
    df = pd.read_csv(p)
    if not args.isdigit():
        # No id provided — show picker.
        ids = [int(x) for x in df.head(15)["topic_id"].tolist()]
        return {
            "text": "Оберіть тему:",
            "reply_markup": _topics_kb(ids),
        }
    tid = int(args)
    row = df[df["topic_id"] == tid]
    if row.empty:
        return f"Тему #{tid} не знайдено."
    r = row.iloc[0]
    text = (
        f"*Тема #{tid}*\n"
        f"Статус: _{STATUS_UA.get(r.get('status',''), r.get('status','?'))}_  "
        f"· модель: `{r.get('model','?')}`\n"
        f"Останні тижні: {r.get('recent_mean','?')}, база {r.get('baseline_mean','?')}\n"
        f"Моментум: {r.get('momentum_pct','?'):+.1f}% · "
        f"Прогноз: {r.get('forecast_pct','?')}\n\n"
        f"Ключові слова: _{r.get('keywords','')}_"
    )
    return {"text": text, "reply_markup": _back_kb()}


def _file_age(p: Path) -> str:
    if not p.exists():
        return "відсутній"
    mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
    delta = datetime.now(timezone.utc) - mtime
    hrs = delta.total_seconds() / 3600
    if hrs < 1:
        return f"{int(delta.total_seconds()/60)} хв тому"
    if hrs < 48:
        return f"{hrs:.1f} год тому"
    return f"{hrs/24:.1f} дн тому"


def cmd_status(chat_id: int, _args: str):
    sources = ["guardian.parquet", "newsapi.parquet",
               "reddit.parquet", "mastodon.parquet", "google_trends.parquet"]
    lines = ["*Статус пайплайну*"]
    for f in sources:
        p = RAW_DIR / f
        n = ""
        if p.exists():
            try:
                cols = ["keyword"] if f == "google_trends.parquet" else ["id"]
                n = f" ({len(pd.read_parquet(p, columns=cols))} рядків)"
            except Exception:
                pass
        lines.append(f"  · `{f}`: {_file_age(p)}{n}")
    lines.append("")
    lines.append(f"`forecasts.parquet`: {_file_age(PROCESSED_DIR / 'forecasts.parquet')}")
    lines.append(f"`emerging_topics.csv`: {_file_age(METRICS_DIR / 'emerging_topics.csv')}")
    lines.append(f"`emerging_trends.csv`: {_file_age(METRICS_DIR / 'emerging_trends.csv')}")
    return {"text": "\n".join(lines), "reply_markup": _back_kb()}


def _trigger_workflow(workflow: str) -> str:
    pat = os.getenv("GH_PAT")
    repo = os.getenv("GH_REPO")  # "owner/name"
    if not pat or not repo:
        return "❌ GH_PAT або GH_REPO не налаштовано."
    url = f"https://api.github.com/repos/{repo}/actions/workflows/{workflow}/dispatches"
    r = requests.post(
        url,
        json={"ref": "main"},
        headers={
            "Authorization": f"Bearer {pat}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }, timeout=15,
    )
    if r.status_code == 204:
        return f"✅ Запущено `{workflow}`"
    return f"❌ {r.status_code}: {r.text[:200]}"




def cmd_run(chat_id: int, _args: str):
    return {"text": _trigger_workflow("daily.yml"), "reply_markup": _back_kb()}


def cmd_retrain(chat_id: int, _args: str):
    return {"text": _trigger_workflow("weekly.yml"), "reply_markup": _back_kb()}


def cmd_mute(chat_id: int, _args: str):
    s = _load_muted()
    s.add(chat_id)
    _save_muted(s)
    return {
        "text": "🔕 Сповіщення вимкнено в цьому чаті.",
        "reply_markup": _main_menu_kb(chat_id),
    }


def cmd_unmute(chat_id: int, _args: str):
    s = _load_muted()
    s.discard(chat_id)
    _save_muted(s)
    return {
        "text": "🔔 Сповіщення увімкнено знову.",
        "reply_markup": _main_menu_kb(chat_id),
    }


HANDLERS = {
    "/start": cmd_help,
    "/help": cmd_help,
    "/trends": cmd_trends,
    "/topic": cmd_topic,
    "/status": cmd_status,
    "/run": cmd_run,
    "/retrain": cmd_retrain,
    "/mute": cmd_mute,
    "/unmute": cmd_unmute,
}


# ---------------------------------------------------------------------------
def _send_reply(chat_id: int, reply) -> None:
    if isinstance(reply, dict):
        text = reply["text"]
        kb = reply.get("reply_markup")
    else:
        text = str(reply)
        kb = None
    try:
        send_message(chat_id, text, reply_markup=kb)
    except requests.HTTPError as e:
        # Likely Markdown parse error — retry as plain text.
        log.warning("send failed (%s) — retrying without Markdown", e)
        try:
            send_message(chat_id, text, parse_mode="", reply_markup=kb)
        except Exception:
            log.exception("plain-text fallback also failed")


def _dispatch(chat_id: int, cmd: str, args: str):
    handler = HANDLERS.get(cmd)
    if handler is None:
        return f"Невідома команда: `{cmd}`. /help — список."
    try:
        return handler(chat_id, args)
    except Exception as e:
        log.exception("handler %s failed", cmd)
        return f"❌ Помилка: {e}"


def _handle_callback(cb: dict) -> None:
    cb_id = cb.get("id")
    data = cb.get("data") or ""
    msg = cb.get("message") or {}
    chat_id = (msg.get("chat") or {}).get("id")
    if chat_id not in _allowed_chats():
        try:
            _api_call("answerCallbackQuery", callback_query_id=cb_id,
                      text="⛔ Доступ заборонено")
        except Exception:
            pass
        return
    try:
        _api_call("answerCallbackQuery", callback_query_id=cb_id)
    except Exception:
        pass

    kind, _, payload = data.partition(":")
    if kind == "menu":
        cmd_map = {
            "help": "/help", "trends": "/trends", "status": "/status",
            "run": "/run", "retrain": "/retrain",
            "mute": "/mute", "unmute": "/unmute",
        }
        cmd = cmd_map.get(payload)
        if cmd:
            _send_reply(chat_id, _dispatch(chat_id, cmd, ""))
    elif kind == "topic":
        _send_reply(chat_id, _dispatch(chat_id, "/topic", payload))


def _handle_update(update: dict) -> None:
    if "callback_query" in update:
        _handle_callback(update["callback_query"])
        return
    msg = update.get("message") or update.get("edited_message")
    if not msg:
        return
    chat = msg.get("chat", {})
    chat_id = chat.get("id")
    text = (msg.get("text") or "").strip()

    allowed = _allowed_chats()
    if chat_id not in allowed:
        log.warning("Denied chat %s (%s)", chat_id, chat.get("username"))
        try:
            send_message(chat_id, "⛔ Доступ заборонено. Ваш chat id: `%s`" % chat_id)
        except Exception:
            pass
        return

    if not text.startswith("/"):
        return
    cmd, _, args = text.partition(" ")
    cmd = cmd.split("@", 1)[0].lower()  # strip /cmd@botname
    _send_reply(chat_id, _dispatch(chat_id, cmd, args))


def poll_once(timeout: int = 0) -> int:
    """Process all pending updates and return the number handled."""
    offset = _load_offset()
    # getUpdates with long-poll needs a longer HTTP timeout than the default.
    url = API.format(token=_token(), method="getUpdates")
    r = requests.post(url, json={"offset": offset, "timeout": timeout},
                      timeout=timeout + 15)
    r.raise_for_status()
    resp = r.json()
    updates = resp.get("result", [])
    if not updates:
        return 0
    for upd in updates:
        try:
            _handle_update(upd)
        except Exception:
            log.exception("Unhandled error in update")
        offset = max(offset, upd["update_id"] + 1)
    _save_offset(offset)
    return len(updates)


def poll_forever(timeout: int = 30) -> None:
    """Long-poll Telegram forever — for always-on hosting (Container Apps)."""
    log.info("Long-poll mode (timeout=%ss). Press Ctrl+C to stop.", timeout)
    backoff = 1.0
    while True:
        try:
            n = poll_once(timeout=timeout)
            if n:
                log.info("Processed %d updates.", n)
            backoff = 1.0
        except KeyboardInterrupt:
            log.info("Stopped by user.")
            return
        except Exception as e:
            log.warning("poll error: %s — retrying in %.1fs", e, backoff)
            import time as _t
            _t.sleep(backoff)
            backoff = min(backoff * 2, 60.0)


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--send", help="Send a one-off message to all allowed chats.")
    p.add_argument("--poll-once", action="store_true",
                   help="Process pending bot updates and exit.")
    p.add_argument("--poll-forever", action="store_true",
                   help="Long-poll forever (for always-on hosting).")
    p.add_argument("--reset-offset", action="store_true",
                   help="Skip all pending updates (set offset to current head).")
    args = p.parse_args()

    if args.reset_offset:
        # Drain queue without invoking handlers.
        resp = _api_call("getUpdates", timeout=0)
        updates = resp.get("result", [])
        if updates:
            _save_offset(updates[-1]["update_id"] + 1)
        log.info("Offset reset; %d updates discarded.", len(updates))
        return

    if args.send:
        send_alert_to_all(args.send)
        return

    if args.poll_once:
        n = poll_once()
        log.info("Processed %d updates.", n)
        return

    if args.poll_forever:
        poll_forever()
        return

    p.print_help()


if __name__ == "__main__":
    main()
