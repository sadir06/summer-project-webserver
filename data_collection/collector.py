import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_URL = "https://icelec50015.azurewebsites.net"
POLL_INTERVAL_SEC = 4.5
REQUEST_TIMEOUT_SEC = 10

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_FILE = DATA_DIR / "ticks.jsonl"
LOG_FILE = DATA_DIR / "collector.log"


def setup_logging() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def fetch_json(session: requests.Session, path: str) -> dict | list:
    url = f"{BASE_URL}/{path}"
    response = session.get(url, timeout=REQUEST_TIMEOUT_SEC)
    response.raise_for_status()
    return response.json()


def collect_snapshot(session: requests.Session) -> dict:
    sun = fetch_json(session, "sun")
    price = fetch_json(session, "price")
    demand = fetch_json(session, "demand")
    deferables = fetch_json(session, "deferables")
    tick = price.get("tick", demand.get("tick", sun.get("tick")))
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "tick": tick,
        "day": price.get("day", demand.get("day")),
        "sun": sun.get("sun"),
        "demand": demand.get("demand"),
        "buy_price": price.get("buy_price"),
        "sell_price": price.get("sell_price"),
        "deferables": deferables,
    }


def append_record(record: dict) -> None:
    line = json.dumps(record, separators=(",", ":")) + "\n"
    with OUTPUT_FILE.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()


def load_last_tick() -> int | None:
    if not OUTPUT_FILE.exists():
        return None
    last_line = None
    with OUTPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last_line = line
    if not last_line:
        return None
    return json.loads(last_line).get("tick")


def main() -> None:
    setup_logging()
    last_tick = load_last_tick()
    if last_tick is not None:
        logging.info("Resuming after tick %s -> %s", last_tick, OUTPUT_FILE)
    else:
        logging.info("Starting fresh -> %s", OUTPUT_FILE)

    session = requests.Session()

    while True:
        try:
            record = collect_snapshot(session)
            tick = record["tick"]
            if tick != last_tick:
                append_record(record)
                last_tick = tick
                logging.info(
                    "saved tick=%s day=%s demand=%.3f sun=%.3f buy=%s sell=%s",
                    tick,
                    record.get("day"),
                    record.get("demand") or 0.0,
                    record.get("sun") or 0.0,
                    record.get("buy_price"),
                    record.get("sell_price"),
                )
        except requests.RequestException as exc:
            logging.warning("network error: %s", exc)
        except Exception:
            logging.exception("unexpected error")
        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()
