import requests

from hardware.config import CLOUD_BASE_URL, POLL_TIMEOUT_S


def _fetch_json(path: str) -> dict | list:
    response = requests.get(f"{CLOUD_BASE_URL}/{path}", timeout=POLL_TIMEOUT_S * 5)
    response.raise_for_status()
    return response.json()


def fetch_cloud_tick_day() -> tuple[int | None, int | None]:
    """Lightweight poll for game tick/day only (used while waiting for next tick)."""
    price = _fetch_json("price")
    tick = price.get("tick")
    day = price.get("day")
    return (
        int(tick) if tick is not None else None,
        int(day) if day is not None else None,
    )


def fetch_cloud_snapshot() -> dict:
    sun = _fetch_json("sun")
    price = _fetch_json("price")
    demand = _fetch_json("demand")
    deferables = _fetch_json("deferables")

    tick = price.get("tick", demand.get("tick", sun.get("tick") if isinstance(sun, dict) else None))
    day = price.get("day", demand.get("day"))

    return {
        "tick": tick,
        "day": day,
        "sun": sun.get("sun") if isinstance(sun, dict) else sun,
        "demand": demand.get("demand") if isinstance(demand, dict) else None,
        "buy_price": price.get("buy_price"),
        "sell_price": price.get("sell_price"),
        "deferables": deferables if isinstance(deferables, list) else [],
    }
