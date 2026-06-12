import concurrent.futures

from hardware.config import CLOUD_BASE_URL, CLOUD_SNAPSHOT_TIMEOUT_S, CLOUD_TICK_TIMEOUT_S
from hardware.http_client import cloud_get


def _fetch_json(path: str, *, timeout_s: float) -> dict | list:
    response = cloud_get(f"{CLOUD_BASE_URL}/{path}", timeout=timeout_s)
    response.raise_for_status()
    return response.json()


def fetch_cloud_tick_day() -> tuple[int | None, int | None]:
    """Lightweight poll for game tick/day only (used while waiting for next tick)."""
    price = _fetch_json("price", timeout_s=CLOUD_TICK_TIMEOUT_S)
    tick = price.get("tick")
    day = price.get("day")
    return (
        int(tick) if tick is not None else None,
        int(day) if day is not None else None,
    )


def fetch_cloud_snapshot() -> dict:
    """Fetch Azure inputs for inference (price, demand, deferables — no sun)."""
    timeout_s = CLOUD_SNAPSHOT_TIMEOUT_S
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        price_f = pool.submit(_fetch_json, "price", timeout_s=timeout_s)
        demand_f = pool.submit(_fetch_json, "demand", timeout_s=timeout_s)
        defer_f = pool.submit(_fetch_json, "deferables", timeout_s=timeout_s)
        price = price_f.result()
        demand = demand_f.result()
        deferables = defer_f.result()

    tick = price.get("tick", demand.get("tick"))
    day = price.get("day", demand.get("day"))

    return {
        "tick": tick,
        "day": day,
        "demand": demand.get("demand") if isinstance(demand, dict) else None,
        "buy_price": price.get("buy_price"),
        "sell_price": price.get("sell_price"),
        "deferables": deferables if isinstance(deferables, list) else [],
    }
