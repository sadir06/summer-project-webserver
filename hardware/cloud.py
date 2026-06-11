import requests

from hardware.config import CLOUD_BASE_URL, POLL_TIMEOUT_S


def _fetch_json(path: str) -> dict | list:
    response = requests.get(f"{CLOUD_BASE_URL}/{path}", timeout=POLL_TIMEOUT_S * 5)
    response.raise_for_status()
    return response.json()


def _instant_demand_w(demand_payload: dict | list | None) -> float | None:
    """Simulator instant load (W) from cloud GET /demand — not the published load command."""
    if not isinstance(demand_payload, dict):
        return None
    value = demand_payload.get("demand")
    if value is not None:
        return float(value)
    return None


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
        "demand": _instant_demand_w(demand),
        "buy_price": price.get("buy_price"),
        "sell_price": price.get("sell_price"),
        "deferables": deferables if isinstance(deferables, list) else [],
    }


def publish_load_demand(*, day: int, tick: int, total_demand_w: float) -> bool:
    """POST load command for the Load Pico (polls GET /demand on Azure).

    Payload shape:
      {"day": 5937300, "total_demand": 1.16, "tick": 45}
    """
    payload = {
        "day": int(day),
        "tick": int(tick),
        "total_demand": float(total_demand_w),
    }
    try:
        response = requests.post(
            f"{CLOUD_BASE_URL}/demand",
            json=payload,
            timeout=POLL_TIMEOUT_S * 5,
        )
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Error publishing load demand to {CLOUD_BASE_URL}/demand: {e}")
        return False
