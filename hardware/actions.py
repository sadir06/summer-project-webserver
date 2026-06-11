import requests

from hardware.config import (
    ACTION_QUERY_PARAM,
    ACTION_TIMEOUT_S,
    ACTIONS_ENABLED,
    LOAD_DEMAND_PATH,
    PICO_ACTION_ENDPOINTS,
    PICO_URL_DEF_ACTION,
    SC_ACTION_WATTS_SCALE,
)
from hardware.load_demand import set_model_load_demand


def _scale_sc_action(sc_action: float) -> float:
    return float(sc_action) * SC_ACTION_WATTS_SCALE


def send_action(name: str, value: float) -> bool:
    if not ACTIONS_ENABLED:
        return False

    endpoint = PICO_ACTION_ENDPOINTS[name]
    url = f"{endpoint['base_url']}{endpoint['path']}"
    try:
        response = requests.get(
            url,
            params={ACTION_QUERY_PARAM: float(value)},
            timeout=ACTION_TIMEOUT_S,
        )
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Error sending {name} action to {url}: {e}")
        return False


def send_load_demand(*, day: int, tick: int, total_demand_w: float) -> bool:
    """Send model total load (instant + defer) to Load Pico as JSON.

    Payload: {"day": ..., "tick": ..., "total_demand": <watts>}
    Also mirrored on Flask GET /api/load_demand for Pico polling.
    """
    set_model_load_demand(day=day, tick=tick, total_demand_w=total_demand_w)
    payload = {
        "day": int(day),
        "tick": int(tick),
        "total_demand": float(total_demand_w),
    }
    url = f"{PICO_URL_DEF_ACTION}{LOAD_DEMAND_PATH}"
    try:
        response = requests.post(url, json=payload, timeout=ACTION_TIMEOUT_S)
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Error sending load demand to {url}: {e}")
        return False


def send_actions(
    sc_action: float,
    *,
    grid_action: float | None = None,
) -> dict[str, bool]:
    """Send supercap (and optional grid) actions directly to Picos."""
    if not ACTIONS_ENABLED:
        print(
            "actions skipped (ACTIONS_ENABLED=False; Pico firmware has no GET /action yet)"
        )
        results = {"sc": False}
        if grid_action is not None:
            results["grid"] = False
        return results

    results = {
        "sc": send_action("sc", _scale_sc_action(sc_action)),
    }
    if grid_action is not None:
        results["grid"] = send_action("grid", grid_action)
    return results
