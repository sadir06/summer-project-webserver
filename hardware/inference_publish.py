"""Fire-and-forget publish of inference metrics to the local Flask dashboard."""

from __future__ import annotations

import requests

from hardware.config import FLASK_PORT, LAPTOP_IP, SC_ACTION_WATTS_SCALE, TICK_INTERVAL_S

_publish_failures = 0


def _json_safe(value):
    """Convert numpy/torch scalars to native Python types for requests JSON encoding."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return float(value)


def _sanitize_payload(payload: dict) -> dict:
    return {key: _json_safe(val) for key, val in payload.items()}


def estimate_grid_power_w(
    *,
    total_demand_w: float,
    pv_w: float,
    sc_action: float,
    pcout_w: float | None,
) -> tuple[float, float, float]:
    """Return (grid_import_w, grid_export_w, sc_bus_w).

    sc_bus_w > 0 means supercap discharging onto the bus (helps meet load).
    """
    if pcout_w is not None:
        sc_bus_w = float(pcout_w)
    else:
        # Model convention: negative sc_action = discharge → supplies bus
        sc_bus_w = -float(sc_action) * SC_ACTION_WATTS_SCALE

    net_grid_w = float(total_demand_w) - float(pv_w) - sc_bus_w
    grid_import_w = max(0.0, net_grid_w)
    grid_export_w = max(0.0, -net_grid_w)
    return grid_import_w, grid_export_w, sc_bus_w


def tick_profit_cents(
    *,
    grid_import_w: float,
    grid_export_w: float,
    buy_price: float,
    sell_price: float,
    tick_dur_s: float = TICK_INTERVAL_S,
) -> float:
    import_j = grid_import_w * tick_dur_s
    export_j = grid_export_w * tick_dur_s
    return export_j * sell_price - import_j * buy_price


def _flask_base() -> str:
    return f"http://{LAPTOP_IP}:{FLASK_PORT}"


def publish_inference_tick(payload: dict) -> bool:
    """POST tick metrics to Flask (charts + GET /api/load_demand for Load Pico)."""
    global _publish_failures

    body = _sanitize_payload(payload)
    if body.get("total_demand_w") is not None:
        body["total_demand"] = body["total_demand_w"]

    load_ok = False
    charts_ok = False

    try:
        response = requests.post(
            f"{_flask_base()}/api/load_demand",
            json=body,
            timeout=3.0,
        )
        response.raise_for_status()
        load_ok = True
    except Exception as e:
        _publish_failures += 1
        if _publish_failures <= 3 or _publish_failures % 25 == 0:
            print(f"Error POST {_flask_base()}/api/load_demand: {e}")

    try:
        response = requests.post(
            f"{_flask_base()}/api/inference_tick",
            json=body,
            timeout=3.0,
        )
        response.raise_for_status()
        charts_ok = True
    except Exception as e:
        if _publish_failures <= 3 or _publish_failures % 25 == 0:
            print(f"Error POST {_flask_base()}/api/inference_tick: {e}")

    if load_ok or charts_ok:
        _publish_failures = 0
        return True
    return False
