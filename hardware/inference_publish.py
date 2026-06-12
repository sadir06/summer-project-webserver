"""Fire-and-forget publish of inference metrics to the local Flask dashboard."""

from __future__ import annotations

import requests

from hardware.config import FLASK_PORT, SC_ACTION_WATTS_SCALE, TICK_INTERVAL_S


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


def publish_inference_tick(payload: dict) -> None:
    """POST tick metrics to Flask; never raises (dashboard is optional)."""
    try:
        requests.post(
            f"http://127.0.0.1:{FLASK_PORT}/api/inference_tick",
            json=payload,
            timeout=0.4,
        )
    except Exception:
        pass
