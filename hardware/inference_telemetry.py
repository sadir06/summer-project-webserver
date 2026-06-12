"""In-memory ring buffer of live inference tick metrics for the dashboard."""

from __future__ import annotations

import threading
import time
from typing import Any

from hardware.inference_recorder import record_inference_tick

MAX_TICKS = 120

_lock = threading.Lock()
_ticks: list[dict[str, Any]] = []
_summary: dict[str, Any] = {
    "cumulative_profit_cents": 0.0,
    "tick_count": 0,
    "last_tick": None,
    "last_day": None,
    "last_updated": None,
    "active": False,
}


def append_inference_tick(record: dict[str, Any]) -> None:
    """Record one inference tick (called from Flask POST handler)."""
    tick_profit = float(record.get("tick_profit_cents", 0.0))
    with _lock:
        _ticks.append(record)
        if len(_ticks) > MAX_TICKS:
            _ticks.pop(0)
        _summary["cumulative_profit_cents"] += tick_profit
        _summary["tick_count"] += 1
        _summary["last_tick"] = record.get("tick")
        _summary["last_day"] = record.get("day")
        _summary["last_updated"] = record.get("ts") or time.time()
        _summary["active"] = True
    record_inference_tick(record)


def get_inference_telemetry() -> dict[str, Any]:
    with _lock:
        last_ts = _summary.get("last_updated")
        active = bool(
            last_ts is not None and (time.time() - float(last_ts)) < 15.0
        )
        return {
            "ticks": list(_ticks),
            "summary": {**_summary, "active": active},
        }
