"""In-memory ring buffer of live inference tick metrics for the dashboard."""

from __future__ import annotations

import threading
import time
from typing import Any

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


def _tick_key(day: int, tick: int) -> tuple[int, int]:
    return int(day), int(tick)


def append_inference_tick(record: dict[str, Any]) -> bool:
    """Record one inference tick synchronously; ignore stale/out-of-order duplicates."""
    tick_profit = float(record.get("tick_profit_cents", 0.0))
    day = int(record["day"])
    tick = int(record["tick"])

    with _lock:
        last_day = _summary.get("last_day")
        last_tick = _summary.get("last_tick")
        if last_day is not None and last_tick is not None:
            if day < int(last_day) or (day == int(last_day) and tick < int(last_tick)):
                return False
            if day == int(last_day) and tick == int(last_tick):
                if _ticks:
                    old_profit = float(_ticks[-1].get("tick_profit_cents", 0.0))
                    _ticks[-1] = record
                    _summary["cumulative_profit_cents"] += tick_profit - old_profit
                _summary["last_updated"] = record.get("ts") or time.time()
                _summary["active"] = True
                return True

        _ticks.append(record)
        if len(_ticks) > MAX_TICKS:
            _ticks.pop(0)
        _summary["cumulative_profit_cents"] += tick_profit
        _summary["tick_count"] += 1
        _summary["last_tick"] = tick
        _summary["last_day"] = day
        _summary["last_updated"] = record.get("ts") or time.time()
        _summary["active"] = True
    return True


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
