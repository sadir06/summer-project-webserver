"""Model-computed load command for the Load Pico (instant + defer this tick)."""

from __future__ import annotations

import threading

_lock = threading.Lock()
_command: dict[str, int | float | None] = {
    "day": None,
    "tick": None,
    "total_demand": None,
}


def _is_newer_tick(day: int, tick: int) -> bool:
    cur_day = _command["day"]
    cur_tick = _command["tick"]
    if cur_day is None or cur_tick is None:
        return True
    if int(day) > int(cur_day):
        return True
    if int(day) == int(cur_day) and int(tick) > int(cur_tick):
        return True
    return False


def set_model_load_demand(*, day: int, tick: int, total_demand_w: float) -> bool:
    """Update load command; ignore stale/out-of-order ticks from async races."""
    with _lock:
        if not _is_newer_tick(day, tick):
            return False
        _command["day"] = int(day)
        _command["tick"] = int(tick)
        _command["total_demand"] = float(total_demand_w)
        return True


def get_model_load_demand() -> dict[str, int | float | None]:
    with _lock:
        return dict(_command)
