"""Model-computed load command for the Load Pico (instant + defer this tick)."""

from __future__ import annotations

import threading

_lock = threading.Lock()
_command: dict[str, int | float | None] = {
    "day": None,
    "tick": None,
    "total_demand": None,
}


def set_model_load_demand(*, day: int, tick: int, total_demand_w: float) -> None:
    """Updated every inference tick with policy output (watts)."""
    with _lock:
        _command["day"] = int(day)
        _command["tick"] = int(tick)
        _command["total_demand"] = float(total_demand_w)


def get_model_load_demand() -> dict[str, int | float | None]:
    with _lock:
        return dict(_command)
