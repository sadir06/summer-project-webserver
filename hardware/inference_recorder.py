"""Persist live inference tick metrics under data/inference_runs/ for report graphs."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SAVE_EVERY_N_TICKS = 25

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = _PROJECT_ROOT / "data" / "inference_runs"

_lock = threading.Lock()
_active: dict[str, Any] | None = None


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")


def _run_dir_name(run_name: str | None) -> str:
    stamp = _utc_stamp()
    if run_name:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_name.strip())
        safe = safe.strip("_") or "run"
        return f"{stamp}_{safe}"
    return stamp


def is_recording() -> bool:
    with _lock:
        return _active is not None


def get_recording_state() -> dict[str, Any]:
    with _lock:
        if _active is None:
            return {"recording": False, "run_dir": None, "tick_count": 0}
        return {
            "recording": True,
            "run_dir": str(_active["run_dir"]),
            "run_name": _active.get("run_name"),
            "tick_count": _active["tick_count"],
            "started_at": _active.get("started_at"),
            "last_saved_at": _active.get("last_saved_at"),
        }


def start_new_run(
    run_name: str | None = None,
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Create a new run folder and begin appending tick records."""
    run_dir = RUNS_DIR / _run_dir_name(run_name)
    run_dir.mkdir(parents=True, exist_ok=False)

    started_at = time.time()
    meta_payload = {
        "started_at": started_at,
        "started_at_iso": datetime.fromtimestamp(started_at, tz=timezone.utc).isoformat(),
        "run_name": run_name,
        **(meta or {}),
    }
    (run_dir / "meta.json").write_text(
        json.dumps(meta_payload, indent=2),
        encoding="utf-8",
    )
    (run_dir / "ticks.jsonl").touch()

    with _lock:
        global _active
        _active = {
            "run_dir": run_dir,
            "run_name": run_name,
            "started_at": started_at,
            "tick_count": 0,
            "ticks": [],
            "cumulative_profit_cents": 0.0,
            "last_saved_at": None,
            "meta": meta_payload,
        }

    return run_dir


def _write_snapshot_locked() -> None:
    assert _active is not None
    run_dir: Path = _active["run_dir"]
    payload = {
        "meta": {
            **_active["meta"],
            "tick_count": _active["tick_count"],
            "cumulative_profit_cents": _active["cumulative_profit_cents"],
            "last_updated": time.time(),
            "last_updated_iso": datetime.now(timezone.utc).isoformat(),
        },
        "summary": {
            "cumulative_profit_cents": _active["cumulative_profit_cents"],
            "tick_count": _active["tick_count"],
        },
        "ticks": list(_active["ticks"]),
    }
    tmp = run_dir / "latest.json.tmp"
    out = run_dir / "latest.json"
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(out)
    _active["last_saved_at"] = time.time()


def record_inference_tick(record: dict[str, Any]) -> None:
    """Append one tick; rewrite latest.json every SAVE_EVERY_N_TICKS ticks."""
    with _lock:
        if _active is None:
            return

        tick_profit = float(record.get("tick_profit_cents", 0.0))
        _active["cumulative_profit_cents"] += tick_profit
        _active["tick_count"] += 1

        row = {
            **record,
            "cumulative_profit_cents": _active["cumulative_profit_cents"],
            "record_index": _active["tick_count"],
        }
        _active["ticks"].append(row)

        jsonl_path = _active["run_dir"] / "ticks.jsonl"
        with jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, separators=(",", ":")))
            fh.write("\n")

        if _active["tick_count"] % SAVE_EVERY_N_TICKS == 0:
            _write_snapshot_locked()


def finalize_run() -> None:
    """Write a final latest.json snapshot (e.g. on shutdown)."""
    with _lock:
        if _active is None or _active["tick_count"] == 0:
            return
        _write_snapshot_locked()
