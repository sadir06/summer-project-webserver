import json
import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

from env4 import SmartGridEnv

TICKS_PER_DAY = SmartGridEnv.ticksPerDay
MAX_PRICE = 150.0
SPLIT_SEED = 42
TRAIN_RATIO = 0.8


def load_ticks_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def group_complete_days(rows: list[dict]) -> dict[int, list[dict]]:
    """Keep only days with every tick 0..59 present (drop partial head/tail days)."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["day"]].append(row)

    complete = {}
    for day_id, day_rows in grouped.items():
        day_rows.sort(key=lambda r: r["tick"])
        ticks = {r["tick"] for r in day_rows}
        if len(ticks) == TICKS_PER_DAY and min(ticks) == 0 and max(ticks) == TICKS_PER_DAY - 1:
            complete[day_id] = day_rows
    return complete


def ensure_train_test_split(complete_days: dict[int, list[dict]], manifest_path: Path) -> dict:
    day_ids = sorted(complete_days.keys())
    rng = random.Random(SPLIT_SEED)
    shuffled = day_ids[:]
    rng.shuffle(shuffled)

    split_idx = int(len(shuffled) * TRAIN_RATIO)
    train_ids = sorted(shuffled[:split_idx])
    test_ids = sorted(shuffled[split_idx:])

    manifest = {
        "split_seed": SPLIT_SEED,
        "train_ratio": TRAIN_RATIO,
        "total_complete_days": len(day_ids),
        "train_days": len(train_ids),
        "test_days": len(test_ids),
        "train_day_ids": train_ids,
        "test_day_ids": test_ids,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_split_manifest(project_root: Path, complete_days: dict[int, list[dict]]) -> dict:
    manifest_path = project_root / "data_collection" / "data" / "split_manifest.json"
    expected_total = len(complete_days)

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("total_complete_days") == expected_total:
            return manifest

    return ensure_train_test_split(complete_days, manifest_path)


def rows_to_day_profile(day_rows: list[dict]) -> dict:
    by_tick = {r["tick"]: r for r in day_rows}

    irradiance = []
    base_demand = []
    buy_price = []
    sell_price = []

    for t in range(TICKS_PER_DAY):
        row = by_tick[t]
        irradiance.append(SmartGridEnv.normalizeIrradiance(row.get("sun") or 0.0))
        base_demand.append(row.get("demand") or 0.0)
        buy_price.append((row.get("buy_price") or 0.0) / MAX_PRICE)
        sell_price.append((row.get("sell_price") or 0.0) / MAX_PRICE)

    defer_src = by_tick[0].get("deferables") or []
    deferables = [
        {
            "energy": float(d["energy"]),
            "start": int(d["start"]),
            "end": int(d["end"]),
        }
        for d in defer_src
    ]

    return {
        "day_id": day_rows[0]["day"],
        "irradiance": irradiance,
        "baseDemand": base_demand,
        "buyPrice": buy_price,
        "sellPrice": sell_price,
        "deferables": deferables,
    }


def _profiles_for_ids(complete_days: dict[int, list[dict]], day_ids: list[int]) -> list[dict]:
    profiles = []
    for day_id in day_ids:
        if day_id in complete_days:
            profiles.append(rows_to_day_profile(complete_days[day_id]))
    return profiles


def load_day_data(project_root: Path) -> tuple[list[dict], list[dict], dict]:
    data_dir = project_root / "data_collection" / "data"
    ticks_path = data_dir / "ticks.jsonl"

    rows = load_ticks_jsonl(ticks_path)
    complete_days = group_complete_days(rows)
    manifest = load_split_manifest(project_root, complete_days)

    train_profiles = _profiles_for_ids(complete_days, manifest["train_day_ids"])
    test_profiles = _profiles_for_ids(complete_days, manifest["test_day_ids"])

    if not train_profiles:
        raise RuntimeError("No complete train days found in ticks.jsonl")

    return train_profiles, test_profiles, manifest


def load_train_day_profiles(project_root: Path) -> list[dict]:
    train_profiles, _, _ = load_day_data(project_root)
    return train_profiles


class SmartGridEnvReal(SmartGridEnv):
    def __init__(self, day_profiles: list[dict], seed: int = 42):
        super().__init__()
        self.day_profiles = day_profiles
        self._rng = random.Random(seed)

    def reset(self, seed=None, options=None):
        profile = self._rng.choice(self.day_profiles)
        day_options = {
            "irradiance": profile["irradiance"],
            "baseDemand": profile["baseDemand"],
            "buyPrice": profile["buyPrice"],
            "sellPrice": profile["sellPrice"],
            "deferables": deepcopy(profile["deferables"]),
        }
        return super().reset(seed=seed, options=day_options)
