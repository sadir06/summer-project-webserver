import json
from collections import defaultdict
from pathlib import Path

PV_POWER_MAX = 8.0
TICKS_PER_DAY = 60
MAX_PRICE = 150.0


def load_ticks_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def group_complete_days(rows: list[dict]) -> dict[int, list[dict]]:
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


def rows_to_day_profile(day_rows: list[dict]) -> dict:
    by_tick = {r["tick"]: r for r in day_rows}

    pv_gen = []
    base_demand = []
    buy_price = []
    sell_price = []

    for t in range(TICKS_PER_DAY):
        row = by_tick[t]
        sun = row.get("sun") or 0.0
        pv_gen.append((sun / 100.0) * PV_POWER_MAX)
        base_demand.append(row.get("demand") or 0.0)
        buy_price.append((row.get("buy_price") or 0.0) / MAX_PRICE)
        sell_price.append((row.get("sell_price") or 0.0) / MAX_PRICE)

    defer_src = day_rows[0].get("deferables") or []
    def_demand_state = []
    for d in defer_src:
        energy = float(d["energy"])
        def_demand_state.append({
            "startTick": int(d["start"]),
            "endTick": int(d["end"]),
            "remainingEn": energy,
            "originalEn": energy,
            "served": False,
        })

    return {
        "pvGen": pv_gen,
        "baseDemand": base_demand,
        "buyPrice": buy_price,
        "sellPrice": sell_price,
        "defDemandState": def_demand_state,
    }


def load_train_day_profiles(project_root: Path) -> list[dict]:
    ticks_path = project_root / "data_collection" / "data" / "ticks.jsonl"
    manifest_path = project_root / "data_collection" / "data" / "split_manifest.json"

    rows = load_ticks_jsonl(ticks_path)
    days = group_complete_days(rows)

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        train_ids = set(manifest["train_day_ids"])
    else:
        train_ids = set(days.keys())

    profiles = []
    for day_id in sorted(train_ids):
        if day_id in days:
            profiles.append(rows_to_day_profile(days[day_id]))

    if not profiles:
        raise RuntimeError("No train days found — check ticks.jsonl and split_manifest.json")

    return profiles
