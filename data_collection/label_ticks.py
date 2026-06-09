import json
import logging
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

if __package__:
    from .paths import DATA_DIR
else:
    from paths import DATA_DIR

INPUT_FILE = DATA_DIR / "ticks.jsonl"
TRAIN_OUTPUT = DATA_DIR / "labelled_train.jsonl"
TEST_OUTPUT = DATA_DIR / "labelled_test.jsonl"
SPLIT_MANIFEST = DATA_DIR / "split_manifest.json"
LOG_FILE = DATA_DIR / "labeler.log"

TICKS_PER_DAY = 60
TICK_DURATION_SEC = 5
PV_POWER_MAX = 8.0
LOAD_DEMAND_MAX = 4.0
MAX_SC_CHARGE_W = 2.0
MAX_SC_DISCHARGE_W = 2.0
DEF_DUE_SOON_TICKS = 30

SUPERCAP_CF = 0.46
SUPERCAP_MIN_V = 10.0
SUPERCAP_MAX_V = 16.5
MIN_PHYSICAL_SC_EN = 0.5 * SUPERCAP_CF * (SUPERCAP_MIN_V ** 2)
MAX_PHYSICAL_SC_EN = 0.5 * SUPERCAP_CF * (SUPERCAP_MAX_V ** 2)
MAX_USABLE_SC_EN = MAX_PHYSICAL_SC_EN - MIN_PHYSICAL_SC_EN

PRICE_LOOKAHEAD_TICKS = 10
GRID_BUY_POWER_MAX = 4.0
GRID_SELL_POWER_MAX = 4.0
MIN_SELL_SOC = 0.25
BUY_MARGIN = 0.05
SELL_MARGIN = 0.05

MAX_PRICE = 150.0
MAX_DEF_EN = 100.0
MAX_DUE_SOON_EN = 100.0
MAX_PROJECTED_NET_EN = PV_POWER_MAX * TICK_DURATION_SEC * PRICE_LOOKAHEAD_TICKS

TRAIN_RATIO = 0.8
SPLIT_SEED = 42


def setup_logging() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def load_ticks(path: Path) -> list[dict]:
    ticks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ticks.append(json.loads(line))
    return ticks


def group_by_day(ticks: list[dict]) -> dict[int, list[dict]]:
    grouped = defaultdict(list)
    for row in ticks:
        grouped[row["day"]].append(row)

    complete = {}
    dropped = 0

    for day_id, rows in grouped.items():
        rows.sort(key=lambda row: row["tick"])
        tick_values = {r["tick"] for r in rows}

        if (
            len(tick_values) == TICKS_PER_DAY
            and min(tick_values) == 0
            and max(tick_values) == TICKS_PER_DAY - 1
        ):
            complete[day_id] = rows
        else:
            dropped += 1

    logging.info("dropped %d incomplete days out of %d", dropped, len(grouped))
    return complete


def split_days(days: dict[int, list[dict]], train_ratio: float, seed: int) -> tuple[dict, dict]:
    day_ids = sorted(days.keys())
    rng = random.Random(seed)
    shuffled = day_ids[:]
    rng.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train_ids = set(shuffled[:split_idx])
    test_ids = set(shuffled[split_idx:])

    train_days = {day_id: days[day_id] for day_id in train_ids}
    test_days = {day_id: days[day_id] for day_id in test_ids}
    return train_days, test_days


def init_deferables(deferables_from_row: list[dict]) -> list[dict]:
    deferables = []
    for item in deferables_from_row:
        deferables.append({
            "start": item["start"],
            "end": item["end"],
            "remaining_en": item["energy"],
            "original_en": item["energy"],
            "served": False,
        })
    return deferables


def compute_defer_features(defer_state: list[dict], tick: int) -> dict:
    total_def_en = sum(dd["remaining_en"] for dd in defer_state)
    due_soon_en = sum(
        dd["remaining_en"]
        for dd in defer_state
        if dd["remaining_en"] > 0
        and tick <= dd["end"] <= tick + DEF_DUE_SOON_TICKS
    )
    active_def_en = sum(
        dd["remaining_en"]
        for dd in defer_state
        if dd["start"] <= tick
    )

    return {
        "total_def_en": total_def_en,
        "due_soon_en": due_soon_en,
        "active_def_en": active_def_en,
    }


def day_series(day_rows: list[dict]) -> dict[str, list[float]]:
    by_tick = {row["tick"]: row for row in day_rows}
    return {
        "buy_price": [by_tick[t]["buy_price"] for t in range(TICKS_PER_DAY)],
        "sell_price": [by_tick[t]["sell_price"] for t in range(TICKS_PER_DAY)],
        "pv_power_w": [
            (by_tick[t].get("sun") or 0.0) / 100.0 * PV_POWER_MAX
            for t in range(TICKS_PER_DAY)
        ],
        "base_demand_w": [by_tick[t].get("demand") or 0.0 for t in range(TICKS_PER_DAY)],
    }


def get_price_projection(day_series: dict[str, list[float]], tick: int) -> dict:
    end_tick = min(TICKS_PER_DAY, tick + PRICE_LOOKAHEAD_TICKS)

    future_buy = day_series["buy_price"][tick:end_tick]
    future_sell = day_series["sell_price"][tick:end_tick]
    future_pv = day_series["pv_power_w"][tick:end_tick]
    future_demand = day_series["base_demand_w"][tick:end_tick]

    if not future_buy:
        future_buy = [day_series["buy_price"][min(tick, TICKS_PER_DAY - 1)]]
    if not future_sell:
        future_sell = [day_series["sell_price"][min(tick, TICKS_PER_DAY - 1)]]
    if not future_pv:
        future_pv = [day_series["pv_power_w"][min(tick, TICKS_PER_DAY - 1)]]
    if not future_demand:
        future_demand = [day_series["base_demand_w"][min(tick, TICKS_PER_DAY - 1)]]

    avg_future_buy = float(np.mean(future_buy))
    avg_future_sell = float(np.mean(future_sell))
    projected_future_pv_en = float(np.sum(future_pv) * TICK_DURATION_SEC)
    projected_future_demand_en = float(np.sum(future_demand) * TICK_DURATION_SEC)
    projected_future_net_en = projected_future_pv_en - projected_future_demand_en

    return {
        "avg_future_buy": avg_future_buy,
        "avg_future_sell": avg_future_sell,
        "projected_future_net_en": projected_future_net_en,
    }


def row_to_physics(row: dict) -> dict:
    sun = row.get("sun") or 0.0
    pv_power_w = (sun / 100.0) * PV_POWER_MAX
    pv_energy_j = pv_power_w * TICK_DURATION_SEC
    demand_w = row.get("demand") or 0.0
    demand_energy_j = demand_w * TICK_DURATION_SEC
    return {
        "pv_power_w": pv_power_w,
        "base_demand_w": demand_w,
        "pv_energy_j": pv_energy_j,
        "base_energy_j": demand_energy_j,
        "buy_price": row["buy_price"],
        "sell_price": row["sell_price"],
        "tick": row["tick"],
        "day": row["day"],
    }


def build_obs(
    tick: int,
    physics: dict,
    sc_energy_j: float,
    defer_features: dict,
    projection: dict,
) -> list[float]:
    sc_soc = sc_energy_j / MAX_USABLE_SC_EN

    time_angle = 2 * np.pi * tick / TICKS_PER_DAY
    time_sin = np.sin(time_angle)
    time_cos = np.cos(time_angle)

    pv_norm = physics["pv_power_w"] / PV_POWER_MAX
    base_demand_norm = physics["base_demand_w"] / LOAD_DEMAND_MAX
    buy_norm = physics["buy_price"] / MAX_PRICE
    sell_norm = physics["sell_price"] / MAX_PRICE
    total_def_norm = defer_features["total_def_en"] / MAX_DEF_EN
    due_soon_norm = defer_features["due_soon_en"] / MAX_DUE_SOON_EN

    max_balance = max(PV_POWER_MAX, LOAD_DEMAND_MAX)
    net_norm = (physics["pv_power_w"] - physics["base_demand_w"]) / max_balance

    avg_future_buy_norm = projection["avg_future_buy"] / MAX_PRICE
    avg_future_sell_norm = projection["avg_future_sell"] / MAX_PRICE
    cap_free_norm = (MAX_USABLE_SC_EN - sc_energy_j) / MAX_USABLE_SC_EN
    projected_future_net_norm = projection["projected_future_net_en"] / MAX_PROJECTED_NET_EN

    obs = np.array([
        time_sin,
        time_cos,
        pv_norm,
        sc_soc,
        base_demand_norm,
        buy_norm,
        sell_norm,
        total_def_norm,
        due_soon_norm,
        net_norm,
        avg_future_buy_norm,
        avg_future_sell_norm,
        cap_free_norm,
        projected_future_net_norm,
    ], dtype=np.float32)

    return np.clip(obs, -1.0, 1.0).tolist()


def should_buy_for_storage(buy_price: float, avg_future_buy: float, sc_energy_j: float) -> bool:
    cap_space = MAX_USABLE_SC_EN - sc_energy_j
    if cap_space <= 0.0:
        return False
    return buy_price < (avg_future_buy * (1.0 - BUY_MARGIN))


def should_sell_from_storage(
    sell_price: float,
    avg_future_sell: float,
    projected_future_net_en: float,
    sc_energy_j: float,
) -> bool:
    sc_soc = sc_energy_j / MAX_USABLE_SC_EN
    if sc_soc <= MIN_SELL_SOC:
        return False
    favourable_sell = sell_price > (avg_future_sell * (1.0 + SELL_MARGIN))
    future_comfortable = projected_future_net_en >= 0.0
    return favourable_sell and future_comfortable


def naive_policy(physics: dict, defer_features: dict) -> tuple[float, float]:
    sc_action = 0.0
    demand_energy = physics["base_energy_j"] + defer_features["active_def_en"]
    net_energy = demand_energy - physics["pv_energy_j"]
    if net_energy > 0:
        grid_action = 1.0
    elif net_energy < 0:
        grid_action = -1.0
    else:
        grid_action = 0.0
    return grid_action, sc_action


def oracle_policy(
    physics: dict,
    defer_features: dict,
    sc_energy_j: float,
    projection: dict,
) -> tuple[float, float]:
    sun_frac = physics["pv_power_w"] / PV_POWER_MAX
    sc_soc = sc_energy_j / MAX_USABLE_SC_EN
    demand_energy = physics["base_energy_j"] + defer_features["active_def_en"]
    net_energy = demand_energy - physics["pv_energy_j"]

    sc_action = 0.0
    grid_action = 0.0

    if sun_frac > 0.4 and sc_soc < 0.85:
        sc_action = 0.8
    elif (
        physics["buy_price"] > projection["avg_future_buy"]
        and sc_soc > 0.15
    ):
        sc_action = -0.8

    if net_energy > 0:
        grid_action = 1.0
    elif net_energy < 0:
        if physics["sell_price"] > physics["buy_price"]:
            grid_action = -0.8
        elif should_sell_from_storage(
            physics["sell_price"],
            projection["avg_future_sell"],
            projection["projected_future_net_en"],
            sc_energy_j,
        ):
            grid_action = -0.5
        else:
            grid_action = 0.0

    if defer_features["due_soon_en"] > 0 and net_energy > 0:
        grid_action = 1.0

    return (
        float(np.clip(grid_action, -1.0, 1.0)),
        float(np.clip(sc_action, -1.0, 1.0)),
    )


def simulate_step(
    sc_energy_j: float,
    physics: dict,
    defer_features: dict,
    projection: dict,
    grid_action: float,
    sc_action: float,
) -> dict:
    grid_action = max(-1.0, min(1.0, grid_action))
    sc_action = max(-1.0, min(1.0, sc_action))

    demand_energy_j = physics["base_energy_j"] + defer_features["active_def_en"]

    if sc_action >= 0:
        sc_power_w = sc_action * MAX_SC_CHARGE_W
    else:
        sc_power_w = sc_action * MAX_SC_DISCHARGE_W

    sc_energy_delta = sc_power_w * TICK_DURATION_SEC

    if sc_energy_delta > 0:
        available_space = MAX_USABLE_SC_EN - sc_energy_j
        actual_charge = min(sc_energy_delta, available_space)
        sc_energy_j += actual_charge
        sc_flow = -actual_charge
    else:
        actual_discharge = min(abs(sc_energy_delta), sc_energy_j)
        sc_energy_j -= actual_discharge
        sc_flow = actual_discharge

    net_demand = demand_energy_j - physics["pv_energy_j"] - sc_flow

    arbitrage_import_j = 0.0
    arbitrage_export_j = 0.0
    bought_for_storage = False

    if should_buy_for_storage(
        physics["buy_price"],
        projection["avg_future_buy"],
        sc_energy_j,
    ):
        cap_space = MAX_USABLE_SC_EN - sc_energy_j
        max_buy_store_en = GRID_BUY_POWER_MAX * TICK_DURATION_SEC
        actual_buy_store_en = min(max_buy_store_en, cap_space)
        sc_energy_j += actual_buy_store_en
        arbitrage_import_j += actual_buy_store_en
        bought_for_storage = actual_buy_store_en > 0.0

    if (not bought_for_storage) and should_sell_from_storage(
        physics["sell_price"],
        projection["avg_future_sell"],
        projection["projected_future_net_en"],
        sc_energy_j,
    ):
        reserve_energy = MIN_SELL_SOC * MAX_USABLE_SC_EN
        sellable_energy = max(0.0, sc_energy_j - reserve_energy)
        max_sell_en = GRID_SELL_POWER_MAX * TICK_DURATION_SEC
        actual_sell_en = min(max_sell_en, sellable_energy)
        sc_energy_j -= actual_sell_en
        arbitrage_export_j += actual_sell_en

    imported_j = arbitrage_import_j
    exported_j = arbitrage_export_j
    demand_import_j = 0.0

    if net_demand > 0:
        if grid_action >= 0:
            demand_import_j = grid_action * net_demand
            imported_j += demand_import_j
        unmet_j = max(0.0, net_demand - demand_import_j)
    else:
        surplus = abs(net_demand)
        unmet_j = 0.0
        if grid_action < 0:
            exported_j += abs(grid_action) * surplus

    tick_cost = imported_j * physics["buy_price"] - exported_j * physics["sell_price"]

    return {
        "sc_energy_j": sc_energy_j,
        "imported_j": imported_j,
        "exported_j": exported_j,
        "arbitrage_import_j": arbitrage_import_j,
        "arbitrage_export_j": arbitrage_export_j,
        "unmet_j": unmet_j,
        "tick_cost": tick_cost,
        "net_demand": net_demand,
    }


def label_day(day_id: int, rows: list[dict]) -> list[dict]:
    series = day_series(rows)
    sc_energy_j = 0.0
    defer_state = init_deferables(rows[0]["deferables"])
    labeled = []

    for row in rows:
        tick = row["tick"]
        physics = row_to_physics(row)
        defer_features = compute_defer_features(defer_state, tick)
        projection = get_price_projection(series, tick)
        sc_soc = sc_energy_j / MAX_USABLE_SC_EN
        obs = build_obs(tick, physics, sc_energy_j, defer_features, projection)

        grid_naive, sc_naive = naive_policy(physics, defer_features)
        grid_oracle, sc_oracle = oracle_policy(physics, defer_features, sc_energy_j, projection)

        sim = simulate_step(
            sc_energy_j,
            physics,
            defer_features,
            projection,
            grid_oracle,
            sc_oracle,
        )
        sc_energy_j = sim["sc_energy_j"]

        labeled.append({
            "recorded_at_utc": row.get("recorded_at_utc"),
            "day": day_id,
            "tick": tick,
            "sun": row.get("sun"),
            "demand": row.get("demand"),
            "buy_price": row.get("buy_price"),
            "sell_price": row.get("sell_price"),
            "deferables": row.get("deferables"),
            "obs": obs,
            "sc_soc": sc_soc,
            "grid_action_naive": grid_naive,
            "sc_action_naive": sc_naive,
            "grid_action": grid_oracle,
            "sc_action": sc_oracle,
            "imported_j": sim["imported_j"],
            "exported_j": sim["exported_j"],
            "arbitrage_import_j": sim["arbitrage_import_j"],
            "arbitrage_export_j": sim["arbitrage_export_j"],
            "unmet_j": sim["unmet_j"],
            "tick_cost": sim["tick_cost"],
            "labeled_at_utc": datetime.now(timezone.utc).isoformat(),
        })

    return labeled


def write_labeled_days(days: dict[int, list[dict]], output_path: Path) -> int:
    row_count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for day_id in sorted(days.keys()):
            labeled_rows = label_day(day_id, days[day_id])
            for row in labeled_rows:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
            row_count += len(labeled_rows)
            logging.info("labeled day=%s (%s ticks) -> %s", day_id, len(labeled_rows), output_path.name)
    return row_count


def main() -> None:
    setup_logging()
    logging.info("loading ticks from %s", INPUT_FILE)
    ticks = load_ticks(INPUT_FILE)
    logging.info("loaded %s raw ticks", len(ticks))

    days = group_by_day(ticks)
    train_days, test_days = split_days(days, TRAIN_RATIO, SPLIT_SEED)

    manifest = {
        "split_seed": SPLIT_SEED,
        "train_ratio": TRAIN_RATIO,
        "total_complete_days": len(days),
        "train_days": len(train_days),
        "test_days": len(test_days),
        "train_day_ids": sorted(train_days.keys()),
        "test_day_ids": sorted(test_days.keys()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    SPLIT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    train_rows = write_labeled_days(train_days, TRAIN_OUTPUT)
    test_rows = write_labeled_days(test_days, TEST_OUTPUT)

    logging.info(
        "done: train=%s rows (%s days), test=%s rows (%s days)",
        train_rows,
        len(train_days),
        test_rows,
        len(test_days),
    )


if __name__ == "__main__":
    main()
