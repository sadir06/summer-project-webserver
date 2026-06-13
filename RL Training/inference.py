import argparse
import concurrent.futures
import random
import sys
import time

import requests
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from load_days import load_day_data, profile_to_reset_options
from live_inputs import (
    CrossDayPriceArchive,
    DaySeriesBuffer,
    build_observation,
    compute_defer_power_served_w,
    compute_pico_demand_power,
    effective_sc_action_for_voltage,
    parse_deferables,
    voltage_to_supercap_en,
)
from price_obs import cross_day_options_for_profile
from networks import PolicyNet

RL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RL_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hardware.actions import send_actions, send_load_demand  # noqa: E402
from hardware.cloud import fetch_cloud_snapshot, fetch_cloud_tick_day  # noqa: E402
from hardware.config import (  # noqa: E402
    FLASK_PORT,
    INFERENCE_ACTION_TIMEOUT_S,
    LAPTOP_IP,
    TICK_INTERVAL_S,
)
from hardware.poller import read_hardware_for_inference  # noqa: E402
from hardware.inference_publish import (  # noqa: E402
    estimate_grid_power_w,
    publish_inference_tick,
    tick_profit_cents,
)
TICK_POLL_S = 0.25


def _fatal_tick_sync(msg: str) -> None:
    print(f"FATAL: {msg}")
    sys.exit(1)


def _check_no_tick_skip(
    *,
    last_tick: int | None,
    last_day: int | None,
    tick: int,
    day: int,
    context: str,
) -> None:
    if last_tick is None:
        return
    if day != last_day:
        return
    if tick > last_tick + 1:
        missed = tick - last_tick - 1
        _fatal_tick_sync(
            f"{context}: Azure tick jumped {last_tick} -> {tick} "
            f"(missed {missed} tick(s) on day {day})"
        )
    if tick <= last_tick:
        _fatal_tick_sync(
            f"{context}: Azure tick went backwards or repeated "
            f"(last={last_tick}, got={tick}, day={day})"
        )


def _fetch_inputs_for_tick() -> tuple[dict, dict]:
    """Parallel Azure snapshot + fast Pico poll (wall time ≈ max of the two)."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        cloud_future = pool.submit(fetch_cloud_snapshot)
        hardware_future = pool.submit(read_hardware_for_inference)
        return cloud_future.result(), hardware_future.result()


def _wait_and_fetch_inputs(
    last_tick: int | None,
    last_day: int | None,
) -> tuple[dict, dict, float, float]:
    """Wait for Azure tick advance, then fetch snapshot+hardware immediately (one tick)."""
    wait_start = time.perf_counter()
    while True:
        try:
            tick_poll, day_poll = fetch_cloud_tick_day()
        except Exception as e:
            print(f"tick wait error: {e}")
            time.sleep(1)
            continue
        if tick_poll is None or day_poll is None:
            time.sleep(TICK_POLL_S)
            continue

        advanced = last_tick is None or day_poll != last_day or tick_poll > last_tick
        if not advanced:
            time.sleep(TICK_POLL_S)
            continue

        if last_tick is not None:
            _check_no_tick_skip(
                last_tick=last_tick,
                last_day=last_day,
                tick=tick_poll,
                day=day_poll,
                context="while waiting for next tick",
            )
        break

    wait_ms = (time.perf_counter() - wait_start) * 1000.0
    fetch_start = time.perf_counter()
    cloud, hardware = _fetch_inputs_for_tick()
    fetch_ms = (time.perf_counter() - fetch_start) * 1000.0
    tick_i = int(cloud["tick"])
    day_i = int(cloud["day"])

    if last_tick is not None:
        _check_no_tick_skip(
            last_tick=last_tick,
            last_day=last_day,
            tick=tick_i,
            day=day_i,
            context="after input fetch",
        )

    return cloud, hardware, wait_ms, fetch_ms


def parse_args():
    parser = argparse.ArgumentParser(description="Run policy inference (live hardware or eval)")
    parser.add_argument(
        "--mode",
        choices=["live", "eval"],
        default="live",
        help="live=continuous hardware loop (default), eval=single tick from dataset",
    )
    parser.add_argument(
        "--prototype",
        type=int,
        default=3,
        choices=[1, 2, 3, 4],
        help="1=env4, 2=proto2, 3=proto3, 4=proto4 voltage limits (default 3)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Day picker seed (eval mode only)")
    parser.add_argument(
        "--new-run",
        action="store_true",
        help="Start a new inference recording run on Flask (data/inference_runs/)",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional label for the inference run folder",
    )
    return parser.parse_args()


def load_env_module(prototype: int):
    if prototype == 4:
        from env4_prototype_4 import SmartGridEnv

        return SmartGridEnv
    if prototype == 3:
        from env4_prototype_3 import SmartGridEnv

        return SmartGridEnv
    if prototype == 2:
        from env4_prototype_2 import SmartGridEnv

        return SmartGridEnv
    from env4 import SmartGridEnv

    return SmartGridEnv


def act_dim_for_prototype(prototype: int) -> int:
    return 2 if prototype >= 3 else 3


def pick_day_profile(seed: int = 0, prototype: int = 2):
    train_profiles, test_profiles, manifest = load_day_data(
        PROJECT_ROOT, prototype=prototype
    )
    rng = random.Random(seed)
    pool = test_profiles if test_profiles else train_profiles
    profile = rng.choice(pool)
    return profile, manifest


def clip_action(raw_action: np.ndarray, prototype: int) -> np.ndarray:
    action = raw_action.astype(np.float32).copy()
    if prototype >= 3:
        action[0] = np.clip(action[0], -1.0, 1.0)
        action[1] = np.clip((action[1] + 1.0) / 2.0, 0.0, 1.0)
        return action
    # Prototype 2: only negative grid_action exports PV surplus; +1 does nothing.
    action[0] = np.clip(action[0], -1.0, 0.0)
    action[1] = np.clip(action[1], -1.0, 1.0)
    action[2] = np.clip((action[2] + 1.0) / 2.0, 0.0, 1.0)
    return action


def resolve_checkpoint(prototype: int) -> Path:
    """Newest policy*.pth in checkpoints/prototype_N/ (by file mtime)."""
    checkpoint_dir = RL_DIR / "checkpoints" / f"prototype_{prototype}"
    candidates = sorted(
        checkpoint_dir.glob("policy*.pth"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found in RL Training/checkpoints/prototype_{prototype}/"
        )
    return candidates[0]


def load_policy(prototype: int, device: torch.device, obs_dim: int) -> PolicyNet:
    checkpoint_path = resolve_checkpoint(prototype)
    policy = PolicyNet(obs_dim=obs_dim, act_dim=act_dim_for_prototype(prototype)).to(device)
    policy.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    policy.eval()
    return policy, checkpoint_path


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def infer_tick(
    policy: PolicyNet, obs: np.ndarray, device: torch.device, prototype: int
) -> np.ndarray:
    action, _infer_ms = timed_infer_tick(policy, obs, device, prototype)
    return action


def timed_infer_tick(
    policy: PolicyNet, obs: np.ndarray, device: torch.device, prototype: int
) -> tuple[np.ndarray, float]:
    sync_device(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        mu, _ = policy(torch.tensor(obs, dtype=torch.float32, device=device))
        action = clip_action(mu.cpu().numpy(), prototype)
    sync_device(device)
    return action, (time.perf_counter() - t0) * 1000.0


def run_eval(args, env_class, device: torch.device) -> None:
    profile, manifest = pick_day_profile(seed=args.seed, prototype=args.prototype)

    print(f"mode=eval prototype={args.prototype}")
    print(f"split: train={manifest['train_days']} test={manifest['test_days']}")
    print(f"day_id={profile.get('day_id', 'unknown')}")
    print(f"deferables={profile['deferables']}")
    if args.prototype >= 2:
        print(f"tick0_pv_w={profile['pvGen'][0]:.3f}")
    else:
        print(f"tick0_irradiance={profile['irradiance'][0]:.3f}")
    print(f"tick0_demand_w={profile['baseDemand'][0]:.3f}")
    if args.prototype >= 3:
        print(f"tick0_buy_cents_J={profile['buyPrice'][0]:.3f}")
        print(f"tick0_sell_cents_J={profile['sellPrice'][0]:.3f}")
    else:
        print(f"tick0_buy_norm={profile['buyPrice'][0]:.3f}")
        print(f"tick0_sell_norm={profile['sellPrice'][0]:.3f}")

    env = env_class()
    cross_day = None
    if args.prototype >= 4:
        _train, test_profiles, _manifest = load_day_data(
            PROJECT_ROOT, prototype=args.prototype
        )
        pool = sorted(test_profiles or _train, key=lambda p: p["day_id"])
        cross_day = cross_day_options_for_profile(
            profile,
            pool,
            env_class.crossDayHistoryDays,
            env_class.ticksPerDay,
        )
    reset_options = profile_to_reset_options(profile, args.prototype, cross_day=cross_day)
    reset_options["deferables"] = deepcopy(reset_options["deferables"])
    obs, _info = env.reset(seed=42, options=reset_options)

    print(f"tick0_pv_w={env.dailyData['pvGen'][0]:.3f}")
    print(f"supercap_start={env.supercapEn:.3f}")

    policy, checkpoint_path = load_policy(
        args.prototype, device, int(getattr(env_class, "obsDim", 14))
    )
    print(f"checkpoint={checkpoint_path}")

    action = infer_tick(policy, obs, device, args.prototype)
    print("model_actions:")
    if args.prototype >= 3:
        print(f"  sc_action={action[0]:+.3f}")
        print(f"  def_action={action[1]:+.3f}")
    elif args.prototype >= 2:
        print(f"  surplus_export_action={action[0]:+.3f}")
        print(f"  sc_action={action[1]:+.3f}")
        print(f"  def_action={action[2]:+.3f}")
    else:
        print(f"  grid_action={action[0]:+.3f}")
        print(f"  sc_action={action[1]:+.3f}")
        print(f"  def_action={action[2]:+.3f}")

    _next_obs, reward, _terminated, _truncated, step_info = env.step(action)
    print(f"reward_after_tick={reward:.3f}")
    if args.prototype >= 3:
        print(f"tick_profit_cents={step_info.get('tickProfitCents', 0):.3f}")
        print(f"total_profit_cents={step_info.get('totalProfitCents', 0):.3f}")
        print(f"grid_to_demand={step_info.get('gridToDemand', 0):.3f}")
        print(f"sc_to_demand={step_info.get('scToDemand', 0):.3f}")
        print(f"grid_to_sc={step_info.get('gridToSc', 0):.3f}")
        print(f"sc_to_grid={step_info.get('scToGrid', 0):.3f}")
    else:
        print(f"total_cost_after_tick={step_info['totalCost']:.3f}")
        if args.prototype >= 2:
            print(f"grid_to_demand={step_info.get('gridToDemand', 0):.3f}")
            print(f"sc_to_demand={step_info.get('scToDemand', 0):.3f}")
            print(f"arbitrage_import={step_info.get('arbitrageImportEn', 0):.3f}")
    print(f"supercap_after_tick={env.supercapEn:.3f}")


def run_live(args, env_class, device: torch.device) -> None:
    if args.prototype < 2:
        raise SystemExit("Live mode requires --prototype 2 or 3 (hardware PV inputs).")

    policy, checkpoint_path = load_policy(
        args.prototype, device, int(getattr(env_class, "obsDim", 14))
    )
    day_buffer = DaySeriesBuffer()
    cross_day_archive = CrossDayPriceArchive(
        max_days=getattr(env_class, "crossDayHistoryDays", 7)
    )
    last_tick = None
    last_day = None

    print(f"mode=live prototype={args.prototype}")
    print(f"checkpoint={checkpoint_path}")
    print(f"device={device}")
    print(f"tick_interval={TICK_INTERVAL_S}s")

    if args.new_run:
        try:
            response = requests.post(
                f"http://127.0.0.1:{FLASK_PORT}/api/inference_run/start",
                json={
                    "run_name": args.run_name,
                    "meta": {
                        "prototype": args.prototype,
                        "checkpoint": str(checkpoint_path),
                        "mode": "live",
                    },
                },
                timeout=2.0,
            )
            response.raise_for_status()
            run_dir = response.json().get("run_dir", "?")
            print(f"inference recording started -> {run_dir}")
        except Exception as e:
            print(
                f"warning: could not start inference recording on Flask "
                f"(is app running with --new-inference-run?): {e}"
            )
    print(
        f"MPPT Pico needs Flask running: "
        f"http://{LAPTOP_IP}:{FLASK_PORT}/api/sun_data "
        f"(set laptop Wi-Fi IP to {LAPTOP_IP} on the hotspot)"
    )
    print(
        f"Dashboard telemetry: POST http://{LAPTOP_IP}:{FLASK_PORT}/api/inference_tick "
        f"(start Flask first for live charts)"
    )
    print("waiting for cloud + hardware inputs...")
    print(f"sync: one inference step per Azure tick (~{TICK_INTERVAL_S}s on game server)")
    print("input: Azure price+demand+deferables, Pico pvout+vcap only (no sun, no cap /p)")
    print(
        "policy: exit on tick skip or if fetch+infer+flask exceeds 5s "
        "(Azure wait time is not counted)"
    )

    while True:
        try:
            cloud, hardware, wait_ms, fetch_ms = _wait_and_fetch_inputs(last_tick, last_day)
        except Exception as e:
            _fatal_tick_sync(f"input fetch failed: {e}")

        tick = cloud.get("tick")
        day = cloud.get("day")
        if tick is None or day is None:
            _fatal_tick_sync("Azure snapshot missing tick or day")

        tick_i, day_i = int(tick), int(day)

        if last_day is not None and day_i != last_day:
            cross_day_archive.complete_day_from_buffer(
                day_buffer.series, env_class.ticksPerDay
            )
        day_buffer.reset_if_new_day(day)

        demand = float(cloud.get("demand") or 0.0)
        buy_price = float(cloud.get("buy_price") or 0.0)
        sell_price = float(cloud.get("sell_price") or 0.0)
        defer_state = parse_deferables(cloud.get("deferables") or [])

        pvout_w = hardware.get("pvout")
        vcap_v = hardware.get("vcap")
        pv_w = float(pvout_w) if pvout_w is not None else 0.0
        vcap_f = float(vcap_v) if vcap_v is not None else None
        supercap_en = (
            voltage_to_supercap_en(float(vcap_v), env_class)
            if vcap_v is not None
            else 0.0
        )

        process_start = time.perf_counter()
        obs = build_observation(
            env_class=env_class,
            tick=int(tick),
            demand_w=demand,
            buy_price_raw=buy_price,
            sell_price_raw=sell_price,
            pv_w=pv_w,
            supercap_en=supercap_en,
            defer_state=defer_state,
            day_buffer=day_buffer,
            cross_day_archive=cross_day_archive,
        )
        action, infer_ms = timed_infer_tick(policy, obs, device, args.prototype)
        process_ms = (time.perf_counter() - process_start) * 1000.0

        if args.prototype >= 3:
            sc_action = action[0]
            def_action = action[1]
            grid_action = None
        else:
            grid_action = action[0]
            sc_action = action[1]
            def_action = action[2]

        demand_output = compute_pico_demand_power(
            instant_demand=demand,
            defer_state=defer_state,
            def_action=def_action,
            tick_dur_s=TICK_INTERVAL_S,
            load_max_w=env_class.loadDemandMax,
        )
        defer_power_w = compute_defer_power_served_w(
            demand,
            defer_state,
            def_action,
            TICK_INTERVAL_S,
            load_max_w=env_class.loadDemandMax,
        )
        sc_action_grid = (
            effective_sc_action_for_voltage(sc_action, vcap_f, env_class)
            if args.prototype >= 4
            else sc_action
        )
        grid_import_w, grid_export_w, sc_bus_w = estimate_grid_power_w(
            total_demand_w=demand_output,
            pv_w=pv_w,
            sc_action=sc_action_grid,
            pcout_w=None,
        )
        profit_cents = tick_profit_cents(
            grid_import_w=grid_import_w,
            grid_export_w=grid_export_w,
            buy_price=buy_price,
            sell_price=sell_price,
        )

        publish_start = time.perf_counter()
        flask_ok = publish_inference_tick(
            {
                "tick": int(tick),
                "day": int(day),
                "instant_demand_w": demand,
                "defer_power_w": defer_power_w,
                "total_demand_w": demand_output,
                "pv_w": pv_w,
                "sc_action": float(sc_action),
                "sc_bus_w": sc_bus_w,
                "pcout_w": sc_bus_w,
                "vcap_v": vcap_f,
                "grid_import_w": grid_import_w,
                "grid_export_w": grid_export_w,
                "buy_price": buy_price,
                "sell_price": sell_price,
                "tick_profit_cents": profit_cents,
            }
        )
        publish_ms = (time.perf_counter() - publish_start) * 1000.0
        output_ms = publish_ms

        results = send_actions(
            sc_action_grid if args.prototype >= 4 else sc_action,
            grid_action=grid_action,
            timeout_s=INFERENCE_ACTION_TIMEOUT_S,
        )
        results["load"] = flask_ok
        results["load_pico"] = send_load_demand(
            day=int(day),
            tick=int(tick),
            total_demand_w=demand_output,
        )

        work_ms = fetch_ms + process_ms + output_ms
        budget_ms = TICK_INTERVAL_S * 1000.0
        if work_ms > budget_ms:
            print(
                f"  bench: wait={wait_ms:.1f}ms fetch={fetch_ms:.1f}ms "
                f"process={process_ms:.1f}ms flask={publish_ms:.1f}ms "
                f"work={work_ms:.1f}ms budget={budget_ms:.0f}ms"
            )
            _fatal_tick_sync(
                f"tick {tick_i} fetch+infer+flask took {work_ms:.0f}ms "
                f"(budget {budget_ms:.0f}ms) — would skip next Azure tick"
            )

        print(
            f"tick={tick} day={day} pv_w={pv_w:.3f} demand={demand:.3f} "
            f"sc_en={supercap_en:.3f}"
        )
        print(
            f"  bench: wait={wait_ms:.1f}ms fetch={fetch_ms:.1f}ms "
            f"process={process_ms:.1f}ms (infer={infer_ms:.1f}ms) "
            f"flask={publish_ms:.1f}ms pico=sync work={work_ms:.1f}ms "
            f"budget={budget_ms:.0f}ms headroom={budget_ms - work_ms:.1f}ms"
        )

        if args.prototype >= 3:
            action_line = (
                f"  actions: sc={sc_action:+.3f} def_frac={def_action:+.3f} "
                f"demand_out={demand_output:.3f} (instant={demand:.3f} "
                f"+ defer={defer_power_w:.3f}) sent={results}"
            )
        else:
            action_line = (
                f"  actions: grid={grid_action:+.3f} sc={sc_action:+.3f} "
                f"def_frac={def_action:+.3f} demand_out={demand_output:.3f} "
                f"(instant={demand:.3f} + defer={defer_power_w:.3f}) "
                f"sent={results}"
            )
        print(action_line)

        last_tick = tick_i
        last_day = day_i


def main():
    args = parse_args()
    env_class = load_env_module(args.prototype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "eval":
        run_eval(args, env_class, device)
    else:
        run_live(args, env_class, device)


if __name__ == "__main__":
    main()
