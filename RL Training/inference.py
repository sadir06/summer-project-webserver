import argparse
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from load_days import load_day_data, profile_to_reset_options
from live_inputs import (
    DaySeriesBuffer,
    build_observation,
    compute_pico_demand_power,
    parse_deferables,
    total_deferrable_energy,
    voltage_to_supercap_en,
)
from networks import PolicyNet

RL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RL_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hardware.actions import send_actions  # noqa: E402
from hardware.cloud import fetch_cloud_snapshot  # noqa: E402
from hardware.config import FLASK_PORT, LAPTOP_IP, TICK_INTERVAL_S  # noqa: E402
from hardware.poller import read_hardware  # noqa: E402


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
        default=2,
        choices=[1, 2, 3, 4],
        help="1=env4, 2=proto2, 3=proto3, 4=proto4 (default 2)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Day picker seed (eval mode only)")
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
    final_ckpt = checkpoint_dir / "policy_final.pth"
    if final_ckpt in candidates:
        return final_ckpt
    return candidates[0]


def load_policy(prototype: int, device: torch.device) -> PolicyNet:
    checkpoint_path = resolve_checkpoint(prototype)
    policy = PolicyNet(obs_dim=14, act_dim=act_dim_for_prototype(prototype)).to(device)
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
    reset_options = profile_to_reset_options(profile, args.prototype)
    reset_options["deferables"] = deepcopy(reset_options["deferables"])
    obs, _info = env.reset(seed=42, options=reset_options)

    print(f"tick0_pv_w={env.dailyData['pvGen'][0]:.3f}")
    print(f"supercap_start={env.supercapEn:.3f}")

    policy, checkpoint_path = load_policy(args.prototype, device)
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

    policy, checkpoint_path = load_policy(args.prototype, device)
    day_buffer = DaySeriesBuffer()
    last_tick = None
    last_day = None

    print(f"mode=live prototype={args.prototype}")
    print(f"checkpoint={checkpoint_path}")
    print(f"device={device}")
    print(f"tick_interval={TICK_INTERVAL_S}s")
    print(
        f"MPPT Pico needs Flask running: "
        f"http://{LAPTOP_IP}:{FLASK_PORT}/api/sun_data "
        f"(set laptop Wi-Fi IP to {LAPTOP_IP} on the hotspot)"
    )
    print("waiting for cloud + hardware inputs...")

    while True:
        loop_start = time.perf_counter()
        try:
            input_start = time.perf_counter()
            cloud = fetch_cloud_snapshot()
            hardware = read_hardware()
            input_ms = (time.perf_counter() - input_start) * 1000.0
        except Exception as e:
            print(f"poll error: {e}")
            time.sleep(1)
            continue

        tick = cloud.get("tick")
        day = cloud.get("day")
        if tick is None:
            time.sleep(0.5)
            continue

        day_buffer.reset_if_new_day(day)

        if tick == last_tick and day == last_day:
            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0.2, 1.0 - elapsed))
            continue

        demand = float(cloud.get("demand") or 0.0)
        buy_price = float(cloud.get("buy_price") or 0.0)
        sell_price = float(cloud.get("sell_price") or 0.0)
        defer_state = parse_deferables(cloud.get("deferables") or [])

        pvout_w = hardware.get("pvout")
        vcap_v = hardware.get("vcap")
        pv_w = float(pvout_w) if pvout_w is not None else 0.0
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
        )

        output_start = time.perf_counter()
        results = send_actions(sc_action, demand_output, grid_action=grid_action)
        output_ms = (time.perf_counter() - output_start) * 1000.0

        total_ms = input_ms + process_ms + output_ms
        budget_ms = TICK_INTERVAL_S * 1000.0

        print(
            f"tick={tick} day={day} pv_w={pv_w:.3f} demand={demand:.3f} "
            f"sc_en={supercap_en:.3f}"
        )
        print(
            f"  bench: input={input_ms:.1f}ms "
            f"process={process_ms:.1f}ms (infer={infer_ms:.1f}ms) "
            f"output={output_ms:.1f}ms total={total_ms:.1f}ms "
            f"budget={budget_ms:.0f}ms headroom={budget_ms - total_ms:.1f}ms"
        )
        defer_en = total_deferrable_energy(defer_state)
        if args.prototype >= 3:
            action_line = (
                f"  actions: sc={sc_action:+.3f} def_frac={def_action:+.3f} "
                f"demand_out={demand_output:.3f} (instant={demand:.3f} "
                f"+ defer={defer_en * def_action / TICK_INTERVAL_S:.3f}) sent={results}"
            )
        else:
            action_line = (
                f"  actions: grid={grid_action:+.3f} sc={sc_action:+.3f} "
                f"def_frac={def_action:+.3f} demand_out={demand_output:.3f} "
                f"(instant={demand:.3f} + defer={defer_en * def_action / TICK_INTERVAL_S:.3f}) "
                f"sent={results}"
            )
        print(action_line)

        last_tick = tick
        last_day = day

        elapsed = time.perf_counter() - loop_start
        sleep_s = max(0.0, TICK_INTERVAL_S - elapsed)
        if sleep_s > 0:
            time.sleep(sleep_s)


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
