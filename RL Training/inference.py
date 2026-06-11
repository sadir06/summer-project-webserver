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
    parse_deferables,
    voltage_to_supercap_en,
)
from networks import PolicyNet

RL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RL_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hardware.actions import send_actions  # noqa: E402
from hardware.cloud import fetch_cloud_snapshot  # noqa: E402
from hardware.config import TICK_INTERVAL_S  # noqa: E402
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
        choices=[1, 2],
        help="1=env4.py, 2=env4_prototype_2.py (default)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Day picker seed (eval mode only)")
    return parser.parse_args()


def load_env_module(prototype: int):
    if prototype == 2:
        from env4_prototype_2 import SmartGridEnv

        return SmartGridEnv
    from env4 import SmartGridEnv

    return SmartGridEnv


def pick_day_profile(seed: int = 0):
    train_profiles, test_profiles, manifest = load_day_data(PROJECT_ROOT)
    rng = random.Random(seed)
    pool = test_profiles if test_profiles else train_profiles
    profile = rng.choice(pool)
    return profile, manifest


def clip_action(raw_action: np.ndarray) -> np.ndarray:
    action = raw_action.astype(np.float32).copy()
    action[0] = np.clip(action[0], -1.0, 1.0)
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
    policy = PolicyNet(obs_dim=14, act_dim=3).to(device)
    policy.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    policy.eval()
    return policy, checkpoint_path


def infer_tick(policy: PolicyNet, obs: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        mu, _ = policy(torch.tensor(obs, dtype=torch.float32, device=device))
    return clip_action(mu.cpu().numpy())


def run_eval(args, env_class, device: torch.device) -> None:
    profile, manifest = pick_day_profile(seed=args.seed)

    print(f"mode=eval prototype={args.prototype}")
    print(f"split: train={manifest['train_days']} test={manifest['test_days']}")
    print(f"day_id={profile.get('day_id', 'unknown')}")
    print(f"deferables={profile['deferables']}")
    if args.prototype >= 2:
        print(f"tick0_pv_kw={profile['pvGen'][0]:.3f}")
    else:
        print(f"tick0_irradiance={profile['irradiance'][0]:.3f}")
    print(f"tick0_demand_kw={profile['baseDemand'][0]:.3f}")
    print(f"tick0_buy_norm={profile['buyPrice'][0]:.3f}")
    print(f"tick0_sell_norm={profile['sellPrice'][0]:.3f}")

    env = env_class()
    reset_options = profile_to_reset_options(profile, args.prototype)
    reset_options["deferables"] = deepcopy(reset_options["deferables"])
    obs, _info = env.reset(seed=42, options=reset_options)

    print(f"tick0_pv_kw={env.dailyData['pvGen'][0]:.3f}")
    print(f"supercap_start={env.supercapEn:.3f}")

    policy, checkpoint_path = load_policy(args.prototype, device)
    print(f"checkpoint={checkpoint_path}")

    action = infer_tick(policy, obs, device)
    print("model_actions:")
    if args.prototype >= 2:
        print(f"  surplus_export_action={action[0]:+.3f}")
        print(f"  sc_action={action[1]:+.3f}")
        print(f"  def_action={action[2]:+.3f}")
    else:
        print(f"  grid_action={action[0]:+.3f}")
        print(f"  sc_action={action[1]:+.3f}")
        print(f"  def_action={action[2]:+.3f}")

    _next_obs, reward, _terminated, _truncated, step_info = env.step(action)
    print(f"reward_after_tick={reward:.3f}")
    print(f"total_cost_after_tick={step_info['totalCost']:.3f}")
    if args.prototype >= 2:
        print(f"grid_to_demand={step_info.get('gridToDemand', 0):.3f}")
        print(f"sc_to_demand={step_info.get('scToDemand', 0):.3f}")
        print(f"arbitrage_import={step_info.get('arbitrageImportEn', 0):.3f}")
    print(f"supercap_after_tick={env.supercapEn:.3f}")


def run_live(args, env_class, device: torch.device) -> None:
    if args.prototype < 2:
        raise SystemExit("Live mode requires --prototype 2 (hardware PV inputs).")

    policy, checkpoint_path = load_policy(args.prototype, device)
    day_buffer = DaySeriesBuffer()
    last_tick = None
    last_day = None

    print(f"mode=live prototype={args.prototype}")
    print(f"checkpoint={checkpoint_path}")
    print(f"tick_interval={TICK_INTERVAL_S}s")
    print("waiting for cloud + hardware inputs...")

    while True:
        loop_start = time.monotonic()
        try:
            cloud = fetch_cloud_snapshot()
            hardware = read_hardware()
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
            elapsed = time.monotonic() - loop_start
            time.sleep(max(0.2, 1.0 - elapsed))
            continue

        demand = float(cloud.get("demand") or 0.0)
        buy_price = float(cloud.get("buy_price") or 0.0)
        sell_price = float(cloud.get("sell_price") or 0.0)
        defer_state = parse_deferables(cloud.get("deferables") or [])

        pvout_w = hardware.get("pvout")
        vcap_v = hardware.get("vcap")
        pv_kw = (float(pvout_w) / 1000.0) if pvout_w is not None else 0.0
        supercap_en = (
            voltage_to_supercap_en(float(vcap_v), env_class)
            if vcap_v is not None
            else 0.0
        )

        obs = build_observation(
            env_class=env_class,
            tick=int(tick),
            demand_kw=demand,
            buy_price_raw=buy_price,
            sell_price_raw=sell_price,
            pv_kw=pv_kw,
            supercap_en=supercap_en,
            defer_state=defer_state,
            day_buffer=day_buffer,
        )

        infer_start = time.monotonic()
        action = infer_tick(policy, obs, device)
        infer_ms = (time.monotonic() - infer_start) * 1000.0

        results = send_actions(action[0], action[1], action[2])

        print(
            f"tick={tick} day={day} pv_kw={pv_kw:.3f} demand={demand:.3f} "
            f"sc_en={supercap_en:.3f} infer_ms={infer_ms:.1f}"
        )
        print(
            f"  actions: grid={action[0]:+.3f} sc={action[1]:+.3f} def={action[2]:+.3f} "
            f"sent={results}"
        )

        last_tick = tick
        last_day = day

        elapsed = time.monotonic() - loop_start
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
