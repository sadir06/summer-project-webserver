"""Evaluate a trained policy on the held-out test days (20% split).

Runs full 60-tick rollouts per day using deterministic policy mean actions.
Reports per-day and total profit in cents, plus timing breakdown.

Does not touch live hardware or inference.py.

Example:
  cd "RL Training"
  python eval_test_set.py --prototype 3
  python eval_test_set.py --prototype 3 --checkpoint checkpoints/prototype_3/policy_latest.pth
"""

from __future__ import annotations

import argparse
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from load_days import load_day_data, profile_to_reset_options
from price_obs import cross_day_options_for_profile
from networks import PolicyNet

RL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RL_DIR.parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate policy on all held-out test days from ticks.jsonl"
    )
    parser.add_argument(
        "--prototype",
        type=int,
        default=3,
        choices=[1, 2, 3, 4],
        help="Environment prototype (default 3)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to policy .pth (default: newest policy*.pth in prototype_N/)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only first N test days (debug)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
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


def clip_action(raw_action: np.ndarray, prototype: int) -> np.ndarray:
    action = raw_action.astype(np.float32).copy()
    if prototype >= 3:
        action[0] = np.clip(action[0], -1.0, 1.0)
        action[1] = np.clip((action[1] + 1.0) / 2.0, 0.0, 1.0)
        return action
    action[0] = np.clip(action[0], -1.0, 0.0)
    action[1] = np.clip(action[1], -1.0, 1.0)
    action[2] = np.clip((action[2] + 1.0) / 2.0, 0.0, 1.0)
    return action


def resolve_checkpoint(prototype: int, checkpoint_arg: str | None) -> Path:
    if checkpoint_arg:
        path = Path(checkpoint_arg)
        if not path.is_absolute():
            path = RL_DIR / path
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    checkpoint_dir = RL_DIR / "checkpoints" / f"prototype_{prototype}"
    candidates = sorted(
        checkpoint_dir.glob("policy*.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint in {checkpoint_dir}")
    return candidates[0]


def load_policy(
    prototype: int, checkpoint_path: Path, device: torch.device, obs_dim: int
) -> PolicyNet:
    policy = PolicyNet(obs_dim=obs_dim, act_dim=act_dim_for_prototype(prototype)).to(device)
    policy.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    policy.eval()
    return policy


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def profit_from_info(info: dict, prototype: int) -> float:
    if prototype >= 3:
        return float(info.get("totalProfitCents", 0.0))
    return -float(info.get("totalCost", 0.0))


def run_day(
    env,
    policy: PolicyNet,
    device: torch.device,
    prototype: int,
    profile: dict,
    *,
    sorted_profiles: list[dict] | None = None,
) -> dict:
    cross_day = None
    if prototype >= 4 and sorted_profiles is not None:
        from env4_prototype_4 import SmartGridEnv as SmartGridEnvProto4

        cross_day = cross_day_options_for_profile(
            profile,
            sorted_profiles,
            SmartGridEnvProto4.crossDayHistoryDays,
            env.ticksPerDay,
        )
    reset_options = profile_to_reset_options(profile, prototype, cross_day=cross_day)
    reset_options["deferables"] = deepcopy(reset_options["deferables"])
    obs, _info = env.reset(options=reset_options)

    day_import_cents = 0.0
    day_export_cents = 0.0
    day_reward = 0.0
    infer_s = 0.0
    env_s = 0.0

    ticks = env.ticksPerDay
    for _ in range(ticks):
        sync_device(device)
        t_infer = time.perf_counter()
        with torch.inference_mode():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            mu, _std = policy(obs_t)
            action = clip_action(mu.cpu().numpy(), prototype)
        sync_device(device)
        infer_s += time.perf_counter() - t_infer

        t_env = time.perf_counter()
        obs, reward, terminated, _truncated, info = env.step(action)
        env_s += time.perf_counter() - t_env

        day_reward += float(reward)
        day_import_cents += float(info.get("costThisTick", 0.0))
        day_export_cents += float(info.get("profitThisTick", 0.0))
        if terminated:
            break

    def_done = sum(
        1 for dd in env.dailyData["defDemandState"] if dd.get("served", False)
    )
    def_missed = sum(
        1 for dd in env.dailyData["defDemandState"] if dd.get("missed", False)
    )
    profit_cents = profit_from_info(info, prototype)

    return {
        "day_id": profile.get("day_id"),
        "profit_cents": profit_cents,
        "import_cents": day_import_cents,
        "export_cents": day_export_cents,
        "reward": day_reward,
        "def_done": def_done,
        "def_missed": def_missed,
        "infer_s": infer_s,
        "env_s": env_s,
        "total_s": infer_s + env_s,
    }


def main():
    args = parse_args()
    device = torch.device(args.device)
    env_class = load_env_module(args.prototype)
    checkpoint_path = resolve_checkpoint(args.prototype, args.checkpoint)
    obs_dim = int(getattr(env_class, "obsDim", 14))
    policy = load_policy(args.prototype, checkpoint_path, device, obs_dim)

    _train, test_profiles, manifest = load_day_data(PROJECT_ROOT, prototype=args.prototype)
    if not test_profiles:
        raise RuntimeError("No test profiles in split manifest")

    sorted_test = sorted(test_profiles, key=lambda p: p["day_id"])

    if args.limit is not None:
        test_profiles = test_profiles[: args.limit]

    env = env_class()
    n_days = len(test_profiles)
    n_ticks = n_days * env.ticksPerDay

    print(f"prototype={args.prototype} env={env_class.__module__} obs_dim={obs_dim}")
    print(f"checkpoint={checkpoint_path}")
    print(f"device={device}")
    print(
        f"test_days={n_days} (of {manifest['test_days']} held-out, "
        f"split_seed={manifest['split_seed']})"
    )
    print(f"profit_unit=cents per simulated day (60 ticks x 5s)")
    print("action=policy mean (deterministic, no sampling)")
    print("-" * 72)

    wall_start = time.perf_counter()
    results = []
    for i, profile in enumerate(test_profiles):
        day_result = run_day(
            env,
            policy,
            device,
            args.prototype,
            profile,
            sorted_profiles=sorted_test if args.prototype >= 4 else None,
        )
        results.append(day_result)
        if (i + 1) % 50 == 0 or i + 1 == n_days:
            print(f"  evaluated {i + 1}/{n_days} days...")

    wall_s = time.perf_counter() - wall_start

    profits = np.array([r["profit_cents"] for r in results], dtype=np.float64)
    imports = np.array([r["import_cents"] for r in results], dtype=np.float64)
    exports = np.array([r["export_cents"] for r in results], dtype=np.float64)
    def_done = np.array([r["def_done"] for r in results], dtype=np.float64)
    infer_total = sum(r["infer_s"] for r in results)
    env_total = sum(r["env_s"] for r in results)

    total_profit = float(profits.sum())
    total_import = float(imports.sum())
    total_export = float(exports.sum())
    days_positive = int((profits > 0).sum())
    days_all_def = int((def_done >= 3).sum())

    print()
    print("=== Test-set results (held-out 20%) ===")
    print(f"days evaluated:        {n_days}")
    print(f"total profit:          {total_profit:+,.1f} cents  (${total_profit / 100:,.2f})")
    print(f"mean profit/day:       {profits.mean():+.1f} cents  (${profits.mean() / 100:+.2f})")
    print(f"median profit/day:     {np.median(profits):+.1f} cents")
    print(f"min / max profit/day:  {profits.min():+.1f} / {profits.max():+.1f} cents")
    print(f"days with profit > 0:  {days_positive}/{n_days} ({100 * days_positive / n_days:.1f}%)")
    print()
    print(f"total grid import:     {total_import:,.1f} cents  (${total_import / 100:,.2f})")
    print(f"total grid export:     {total_export:,.1f} cents  (${total_export / 100:,.2f})")
    print(f"net (export - import): {total_export - total_import:+,.1f} cents")
    print()
    print(f"defer tasks completed: mean {def_done.mean():.2f}/3 per day")
    print(f"days all 3 defer done: {days_all_def}/{n_days}")
    print(f"days with missed defer: {n_days - days_all_def} (approx)")
    print()
    print("=== Timing ===")
    print(f"wall clock total:      {wall_s:.2f} s")
    print(f"env step total:        {env_total:.2f} s  ({100 * env_total / wall_s:.1f}% of wall)")
    print(f"policy infer total:    {infer_total:.2f} s  ({100 * infer_total / wall_s:.1f}% of wall)")
    print(f"per day (mean):        {wall_s / n_days * 1000:.1f} ms")
    print(f"per tick (mean):       {wall_s / n_ticks * 1000:.3f} ms")
    print(f"infer per tick:        {infer_total / n_ticks * 1000:.3f} ms")

    best_idx = int(profits.argmax())
    worst_idx = int(profits.argmin())
    print()
    print(f"best day:  id={results[best_idx]['day_id']} profit={profits[best_idx]:+.1f} cents")
    print(f"worst day: id={results[worst_idx]['day_id']} profit={profits[worst_idx]:+.1f} cents")


if __name__ == "__main__":
    main()
