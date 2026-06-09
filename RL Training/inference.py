import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from env4 import SmartGridEnv
from load_days import load_day_data
from networks import PolicyNet

RL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RL_DIR.parent
CHECKPOINT_DIR = RL_DIR / "checkpoints"

def pick_day_profile(seed: int = 0):
    """Load one real day in"""
    train_profiles, test_profiles, manifest = load_day_data(PROJECT_ROOT)
    rng = random.Random(seed)
    pool = test_profiles if test_profiles else train_profiles
    profile = rng.choice(pool)
    return profile, manifest

def clip_action(raw_action: np.ndarray) -> np.ndarray:
    """Clip action to the valid range [-1.0, 1.0]"""
    action = raw_action.astype(np.float32).copy()
    action[0] = np.clip(action[0], -1.0, 1.0) # grid action
    action[1] = np.clip(action[1], -1.0, 1.0) # super capacitor action
    action[2] = np.clip((action[2] + 1.0) / 2.0, 0.0, 1.0) # deferrable action
    return action

def resolve_checkpoint() -> Path:
    candidates = [
        CHECKPOINT_DIR / "policy_latest.pth",
        CHECKPOINT_DIR / "policy_final.pth",
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        raise FileNotFoundError("No checkpoint found in RL Training/checkpoints/")
    return max(existing, key=lambda path: path.stat().st_mtime)


def load_policy(device: torch.device) -> PolicyNet:
    checkpoint_path = resolve_checkpoint()
    policy = PolicyNet(obs_dim=14, act_dim=3).to(device)
    policy.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    policy.eval() # Runs inference
    return policy

def infer_tick(policy: PolicyNet, obs: np.ndarray, device: torch.device) -> np.ndarray:
    """Run inference for one tick"""
    with torch.no_grad(): # Disable gradient tracking
        mu, _ = policy(torch.tensor(obs, dtype=torch.float32, device=device))
    return clip_action(mu.cpu().numpy()) # Clip the action to the valid range [-1.0, 1.0], and return the action as a numpy array to the cpu

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    profile, manifest = pick_day_profile()

    print(f"split: train={manifest['train_days']} test={manifest['test_days']}")
    print(f"day_id={profile.get('day_id', 'unknown')}")
    print(f"deferables={profile['deferables']}")
    print(f"tick0_irradiance={profile['irradiance'][0]:.3f}")
    print(f"tick0_demand_kw={profile['baseDemand'][0]:.3f}")
    print(f"tick0_buy_norm={profile['buyPrice'][0]:.3f}")
    print(f"tick0_sell_norm={profile['sellPrice'][0]:.3f}")

    env = SmartGridEnv()
    obs, info = env.reset(
        seed=42,
        options={
            "irradiance": profile["irradiance"],
            "baseDemand": profile["baseDemand"],
            "buyPrice": profile["buyPrice"],
            "sellPrice": profile["sellPrice"],
            "deferables": deepcopy(profile["deferables"]),
        },
    )

    print(f"tick0_pv_kw={env.dailyData['pvGen'][0]:.3f}")
    print(f"supercap_start={env.supercapEn:.3f}")
    
    checkpoint_path = resolve_checkpoint()
    print(f"checkpoint={checkpoint_path.name}")

    policy = load_policy(device)
    action = infer_tick(policy, obs, device)
    
    print("model_actions:")
    print(f"  grid_action={action[0]:+.3f}")
    print(f"  sc_action={action[1]:+.3f}")
    print(f"  def_action={action[2]:+.3f}")

    next_obs, reward, terminated, truncated, step_info = env.step(action)
    print(f"reward_after_tick={reward:.3f}")
    print(f"total_cost_after_tick={step_info['totalCost']:.3f}")
    print(f"supercap_after_tick={env.supercapEn:.3f}")


if __name__ == "__main__":
    main()