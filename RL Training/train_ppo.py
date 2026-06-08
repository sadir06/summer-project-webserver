import importlib.util
import random
import sys
import time
from collections import deque
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from load_days import load_train_day_profiles
from networks import PolicyNet, ValueNet

RL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RL_DIR.parent

env_path = RL_DIR / "Env_ver3.py"
spec = importlib.util.spec_from_file_location("env_ver3", env_path)
env_mod = importlib.util.module_from_spec(spec)
sys.modules["env_ver3"] = env_mod
spec.loader.exec_module(env_mod)
SmartGridEnv = env_mod.SmartGridEnv

CHECKPOINT_DIR = RL_DIR / "checkpoints"
LOG_FILE = RL_DIR / "train_ppo.log"

OBS_DIM = 14
ACT_DIM = 2

# Phase 2: fine-tune cost optimization after phase-1 blackout avoidance (updates 1-100).
TOTAL_UPDATES = 400
EPISODES_PER_ROLLOUT = 32
PPO_EPOCHS = 4
CLIP_EPS = 0.2
GAMMA = 0.99
GAE_LAMBDA = 0.95
LR = 1e-4
VALUE_COEF = 0.5
ENTROPY_COEF = 0.02
MAX_GRAD_NORM = 0.5
REWARD_SCALE = 0.01
SEED = 42
RESUME_FROM_CHECKPOINT = True
ROLLING_WINDOW = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SmartGridEnvReal(SmartGridEnv):
    def __init__(self, day_profiles: list[dict]):
        super().__init__()
        self.day_profiles = day_profiles
        self._rng = random.Random(SEED)

        # Real deferrable loads are much larger than the synthetic reset defaults.
        self.maxDefEn = 100.0
        self.maxDueSoon = 100.0

        # Softer than -100 so the agent can trade import cost vs. small unmet gaps.
        self.penaltyDemand = -15.0

    def reset(self, seed=None, options=None):
        super(SmartGridEnv, self).reset(seed=seed)

        if seed is not None:
            self._rng = random.Random(seed)

        profile = self._rng.choice(self.day_profiles)

        self.curTick = 0
        self.supercapEn = 0.0
        self.totalCost = 0.0
        self.totalUnmetDemand = 0.0
        self.prevUnmetDemand = 0.0

        self.dailyData = {
            "pvGen": profile["pvGen"][:],
            "baseDemand": profile["baseDemand"][:],
            "buyPrice": profile["buyPrice"][:],
            "sellPrice": profile["sellPrice"][:],
            "defDemandState": deepcopy(profile["defDemandState"]),
        }

        return self.getObs(), self.getInfo()


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def collect_rollout(env, policy, value_net, episodes: int, device: torch.device):
    obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
    ep_returns = []
    ep_costs = []
    ep_unmet = []
    ep_grid_actions = []
    ep_sc_actions = []

    policy.eval()
    value_net.eval()

    for _ in range(episodes):
        obs, _info = env.reset()
        ep_r = 0.0
        ep_cost = 0.0
        ep_u = 0.0
        grid_actions = []
        sc_actions = []

        for _tick in range(env.ticksPerDay):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)

            with torch.no_grad():
                mu, std = policy(obs_t)
                dist = torch.distributions.Normal(mu, std)
                raw_action = dist.sample()
                logp = dist.log_prob(raw_action).sum()
                v = value_net(obs_t)

            action = torch.clamp(raw_action, -1.0, 1.0).cpu().numpy().astype(np.float32)
            grid_actions.append(float(action[0]))
            sc_actions.append(float(action[1]))

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            scaled_reward = float(reward) * REWARD_SCALE

            obs_buf.append(obs)
            act_buf.append(raw_action.cpu().numpy())
            logp_buf.append(logp.item())
            rew_buf.append(scaled_reward)
            val_buf.append(v.item())
            done_buf.append(float(done))

            ep_r += reward
            ep_cost = info.get("totalCost", ep_cost)
            ep_u = info.get("totalUnmetDemand", ep_u)
            obs = next_obs

            if done:
                break

        ep_returns.append(ep_r)
        ep_costs.append(ep_cost)
        ep_unmet.append(ep_u)
        ep_grid_actions.append(float(np.mean(grid_actions)))
        ep_sc_actions.append(float(np.mean(sc_actions)))

    return {
        "obs": torch.tensor(np.array(obs_buf), dtype=torch.float32, device=device),
        "actions": torch.tensor(np.array(act_buf), dtype=torch.float32, device=device),
        "logprobs": torch.tensor(np.array(logp_buf), dtype=torch.float32, device=device),
        "rewards": torch.tensor(np.array(rew_buf), dtype=torch.float32, device=device),
        "values": torch.tensor(np.array(val_buf), dtype=torch.float32, device=device),
        "dones": torch.tensor(np.array(done_buf), dtype=torch.float32, device=device),
        "ep_returns": ep_returns,
        "ep_costs": ep_costs,
        "ep_unmet": ep_unmet,
        "ep_grid_actions": ep_grid_actions,
        "ep_sc_actions": ep_sc_actions,
    }


def compute_gae(rewards, values, dones, gamma, lam):
    T = len(rewards)
    advantages = torch.zeros(T, device=rewards.device)
    gae = 0.0

    for t in reversed(range(T)):
        if dones[t]:
            next_v = 0.0
            gae = 0.0
        else:
            next_v = values[t + 1]

        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae

    returns = advantages + values
    return advantages, returns


def ppo_update(policy, value_net, opt_policy, opt_value, batch):
    obs = batch["obs"]
    actions = batch["actions"]
    old_logprobs = batch["logprobs"]
    rewards = batch["rewards"]
    values = batch["values"]
    dones = batch["dones"]

    advantages, returns = compute_gae(rewards, values, dones, GAMMA, GAE_LAMBDA)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    policy.train()
    value_net.train()

    policy_loss_val = 0.0
    value_loss_val = 0.0
    entropy_val = 0.0

    for _ in range(PPO_EPOCHS):
        mu, std = policy(obs)
        dist = torch.distributions.Normal(mu, std)
        new_logprobs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()

        ratio = torch.exp(new_logprobs - old_logprobs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages
        policy_loss = -torch.min(surr1, surr2).mean() - ENTROPY_COEF * entropy

        v_pred = value_net(obs)
        value_loss = nn.functional.smooth_l1_loss(v_pred, returns)

        opt_policy.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
        opt_policy.step()

        opt_value.zero_grad()
        (VALUE_COEF * value_loss).backward()
        nn.utils.clip_grad_norm_(value_net.parameters(), MAX_GRAD_NORM)
        opt_value.step()

        policy_loss_val = float(policy_loss.item())
        value_loss_val = float(value_loss.item())
        entropy_val = float(entropy.item())

    return policy_loss_val, value_loss_val, entropy_val


def load_checkpoint(policy, value_net, opt_policy, opt_value):
    policy_path = CHECKPOINT_DIR / "policy_latest.pth"
    value_path = CHECKPOINT_DIR / "value_latest.pth"
    if not policy_path.exists() or not value_path.exists():
        return False

    policy.load_state_dict(torch.load(policy_path, map_location=DEVICE, weights_only=True))
    value_net.load_state_dict(torch.load(value_path, map_location=DEVICE, weights_only=True))
    return True


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    log("=== phase 2 training start ===")
    log(
        f"device={DEVICE} lr={LR} reward_scale={REWARD_SCALE} "
        f"penalty={-15} entropy={ENTROPY_COEF} resume={RESUME_FROM_CHECKPOINT}"
    )
    if DEVICE.type == "cuda":
        log(f"gpu={torch.cuda.get_device_name(0)}")

    log("loading train day profiles...")
    day_profiles = load_train_day_profiles(PROJECT_ROOT)
    log(f"loaded {len(day_profiles)} train days")

    env = SmartGridEnvReal(day_profiles)

    policy = PolicyNet(OBS_DIM, ACT_DIM).to(DEVICE)
    value_net = ValueNet(OBS_DIM).to(DEVICE)

    opt_policy = optim.Adam(policy.parameters(), lr=LR)
    opt_value = optim.Adam(value_net.parameters(), lr=LR)

    if RESUME_FROM_CHECKPOINT:
        if load_checkpoint(policy, value_net, opt_policy, opt_value):
            log("loaded policy_latest.pth and value_latest.pth (warm start from update ~100)")
        else:
            log("no checkpoint found — starting from scratch")

    return_history = deque(maxlen=ROLLING_WINDOW)
    cost_history = deque(maxlen=ROLLING_WINDOW)

    for update in range(1, TOTAL_UPDATES + 1):
        batch = collect_rollout(env, policy, value_net, EPISODES_PER_ROLLOUT, DEVICE)
        p_loss, v_loss, entropy = ppo_update(
            policy, value_net, opt_policy, opt_value, batch
        )

        mean_return = float(np.mean(batch["ep_returns"]))
        mean_cost = float(np.mean(batch["ep_costs"]))
        mean_unmet = float(np.mean(batch["ep_unmet"]))
        mean_grid = float(np.mean(batch["ep_grid_actions"]))
        mean_sc = float(np.mean(batch["ep_sc_actions"]))

        return_history.append(mean_return)
        cost_history.append(mean_cost)
        roll_return = float(np.mean(return_history))
        roll_cost = float(np.mean(cost_history))

        log(
            f"update {update}/{TOTAL_UPDATES} "
            f"return={mean_return:.2f} cost={mean_cost:.2f} "
            f"roll_return={roll_return:.2f} roll_cost={roll_cost:.2f} "
            f"unmet={mean_unmet:.1f} grid={mean_grid:+.3f} sc={mean_sc:+.3f} "
            f"p_loss={p_loss:.4f} v_loss={v_loss:.4f} entropy={entropy:.4f}"
        )

        if update % 25 == 0:
            torch.save(policy.state_dict(), CHECKPOINT_DIR / "policy_latest.pth")
            torch.save(value_net.state_dict(), CHECKPOINT_DIR / "value_latest.pth")
            log("saved checkpoints")

    torch.save(policy.state_dict(), CHECKPOINT_DIR / "policy_final.pth")
    torch.save(value_net.state_dict(), CHECKPOINT_DIR / "value_final.pth")
    log("phase 2 training complete")


if __name__ == "__main__":
    main()
