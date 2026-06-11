import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from load_days import load_day_data, make_real_env
from networks import PolicyNet, ValueNet

RL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RL_DIR.parent

OBS_DIM = 14

TOTAL_UPDATES = 500
EPISODES_PER_ROLLOUT = 32
NUM_ENVS = 16
PPO_EPOCHS = 4
CLIP_EPS = 0.2
GAMMA = 0.99
GAE_LAMBDA = 0.95
LR = 3e-4
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5
SEED = 42
ROLLING_WINDOW = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on SmartGridEnv")
    parser.add_argument(
        "--prototype",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="1=env4, 2=proto2, 3=proto3 (frozen, default 2)",
    )
    parser.add_argument("--updates", type=int, default=TOTAL_UPDATES)
    parser.add_argument("--episodes", type=int, default=EPISODES_PER_ROLLOUT)
    parser.add_argument(
        "--num-envs",
        type=int,
        default=NUM_ENVS,
        help="Parallel envs for batched GPU inference (CPU-bound sim)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile policy/value nets (PyTorch 2+)",
    )
    return parser.parse_args()


def act_dim_for_prototype(prototype: int) -> int:
    return 2 if prototype >= 3 else 3


def load_env_module(prototype: int):
    if prototype == 3:
        from env4_prototype_3 import SmartGridEnv

        return SmartGridEnv, "env4_prototype_3"
    if prototype == 2:
        from env4_prototype_2 import SmartGridEnv

        return SmartGridEnv, "env4_prototype_2"
    from env4 import SmartGridEnv

    return SmartGridEnv, "env4"


def log(msg: str, log_file: Path) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def clip_action(raw_action: np.ndarray, prototype: int) -> np.ndarray:
    action = raw_action.astype(np.float32).copy()
    if prototype >= 3:
        action[0] = np.clip(action[0], -1.0, 1.0)
        action[1] = np.clip((action[1] + 1.0) / 2.0, 0.0, 1.0)
        return action
    action[0] = np.clip(action[0], -1.0, 1.0)
    action[1] = np.clip(action[1], -1.0, 1.0)
    action[2] = np.clip((action[2] + 1.0) / 2.0, 0.0, 1.0)
    return action


def collect_rollout_batched(
    envs,
    policy,
    value_net,
    episodes: int,
    device: torch.device,
    prototype: int,
):
    """Roll out with N parallel envs and one batched GPU forward per step."""
    num_envs = len(envs)
    ticks_per_day = envs[0].ticksPerDay
    target_episodes = episodes
    finished_episodes = 0

    obs_list = []
    act_list = []
    logp_list = []
    rew_list = []
    val_list = []
    done_list = []

    ep_costs = []
    ep_import = []
    ep_export = []
    ep_rewards = []
    ep_unmet_base = []
    ep_unmet_def = []
    ep_def_done = []

    obs = np.zeros((num_envs, OBS_DIM), dtype=np.float32)
    for i, env in enumerate(envs):
        obs[i], _ = env.reset()

    day_import = np.zeros(num_envs, dtype=np.float64)
    day_export = np.zeros(num_envs, dtype=np.float64)
    day_reward = np.zeros(num_envs, dtype=np.float64)

    policy.eval()
    value_net.eval()

    with torch.inference_mode():
        while finished_episodes < target_episodes:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            mu, std = policy(obs_t)
            dist = torch.distributions.Normal(mu, std)
            raw_action = dist.sample()
            logp = dist.log_prob(raw_action).sum(dim=-1)
            values = value_net(obs_t)

            raw_np = raw_action.cpu().numpy()
            logp_np = logp.cpu().numpy()
            val_np = values.cpu().numpy()

            for i, env in enumerate(envs):
                if finished_episodes >= target_episodes:
                    break

                action = clip_action(raw_np[i], prototype)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                obs_list.append(obs[i].copy())
                act_list.append(raw_np[i].copy())
                logp_list.append(float(logp_np[i]))
                rew_list.append(float(reward))
                val_list.append(float(val_np[i]))
                done_list.append(float(done))

                day_import[i] += info.get("costThisTick", 0.0)
                day_export[i] += info.get("profitThisTick", 0.0)
                day_reward[i] += float(reward)
                obs[i] = next_obs

                if done:
                    finished_episodes += 1
                    if prototype >= 3:
                        ep_costs.append(-info.get("totalProfitCents", 0.0))
                    else:
                        ep_costs.append(info.get("totalCost", 0.0))
                    ep_import.append(float(day_import[i]))
                    ep_export.append(float(day_export[i]))
                    ep_rewards.append(float(day_reward[i]))
                    ep_unmet_base.append(info.get("totalUnmetBaseDemand", 0.0))
                    ep_unmet_def.append(info.get("totalUnmetDefDemand", 0.0))
                    ep_def_done.append(
                        sum(
                            1
                            for dd in env.dailyData["defDemandState"]
                            if dd.get("served", False)
                        )
                    )
                    day_import[i] = 0.0
                    day_export[i] = 0.0
                    day_reward[i] = 0.0
                    obs[i], _ = env.reset()

    return {
        "obs": torch.tensor(np.array(obs_list), dtype=torch.float32, device=device),
        "actions": torch.tensor(np.array(act_list), dtype=torch.float32, device=device),
        "logprobs": torch.tensor(np.array(logp_list), dtype=torch.float32, device=device),
        "rewards": torch.tensor(np.array(rew_list), dtype=torch.float32, device=device),
        "values": torch.tensor(np.array(val_list), dtype=torch.float32, device=device),
        "dones": torch.tensor(np.array(done_list), dtype=torch.float32, device=device),
        "ep_costs": ep_costs,
        "ep_import": ep_import,
        "ep_export": ep_export,
        "ep_rewards": ep_rewards,
        "ep_unmet_base": ep_unmet_base,
        "ep_unmet_def": ep_unmet_def,
        "ep_def_done": ep_def_done,
        "steps": len(obs_list),
        "ticks_per_day": ticks_per_day,
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
        value_loss = nn.functional.mse_loss(v_pred, returns)

        opt_policy.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
        opt_policy.step()

        opt_value.zero_grad()
        (VALUE_COEF * value_loss).backward()
        nn.utils.clip_grad_norm_(value_net.parameters(), MAX_GRAD_NORM)
        opt_value.step()

    return float(policy_loss.item()), float(value_loss.item())


def main():
    args = parse_args()
    act_dim = act_dim_for_prototype(args.prototype)
    env_class, env_name = load_env_module(args.prototype)
    checkpoint_dir = RL_DIR / "checkpoints" / f"prototype_{args.prototype}"
    log_file = RL_DIR / "logs" / f"train_ppo_prototype_{args.prototype}.log"

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("", encoding="utf-8")

    log(f"device={DEVICE}", log_file)
    log(f"prototype={args.prototype} env={env_name}", log_file)
    log(
        f"updates={args.updates} episodes/rollout={args.episodes} num_envs={args.num_envs}",
        log_file,
    )
    if DEVICE.type == "cuda":
        log(f"gpu={torch.cuda.get_device_name(0)}", log_file)

    train_profiles, _test_profiles, manifest = load_day_data(
        PROJECT_ROOT, prototype=args.prototype
    )
    log(
        f"data: {manifest['total_complete_days']} complete days | "
        f"train={manifest['train_days']} test={manifest['test_days']} "
        f"(80/20 seed={manifest['split_seed']})",
        log_file,
    )
    if args.prototype >= 3:
        log(
            "metrics: profit_cents=totalProfitCents (higher=better) | "
            "import=grid spend cents | export=grid earnings cents | "
            "reward=tick_profit_cents/profitRewardScale + defer penalties | "
            "def_done=tasks completed/3",
            log_file,
        )
    else:
        log(
            "metrics: net_profit=-totalCost (higher=better) | "
            "import=grid spend | export=grid earnings | "
            "unmet=Joules not served (lower=better) | def_done=tasks completed/3",
            log_file,
        )

    EnvClass = make_real_env(
        env_class, train_profiles, seed=SEED, prototype=args.prototype
    )
    envs = [EnvClass() for _ in range(max(1, args.num_envs))]
    policy = PolicyNet(OBS_DIM, act_dim).to(DEVICE)
    value_net = ValueNet(OBS_DIM).to(DEVICE)
    if args.compile and hasattr(torch, "compile"):
        policy = torch.compile(policy)
        value_net = torch.compile(value_net)
        log("torch.compile enabled", log_file)

    opt_policy = optim.Adam(policy.parameters(), lr=LR)
    opt_value = optim.Adam(value_net.parameters(), lr=LR)

    profit_history = deque(maxlen=ROLLING_WINDOW)
    unmet_history = deque(maxlen=ROLLING_WINDOW)

    for update in range(1, args.updates + 1):
        t0 = time.perf_counter()
        batch = collect_rollout_batched(
            envs, policy, value_net, args.episodes, DEVICE, args.prototype
        )
        _p_loss, _v_loss = ppo_update(policy, value_net, opt_policy, opt_value, batch)
        rollout_s = time.perf_counter() - t0

        avg_cost = float(np.mean(batch["ep_costs"]))
        avg_profit = -avg_cost
        avg_import = float(np.mean(batch["ep_import"]))
        avg_export = float(np.mean(batch["ep_export"]))
        avg_reward = float(np.mean(batch["ep_rewards"]))
        avg_unmet_base = float(np.mean(batch["ep_unmet_base"]))
        avg_unmet_def = float(np.mean(batch["ep_unmet_def"]))
        avg_unmet = avg_unmet_base + avg_unmet_def
        avg_def_done = float(np.mean(batch["ep_def_done"]))

        profit_history.append(avg_profit)
        unmet_history.append(avg_unmet)
        roll_profit = float(np.mean(profit_history))
        roll_unmet = float(np.mean(unmet_history))

        log(
            f"update {update}/{args.updates} | "
            f"profit={avg_profit:+.1f} roll_profit={roll_profit:+.1f} | "
            f"import={avg_import:.1f} export={avg_export:.1f} | "
            f"unmet={avg_unmet:.0f} roll_unmet={roll_unmet:.0f} | "
            f"def_done={avg_def_done:.1f}/3 reward={avg_reward:.0f} | "
            f"rollout={rollout_s:.1f}s steps={batch['steps']}",
            log_file,
        )

        if update % 25 == 0:
            torch.save(policy.state_dict(), checkpoint_dir / "policy_latest.pth")
            torch.save(value_net.state_dict(), checkpoint_dir / "value_latest.pth")
            log(f"saved checkpoints -> {checkpoint_dir}", log_file)

    torch.save(policy.state_dict(), checkpoint_dir / "policy_final.pth")
    torch.save(value_net.state_dict(), checkpoint_dir / "value_final.pth")
    log(f"training complete -> {checkpoint_dir / 'policy_final.pth'}", log_file)


if __name__ == "__main__":
    main()
