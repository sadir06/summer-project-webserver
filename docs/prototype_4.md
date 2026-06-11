# Prototype 4 — Defer timing + faster training

**Environment:** `RL Training/env4_prototype_4.py`  
**Checkpoints:** `RL Training/checkpoints/prototype_4/`  
**Log:** `RL Training/logs/train_ppo_prototype_4.log`  
**Train:** `python train_ppo.py --prototype 4 --updates 2000 --episodes 256 --num-envs 32`

---

## Relationship to Prototype 3

Prototype 3 (**frozen**) broke through to **positive daily profit** on the H100 run (~update 174: export > import, **+$10/day** by update 199). That config is unchanged in `env4_prototype_3.py`.

Prototype 4 adds experiments on top without touching the winning proto 3 physics:

| Feature | Prototype 3 (frozen) | Prototype 4 |
|---------|---------------------|-------------|
| `maxDefPower` | 8 W (fixed) | 10 W + **dynamic** `remaining_J / 5s` cap |
| Defer price gate | No — agent chooses every tick | **Yes** — block defer when buy > window average |
| Deadline force-serve | Penalty only | Gate opens when `mustServeNow` |
| Training log | `train_ppo.log` (legacy) | `train_ppo_prototype_4.log` |
| Rollout collector | Single env (old VM run) | **Batched** `--num-envs` parallel envs |

Shared with proto 3: profit reward (`tick_profit_cents / profitRewardScale`), `profitRewardScale=50`, `deferMissPenalty=5`, `defDemandATicks=15`, η=0.8, auto PSU, ticks.jsonl data.

---

## Prototype 3 training result (reference)

From H100 run (do not overwrite `checkpoints/prototype_3/`):

| Update | roll_profit (cents) | ≈ USD/day |
|--------|---------------------|-----------|
| 1 | −4196 | −$42 |
| 162 | −2481 | −$25 |
| 174 | −969 | −$10 |
| **174+** | **crosses 0** | **profitable** |
| 199 | +983 | **+$9.83** |

Export overtook import; agent learned buy-low / sell-high arbitrage. **Patience was the lesson.**

---

## Prototype 4 defer logic

```text
tick_def_power = min(maxDefPower, active_remaining_J / 5s)
defer_gate     = buy <= avg(remaining window buys) OR must_serve_before_deadline
def_capacity   = def_action × defer_gate × tick_def_power × 5s
```

Agent should learn `def_action ≈ 0` when gate is closed (expensive ticks) and burst when gate opens (cheap ticks).

---

## Parallel runs on VM

```bash
# Terminal 1 — let proto 3 finish (existing process)
tail -f "RL Training/logs/train_ppo.log"

# Terminal 2 — new proto 4 run after git pull
cd "/code/summer-project-webserver/RL Training"
source ../.venv/bin/activate
python train_ppo.py --prototype 4 --updates 2000 --episodes 256 --num-envs 32 \
  2>&1 | tee logs/proto4_console.log

tail -f logs/train_ppo_prototype_4.log
```

Checkpoints are separate: `checkpoints/prototype_3/` vs `checkpoints/prototype_4/`.
