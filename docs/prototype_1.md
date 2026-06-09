# Prototype 1 — Model Behaviour & Arbitrage Findings

**Checkpoint:** `RL Training/checkpoints/prototype_1/policy_final.pth` (500 PPO updates)  
**Evaluation:** Full held-out test set via `inference.py --prototype 1` + `env4.SmartGridEnv` rollouts  
**Test days:** 229 (80/20 split from `split_manifest.json`)

---

## Executive summary

The trained policy is a **reliability-first controller**: it maximises grid import and supercap discharge on almost every tick, serves deferrables aggressively, and **eliminates unmet demand** across all 229 test days. It does **not** learn cost-aware or arbitrage-aligned behaviour. Net energy cost is substantially worse than a simple reactive baseline on every test day.

Hardcoded price arbitrage in `env4.py` still fires in the background, but the learned policy **works against** storage charging most of the time, so arbitrage only operates on the minority of ticks where price signals override the agent's discharge bias.

---

## Test-set metrics (RL vs naive baseline)

| Metric | Naive baseline | RL policy |
|--------|----------------|-----------|
| Mean net profit (export − import) | **+14.9** | **−59.5** |
| Mean unmet demand (J) | 98.7 | **0.01** |
| Mean deferrables completed | 2.95 / 3 | **2.99 / 3** |
| Days with zero unmet | 0 / 229 | **229 / 229** |
| Days beating baseline on profit | — | **0 / 229** |
| Days beating baseline on unmet | — | **229 / 229** |

**RL cost distribution** (`totalCost` = import spend − export earnings; higher = worse):

| Percentile | Value |
|------------|-------|
| p5 | 27.7 |
| p50 (median) | 59.4 |
| p95 | 91.6 |
| min / max | 15.0 / 107.3 |

The policy is consistent but uniformly expensive: even the best RL day (~15 cost) is worse than naive's average (~−15 profit equivalent).

---

## Learned action pattern (saturated controls)

Mean actions over 60 ticks × 229 days:

| Action | Mean | Interpretation |
|--------|------|----------------|
| `gridAction` | **+1.00** | Import 100% of residual demand every tick |
| `scAction` | **−0.99** | Discharge supercap at max rate every tick |
| `defAction` | **+0.97** | Allocate ~97% of deferrable serving capacity |

Per-day behaviour is even more extreme:

- **Grid import ticks** (`gridAction > 0.5`): 60 / 60 on every day
- **SC discharge ticks** (`scAction < −0.5`): 60 / 60 on every day
- **SC charge ticks** (`scAction > 0.5`): **0** on every day

The policy has collapsed to a corner of the action space: *"import everything, discharge storage constantly, serve deferrables fully."* This matches training logs where `roll_unmet` fell sharply while `roll_profit` stayed negative from ~update 18 onward.

---

## Hardcoded arbitrage — how it interacts with the policy

Arbitrage is **not** an agent action. It is applied inside `env4.step()` by merging arb buy/sell energy into the same supercap delta as the agent's `scAction`:

```
requestedScDelta = agentScDelta + arbBuyEn − arbSellEn
```

Attribution rules:

- **Arb import** (`arbitrageImportEn`): only when net SC delta is **positive** (charging) *and* `shouldBuyForStorage()` — buy price < 95% of average future buy price, with cap headroom.
- **Arb export** (`arbitrageExportEn`): only when net SC delta is **negative** (discharging) *and* `shouldSellFromStorage()` — sell price > 105% of average future sell *and* projected future net demand is non-negative, *and* arb buy is not also wanted.

Arbitrage also receives a small fixed reward bonus (+0.5 per import/export tick) that the agent does not directly control.

### Observed arbitrage frequency (RL rollouts)

| Stat | Value |
|------|-------|
| Mean arb-import ticks per day | **16.2** / 60 (~27%) |
| Mean arb-export ticks per day | **3.9** / 60 (~6.5%) |
| Total arb-import ticks (229 days) | 3,702 |
| Total arb-export ticks (229 days) | 893 |
| Days with ≥1 arb-import tick | 229 / 229 |
| Days with ≥1 arb-export tick | 229 / 229 |

### The conflict pattern

Because the agent **always** requests max SC discharge, arbitrage buy logic only succeeds when cheap-price ticks produce a **net positive** `requestedScDelta` (arb buy energy outweighs agent discharge). That happens on ~16 ticks per day.

On the remaining ~44 ticks, agent discharge dominates: arbitrage cannot charge the cap even when buy prices are favourable. Arb export fires on ~4 ticks per day when sell is favourable, future demand looks comfortable, and buy arb is not active.

**Net effect:** Arbitrage partially leaks through on cheap ticks but is largely **suppressed** by the policy's discharge saturation. Mean end-of-day supercap energy is **~5.3 J** (starts at 0), accumulated by arb overriding discharge intent — not because the agent learned to store energy.

---

## Why the model behaves this way

1. **Penalty asymmetry.** Unmet base demand costs −100 per Joule in reward shaping; unmet deferrable −30. Import cost is only −1× price per unit energy. The policy quickly learns that over-importing is cheaper than risking any unmet.

2. **No explicit profit objective in deployment.** Training reward mixes cost, unmet penalties, curtailment, and smoothness. The dominant gradient path is "zero unmet" not "minimise `totalCost`."

3. **Arbitrage is opaque to the policy.** The agent observes SC state but does not choose arb; it only sees the combined SC outcome. With saturated discharge, it cannot coordinate with arb buy windows.

4. **Single combined SC update.** Agent and arb share one physical SC actuator. A discharge-biased policy structurally limits arb import frequency regardless of price signal strength.

---

## Implications for Prototype 2

| Area | Finding | Suggested direction |
|------|---------|---------------------|
| Reliability | Strong — 0 unmet on full test set | Keep penalty structure or add hard constraint metric |
| Economics | Failed — 0/229 days beat naive on profit | Add/normalise cost term; consider Lagrangian budget on import |
| Arbitrage | Runs but undermined by SC discharge policy | Decouple arb from agent SC actuator (`env4_prototype_2.py`) |
| Action collapse | All actions saturated | Entropy bonus tuning, action noise at train time, or reward scaling review |
| Evaluation | `inference.py` is single-tick demo | Use full-day rollouts for regression checks |

---

## Reproduction

```bash
cd "RL Training"
python inference.py --prototype 1
python train_ppo.py --prototype 1   # retrain
```

---

*Generated after Prototype 1 training (500 updates) and full test-set inference pass.*
