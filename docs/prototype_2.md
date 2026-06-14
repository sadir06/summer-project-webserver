# Prototype 2: model behaviour and evaluation

**Checkpoint:** `RL Training/checkpoints/prototype_2/policy_final.pth` (500 PPO updates)  
**Environment:** `env4_prototype_2.py`  
**Evaluation:** Full-day rollouts on all 229 held-out test days (`split_manifest.json`, seed 42)  
**Training log:** `RL Training/logs/train_ppo.log` (final `roll_profit` ≈ +254)

---

## Summary

Prototype 2 fixes the core failure mode of Prototype 1: a policy that bought reliability with unlimited grid imports. The new environment serves demand automatically (PV, then supercap, then grid), and the trained policy focuses on surplus export, storage timing, and deferrable scheduling. On the test set it is **strongly profitable**, **never misses demand**, and beats a simple naive baseline on profit by a small average margin. It is not uniformly better than naive on every day, and deferrable completion is marginally weaker.

Compared to Prototype 1 (mean test profit **−59.5**, 0/229 days beating naive), Prototype 2 is a clear step forward.

---

## How the model works

### Actions (3-dimensional, same network as Prototype 1)

| Action | Range | Role in `env4_prototype_2` |
|--------|-------|----------------------------|
| `gridAction` | [−1, 1] | **Export PV surplus only.** Does not control demand imports. |
| `scAction` | [−1, 1] | Discharge to meet demand, or charge from PV surplus if positive. |
| `defAction` | [0, 1] | Fraction of max deferrable serving capacity allocated this tick. |

This is a different semantics map than Prototype 1, where `gridAction` scaled demand imports directly.

### Demand dispatch (hardcoded, not learned)

Each tick, the environment satisfies load in fixed order:

1. PV generation (sampled independently per tick from `Normal(3.5, 2.0)` kW, clipped to [0, 7], seeded per day in `load_days.py`; not derived from `sun`)
2. Supercap discharge (if the agent requests it and energy is available)
3. Grid import (automatic, up to `gridBuyPowerMax × tickDur`)

Unmet demand only appears if residual load exceeds grid capacity in that tick (40 J at defaults). This is why **unmet can be zero from update 1 of training**: reliability is largely enforced by physics, not by something the policy must discover.

### Arbitrage (hardcoded, runs after demand is met)

- **Buy into storage** when prices are favourable and the agent is not strongly discharging (`scAction > −0.2`) or SOC is low.
- **Sell from storage** when sell is favourable and no arb-buy happened that tick.
- Agent and arb no longer share a single merged SC delta as in Prototype 1.

### What the policy learned (test-set averages)

| Signal | Value | Reading |
|--------|-------|---------|
| `gridAction` | **−0.999** | Near-max surplus export every tick |
| `scAction` | **−0.998** | Near-max discharge *request* every tick |
| `defAction` | **0.510** | ~51% deferrable capacity (not saturated) |
| SC discharge requested | 100% of ticks | Intent is always "discharge" |
| SC discharge physical | **37.6%** of ticks | Cap is often empty; requests do nothing |
| PV → SC charge ticks | **0** | Agent never chooses to charge |
| End-of-day SC | **32.5 J** (max ≈ 38 J) | ~85% SOC; arb import fills the cap despite discharge requests |
| Arb import ticks/day | **15.3** | Similar to Prototype 1 |
| Arb export ticks/day | **17.4** | Much higher than Prototype 1 (~4); aligned with export-heavy `gridAction` |

The policy is not "using the supercap to serve load." It requests discharge constantly, but the cap is empty most ticks. Profit comes mainly from **export earnings** (arb + surplus PV), while **import spend** stays near the minimum the env requires (~110 units/day in training logs).

---

## Test-set results (229 days)

Checkpoint: `policy_final.pth`. Naive baseline: export surplus when net load is negative, otherwise serve deferrables at full capacity, no SC action.

| Metric | Naive | RL (Prototype 2) |
|--------|-------|------------------|
| Mean net profit (export − import) | +292.6 | **+301.5** |
| Mean profit delta (RL − naive) | — | **+8.9** |
| Days RL profit > naive | — | **127 / 229** (55%) |
| Days RL profit < naive | — | **102 / 229** (45%) |
| Mean unmet demand (J) | 0.44 | **0.0** |
| Days with zero unmet | 203 / 229 | **229 / 229** |
| Deferrables completed | 2.996 / 3 | 2.991 / 3 |
| Days with any missed deferrable | 1 | 2 |

**RL profit distribution (test):**

| Percentile | Profit |
|------------|--------|
| p5 | 241.2 |
| p50 | 301.6 |
| p95 | 353.5 |
| min / max | 221.5 / 396.6 |

Training rollouts averaged ~+254 profit; test mean is slightly higher (+301). Training batches sample random train days with exploration noise; test eval uses deterministic policy means on held-out days. The gap is plausible and not evidence of log tampering.

---

## Comparison to Prototype 1

| | Prototype 1 | Prototype 2 |
|--|-------------|-------------|
| Mean test profit | −59.5 | **+301.5** |
| Beat naive on profit | 0 / 229 | **127 / 229** |
| Zero unmet days | 229 / 229 | 229 / 229 |
| Primary mechanism | `gridAction = +1` (import all residual demand) | Auto import + export/arbitrage |
| SC strategy | Discharge intent, fought arb | Same discharge intent, but arb runs on separate path |
| Economic outcome | Reliability at any cost | Reliability with net revenue |

Prototype 1 proved the agent could eliminate unmet demand but could not learn cost control when import scaling was entirely its responsibility. Prototype 2 removes that failure mode structurally.

---

## Nuances worth keeping in mind

**1. Zero unmet from day one is expected, not suspicious.** Demand imports are automatic. The policy is not "learning to import everything" as in Prototype 1. The learning problem shifted to export, storage interaction, and deferrables.

**2. Profit plateau during training does not mean the model stopped improving.** Roll profit stabilised around +250 by update ~60 and held there through 500. Test profit (+301) and deferrable completion (2.99 in training at end vs 2.95 at update 50) still moved slightly. The metric is a 10-update rolling mean over stochastic train-day batches.

**3. The policy is not dominant on profit.** It wins 55% of days vs naive. On 45% of days, naive is cheaper. The average gain (+8.9) is real but modest relative to the ~300 profit scale (~3%).

**4. Deferrables are slightly worse than naive.** RL completes 2.991 vs 2.996 tasks on average, with 2 missed-task days vs 1 for naive. `defAction` saturated around 0.51, not 1.0. The policy under-serves deferrables relative to both capacity and the naive baseline.

**5. Discharge spam without physical effect.** `emptyScDischargePenalty` and `maxDischargeSpamPenalty` exist in the env, but the policy still outputs `scAction ≈ −1` every tick. Penalties are too weak or the value net found export-first behaviour sufficient without SC-aware control.

**6. High end-of-day SOC (~85%).** `targetEndSoc` is 0.25 (~9.5 J). The policy ends days near full cap because arb import wins on cheap ticks while discharge requests cannot empty a cap that refills via arb. The end-of-day SOC penalty is not strong enough to pull storage down.

**7. PV is independent of irradiance.** `sun` in `ticks.jsonl` is stored in profiles as `irradiance` but Prototype 2 uses a separate `pvGen` series per day. The agent optimises against synthetic PV, not measured sun correlation. Lab deployment may differ until real PV traces are fed via `options["pvGen"]`.

---

## How to read the logged quantities

From `train_ppo.log` each update line:

- **`profit`** / **`roll_profit`**: negative of mean `totalCost` across the rollout batch. Higher is better. This is export earnings minus import spend.
- **`import`** / **`export`**: summed `costThisTick` and `profitThisTick` per episode in the batch (not normalized per day).
- **`unmet`**: Joules of base + deferrable demand not served. Expect 0 in Prototype 2 training.
- **`def_done`**: deferrable tasks marked served (0–3 per episode).
- **`reward`**: shaped env reward (profit + penalties). Noisy; poor headline metric.

From inference / full-day eval `info`:

- **`gridToDemand`**: automatic import to cover residual load after PV and SC.
- **`arbitrageImportEn`** / **`arbitrageExportEn`**: grid energy attributed to hardcoded arb this tick.
- **`pvChargeEn`**: energy stored from PV surplus via positive `scAction`.
- **`totalCost`**: cumulative import spend minus export earnings (lower is better; negative means net profit).

---

## Further improvements

1. **Deferrable scheduling.** Raise `defAction` toward 1.0 on cheap ticks or near deadlines. Consider a stronger deferrable penalty or curriculum on `def_done`.

2. **SC-aware policy.** Increase `emptyScDischargePenaltyWeight` or add SOC to a more salient reward term so the agent stops requesting discharge into an empty cap.

3. **End-of-day SOC target.** `endSocPenaltyWeight = 5.0` is not pulling SOC from ~85% toward 25%. Retune or tie penalty to arb behaviour you actually want.

4. **Real PV data.** Replace `sample_day_pv_gen()` with measured PV time series when available. Training currently does not use sun-PV correlation from the lab.

5. **Held-out eval in the training loop.** Log test-set profit every N updates to catch overfitting and compare to naive without a manual script.

6. **Profit vs naive as an explicit metric.** 127/229 wins is informative but easy to miss in training logs. A rolling "beat naive %" on a fixed eval subset would make economic progress visible during training.

---

## Reproduction

```bash
cd "RL Training"
python inference.py                    # single tick, loads policy_final.pth
python train_ppo.py                    # retrain prototype 2 (default)
python train_ppo.py --prototype 1      # prototype 1 env
```

Full test-set evaluation was run with `policy_final.pth`, `SmartGridEnv` from `env4_prototype_2`, and profiles from `load_days.load_day_data()`.

---

*Written after Prototype 2 training (500 updates, 2026-06-09) and full test-set evaluation.*
