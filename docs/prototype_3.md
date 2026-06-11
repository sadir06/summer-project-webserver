# Prototype 3 — Profit-Driven Arbitrage (ready for training)

**Environment:** `RL Training/env4_prototype_3.py`  
**Training data:** `data_collection/data/ticks.jsonl` only (not `live_inputs.py`)  
**Checkpoint dir:** `RL Training/checkpoints/prototype_3/` (empty until training)  
**Train command (when approved):** `cd "RL Training" && python train_ppo.py --prototype 3`

---

## Design goals

Prototype 3 removes the learned grid action and hardcoded arbitrage heuristics from Prototype 2. The agent learns **when to charge/discharge the supercap** and **how much deferrable load to serve**, while the PSU automatically imports any shortfall and exports any surplus. Reward is dominated by **real money**: export revenue minus import cost each tick, using raw **cents/J** prices from collected data.

Because sell price is typically ~2× buy price in `ticks.jsonl`, the optimal strategy is: **import and store when buy is low**, **export when sell is high**, and **serve deferrables during cheap windows** before deadlines.

---

## Actions (2-dimensional)

| Action | Range | Role |
|--------|-------|------|
| `scAction` | [−1, 1] | Negative = discharge (demand first, then grid export). Positive = charge (PV surplus first, then grid import). |
| `defAction` | [0, 1] | Fraction of max deferrable serving power this tick. |

No `gridAction`. PSU handles residual import/export automatically (KCL).

---

## Physics

### Units

- Power: **watts**
- Energy per tick: **joules** = W × 5 s
- Prices in `step()`: **raw cents/J** from ticks (buy ~5–95, sell ~10–191)

### Supercap efficiency η = 0.8

- **Charging:** bus energy `E_in` → stored `η × E_in`
- **Discharging:** bus energy `E_out` delivered → removed `E_out / η` from usable storage

### Supercap voltage model

- `supercapMaxV = 16.25`, `supercapMinV = 10.35`, `CF = 0.65`
- Usable energy = physical − minimum offset (same family as Prototype 2, updated max V)

### Dispatch order each tick

1. Serve deferrables (agent `defAction` × max power × 5 s)
2. Meet instant + defer demand: **PV → SC discharge → PSU import**
3. PV surplus: **SC charge → PSU export**
4. Remaining discharge budget → **SC to grid export**
5. Remaining charge budget → **grid import to SC**

### Profit and reward

```text
tick_profit_cents = export_J × sell_cents_J − import_J × buy_cents_J
reward = tick_profit_cents / 100 + defer_deadline_penalties
```

Defer miss penalty: `10 cents/J` per missed energy (scaled by `/100` into reward).

---

## Training data (`load_days.py`, prototype 3)

From each `ticks.jsonl` row:

| Field | Use |
|-------|-----|
| `sun` | PV watts: `(sun/100) × 7` |
| `demand` | Instant load (W) |
| `buy_price`, `sell_price` | Raw cents/J (no ÷150) |
| `deferables` | Task energy + start/end ticks |
| `tick`, `day` | Time encoding + day grouping |

1,702 complete days (60 ticks), 80/20 train/test split (`split_manifest.json`).

---

## Observations (14-dim, same layout as Prototype 2)

| # | Name | Source | Normalisation |
|---|------|--------|---------------|
| 0–1 | time sin/cos | `tick` | trigonometric |
| 2 | pv_norm | sun → PV | ÷ 7 W |
| 3 | sc_energy_norm | sim SC | ÷ max usable J |
| 4 | base_demand_norm | `demand` | ÷ 8 W |
| 5–6 | buy/sell | ticks | ÷ 95 / ÷ 191 |
| 7–8 | defer totals | sim defer state | ÷ 100 J |
| 9 | net_norm | PV − demand | ÷ max(7, 8) |
| 10–11 | future buy/sell | same-day lookahead | ÷ 95 / ÷ 191 |
| 12 | cap_free_norm | sim SC | 1 − soc |
| 13 | projected_future_net | future PV − demand | ÷ (7×5×6) |

---

## Inference (`live_inputs.py` + `inference.py`)

Live loop uses **hardware + cloud**, not `ticks.jsonl`. Observation math matches the env (including cents/J normalisation for prototype 3). Actions sent: **SC + load/def only** (no grid Pico for proto 3).

```text
python inference.py --mode live --prototype 3
python inference.py --mode eval --prototype 3
```

---

## Differences from Prototype 2

| | Prototype 2 | Prototype 3 |
|---|-------------|-------------|
| Actions | 3 (grid, sc, def) | 2 (sc, def) |
| PV training | Random Gaussian | Real `sun` from ticks |
| Prices | Normalised ÷150 | Raw cents/J in physics |
| Grid | Learned export + hardcoded arb | Fully automatic PSU |
| SC efficiency | None | η = 0.8 |
| Reward | Shaped + profit (normalised) | **Profit cents / 100** + defer penalties |
| Unmet instant demand | Huge penalty | None (PSU imports) |
