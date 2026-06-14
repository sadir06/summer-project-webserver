# Prototype 4: hardware efficiencies and SC voltage-limit penalty

**Environment:** `RL Training/env4_prototype_4.py`  
**Checkpoint dir:** `RL Training/checkpoints/prototype_4/`

## Changes from Prototype 3

### Supercap (η = 0.77)

- Charge: stored += η × E_bus absorbed  
- Discharge: deliver E_bus, remove E_bus/η from storage  
- Voltage limits: **10.25 V – 16.0 V**  
- Reward penalty if commanding charge at max V or discharge at min V  

### Total load cap (8 W / tick)

- **Instant demand** is always fully served (non-negotiable).
- **Defer** power = `def_action × (8 W − instant_demand)`, capped by remaining defer energy.
- Example: instant 5 W + `def_action=1` → at most **3 W** defer, not 4 W.
- `def_en_served`, load η, and profit use **actual** defer J served; `remainingEn` updates accordingly.

### Grid PSU efficiency (import and export)

Step on **bus power** |P_bus| (W):

| |P_bus| | η_grid |
|--------|--------|
| < 0.5 W | 0.55 |
| ≥ 0.5 W | 0.90 |

- **Import:** metered J = E_bus / η → pay more cents/J when η is low  
- **Export:** credited J = E_bus × η → earn less when η is low  

PSU power caps (8 W) apply to **bus** joules; profit uses metered joules.

### Load efficiency (separate from grid)

Measured curve (linear interp between lab points 0.1–8 W):

- Low load is costly: ~50% at 0.1 W, ~69% at 0.5 W  
- Peak ~81% around 4–5 W  
- ~77% at 8 W  

**Bus demand** each tick:

```text
E_bus_demand = E_load_terminal / η_load(P_load)
```

where `E_load_terminal = instant_demand×5s + defer_served_J`, `P_load = E_load_terminal / 5s`.

### Profit

```text
tick_profit = exported_metered_J × sell − imported_metered_J × buy
```

All SC and load physics use **actual** bus J moved; commanded `sc_action` never enters profit directly.

### Reward

```text
reward = tick_profit_cents / 100 − defer_miss_penalty − sc_limit_penalty
defer_miss_penalty = missed_J × 200 cents/J / 100   # was 10; must beat grid profit
```

Typical defer task is **~37 J** (max 50 J). Missing one full task costs about a full day of grid profit in reward units.

Log columns: `def_done` (tasks served / 3), `def_missed` (tasks missed / day), `profit` (grid only, excludes defer penalties).

## Training data (prototype 4)

From `data_collection/data/ticks.jsonl` (80/20 split, seed 42):

| Field | Source |
|-------|--------|
| **PV** | Uniform random **[0, 7] W** per tick, not from `sun` |
| **Demand** | `demand` from ticks (instant load, W) |
| **Prices** | Raw `buy_price` / `sell_price` (cents/J); profit uses E = P × 5 s |
| **Deferables** | `energy`, `start`, `end` from tick 0 row |

Obs price features (dims 11–12, 14): rolling mean over **past** 6 ticks only, no future oracle.

**Extended obs (dims 15–21, prototype 4 only):**

| # | Feature |
|---|---------|
| 15–17 | Cross-day same-tick: mean buy, mean sell, std buy over prior **7** completed days (A) |
| 18–19 | Intraday momentum: Δbuy, Δsell over last **6** ticks (C) |
| 20–21 | Intraday range position: where current buy/sell sits in today's min–max so far (C) |

Total **21-D** observation vector. Prior checkpoints trained on 14-D are incompatible.

## Train (RTX 4070)

```powershell
cd "RL Training"
python train_ppo.py --prototype 4
```

Default **4 parallel envs** (override with `--num-envs`). Checkpoints → `checkpoints/prototype_4/`.

## Eval

```powershell
python eval_test_set.py --prototype 4
```

## Next (discussion)

Arbitrage improvements now that efficiencies match hardware (see training logs).
