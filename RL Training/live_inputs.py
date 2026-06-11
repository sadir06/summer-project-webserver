from __future__ import annotations

import numpy as np

MAX_PRICE_LEGACY = 150.0


def voltage_to_supercap_en(voltage: float, env_class) -> float:
    physical_energy = 0.5 * env_class.supercapCF * (voltage**2)
    usable = physical_energy - env_class.minPhysicalSupercapEn
    return float(np.clip(usable, 0.0, env_class.maxSupercapEn))


def parse_deferables(deferables_raw: list[dict]) -> list[dict]:
    state = []
    for task in deferables_raw or []:
        energy = float(task.get("energy", 0.0))
        state.append(
            {
                "startTick": int(task["start"]),
                "endTick": int(task["end"]),
                "remainingEn": energy,
                "originalEn": energy,
            }
        )
    return state


def _price_obs_norm(env_class, buy_price_raw: float, sell_price_raw: float) -> tuple[float, float]:
    """Map raw cloud cents/J to observation normalisation used by the env."""
    if hasattr(env_class, "maxBuyPriceObs"):
        return (
            float(buy_price_raw) / env_class.maxBuyPriceObs,
            float(sell_price_raw) / env_class.maxSellPriceObs,
        )
    buy_norm = float(buy_price_raw) / MAX_PRICE_LEGACY
    sell_norm = float(sell_price_raw) / MAX_PRICE_LEGACY
    return buy_norm / env_class.maxPrice, sell_norm / env_class.maxPrice


def _future_price_obs_norm(env_class, avg_buy: float, avg_sell: float) -> tuple[float, float]:
    if hasattr(env_class, "maxBuyPriceObs"):
        return avg_buy / env_class.maxBuyPriceObs, avg_sell / env_class.maxSellPriceObs
    return (avg_buy / MAX_PRICE_LEGACY) / env_class.maxPrice, (
        avg_sell / MAX_PRICE_LEGACY
    ) / env_class.maxPrice


class DaySeriesBuffer:
    def __init__(self):
        self.day_id = None
        self.series: dict[int, dict[str, float]] = {}

    def reset_if_new_day(self, day_id) -> None:
        if day_id != self.day_id:
            self.day_id = day_id
            self.series = {}

    def record_tick(self, tick: int, buy_price: float, sell_price: float, demand: float, pv_w: float) -> None:
        self.series[tick] = {
            "buy_price": buy_price,
            "sell_price": sell_price,
            "demand": demand,
            "pv_w": pv_w,
        }

    def get_price_projection(self, cur_tick: int, env_class) -> tuple[float, float, float, float, float]:
        start_tick = cur_tick
        end_tick = min(env_class.ticksPerDay, cur_tick + env_class.priceLookaheadTicks)
        data_tick = min(cur_tick, env_class.ticksPerDay - 1)

        def values_for(field: str, fallback: float) -> list[float]:
            collected = []
            for tick in range(start_tick, end_tick):
                if tick in self.series:
                    collected.append(self.series[tick][field])
            return collected or [fallback]

        current = self.series.get(data_tick, {})
        buy_fallback = current.get("buy_price", 0.0)
        sell_fallback = current.get("sell_price", 0.0)
        pv_fallback = current.get("pv_w", 0.0)
        demand_fallback = current.get("demand", 0.0)

        future_buy = values_for("buy_price", buy_fallback)
        future_sell = values_for("sell_price", sell_fallback)
        future_pv = values_for("pv_w", pv_fallback)
        future_demand = values_for("demand", demand_fallback)

        avg_future_buy = float(np.mean(future_buy))
        avg_future_sell = float(np.mean(future_sell))
        min_future_buy = float(np.min(future_buy))
        max_future_sell = float(np.max(future_sell))
        projected_future_pv_en = float(np.sum(future_pv) * env_class.tickDur)
        projected_future_demand_en = float(np.sum(future_demand) * env_class.tickDur)
        projected_future_net_en = projected_future_pv_en - projected_future_demand_en
        return (
            avg_future_buy,
            avg_future_sell,
            min_future_buy,
            max_future_sell,
            projected_future_net_en,
        )


def build_observation(
    env_class,
    tick: int,
    demand_w: float,
    buy_price_raw: float,
    sell_price_raw: float,
    pv_w: float,
    supercap_en: float,
    defer_state: list[dict],
    day_buffer: DaySeriesBuffer,
) -> np.ndarray:
    data_tick = min(int(tick), env_class.ticksPerDay - 1)
    buy_norm, sell_norm = _price_obs_norm(env_class, buy_price_raw, sell_price_raw)

    day_buffer.record_tick(
        data_tick, float(buy_price_raw), float(sell_price_raw), demand_w, pv_w
    )

    time_angle = 2 * np.pi * data_tick / env_class.ticksPerDay
    time_sin = np.sin(time_angle)
    time_cos = np.cos(time_angle)

    total_def_en = sum(dd["remainingEn"] for dd in defer_state)
    due_soon_en = 0.0
    for dd in defer_state:
        if dd["remainingEn"] > 0 and data_tick <= dd["endTick"] <= (
            data_tick + env_class.defDemandATicks
        ):
            due_soon_en += dd["remainingEn"]

    pv_norm = pv_w / env_class.pvPowerMax
    sc_energy_norm = supercap_en / env_class.maxSupercapEn if env_class.maxSupercapEn > 0 else 0.0
    base_demand_norm = demand_w / env_class.loadDemandMax
    total_def_norm = total_def_en / env_class.maxDefEn
    due_soon_norm = due_soon_en / env_class.maxDueSoon

    net_balance = pv_w - demand_w
    max_balance = max(env_class.pvPowerMax, env_class.loadDemandMax)
    net_norm = net_balance / max_balance

    (
        avg_future_buy,
        avg_future_sell,
        _min_future_buy,
        _max_future_sell,
        projected_future_net_en,
    ) = day_buffer.get_price_projection(data_tick, env_class)

    avg_future_buy_norm, avg_future_sell_norm = _future_price_obs_norm(
        env_class, avg_future_buy, avg_future_sell
    )
    cap_free_norm = (env_class.maxSupercapEn - supercap_en) / env_class.maxSupercapEn
    max_projected_net_en = env_class.pvPowerMax * env_class.tickDur * env_class.priceLookaheadTicks
    if max_projected_net_en <= 0:
        projected_future_net_norm = 0.0
    else:
        projected_future_net_norm = projected_future_net_en / max_projected_net_en

    obs = np.array(
        [
            time_sin,
            time_cos,
            pv_norm,
            sc_energy_norm,
            base_demand_norm,
            buy_norm,
            sell_norm,
            total_def_norm,
            due_soon_norm,
            net_norm,
            avg_future_buy_norm,
            avg_future_sell_norm,
            cap_free_norm,
            projected_future_net_norm,
        ],
        dtype=np.float32,
    )
    return np.clip(obs, env_class().observation_space.low, env_class().observation_space.high)


def total_deferrable_energy(defer_state: list[dict]) -> float:
    return float(sum(dd["remainingEn"] for dd in defer_state))


def compute_pico_demand_power(
    instant_demand: float,
    defer_state: list[dict],
    def_action: float,
    tick_dur_s: float,
) -> float:
    """Load Pico power = instant demand + deferrable power this tick."""
    defer_power = total_deferrable_energy(defer_state) * float(def_action) / tick_dur_s
    return min(float(instant_demand) + defer_power, 9.0)
