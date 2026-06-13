from __future__ import annotations

import numpy as np

from price_obs import (
    CrossDayPriceArchive,
    compute_price_momentum,
    cross_day_features_at_tick,
)

MAX_PRICE_LEGACY = 150.0


def voltage_to_supercap_en(voltage: float, env_class) -> float:
    physical_energy = 0.5 * env_class.supercapCF * (voltage**2)
    usable = physical_energy - env_class.minPhysicalSupercapEn
    return float(np.clip(usable, 0.0, env_class.maxSupercapEn))


def effective_sc_action_for_voltage(
    sc_action: float,
    vcap_v: float | None,
    env_class,
) -> float:
    """Clip SC command when hardware is already at min/max voltage (proto 4+ live profit)."""
    if vcap_v is None:
        return float(sc_action)
    eps = getattr(env_class, "voltageLimitEpsilonV", 0.02)
    v = float(vcap_v)
    if v <= env_class.supercapMinV + eps and sc_action < 0.0:
        return 0.0
    if v >= env_class.supercapMaxV - eps and sc_action > 0.0:
        return 0.0
    return float(sc_action)


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
        """Rolling window over ticks seen so far today (no future oracle)."""
        end_tick = cur_tick + 1
        start_tick = max(0, end_tick - env_class.priceLookaheadTicks)
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


def _today_price_lists(
    series: dict[int, dict[str, float]],
    data_tick: int,
    buy_now: float,
    sell_now: float,
) -> tuple[list[float], list[float]]:
    buy_prices: list[float] = []
    sell_prices: list[float] = []
    for t in range(data_tick + 1):
        row = series.get(t, {})
        buy_prices.append(float(row.get("buy_price", buy_now)))
        sell_prices.append(float(row.get("sell_price", sell_now)))
    return buy_prices, sell_prices


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
    cross_day_archive: CrossDayPriceArchive | None = None,
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

    obs_dim = int(getattr(env_class, "obsDim", 14))
    base_obs = [
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
    ]

    if obs_dim > 14:
        if cross_day_archive is not None:
            cross_mean_buy_norm, cross_mean_sell_norm, cross_std_buy_norm = (
                cross_day_archive.stats_at_tick(
                    data_tick,
                    float(buy_price_raw),
                    float(sell_price_raw),
                    env_class.ticksPerDay,
                    env_class.maxBuyPriceObs,
                    env_class.maxSellPriceObs,
                )
            )
        else:
            cross_mean_buy_norm, cross_mean_sell_norm, cross_std_buy_norm = (
                cross_day_features_at_tick(
                    data_tick,
                    float(buy_price_raw),
                    float(sell_price_raw),
                    None,
                    None,
                    None,
                    env_class.maxBuyPriceObs,
                    env_class.maxSellPriceObs,
                )
            )
        buy_prices, sell_prices = _today_price_lists(
            day_buffer.series, data_tick, float(buy_price_raw), float(sell_price_raw)
        )
        momentum_ticks = getattr(env_class, "priceMomentumTicks", env_class.priceLookaheadTicks)
        buy_momentum_norm, sell_momentum_norm, buy_range_pos, sell_range_pos = (
            compute_price_momentum(
                buy_prices,
                sell_prices,
                data_tick,
                momentum_ticks,
                env_class.maxBuyPriceObs,
                env_class.maxSellPriceObs,
            )
        )
        base_obs.extend(
            [
                cross_mean_buy_norm,
                cross_mean_sell_norm,
                cross_std_buy_norm,
                buy_momentum_norm,
                sell_momentum_norm,
                buy_range_pos,
                sell_range_pos,
            ]
        )

    obs = np.array(base_obs, dtype=np.float32)
    return np.clip(obs, env_class().observation_space.low, env_class().observation_space.high)


def total_deferrable_energy(defer_state: list[dict]) -> float:
    return float(sum(dd["remainingEn"] for dd in defer_state))


def defer_headroom_w(instant_demand_w: float, *, load_max_w: float = 8.0) -> float:
    """Load cap 8 W/tick: instant is served first; defer gets only the remainder."""
    return max(0.0, float(load_max_w) - float(instant_demand_w))


def compute_defer_power_served_w(
    instant_demand_w: float,
    defer_state: list[dict],
    def_action: float,
    tick_dur_s: float,
    *,
    load_max_w: float = 8.0,
) -> float:
    """Defer power actually served this tick (capped by 8W − instant and remaining defer J)."""
    headroom = defer_headroom_w(instant_demand_w, load_max_w=load_max_w)
    requested_w = float(def_action) * headroom
    available_w = total_deferrable_energy(defer_state) / tick_dur_s
    return min(requested_w, available_w)


def compute_pico_demand_power(
    instant_demand: float,
    defer_state: list[dict],
    def_action: float,
    tick_dur_s: float,
    *,
    load_max_w: float = 8.0,
) -> float:
    """Total load power = instant (always) + defer served (≤ headroom below load_max_w)."""
    defer_power = compute_defer_power_served_w(
        instant_demand,
        defer_state,
        def_action,
        tick_dur_s,
        load_max_w=load_max_w,
    )
    return float(instant_demand) + defer_power

