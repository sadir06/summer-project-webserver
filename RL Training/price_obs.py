"""Cross-day and intraday price observation features (prototype 4)."""

from __future__ import annotations

import numpy as np


def build_cross_day_price_series(
    prior_profiles: list[dict],
    ticks_per_day: int,
) -> dict[str, list[float]] | None:
    """Same-tick mean/std buy and mean sell over prior completed days (A)."""
    if not prior_profiles:
        return None

    mean_buy: list[float] = []
    mean_sell: list[float] = []
    std_buy: list[float] = []
    for t in range(ticks_per_day):
        buys = [float(p["buyPrice"][t]) for p in prior_profiles]
        sells = [float(p["sellPrice"][t]) for p in prior_profiles]
        mean_buy.append(float(np.mean(buys)))
        mean_sell.append(float(np.mean(sells)))
        std_buy.append(float(np.std(buys)) if len(buys) > 1 else 0.0)

    return {
        "crossMeanBuy": mean_buy,
        "crossMeanSell": mean_sell,
        "crossStdBuy": std_buy,
    }


def cross_day_options_for_profile(
    profile: dict,
    sorted_profiles: list[dict],
    history_days: int,
    ticks_per_day: int,
) -> dict[str, list[float]] | None:
    day_id = profile["day_id"]
    index_by_day = {p["day_id"]: i for i, p in enumerate(sorted_profiles)}
    idx = index_by_day[day_id]
    prior = sorted_profiles[max(0, idx - history_days) : idx]
    return build_cross_day_price_series(prior, ticks_per_day)


def compute_price_momentum(
    buy_prices: list[float],
    sell_prices: list[float],
    tick: int,
    momentum_ticks: int,
    max_buy_price: float,
    max_sell_price: float,
) -> tuple[float, float, float, float]:
    """Intraday price momentum and range position (C). Uses ticks 0..tick only."""
    t = min(int(tick), len(buy_prices) - 1)
    buy_now = float(buy_prices[t])
    sell_now = float(sell_prices[t])
    t_prev = max(0, t - momentum_ticks)

    buy_momentum = (buy_now - float(buy_prices[t_prev])) / max_buy_price
    sell_momentum = (sell_now - float(sell_prices[t_prev])) / max_sell_price

    seen_buy = [float(x) for x in buy_prices[: t + 1]]
    seen_sell = [float(x) for x in sell_prices[: t + 1]]
    buy_min, buy_max = min(seen_buy), max(seen_buy)
    sell_min, sell_max = min(seen_sell), max(seen_sell)

    buy_span = buy_max - buy_min
    sell_span = sell_max - sell_min
    buy_range_pos = (buy_now - buy_min) / buy_span if buy_span > 1e-6 else 0.5
    sell_range_pos = (sell_now - sell_min) / sell_span if sell_span > 1e-6 else 0.5

    return (
        float(np.clip(buy_momentum, -1.0, 1.0)),
        float(np.clip(sell_momentum, -1.0, 1.0)),
        float(np.clip(buy_range_pos, 0.0, 1.0)),
        float(np.clip(sell_range_pos, 0.0, 1.0)),
    )


def cross_day_features_at_tick(
    tick: int,
    buy_price: float,
    sell_price: float,
    cross_mean_buy: list[float] | None,
    cross_mean_sell: list[float] | None,
    cross_std_buy: list[float] | None,
    max_buy_price: float,
    max_sell_price: float,
) -> tuple[float, float, float]:
    t = min(int(tick), len(cross_mean_buy) - 1) if cross_mean_buy else 0
    if cross_mean_buy and cross_mean_sell and cross_std_buy:
        mean_buy = cross_mean_buy[t]
        mean_sell = cross_mean_sell[t]
        std_buy = cross_std_buy[t]
    else:
        mean_buy = buy_price
        mean_sell = sell_price
        std_buy = 0.0

    return (
        mean_buy / max_buy_price,
        mean_sell / max_sell_price,
        float(np.clip(std_buy / max_buy_price, 0.0, 1.0)),
    )


class CrossDayPriceArchive:
    """Rolling archive of completed days for live / eval cross-day features."""

    def __init__(self, max_days: int = 7):
        self.max_days = max_days
        self._days: list[dict[str, list[float]]] = []

    def complete_day_from_buffer(self, series: dict[int, dict[str, float]], ticks_per_day: int) -> None:
        if not series:
            return
        buy = [float(series.get(t, {}).get("buy_price", 0.0)) for t in range(ticks_per_day)]
        sell = [float(series.get(t, {}).get("sell_price", 0.0)) for t in range(ticks_per_day)]
        self._days.append({"buyPrice": buy, "sellPrice": sell})
        if len(self._days) > self.max_days:
            self._days.pop(0)

    def stats_at_tick(
        self,
        tick: int,
        buy_price: float,
        sell_price: float,
        ticks_per_day: int,
        max_buy_price: float,
        max_sell_price: float,
    ) -> tuple[float, float, float]:
        prior_profiles = [
            {"buyPrice": d["buyPrice"], "sellPrice": d["sellPrice"]} for d in self._days
        ]
        series = build_cross_day_price_series(prior_profiles, ticks_per_day)
        if series is None:
            return cross_day_features_at_tick(
                tick, buy_price, sell_price, None, None, None, max_buy_price, max_sell_price
            )
        return cross_day_features_at_tick(
            tick,
            buy_price,
            sell_price,
            series["crossMeanBuy"],
            series["crossMeanSell"],
            series["crossStdBuy"],
            max_buy_price,
            max_sell_price,
        )
