import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

from price_obs import compute_price_momentum, cross_day_features_at_tick


class SmartGridEnv(gym.Env):
    """Prototype 4: proto3 + SC voltage-limit penalty, hardware efficiencies (SC/grid/load)."""

    # Supercapacitor physical model (hardware voltage limits)
    supercapMinV = 10.25
    supercapMaxV = 16.0
    supercapCF = 0.65
    scEfficiency = 0.77  # measured hardware round-trip efficiency

    # Grid PSU (bidirectional): step η on |P_bus| (W); import metered = E_bus/η, export credit = E_bus×η
    gridEfficiencyLow = 0.55
    gridEfficiencyHigh = 0.90
    gridEfficiencyPowerThresholdW = 0.5

    # Load converter: measured η vs load power (W); bus draw = E_load/η_load(P)
    _LOAD_EFF_POWER_W = np.array(
        [0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0],
        dtype=np.float64,
    )
    _LOAD_EFF_ETA = np.array(
        [0.505, 0.673, 0.694, 0.694, 0.735, 0.794, 0.801, 0.806, 0.810, 0.794, 0.817, 0.818,
         0.807, 0.798, 0.801, 0.794, 0.780, 0.768],
        dtype=np.float64,
    )

    minPhysicalSupercapEn = 0.5 * supercapCF * (supercapMinV**2)
    maxPhysicalSupercapEn = 0.5 * supercapCF * (supercapMaxV**2)
    maxSupercapEn = maxPhysicalSupercapEn - minPhysicalSupercapEn

    # Power limits (watts)
    loadDemandMax = 8.0
    pvPowerMax = 7.0

    tickDur = 5
    ticksPerDay = 60

    defDemandATicks = 30

    maxBuyPriceObs = 95.0
    maxSellPriceObs = 191.0

    maxDefEn = 100.0
    maxDueSoon = 100.0

    maxScCharge = 3.0
    maxScDischarge = 3.0
    maxDefPower = 8.0

    priceLookaheadTicks = 6  # history window length (past ticks only; no future oracle)
    crossDayHistoryDays = 7  # same-tick stats from prior completed days (A)
    priceMomentumTicks = 6  # intraday Δprice lookback (C)

    # obs: 14 base + 3 cross-day + 4 momentum = 21
    obsDim = 21

    gridBuyPowerMax = 8.0
    gridSellPowerMax = 8.0

    profitRewardScale = 100.0
    # Must dominate tick profit so agent never skips defer (typical task ~37 J, max 50 J)
    deferMissPenaltyPerJ_cents = 200
    # Penalty for commanding SC past voltage limits (cents per J of wasted request)
    scLimitPenaltyPerJ_cents = 15.0
    scLimitActionThreshold = 0.05
    voltageLimitEpsilonV = 0.02

    # PV: uniform random per tick in [0, pvPowerMax] W (lab demo; not derived from sun)

    @classmethod
    def samplePvPower(cls, rng: random.Random) -> float:
        return float(rng.uniform(0.0, cls.pvPowerMax))

    @classmethod
    def is_at_min_voltage(cls, voltage: float) -> bool:
        return voltage <= cls.supercapMinV + cls.voltageLimitEpsilonV

    @classmethod
    def is_at_max_voltage(cls, voltage: float) -> bool:
        return voltage >= cls.supercapMaxV - cls.voltageLimitEpsilonV

    @classmethod
    def grid_efficiency(cls, bus_power_w: float) -> float:
        if abs(bus_power_w) < cls.gridEfficiencyPowerThresholdW:
            return cls.gridEfficiencyLow
        return cls.gridEfficiencyHigh

    @classmethod
    def load_efficiency(cls, load_power_w: float) -> float:
        p = float(np.clip(load_power_w, 0.0, cls.loadDemandMax))
        return float(
            np.interp(
                p,
                cls._LOAD_EFF_POWER_W,
                cls._LOAD_EFF_ETA,
                left=float(cls._LOAD_EFF_ETA[0]),
                right=float(cls._LOAD_EFF_ETA[-1]),
            )
        )

    def _meter_grid_import(self, bus_energy_j: float) -> float:
        """Bus J delivered from grid → metered J billed (E_grid = E_bus / η)."""
        if bus_energy_j <= 0.0:
            return 0.0
        p_bus = bus_energy_j / self.tickDur
        return bus_energy_j / self.grid_efficiency(p_bus)

    def _credit_grid_export(self, bus_energy_j: float) -> float:
        """Bus J sent to grid → metered J credited (E_credit = E_bus × η)."""
        if bus_energy_j <= 0.0:
            return 0.0
        p_bus = bus_energy_j / self.tickDur
        return bus_energy_j * self.grid_efficiency(p_bus)

    def __init__(self):
        super().__init__()
        obs_low = np.array(
            [
                -1.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                -1.0,
                -1.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
        obs_high = np.array(self.obsDim * [1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.curTick = 0
        self.supercapEn = 0.0
        self.totalProfitCents = 0.0
        self.totalImportJ = 0.0
        self.totalExportJ = 0.0
        self.seededRandom = None
        self.prevAction = np.zeros(2, dtype=np.float32)
        self.crossMeanBuy: list[float] | None = None
        self.crossMeanSell: list[float] | None = None
        self.crossStdBuy: list[float] | None = None
        self.dailyData = {
            "pvGen": [],
            "baseDemand": [],
            "buyPrice": [],
            "sellPrice": [],
            "defDemandState": [],
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.seededRandom = random.Random(
            seed if seed is not None else random.randint(0, 2**31 - 1)
        )
        self.curTick = 0
        self.supercapEn = 0.0
        self.totalProfitCents = 0.0
        self.totalImportJ = 0.0
        self.totalExportJ = 0.0
        self.prevAction = np.zeros(2, dtype=np.float32)

        options = options or {}
        pv_data = options.get("pvGen")
        base_demand_data = options.get("baseDemand")
        buy_price_data = options.get("buyPrice")
        sell_price_data = options.get("sellPrice")
        deferables_data = options.get("deferables")
        self.crossMeanBuy = options.get("crossMeanBuy")
        self.crossMeanSell = options.get("crossMeanSell")
        self.crossStdBuy = options.get("crossStdBuy")

        self.dailyData = {
            "pvGen": [],
            "baseDemand": [],
            "buyPrice": [],
            "sellPrice": [],
            "defDemandState": [],
        }

        for tick in range(self.ticksPerDay):
            if pv_data is not None:
                pv_val = float(pv_data[tick])
            else:
                pv_val = self.samplePvPower(self.seededRandom)
            if base_demand_data is not None:
                base_demand_val = float(base_demand_data[tick])
            else:
                base_demand_val = self.seededRandom.uniform(0.5, self.loadDemandMax)
            if buy_price_data is not None:
                buy_price_val = float(buy_price_data[tick])
            else:
                buy_price_val = self.seededRandom.uniform(5.0, self.maxBuyPriceObs)
            if sell_price_data is not None:
                sell_price_val = float(sell_price_data[tick])
            else:
                sell_price_val = self.seededRandom.uniform(10.0, self.maxSellPriceObs)

            self.dailyData["pvGen"].append(pv_val)
            self.dailyData["baseDemand"].append(base_demand_val)
            self.dailyData["buyPrice"].append(buy_price_val)
            self.dailyData["sellPrice"].append(sell_price_val)

        if deferables_data is not None:
            for task in deferables_data:
                energy = float(task["energy"])
                self.dailyData["defDemandState"].append(
                    {
                        "startTick": int(task["start"]),
                        "endTick": int(task["end"]),
                        "remainingEn": energy,
                        "originalEn": energy,
                        "served": False,
                        "missed": False,
                    }
                )

        return self.getObs(), self.getInfo()

    def getPriceProjection(self):
        """Rolling window over past ticks only (no future prices/PV/demand)."""
        end_tick = self.curTick + 1
        start_tick = max(0, end_tick - self.priceLookaheadTicks)
        data_tick = min(self.curTick, self.ticksPerDay - 1)

        hist_buy = self.dailyData["buyPrice"][start_tick:end_tick]
        hist_sell = self.dailyData["sellPrice"][start_tick:end_tick]
        hist_pv = self.dailyData["pvGen"][start_tick:end_tick]
        hist_demand = self.dailyData["baseDemand"][start_tick:end_tick]

        if not hist_buy:
            hist_buy = [self.dailyData["buyPrice"][data_tick]]
        if not hist_sell:
            hist_sell = [self.dailyData["sellPrice"][data_tick]]
        if not hist_pv:
            hist_pv = [self.dailyData["pvGen"][data_tick]]
        if not hist_demand:
            hist_demand = [self.dailyData["baseDemand"][data_tick]]

        avg_recent_buy = float(np.mean(hist_buy))
        avg_recent_sell = float(np.mean(hist_sell))
        projected_recent_net_en = float(
            np.sum((np.array(hist_pv) - np.array(hist_demand)) * self.tickDur)
        )
        return avg_recent_buy, avg_recent_sell, projected_recent_net_en

    def getSupercapPhysicalEnergy(self):
        return self.minPhysicalSupercapEn + self.supercapEn

    def getSupercapVoltage(self):
        physical_energy = self.getSupercapPhysicalEnergy()
        voltage = np.sqrt((2.0 * physical_energy) / self.supercapCF)
        return float(np.clip(voltage, self.supercapMinV, self.supercapMaxV))

    def _charge_supercap_from_bus(self, bus_energy_j: float) -> float:
        if bus_energy_j <= 0.0:
            return 0.0
        cap_space = self.maxSupercapEn - self.supercapEn
        if cap_space <= 0.0:
            return 0.0
        max_store = cap_space / self.scEfficiency
        absorbed = min(bus_energy_j, max_store)
        self.supercapEn += absorbed * self.scEfficiency
        return absorbed

    def _discharge_supercap_to_bus(self, bus_energy_j: float) -> float:
        if bus_energy_j <= 0.0 or self.supercapEn <= 0.0:
            return 0.0
        max_deliver = self.supercapEn * self.scEfficiency
        delivered = min(bus_energy_j, max_deliver)
        self.supercapEn -= delivered / self.scEfficiency
        return delivered

    def _sc_limit_wasted_j(self, sc_action: float, voltage: float) -> float:
        """Joules of SC action that hardware would reject (already at min/max V)."""
        wasted = 0.0
        if sc_action < -self.scLimitActionThreshold and self.is_at_min_voltage(voltage):
            wasted += abs(sc_action) * self.maxScDischarge * self.tickDur
        if sc_action > self.scLimitActionThreshold and self.is_at_max_voltage(voltage):
            wasted += sc_action * self.maxScCharge * self.tickDur
        return wasted

    def _serve_deferrables(self, def_capacity_j: float) -> float:
        served_total = 0.0
        remaining_capacity = def_capacity_j
        for dd in self.dailyData["defDemandState"]:
            if remaining_capacity <= 0.0:
                break
            if dd["remainingEn"] <= 0.0:
                continue
            if not (dd["startTick"] <= self.curTick <= dd["endTick"]):
                continue
            served = min(dd["remainingEn"], remaining_capacity)
            dd["remainingEn"] -= served
            remaining_capacity -= served
            served_total += served
        return served_total

    def _defer_deadline_penalty_j(self) -> float:
        penalty_j = 0.0
        for dd in self.dailyData["defDemandState"]:
            if dd["remainingEn"] <= 0.0:
                continue
            if self.curTick > dd["endTick"]:
                penalty_j += dd["remainingEn"]
                dd["remainingEn"] = 0.0
                dd["missed"] = True
        return penalty_j

    def getObs(self):
        data_tick = min(self.curTick, self.ticksPerDay - 1)
        pv_gen = self.dailyData["pvGen"][data_tick]
        base_demand = self.dailyData["baseDemand"][data_tick]
        buy_price = self.dailyData["buyPrice"][data_tick]
        sell_price = self.dailyData["sellPrice"][data_tick]

        time_angle = 2 * np.pi * data_tick / self.ticksPerDay
        time_sin = np.sin(time_angle)
        time_cos = np.cos(time_angle)

        total_def_en = sum(dd["remainingEn"] for dd in self.dailyData["defDemandState"])
        due_soon_en = 0.0
        for dd in self.dailyData["defDemandState"]:
            if dd["remainingEn"] > 0 and data_tick <= dd["endTick"] <= (
                data_tick + self.defDemandATicks
            ):
                due_soon_en += dd["remainingEn"]

        pv_norm = pv_gen / self.pvPowerMax
        sc_energy_norm = self.supercapEn / self.maxSupercapEn
        base_demand_norm = base_demand / self.loadDemandMax
        buy_norm = buy_price / self.maxBuyPriceObs
        sell_norm = sell_price / self.maxSellPriceObs
        total_def_norm = total_def_en / self.maxDefEn
        due_soon_norm = due_soon_en / self.maxDueSoon

        max_balance = max(self.pvPowerMax, self.loadDemandMax)
        net_norm = (pv_gen - base_demand) / max_balance

        avg_future_buy, avg_future_sell, projected_future_net_en = self.getPriceProjection()
        avg_future_buy_norm = avg_future_buy / self.maxBuyPriceObs
        avg_future_sell_norm = avg_future_sell / self.maxSellPriceObs
        cap_free_norm = (self.maxSupercapEn - self.supercapEn) / self.maxSupercapEn
        max_projected_net_en = self.pvPowerMax * self.tickDur * self.priceLookaheadTicks
        projected_future_net_norm = (
            projected_future_net_en / max_projected_net_en
            if max_projected_net_en > 0.0
            else 0.0
        )

        cross_mean_buy_norm, cross_mean_sell_norm, cross_std_buy_norm = cross_day_features_at_tick(
            data_tick,
            buy_price,
            sell_price,
            self.crossMeanBuy,
            self.crossMeanSell,
            self.crossStdBuy,
            self.maxBuyPriceObs,
            self.maxSellPriceObs,
        )
        buy_momentum_norm, sell_momentum_norm, buy_range_pos, sell_range_pos = (
            compute_price_momentum(
                self.dailyData["buyPrice"],
                self.dailyData["sellPrice"],
                data_tick,
                self.priceMomentumTicks,
                self.maxBuyPriceObs,
                self.maxSellPriceObs,
            )
        )

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
                cross_mean_buy_norm,
                cross_mean_sell_norm,
                cross_std_buy_norm,
                buy_momentum_norm,
                sell_momentum_norm,
                buy_range_pos,
                sell_range_pos,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def getInfo(self):
        data_tick = min(self.curTick, self.ticksPerDay - 1)
        avg_future_buy, avg_future_sell, projected_future_net_en = self.getPriceProjection()
        return {
            "curTick": self.curTick,
            "supercapEn": self.supercapEn,
            "supercapSoc": self.supercapEn / self.maxSupercapEn,
            "supercapVoltage": self.getSupercapVoltage(),
            "totalProfitCents": self.totalProfitCents,
            "totalCost": -self.totalProfitCents,
            "buyPrice": self.dailyData["buyPrice"][data_tick],
            "sellPrice": self.dailyData["sellPrice"][data_tick],
            "avgFutureBuy": avg_future_buy,
            "avgFutureSell": avg_future_sell,
            "projectedFutureNetEn": projected_future_net_en,
            "remainingDeferrableEn": sum(
                dd["remainingEn"] for dd in self.dailyData["defDemandState"]
            ),
        }

    def step(self, action):
        if self.curTick >= self.ticksPerDay:
            return self.getObs(), 0.0, True, False, self.getInfo()

        action = np.array(action, dtype=np.float32)
        sc_action = float(np.clip(action[0], -1.0, 1.0))
        def_action = float(np.clip(action[1], 0.0, 1.0))

        pv_gen = self.dailyData["pvGen"][self.curTick]
        base_demand = self.dailyData["baseDemand"][self.curTick]
        buy_price = self.dailyData["buyPrice"][self.curTick]
        sell_price = self.dailyData["sellPrice"][self.curTick]

        sc_voltage_before = self.getSupercapVoltage()
        sc_wasted_j = self._sc_limit_wasted_j(sc_action, sc_voltage_before)
        sc_limit_penalty_cents = sc_wasted_j * self.scLimitPenaltyPerJ_cents

        pv_en = pv_gen * self.tickDur
        base_demand_en = base_demand * self.tickDur

        defer_headroom_w = max(0.0, self.loadDemandMax - base_demand)
        defer_power_requested_w = def_action * defer_headroom_w
        def_capacity = defer_power_requested_w * self.tickDur
        def_en_served = self._serve_deferrables(def_capacity)
        def_power_served_w = def_en_served / self.tickDur
        load_terminal_en = base_demand_en + def_en_served
        load_power_w = load_terminal_en / self.tickDur
        load_eta = self.load_efficiency(load_power_w)
        bus_demand_en = load_terminal_en / load_eta if load_eta > 0.0 else load_terminal_en

        imported_en = 0.0
        exported_en = 0.0
        grid_import_bus_j = 0.0
        grid_export_bus_j = 0.0

        pv_to_demand = min(pv_en, bus_demand_en)
        remaining_demand = bus_demand_en - pv_to_demand
        remaining_pv = pv_en - pv_to_demand

        requested_discharge_j = (
            abs(sc_action) * self.maxScDischarge * self.tickDur if sc_action < 0.0 else 0.0
        )
        requested_charge_j = (
            sc_action * self.maxScCharge * self.tickDur if sc_action > 0.0 else 0.0
        )
        discharge_budget_j = requested_discharge_j
        charge_budget_j = requested_charge_j

        sc_to_demand = 0.0
        if remaining_demand > 0.0 and discharge_budget_j > 0.0:
            sc_to_demand = self._discharge_supercap_to_bus(
                min(discharge_budget_j, remaining_demand)
            )
            discharge_budget_j -= sc_to_demand
            remaining_demand -= sc_to_demand

        grid_to_demand = 0.0
        if remaining_demand > 0.0:
            import_cap_bus_j = max(
                0.0, self.gridBuyPowerMax * self.tickDur - grid_import_bus_j
            )
            grid_to_demand = min(remaining_demand, import_cap_bus_j)
            grid_import_bus_j += grid_to_demand
            imported_en += self._meter_grid_import(grid_to_demand)
            remaining_demand -= grid_to_demand

        pv_charge_en = 0.0
        if remaining_pv > 0.0 and charge_budget_j > 0.0:
            pv_charge_en = self._charge_supercap_from_bus(
                min(charge_budget_j, remaining_pv)
            )
            charge_budget_j -= pv_charge_en
            remaining_pv -= pv_charge_en

        exported_pv_en = 0.0
        if remaining_pv > 0.0:
            export_cap_bus_j = max(
                0.0, self.gridSellPowerMax * self.tickDur - grid_export_bus_j
            )
            exported_pv_en = min(remaining_pv, export_cap_bus_j)
            grid_export_bus_j += exported_pv_en
            exported_en += self._credit_grid_export(exported_pv_en)
            remaining_pv -= exported_pv_en

        sc_to_grid = 0.0
        if discharge_budget_j > 0.0:
            export_cap_bus_j = max(
                0.0, self.gridSellPowerMax * self.tickDur - grid_export_bus_j
            )
            sc_to_grid = self._discharge_supercap_to_bus(
                min(discharge_budget_j, export_cap_bus_j)
            )
            grid_export_bus_j += sc_to_grid
            exported_en += self._credit_grid_export(sc_to_grid)

        grid_to_sc = 0.0
        if charge_budget_j > 0.0:
            import_cap_bus_j = max(
                0.0, self.gridBuyPowerMax * self.tickDur - grid_import_bus_j
            )
            grid_to_sc = min(charge_budget_j, import_cap_bus_j)
            absorbed = self._charge_supercap_from_bus(grid_to_sc)
            grid_import_bus_j += absorbed
            imported_en += self._meter_grid_import(absorbed)
            grid_to_sc = absorbed

        missed_def_j = self._defer_deadline_penalty_j()

        # Profit: imported_en / exported_en are METERED (grid-billed) joules after η.
        # Bus dispatch uses bus_demand_en (load_terminal/η_load), SC actual J, grid η on transfers.
        actual_sc_discharge_bus_j = sc_to_demand + sc_to_grid
        actual_sc_charge_bus_j = pv_charge_en + grid_to_sc

        tick_profit_cents = exported_en * sell_price - imported_en * buy_price
        self.totalProfitCents += tick_profit_cents
        self.totalImportJ += imported_en
        self.totalExportJ += exported_en

        reward = tick_profit_cents / self.profitRewardScale
        if missed_def_j > 0.0:
            reward -= (missed_def_j * self.deferMissPenaltyPerJ_cents) / self.profitRewardScale
        if sc_limit_penalty_cents > 0.0:
            reward -= sc_limit_penalty_cents / self.profitRewardScale

        self.prevAction = np.array([sc_action, def_action], dtype=np.float32)
        self.curTick += 1
        terminated = self.curTick >= self.ticksPerDay

        if terminated:
            end_missed_j = 0.0
            for dd in self.dailyData["defDemandState"]:
                if dd["remainingEn"] > 0.0:
                    end_missed_j += dd["remainingEn"]
                    dd["missed"] = True
                    dd["remainingEn"] = 0.0
            if end_missed_j > 0.0:
                reward -= (
                    end_missed_j * self.deferMissPenaltyPerJ_cents
                ) / self.profitRewardScale
            for dd in self.dailyData["defDemandState"]:
                if dd["remainingEn"] <= 0 and not dd.get("served") and not dd.get("missed"):
                    dd["served"] = True

        info = self.getInfo()
        info.update(
            {
                "importedEn": imported_en,
                "exportedEn": exported_en,
                "tickProfitCents": tick_profit_cents,
                "netProfitThisTick": tick_profit_cents,
                "costThisTick": imported_en * buy_price,
                "profitThisTick": exported_en * sell_price,
                "defEnServedThisTick": def_en_served,
                "deferHeadroomW": defer_headroom_w,
                "deferPowerRequestedW": defer_power_requested_w,
                "deferPowerServedW": def_power_served_w,
                "loadTerminalEn": load_terminal_en,
                "loadPowerW": load_power_w,
                "loadEfficiency": load_eta,
                "busDemandEn": bus_demand_en,
                "gridImportBusJ": grid_import_bus_j,
                "gridExportBusJ": grid_export_bus_j,
                "missedDefJ": missed_def_j,
                "pvToDemand": pv_to_demand,
                "scToDemand": sc_to_demand,
                "gridToDemand": grid_to_demand,
                "pvChargeEn": pv_charge_en,
                "exportedPvEn": exported_pv_en,
                "scToGrid": sc_to_grid,
                "gridToSc": grid_to_sc,
                "scAction": sc_action,
                "defAction": def_action,
                "scVoltageBefore": sc_voltage_before,
                "scWastedJ": sc_wasted_j,
                "scLimitPenaltyCents": sc_limit_penalty_cents,
                "requestedScDischargeJ": requested_discharge_j,
                "requestedScChargeJ": requested_charge_j,
                "actualScDischargeBusJ": actual_sc_discharge_bus_j,
                "actualScChargeBusJ": actual_sc_charge_bus_j,
                "actualScDischargeBusW": actual_sc_discharge_bus_j / self.tickDur,
                "actualScChargeBusW": actual_sc_charge_bus_j / self.tickDur,
            }
        )
        return self.getObs(), reward, terminated, False, info
