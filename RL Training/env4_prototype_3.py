import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class SmartGridEnv(gym.Env):
    """Prototype 3 environment (frozen)."""
    supercapMinV = 10.35
    supercapMaxV = 16.25
    supercapCF = 0.65
    scEfficiency = 0.8

    minPhysicalSupercapEn = 0.5 * supercapCF * (supercapMinV**2)
    maxPhysicalSupercapEn = 0.5 * supercapCF * (supercapMaxV**2)
    maxSupercapEn = maxPhysicalSupercapEn - minPhysicalSupercapEn

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

    priceLookaheadTicks = 6

    gridBuyPowerMax = 8.0
    gridSellPowerMax = 8.0

    profitRewardScale = 100.0
    deferMissPenaltyPerJ_cents = 10

    @classmethod
    def sun_to_pv_w(cls, sun: float) -> float:
        """Map sun reading to PV watts."""
        return float(np.clip((sun / 100.0) * cls.pvPowerMax, 0.0, cls.pvPowerMax))

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
            ],
            dtype=np.float32,
        )
        obs_high = np.array(14 * [1.0], dtype=np.float32)
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
                sun = self.seededRandom.uniform(0.0, 100.0)
                pv_val = self.sun_to_pv_w(sun)
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
        start_tick = self.curTick
        end_tick = min(self.ticksPerDay, self.curTick + self.priceLookaheadTicks)
        data_tick = min(self.curTick, self.ticksPerDay - 1)

        future_buy = self.dailyData["buyPrice"][start_tick:end_tick]
        future_sell = self.dailyData["sellPrice"][start_tick:end_tick]
        future_pv = self.dailyData["pvGen"][start_tick:end_tick]
        future_demand = self.dailyData["baseDemand"][start_tick:end_tick]

        if not future_buy:
            future_buy = [self.dailyData["buyPrice"][data_tick]]
        if not future_sell:
            future_sell = [self.dailyData["sellPrice"][data_tick]]
        if not future_pv:
            future_pv = [self.dailyData["pvGen"][data_tick]]
        if not future_demand:
            future_demand = [self.dailyData["baseDemand"][data_tick]]

        avg_future_buy = float(np.mean(future_buy))
        avg_future_sell = float(np.mean(future_sell))
        projected_future_pv_en = float(np.sum(future_pv) * self.tickDur)
        projected_future_demand_en = float(np.sum(future_demand) * self.tickDur)
        projected_future_net_en = projected_future_pv_en - projected_future_demand_en
        return avg_future_buy, avg_future_sell, projected_future_net_en

    def getSupercapPhysicalEnergy(self):
        return self.minPhysicalSupercapEn + self.supercapEn

    def getSupercapVoltage(self):
        physical_energy = self.getSupercapPhysicalEnergy()
        voltage = np.sqrt((2.0 * physical_energy) / self.supercapCF)
        return float(np.clip(voltage, self.supercapMinV, self.supercapMaxV))

    def _charge_supercap_from_bus(self, bus_energy_j: float) -> float:
        """Charge from bus energy."""
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
        """Discharge to bus energy."""
        if bus_energy_j <= 0.0 or self.supercapEn <= 0.0:
            return 0.0
        max_deliver = self.supercapEn * self.scEfficiency
        delivered = min(bus_energy_j, max_deliver)
        self.supercapEn -= delivered / self.scEfficiency
        return delivered

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

        pv_en = pv_gen * self.tickDur
        base_demand_en = base_demand * self.tickDur

        def_capacity = def_action * self.maxDefPower * self.tickDur
        def_en_served = self._serve_deferrables(def_capacity)
        demand_en = base_demand_en + def_en_served

        imported_en = 0.0
        exported_en = 0.0

        pv_to_demand = min(pv_en, demand_en)
        remaining_demand = demand_en - pv_to_demand
        remaining_pv = pv_en - pv_to_demand

        discharge_budget_j = (
            abs(sc_action) * self.maxScDischarge * self.tickDur if sc_action < 0.0 else 0.0
        )
        charge_budget_j = (
            sc_action * self.maxScCharge * self.tickDur if sc_action > 0.0 else 0.0
        )

        sc_to_demand = 0.0
        if remaining_demand > 0.0 and discharge_budget_j > 0.0:
            sc_to_demand = self._discharge_supercap_to_bus(
                min(discharge_budget_j, remaining_demand)
            )
            discharge_budget_j -= sc_to_demand
            remaining_demand -= sc_to_demand

        grid_to_demand = 0.0
        if remaining_demand > 0.0:
            grid_to_demand = min(remaining_demand, self.gridBuyPowerMax * self.tickDur)
            imported_en += grid_to_demand
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
            exported_pv_en = min(remaining_pv, self.gridSellPowerMax * self.tickDur)
            exported_en += exported_pv_en
            remaining_pv -= exported_pv_en

        sc_to_grid = 0.0
        if discharge_budget_j > 0.0:
            export_capacity_left = max(
                0.0, self.gridSellPowerMax * self.tickDur - exported_en
            )
            sc_to_grid = self._discharge_supercap_to_bus(
                min(discharge_budget_j, export_capacity_left)
            )
            exported_en += sc_to_grid

        grid_to_sc = 0.0
        if charge_budget_j > 0.0:
            import_capacity_left = max(
                0.0, self.gridBuyPowerMax * self.tickDur - imported_en
            )
            grid_to_sc = min(charge_budget_j, import_capacity_left)
            absorbed = self._charge_supercap_from_bus(grid_to_sc)
            imported_en += absorbed
            grid_to_sc = absorbed

        missed_def_j = self._defer_deadline_penalty_j()

        tick_profit_cents = exported_en * sell_price - imported_en * buy_price
        self.totalProfitCents += tick_profit_cents
        self.totalImportJ += imported_en
        self.totalExportJ += exported_en

        reward = tick_profit_cents / self.profitRewardScale
        if missed_def_j > 0.0:
            reward -= (missed_def_j * self.deferMissPenaltyPerJ_cents) / self.profitRewardScale

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
            }
        )
        return self.getObs(), reward, terminated, False, info
