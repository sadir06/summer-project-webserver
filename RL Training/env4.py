import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class SmartGridEnv(gym.Env):
    # Supercapacitor physical energy model
    supercapMinV = 10.35
    supercapMaxV = 16.13
    supercapCF = 0.5

    minPhysicalSupercapEn = 0.5 * supercapCF * (supercapMinV ** 2)
    maxPhysicalSupercapEn = 0.5 * supercapCF * (supercapMaxV ** 2)
    maxSupercapEn = maxPhysicalSupercapEn - minPhysicalSupercapEn

    # Main system limits
    loadDemandMax = 8
    pvPowerMax = 7

    tickDur = 5
    ticksPerDay = 60

    # Separate demand penalties
    penaltyBaseDemand = -100
    penaltyDefDemand = -30
    penaltyCapOvUn = -10

    defDemandATicks = 30

    maxPrice = 1.0
    maxDefEn = 100.0
    maxDueSoon = 100.0

    maxScCharge = 3
    maxScDischarge = 3

    # Maximum deferable demand serving power
    maxDefPower = 8

    priceLookaheadTicks = 6

    gridBuyPowerMax = 8
    gridSellPowerMax = 8

    # Kept but deliberately ignored in sell logic
    minSellSoc = 0

    buyMargin = 0.05
    sellMargin = 0.05

    # Reward shaping terms
    pvCurtailmentPenalty = 0.05
    balancePenalty = 0.01
    actionSmoothPenalty = 0.02

    # PV cell noise (irradiance -> V/I -> power is not deterministic)
    pvNoiseStd = 0.08

    @staticmethod
    def normalizeIrradiance(sun: float) -> float:
        if sun > 1.0:
            return float(np.clip(sun / 100.0, 0.0, 1.0))
        return float(np.clip(sun, 0.0, 1.0))

    @classmethod
    def computePvPowerFromIrradiance(cls, irradiance: float, rng: random.Random) -> float:
        """PV power from irradiance via nonlinear V-I behaviour + Gaussian noise."""
        g = float(np.clip(irradiance, 0.0, 1.0))
        if g <= 0.0:
            return 0.0

        voc_norm = 0.82 + 0.18 * (g ** 0.15)
        isc_norm = g ** 0.95
        vmp = 0.78 * voc_norm
        imp = 0.88 * isc_norm
        p_norm = vmp * imp

        noise = rng.gauss(1.0, cls.pvNoiseStd)
        power = cls.pvPowerMax * p_norm * noise
        return float(np.clip(power, 0.0, cls.pvPowerMax))

    # Environment setup
    def __init__(self):
        super().__init__()

        obsLow = np.array([
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
        ], dtype=np.float32)

        obsHigh = np.array(14 * [1.0], dtype=np.float32)

        self.obsSpace = spaces.Box(
            low=obsLow,
            high=obsHigh,
            dtype=np.float32
        )
        self.observation_space = self.obsSpace

        # Actions: gridAction, scAction, defAction
        self.actionSpace = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = self.actionSpace

        self.curTick = 0
        self.supercapEn = 0.0
        self.totalCost = 0.0
        self.totalUnmetBaseDemand = 0.0
        self.totalUnmetDefDemand = 0.0
        self.prevUnmetBaseDemand = 0.0
        self.prevUnmetDefDemand = 0.0
        self.seededRandom = None
        self.prevAction = np.zeros(3, dtype=np.float32)

        self.dailyData = {
            "pvGen": [],
            "baseDemand": [],
            "buyPrice": [],
            "sellPrice": [],
            "defDemandState": []
        }

    # Reset one simulated day
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.seededRandom = random.Random(
            seed if seed is not None else random.randint(0, 2**31 - 1)
        )

        self.curTick = 0
        self.supercapEn = 0.0
        self.totalCost = 0.0
        self.totalUnmetBaseDemand = 0.0
        self.totalUnmetDefDemand = 0.0
        self.prevUnmetBaseDemand = 0.0
        self.prevUnmetDefDemand = 0.0
        self.prevAction = np.zeros(3, dtype=np.float32)

        if options is None:
            options = {}

        irradianceData = options.get("irradiance", None)
        baseDemandData = options.get("baseDemand", None)
        buyPriceData = options.get("buyPrice", None)
        sellPriceData = options.get("sellPrice", None)
        deferablesData = options.get("deferables", None)

        self.dailyData = {
            "pvGen": [],
            "baseDemand": [],
            "buyPrice": [],
            "sellPrice": [],
            "defDemandState": []
        }

        for tick in range(self.ticksPerDay):
            if irradianceData is not None:
                irr = self.normalizeIrradiance(float(irradianceData[tick]))
            else:
                irr = self.seededRandom.uniform(0.0, 1.0)

            pvVal = self.computePvPowerFromIrradiance(irr, self.seededRandom)

            if baseDemandData is not None:
                baseDemandVal = float(baseDemandData[tick])
            else:
                baseDemandVal = self.seededRandom.uniform(1.0, self.loadDemandMax)

            if buyPriceData is not None:
                buyPriceVal = float(buyPriceData[tick])
            else:
                buyPriceVal = self.seededRandom.uniform(0.2, 1.0)

            if sellPriceData is not None:
                sellPriceVal = float(sellPriceData[tick])
            else:
                sellPriceVal = self.seededRandom.uniform(0.05, 0.5)

            self.dailyData["pvGen"].append(pvVal)
            self.dailyData["baseDemand"].append(baseDemandVal)
            self.dailyData["buyPrice"].append(buyPriceVal)
            self.dailyData["sellPrice"].append(sellPriceVal)

        if deferablesData is not None:
            for task in deferablesData:
                energy = float(task["energy"])

                self.dailyData["defDemandState"].append({
                    "startTick": int(task["start"]),
                    "endTick": int(task["end"]),
                    "remainingEn": energy,
                    "originalEn": energy,
                    "served": False,
                    "missed": False
                })

        obs = self.getObs()
        info = self.getInfo()

        return obs, info

    # Look ahead at future price and net energy
    def getPriceProjection(self):
        startTick = self.curTick
        endTick = min(self.ticksPerDay, self.curTick + self.priceLookaheadTicks)
        dataTick = min(self.curTick, self.ticksPerDay - 1)

        futureBuyPrices = self.dailyData["buyPrice"][startTick:endTick]
        futureSellPrices = self.dailyData["sellPrice"][startTick:endTick]
        futurePv = self.dailyData["pvGen"][startTick:endTick]
        futureDemand = self.dailyData["baseDemand"][startTick:endTick]

        if len(futureBuyPrices) == 0:
            futureBuyPrices = [self.dailyData["buyPrice"][dataTick]]
        if len(futureSellPrices) == 0:
            futureSellPrices = [self.dailyData["sellPrice"][dataTick]]
        if len(futurePv) == 0:
            futurePv = [self.dailyData["pvGen"][dataTick]]
        if len(futureDemand) == 0:
            futureDemand = [self.dailyData["baseDemand"][dataTick]]

        avgFutureBuy = float(np.mean(futureBuyPrices))
        avgFutureSell = float(np.mean(futureSellPrices))
        minFutureBuy = float(np.min(futureBuyPrices))
        maxFutureSell = float(np.max(futureSellPrices))

        projectedFuturePvEn = float(np.sum(futurePv) * self.tickDur)
        projectedFutureDemandEn = float(np.sum(futureDemand) * self.tickDur)
        projectedFutureNetEn = projectedFuturePvEn - projectedFutureDemandEn

        return (
            avgFutureBuy,
            avgFutureSell,
            minFutureBuy,
            maxFutureSell,
            projectedFutureNetEn
        )

    # Decide whether to buy cheap grid energy for storage
    def shouldBuyForStorage(self, buyPrice, avgFutureBuy):
        capSpace = self.maxSupercapEn - self.supercapEn

        if capSpace <= 0.0:
            return False

        favourableBuyPrice = buyPrice < (avgFutureBuy * (1.0 - self.buyMargin))
        return favourableBuyPrice

    # Decide whether to sell stored energy
    def shouldSellFromStorage(self, sellPrice, avgFutureSell, projectedFutureNetEn):
        if self.supercapEn <= 0.0:
            return False

        favourableSellPrice = sellPrice > (avgFutureSell * (1.0 + self.sellMargin))
        futureLooksComfortable = projectedFutureNetEn >= 0.0

        return favourableSellPrice and futureLooksComfortable

    # Decide whether deferable demand should be served this tick
    def shouldServeDeferrableNow(self, dd, buyPrice):
        if dd["remainingEn"] <= 0.0:
            return False

        if not (dd["startTick"] <= self.curTick <= dd["endTick"]):
            return False

        maxDefEnPerTick = self.maxDefPower * self.tickDur
        remainingTicks = max(1, dd["endTick"] - self.curTick + 1)
        ticksNeeded = int(np.ceil(dd["remainingEn"] / maxDefEnPerTick))
        mustServeNow = ticksNeeded >= remainingTicks

        remainingWindowPrices = self.dailyData["buyPrice"][
            self.curTick : dd["endTick"] + 1
        ]

        if len(remainingWindowPrices) == 0:
            remainingWindowPrices = [buyPrice]

        avgRemainingPrice = float(np.mean(remainingWindowPrices))
        priceIsReasonable = buyPrice <= avgRemainingPrice

        return priceIsReasonable or mustServeNow

    # Convert usable supercap energy into physical energy
    def getSupercapPhysicalEnergy(self):
        return self.minPhysicalSupercapEn + self.supercapEn

    # Estimate supercap voltage from stored energy
    def getSupercapVoltage(self):
        physicalEnergy = self.getSupercapPhysicalEnergy()
        voltage = np.sqrt((2.0 * physicalEnergy) / self.supercapCF)

        return float(np.clip(
            voltage,
            self.supercapMinV,
            self.supercapMaxV
        ))

    # Build observation vector
    def getObs(self):
        dataTick = min(self.curTick, self.ticksPerDay - 1)

        pvGen = self.dailyData["pvGen"][dataTick]
        baseDemand = self.dailyData["baseDemand"][dataTick]
        buyPrice = self.dailyData["buyPrice"][dataTick]
        sellPrice = self.dailyData["sellPrice"][dataTick]

        timeAngle = 2 * np.pi * dataTick / self.ticksPerDay
        timeSin = np.sin(timeAngle)
        timeCos = np.cos(timeAngle)

        totalDefEn = sum(
            dd["remainingEn"]
            for dd in self.dailyData["defDemandState"]
        )

        dueSoonEn = 0.0
        for dd in self.dailyData["defDemandState"]:
            if dd["remainingEn"] > 0:
                if dataTick <= dd["endTick"] <= (dataTick + self.defDemandATicks):
                    dueSoonEn += dd["remainingEn"]

        pvNorm = pvGen / self.pvPowerMax
        scSoc = self.supercapEn / self.maxSupercapEn
        baseDemandNorm = baseDemand / self.loadDemandMax
        buyNorm = buyPrice / self.maxPrice
        sellNorm = sellPrice / self.maxPrice
        totalDefNorm = totalDefEn / self.maxDefEn
        dueSoonNorm = dueSoonEn / self.maxDueSoon

        netBalance = pvGen - baseDemand
        maxBalance = max(self.pvPowerMax, self.loadDemandMax)
        netNorm = netBalance / maxBalance

        (
            avgFutureBuy,
            avgFutureSell,
            minFutureBuy,
            maxFutureSell,
            projectedFutureNetEn
        ) = self.getPriceProjection()

        avgFutureBuyNorm = avgFutureBuy / self.maxPrice
        avgFutureSellNorm = avgFutureSell / self.maxPrice
        capFreeNorm = (self.maxSupercapEn - self.supercapEn) / self.maxSupercapEn

        maxProjectedNetEn = self.pvPowerMax * self.tickDur * self.priceLookaheadTicks
        if maxProjectedNetEn <= 0:
            projectedFutureNetNorm = 0.0
        else:
            projectedFutureNetNorm = projectedFutureNetEn / maxProjectedNetEn

        obs = np.array([
            timeSin,
            timeCos,
            pvNorm,
            scSoc,
            baseDemandNorm,
            buyNorm,
            sellNorm,
            totalDefNorm,
            dueSoonNorm,
            netNorm,
            avgFutureBuyNorm,
            avgFutureSellNorm,
            capFreeNorm,
            projectedFutureNetNorm
        ], dtype=np.float32)

        return np.clip(
            obs,
            self.observation_space.low,
            self.observation_space.high
        )

    # Build debug information dictionary
    def getInfo(self):
        dataTick = min(self.curTick, self.ticksPerDay - 1)

        (
            avgFutureBuy,
            avgFutureSell,
            minFutureBuy,
            maxFutureSell,
            projectedFutureNetEn
        ) = self.getPriceProjection()

        info = {
            "curTick": self.curTick,
            "supercapEn": self.supercapEn,
            "supercapSoc": self.supercapEn / self.maxSupercapEn,
            "supercapFreeEn": self.maxSupercapEn - self.supercapEn,
            "supercapVoltage": self.getSupercapVoltage(),
            "supercapPhysicalEn": self.getSupercapPhysicalEnergy(),
            "supercapUsableEn": self.supercapEn,
            "totalCost": self.totalCost,
            "totalUnmetBaseDemand": self.totalUnmetBaseDemand,
            "totalUnmetDefDemand": self.totalUnmetDefDemand,
            "unmetBaseDemandThisTick": (
                self.totalUnmetBaseDemand - self.prevUnmetBaseDemand
            ),
            "unmetDefDemandThisTick": (
                self.totalUnmetDefDemand - self.prevUnmetDefDemand
            ),
            "buyPrice": self.dailyData["buyPrice"][dataTick],
            "sellPrice": self.dailyData["sellPrice"][dataTick],
            "avgFutureBuy": avgFutureBuy,
            "avgFutureSell": avgFutureSell,
            "minFutureBuy": minFutureBuy,
            "maxFutureSell": maxFutureSell,
            "projectedFutureNetEn": projectedFutureNetEn,
            "remainingDeferrableEn": sum(
                dd["remainingEn"]
                for dd in self.dailyData["defDemandState"]
            )
        }

        return info

    # Run one environment tick
    def step(self, action):
        if self.curTick >= self.ticksPerDay:
            obs = self.getObs()
            info = self.getInfo()
            return (obs, 0.0, True, False, info)

        action = np.array(action, dtype=np.float32)
        gridAction = np.clip(action[0], -1.0, 1.0)
        scAction = np.clip(action[1], -1.0, 1.0)
        defAction = np.clip(action[2], 0.0, 1.0)
        actionChange = np.sum(np.abs(action - self.prevAction))

        pvGen = self.dailyData["pvGen"][self.curTick]
        baseDemand = self.dailyData["baseDemand"][self.curTick]
        buyPrice = self.dailyData["buyPrice"][self.curTick]
        sellPrice = self.dailyData["sellPrice"][self.curTick]

        (
            avgFutureBuy,
            avgFutureSell,
            minFutureBuy,
            maxFutureSell,
            projectedFutureNetEn
        ) = self.getPriceProjection()

        pvEn = pvGen * self.tickDur
        baseDemandEn = baseDemand * self.tickDur
        defEnServedThisTick = 0.0
        defDeadlinePenalty = 0.0
        defCapacityLeft = defAction * self.maxDefPower * self.tickDur

        for dd in self.dailyData["defDemandState"]:
            if dd["remainingEn"] <= 0.0:
                continue

            if self.curTick > dd["endTick"]:
                defDeadlinePenalty += dd["remainingEn"] * self.penaltyDefDemand
                dd["remainingEn"] = 0.0
                dd["missed"] = True
                continue

            if self.shouldServeDeferrableNow(dd, buyPrice) and defCapacityLeft > 0.0:
                served = min(dd["remainingEn"], defCapacityLeft)
                dd["remainingEn"] -= served
                defCapacityLeft -= served
                defEnServedThisTick += served

        demandEn = baseDemandEn + defEnServedThisTick

        if scAction >= 0:
            scPowerWanted = scAction * self.maxScCharge
        else:
            scPowerWanted = scAction * self.maxScDischarge

        agentScDelta = scPowerWanted * self.tickDur

        arbBuyEn = 0.0
        arbSellEn = 0.0
        wantArbBuy = self.shouldBuyForStorage(buyPrice, avgFutureBuy)
        wantArbSell = self.shouldSellFromStorage(
            sellPrice,
            avgFutureSell,
            projectedFutureNetEn
        )

        if wantArbBuy:
            arbBuyEn = self.gridBuyPowerMax * self.tickDur

        if (not wantArbBuy) and wantArbSell:
            arbSellEn = self.gridSellPowerMax * self.tickDur

        requestedScDelta = agentScDelta + arbBuyEn - arbSellEn
        newSupercapEn = float(np.clip(
            self.supercapEn + requestedScDelta,
            0.0,
            self.maxSupercapEn
        ))
        actualScDelta = newSupercapEn - self.supercapEn

        scPenalty = 0.0
        if abs(actualScDelta - requestedScDelta) > 1e-6:
            scPenalty += self.penaltyCapOvUn

        self.supercapEn = newSupercapEn
        scFlow = -actualScDelta

        if actualScDelta > 0.0:
            arbitrageImportEn = min(arbBuyEn, actualScDelta) if wantArbBuy else 0.0
        else:
            arbitrageImportEn = 0.0

        if actualScDelta < 0.0:
            arbitrageExportEn = (
                min(arbSellEn, abs(actualScDelta))
                if wantArbSell and not wantArbBuy
                else 0.0
            )
        else:
            arbitrageExportEn = 0.0

        netDemand = demandEn - pvEn - scFlow
        importedEn = arbitrageImportEn
        exportedEn = arbitrageExportEn
        unmetBaseDemand = 0.0
        unmetDefDemand = 0.0
        curtailedPvEn = 0.0
        exportedSurplusEn = 0.0
        remainingImbalance = 0.0

        if netDemand > 0:
            if gridAction >= 0:
                demandImportEn = gridAction * netDemand
                importedEn += demandImportEn
            else:
                demandImportEn = 0.0

            unmetDemand = max(0.0, netDemand - demandImportEn)
            remainingImbalance = unmetDemand
            unmetBaseDemand = min(unmetDemand, baseDemandEn)
            unmetDefDemand = max(0.0, unmetDemand - unmetBaseDemand)

            self.totalUnmetBaseDemand += unmetBaseDemand
            self.totalUnmetDefDemand += unmetDefDemand
        else:
            surplus = abs(netDemand)

            if gridAction < 0:
                exportedSurplusEn = abs(gridAction) * surplus
                exportedEn += exportedSurplusEn

            curtailedPvEn = max(0.0, surplus - exportedSurplusEn)
            remainingImbalance = curtailedPvEn

        cost = importedEn * buyPrice
        profit = exportedEn * sellPrice
        self.totalCost += (cost - profit)

        unmetBasePenalty = (
            (self.totalUnmetBaseDemand - self.prevUnmetBaseDemand)
            * self.penaltyBaseDemand
        )
        unmetDefPenalty = (
            (self.totalUnmetDefDemand - self.prevUnmetDefDemand)
            * self.penaltyDefDemand
        )

        curtailmentPenalty = -self.pvCurtailmentPenalty * curtailedPvEn
        balanceError = remainingImbalance
        busBalancePenalty = -self.balancePenalty * balanceError
        smoothnessPenalty = -self.actionSmoothPenalty * actionChange

        reward = (
            -cost
            + profit
            + unmetBasePenalty
            + unmetDefPenalty
            + scPenalty
            + defDeadlinePenalty
            + curtailmentPenalty
            + busBalancePenalty
            + smoothnessPenalty
        )

        if arbitrageImportEn > 0.0:
            reward += 0.5
        if arbitrageExportEn > 0.0:
            reward += 0.5

        defCompleted = 0.0
        for dd in self.dailyData["defDemandState"]:
            if (
                dd["remainingEn"] <= 0
                and not dd.get("served", False)
                and not dd.get("missed", False)
            ):
                defCompleted += 1.0
                dd["served"] = True

        reward += defCompleted * 5.0

        self.prevUnmetBaseDemand = self.totalUnmetBaseDemand
        self.prevUnmetDefDemand = self.totalUnmetDefDemand
        self.prevAction = np.array([gridAction, scAction, defAction], dtype=np.float32)
        self.curTick += 1

        terminated = self.curTick >= self.ticksPerDay
        if terminated:
            remainingDefEn = sum(
                dd["remainingEn"]
                for dd in self.dailyData["defDemandState"]
                if not dd.get("served", False)
                and not dd.get("missed", False)
            )

            if remainingDefEn > 0.0:
                reward += remainingDefEn * self.penaltyDefDemand

                for dd in self.dailyData["defDemandState"]:
                    if dd["remainingEn"] > 0.0:
                        dd["missed"] = True
                        dd["remainingEn"] = 0.0

        obs = self.getObs()
        info = self.getInfo()

        info["importedEn"] = importedEn
        info["exportedEn"] = exportedEn
        info["arbitrageImportEn"] = arbitrageImportEn
        info["arbitrageExportEn"] = arbitrageExportEn
        info["costThisTick"] = cost
        info["profitThisTick"] = profit
        info["defEnServedThisTick"] = defEnServedThisTick
        info["defDeadlinePenalty"] = defDeadlinePenalty
        info["unmetBaseDemand"] = unmetBaseDemand
        info["unmetDefDemand"] = unmetDefDemand
        info["curtailedPvEn"] = curtailedPvEn
        info["balanceError"] = balanceError
        info["remainingImbalance"] = remainingImbalance
        info["actionChange"] = actionChange

        return (obs, reward, terminated, False, info)
