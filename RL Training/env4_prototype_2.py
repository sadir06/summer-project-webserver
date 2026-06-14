import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class SmartGridEnv(gym.Env):
    supercapMinV = 10.35
    supercapMaxV = 16.13
    supercapCF = 0.5

    minPhysicalSupercapEn = 0.5 * supercapCF * (supercapMinV**2)
    maxPhysicalSupercapEn = 0.5 * supercapCF * (supercapMaxV**2)
    maxSupercapEn = maxPhysicalSupercapEn - minPhysicalSupercapEn

    loadDemandMax = 8
    pvPowerMax = 7

    tickDur = 5
    ticksPerDay = 60

    penaltyBaseDemand = -100
    penaltyDefDemand = -30
    penaltyCapOvUn = -10
    demandHardConstraintPenalty = -10000

    defDemandATicks = 30

    maxPrice = 1.0
    maxDefEn = 100.0
    maxDueSoon = 100.0

    maxScCharge = 3
    maxScDischarge = 3

    maxDefPower = 8

    priceLookaheadTicks = 6

    gridBuyPowerMax = 8
    gridSellPowerMax = 8

    minSellSoc = 0

    buyMargin = 0.05
    sellMargin = 0.05
    storageArbitrageMargin = 0.10

    pvCurtailmentPenalty = 0.05
    balancePenalty = 0.01
    actionSmoothPenalty = 0.02

    pvGenMean = 3.5
    pvGenStd = 2.0

    emptyScDischargePenaltyWeight = 2.0
    maxDischargeSpamPenaltyWeight = 1.0
    pvChargeRewardWeight = 0.1
    curtailmentWithSpacePenaltyWeight = 0.2
    unnecessaryImportPenaltyWeight = 0.2
    arbitrageEnergyRewardWeight = 0.05
    targetEndSoc = 0.25
    endSocPenaltyWeight = 5.0

    @classmethod
    def samplePvPower(cls, rng: random.Random) -> float:
        """Sample PV power (kW) from a normal distribution, clipped to system limits."""
        power = rng.gauss(cls.pvGenMean, cls.pvGenStd)
        return float(np.clip(power, 0.0, cls.pvPowerMax))

    def __init__(self):
        super().__init__()
        obsLow = np.array(
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
        obsHigh = np.array(14 * [1.0], dtype=np.float32)
        self.obsSpace = spaces.Box(low=obsLow, high=obsHigh, dtype=np.float32)
        self.observation_space = self.obsSpace

        self.actionSpace = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
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
            "defDemandState": [],
        }

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
        pvData = options.get("pvGen", None)
        baseDemandData = options.get("baseDemand", None)
        buyPriceData = options.get("buyPrice", None)
        sellPriceData = options.get("sellPrice", None)
        deferablesData = options.get("deferables", None)
        self.dailyData = {
            "pvGen": [],
            "baseDemand": [],
            "buyPrice": [],
            "sellPrice": [],
            "defDemandState": [],
        }
        for tick in range(self.ticksPerDay):
            if pvData is not None:
                pvVal = float(pvData[tick])
            else:
                pvVal = self.samplePvPower(self.seededRandom)
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
        obs = self.getObs()
        info = self.getInfo()
        return obs, info

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
            projectedFutureNetEn,
        )

    def shouldBuyForStorage(self, buyPrice, avgFutureBuy, maxFutureSell):
        capSpace = self.maxSupercapEn - self.supercapEn
        if capSpace <= 0.0:
            return False
        cheapComparedToFutureBuy = buyPrice < (avgFutureBuy * (1.0 - self.buyMargin))
        profitableFutureExport = maxFutureSell > (
            buyPrice * (1.0 + self.storageArbitrageMargin)
        )
        return cheapComparedToFutureBuy or profitableFutureExport

    def shouldSellFromStorage(
        self, sellPrice, avgFutureSell, avgFutureBuy, projectedFutureNetEn
    ):
        if self.supercapEn <= 0.0:
            return False
        favourableSellPrice = sellPrice > (avgFutureSell * (1.0 + self.sellMargin))
        betterThanFutureBuyValue = sellPrice > avgFutureBuy
        futureLooksComfortable = projectedFutureNetEn >= 0.0
        return favourableSellPrice and (
            betterThanFutureBuyValue or futureLooksComfortable
        )

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

    def getSupercapPhysicalEnergy(self):
        return self.minPhysicalSupercapEn + self.supercapEn

    def getSupercapVoltage(self):
        physicalEnergy = self.getSupercapPhysicalEnergy()
        voltage = np.sqrt((2.0 * physicalEnergy) / self.supercapCF)
        return float(np.clip(voltage, self.supercapMinV, self.supercapMaxV))

    def getObs(self):
        dataTick = min(self.curTick, self.ticksPerDay - 1)
        pvGen = self.dailyData["pvGen"][dataTick]
        baseDemand = self.dailyData["baseDemand"][dataTick]
        buyPrice = self.dailyData["buyPrice"][dataTick]
        sellPrice = self.dailyData["sellPrice"][dataTick]
        timeAngle = 2 * np.pi * dataTick / self.ticksPerDay
        timeSin = np.sin(timeAngle)
        timeCos = np.cos(timeAngle)
        totalDefEn = sum(dd["remainingEn"] for dd in self.dailyData["defDemandState"])
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
            projectedFutureNetEn,
        ) = self.getPriceProjection()
        avgFutureBuyNorm = avgFutureBuy / self.maxPrice
        avgFutureSellNorm = avgFutureSell / self.maxPrice
        capFreeNorm = (self.maxSupercapEn - self.supercapEn) / self.maxSupercapEn
        maxProjectedNetEn = self.pvPowerMax * self.tickDur * self.priceLookaheadTicks
        if maxProjectedNetEn <= 0:
            projectedFutureNetNorm = 0.0
        else:
            projectedFutureNetNorm = projectedFutureNetEn / maxProjectedNetEn
        obs = np.array(
            [
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
                projectedFutureNetNorm,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def getInfo(self):
        dataTick = min(self.curTick, self.ticksPerDay - 1)
        (
            avgFutureBuy,
            avgFutureSell,
            minFutureBuy,
            maxFutureSell,
            projectedFutureNetEn,
        ) = self.getPriceProjection()
        info = {
            "curTick": self.curTick,
            "supercapEn": self.supercapEn,
            "supercapSoc": (self.supercapEn / self.maxSupercapEn),
            "supercapFreeEn": (self.maxSupercapEn - self.supercapEn),
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
                dd["remainingEn"] for dd in self.dailyData["defDemandState"]
            ),
        }
        return info

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
            projectedFutureNetEn,
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
        importedEn = 0.0
        exportedEn = 0.0
        arbitrageImportEn = 0.0
        arbitrageExportEn = 0.0
        curtailedPvEn = 0.0
        unmetBaseDemand = 0.0
        unmetDefDemand = 0.0

        pvToDemand = min(pvEn, demandEn)
        remainingDemand = demandEn - pvToDemand
        remainingPv = pvEn - pvToDemand
        scToDemand = 0.0
        emptyScDischargePenalty = 0.0
        maxDischargeSpamPenalty = 0.0
        if scAction < -0.8 and self.supercapEn < 0.1 * self.maxSupercapEn:
            maxDischargeSpamPenalty = -self.maxDischargeSpamPenaltyWeight
        if remainingDemand > 0.0 and scAction < 0.0:
            requestedScDischarge = abs(scAction) * self.maxScDischarge * self.tickDur
            if self.supercapEn <= 0.0:
                emptyScDischargePenalty = -self.emptyScDischargePenaltyWeight * abs(
                    scAction
                )
            scToDemand = min(requestedScDischarge, self.supercapEn, remainingDemand)
            self.supercapEn -= scToDemand
            remainingDemand -= scToDemand
        gridToDemand = 0.0
        if remainingDemand > 0.0:
            maxGridDemandEn = self.gridBuyPowerMax * self.tickDur
            gridToDemand = min(remainingDemand, maxGridDemandEn)
            importedEn += gridToDemand
            remainingDemand -= gridToDemand
        if remainingDemand > 0.0:
            unmetDemand = remainingDemand
            unmetDefDemand = min(unmetDemand, defEnServedThisTick)
            unmetBaseDemand = max(0.0, unmetDemand - unmetDefDemand)
            self.totalUnmetBaseDemand += unmetBaseDemand
            self.totalUnmetDefDemand += unmetDefDemand

        pvChargeEn = 0.0
        scPenalty = 0.0
        if remainingPv > 0.0 and scAction > 0.0:
            requestedScCharge = scAction * self.maxScCharge * self.tickDur
            availableSpace = self.maxSupercapEn - self.supercapEn
            pvChargeEn = min(requestedScCharge, remainingPv, availableSpace)
            self.supercapEn += pvChargeEn
            remainingPv -= pvChargeEn
            if pvChargeEn < min(requestedScCharge, remainingPv + pvChargeEn):
                scPenalty += self.penaltyCapOvUn

        boughtForStorage = False
        if self.shouldBuyForStorage(buyPrice, avgFutureBuy, maxFutureSell) and (
            scAction > -0.2 or self.supercapEn < 0.2 * self.maxSupercapEn
        ):
            capSpace = self.maxSupercapEn - self.supercapEn
            gridImportCapacityLeft = max(
                0.0, (self.gridBuyPowerMax * self.tickDur) - importedEn
            )
            maxBuyStoreEn = self.gridBuyPowerMax * self.tickDur
            actualBuyStoreEn = min(maxBuyStoreEn, capSpace, gridImportCapacityLeft)
            self.supercapEn += actualBuyStoreEn
            arbitrageImportEn += actualBuyStoreEn
            importedEn += actualBuyStoreEn
            boughtForStorage = actualBuyStoreEn > 0.0

        exportedSurplusEn = 0.0
        if remainingPv > 0.0:
            if gridAction < 0.0:
                gridExportCapacityLeft = max(
                    0.0, (self.gridSellPowerMax * self.tickDur) - exportedEn
                )
                requestedSurplusExportEn = abs(gridAction) * remainingPv
                exportedSurplusEn = min(
                    requestedSurplusExportEn, gridExportCapacityLeft
                )
                exportedEn += exportedSurplusEn
            curtailedPvEn = max(0.0, remainingPv - exportedSurplusEn)

        if (not boughtForStorage) and self.shouldSellFromStorage(
            sellPrice, avgFutureSell, avgFutureBuy, projectedFutureNetEn
        ):
            sellableEnergy = max(0.0, self.supercapEn)
            gridExportCapacityLeft = max(
                0.0, (self.gridSellPowerMax * self.tickDur) - exportedEn
            )
            maxSellEn = self.gridSellPowerMax * self.tickDur
            actualSellEn = min(maxSellEn, sellableEnergy, gridExportCapacityLeft)
            self.supercapEn -= actualSellEn
            arbitrageExportEn += actualSellEn
            exportedEn += actualSellEn
        cost = importedEn * buyPrice
        profit = exportedEn * sellPrice
        netProfit = profit - cost
        unnecessaryImportPenalty = 0.0
        if arbitrageImportEn <= 0.0 and importedEn > gridToDemand:
            unnecessaryImportPenalty = -self.unnecessaryImportPenaltyWeight * (
                importedEn - gridToDemand
            )
        pvChargeReward = 0.0
        if pvChargeEn > 0.0:
            pvChargeReward = self.pvChargeRewardWeight * pvChargeEn
        curtailmentWithSpacePenalty = 0.0
        if curtailedPvEn > 0.0 and self.supercapEn < self.maxSupercapEn:
            curtailmentWithSpacePenalty = (
                -self.curtailmentWithSpacePenaltyWeight * curtailedPvEn
            )
        self.totalCost += cost - profit
        unmetBasePenalty = (
            self.totalUnmetBaseDemand - self.prevUnmetBaseDemand
        ) * self.penaltyBaseDemand
        unmetDefPenalty = (
            self.totalUnmetDefDemand - self.prevUnmetDefDemand
        ) * self.penaltyDefDemand
        hardConstraintPenalty = 0.0
        if unmetBaseDemand > 0.0 or unmetDefDemand > 0.0:
            hardConstraintPenalty = self.demandHardConstraintPenalty
        curtailmentPenalty = -self.pvCurtailmentPenalty * curtailedPvEn
        balanceError = unmetBaseDemand + unmetDefDemand + curtailedPvEn
        busBalancePenalty = -self.balancePenalty * balanceError
        smoothnessPenalty = -self.actionSmoothPenalty * actionChange
        reward = (
            netProfit
            + unmetBasePenalty
            + unmetDefPenalty
            + hardConstraintPenalty
            + scPenalty
            + defDeadlinePenalty
            + curtailmentPenalty
            + busBalancePenalty
            + smoothnessPenalty
            + emptyScDischargePenalty
            + maxDischargeSpamPenalty
            + unnecessaryImportPenalty
            + pvChargeReward
            + curtailmentWithSpacePenalty
        )
        if arbitrageImportEn > 0.0:
            reward += self.arbitrageEnergyRewardWeight * arbitrageImportEn
        if arbitrageExportEn > 0.0:
            reward += self.arbitrageEnergyRewardWeight * arbitrageExportEn
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
                if not dd.get("served", False) and not dd.get("missed", False)
            )
            if remainingDefEn > 0.0:
                reward += remainingDefEn * self.penaltyDefDemand
                for dd in self.dailyData["defDemandState"]:
                    if dd["remainingEn"] > 0.0:
                        dd["missed"] = True
                        dd["remainingEn"] = 0.0
            endSoc = self.supercapEn / self.maxSupercapEn
            reward -= abs(endSoc - self.targetEndSoc) * self.endSocPenaltyWeight
        obs = self.getObs()
        info = self.getInfo()
        info["importedEn"] = importedEn
        info["exportedEn"] = exportedEn
        info["arbitrageImportEn"] = arbitrageImportEn
        info["arbitrageExportEn"] = arbitrageExportEn
        info["costThisTick"] = cost
        info["profitThisTick"] = profit
        info["netProfitThisTick"] = netProfit
        info["defEnServedThisTick"] = defEnServedThisTick
        info["defDeadlinePenalty"] = defDeadlinePenalty
        info["unmetBaseDemand"] = unmetBaseDemand
        info["unmetDefDemand"] = unmetDefDemand
        info["curtailedPvEn"] = curtailedPvEn
        info["balanceError"] = balanceError
        info["remainingImbalance"] = balanceError
        info["actionChange"] = actionChange
        info["pvToDemand"] = pvToDemand
        info["scToDemand"] = scToDemand
        info["gridToDemand"] = gridToDemand
        info["pvChargeEn"] = pvChargeEn
        info["exportedSurplusEn"] = exportedSurplusEn
        info["emptyScDischargePenalty"] = emptyScDischargePenalty
        info["maxDischargeSpamPenalty"] = maxDischargeSpamPenalty
        info["unnecessaryImportPenalty"] = unnecessaryImportPenalty
        info["pvChargeReward"] = pvChargeReward
        info["curtailmentWithSpacePenalty"] = curtailmentWithSpacePenalty
        return (obs, reward, terminated, False, info)
