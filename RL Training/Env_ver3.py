import gymnasium as gym # RL Env
from gymnasium import spaces # Observation or action ranges
import torch as py # Training
import numpy as np
import random # Seeded random gen


class SmartGridEnv(gym.Env):

    supercapMinV = 10.0
    # NEW: Minimum usable capacitor voltage.
    # Energy below this voltage is treated as inaccessible.

    supercapMaxV = 16.5
    # NEW: Maximum usable capacitor voltage.

    supercapCF = 0.46

    minPhysicalSupercapEn = (
        0.5 * supercapCF * (supercapMinV ** 2)
    )
    # NEW: Physical energy already present at 10 V.
    # This energy exists in the capacitor but cannot be used by the controller.

    maxPhysicalSupercapEn = (
        0.5 * supercapCF * (supercapMaxV ** 2)
    )
    # NEW: Physical energy present at 16.5 V.

    maxSupercapEn = (
        maxPhysicalSupercapEn - minPhysicalSupercapEn
    )
    # NEW: Usable capacitor energy between 10 V and 16.5 V only.

    loadDemandMax = 4 # Power in Watts
    pvPowerMax = 8 # Watts

    tickDur = 5 # Seconds
    ticksPerDay = 60

    penaltyDemand = -100 # Heavy penalty when load not sufficiently supplied
    penaltyCapOvUn = -10 # Lighter penalty for capacitor overcharge/undercharge

    defDemandATicks = 30 # Deferable demand window ticks

    maxPrice = 1.0
    maxDefEn = 1.0 # Joules
    maxDueSoon = 1.0 # Joules

    maxScCharge = 2
    maxScDischarge = 2 # Placeholders

    priceLookaheadTicks = 10 
    #Number of future ticks the environment checks when estimating whether
    # the current buy/sell price is good or bad.
    # Since tickDur = 5 s, this means 10 ticks = 50 seconds of projection.

    gridBuyPowerMax = 4.0 
    # NEW: Maximum extra grid power allowed for buying cheap energy to store.
    # This is separate from importing energy to satisfy demand.

    gridSellPowerMax = 4.0 
    # NEW: Maximum power allowed when selling stored capacitor energy to the grid.

    minSellSoc = 0.25 
    #  Prevents the capacitor from being emptied completely.
    # Stored energy can only be sold if SOC is above 25%. Need to figure out what is optimal


    buyMargin = 0.05 
    # NEW: Requires current buy price to be at least 5% cheaper than the future average.
    # This prevents the controller from buying due to tiny/random price differences.

    sellMargin = 0.05 
    # NEW: Requires current sell price to be at least 5% better than the future average.
    # This prevents selling unless the price is meaningfully favourable.


    def __init__(self):

        super().__init__()

        obsLow = np.array([
            -1.0, # Time Cosine
            -1.0, # Time Sine
             0.0, # Normalised PV Power
             0.0, # Super Capacitor State of charge
             0.0, # Normalised Base Demand
             0.0, # Normalised Buy Price
             0.0, # Normalised Sell Price
             0.0, # Normalised deferable demand
             0.0, # Normalised urgent demand
            -1.0, # Normalised net Power Balance

             0.0, 
             # NEW: Normalised average future buy price.
             # Allows the agent to compare current buy price against projected buy price.

             0.0, 
             # Normalised average future sell price.
             # Allows the agent to compare current sell price against projected sell price.

             0.0, 
             # Normalised capacitor free space.
             # Tells the agent whether there is room to store cheap imported energy.

            -1.0, 
             # Normalised projected future net energy balance.
             # Positive means future PV is expected to exceed demand.
             # Negative means future demand is expected to exceed PV.
        ], dtype=np.float32)

        obsHigh = np.array(
            14 * [1.0], dtype=np.float32
        )
        # Observation size increased from 10 to 14.
        # This is because the following was added:
        # 1) avg future buy price
        # 2) avg future sell price
        # 3) capacitor free space
        # 4) projected future net energy

        self.obsSpace = spaces.Box(low=obsLow, high=obsHigh, dtype=np.float32)

        self.observation_space = self.obsSpace

        self.actionSpace = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.action_space = self.actionSpace

        self.curTick = 0

        self.supercapEn = 0.0

        self.totalCost = 0.0

        self.totalUnmetDemand = 0.0
        self.prevUnmetDemand = 0.0

        self.seededRandom = None

        self.dailyData = {
            "pvGen" : [],
            "baseDemand" : [],
            "buyPrice" : [],
            "sellPrice" : [],
            "defDemandState" : []
        }


    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.seededRandom = random.Random(seed)

        self.curTick = 0

        self.supercapEn = 0.0

        self.totalCost = 0.0

        self.totalUnmetDemand = 0.0
        self.prevUnmetDemand = 0.0

        self.dailyData = {
            "pvGen" : [],
            "baseDemand" : [],
            "buyPrice" : [],
            "sellPrice" : [],
            "defDemandState" : []
        }

        for tick in range(self.ticksPerDay):

            timeRatio = (tick / self.ticksPerDay)

            pvVal = (self.pvPowerMax) * max(0.0, np.sin(np.pi * timeRatio))

            baseDemandVal = self.seededRandom.uniform(1.0, self.loadDemandMax)

            buyPriceVal = self.seededRandom.uniform(0.2, 1.0)

            sellPriceVal = self.seededRandom.uniform(0.05, 0.5)

            self.dailyData["pvGen"].append(pvVal)
            self.dailyData["baseDemand"].append(baseDemandVal)
            self.dailyData["buyPrice"].append(buyPriceVal)
            self.dailyData["sellPrice"].append(sellPriceVal)

            # Creates random flexible demand tasks each tick during reset.
            if self.seededRandom.random() < 0.15:

                startTick = tick

                duration = self.seededRandom.randint(5, 20)

                endTick = min(self.ticksPerDay - 1, startTick + duration)

                energy = self.seededRandom.uniform(0.5, 3.0)

                self.dailyData["defDemandState"].append({
                    "startTick": startTick,
                    "endTick": endTick,
                    "remainingEn": energy,
                    "originalEn": energy,
                    "served": False
                })

        obs = self.getObs()
        info = self.getInfo()

        return obs, info


    def getPriceProjection(self):

        startTick = self.curTick

        endTick = min(
            self.ticksPerDay,
            self.curTick + self.priceLookaheadTicks
        )

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


    def shouldBuyForStorage(self, buyPrice, avgFutureBuy):
        
        capSpace = self.maxSupercapEn - self.supercapEn

        if capSpace <= 0.0:
            return False

        favourableBuyPrice = buyPrice < (avgFutureBuy * (1.0 - self.buyMargin))

        return favourableBuyPrice


    def shouldSellFromStorage(self, sellPrice, avgFutureSell, projectedFutureNetEn):

        scSoc = self.supercapEn / self.maxSupercapEn

        if scSoc <= self.minSellSoc:
            return False

        favourableSellPrice = sellPrice > (avgFutureSell * (1.0 + self.sellMargin))

        futureLooksComfortable = projectedFutureNetEn >= 0.0

        return favourableSellPrice and futureLooksComfortable


    def getSupercapPhysicalEnergy(self):

        return self.minPhysicalSupercapEn + self.supercapEn
        # NEW: supercapEn = 0 means physical capacitor is already at 10 V.


    def getSupercapVoltage(self):

        physicalEnergy = self.getSupercapPhysicalEnergy()

        voltage = np.sqrt(
            (2.0 * physicalEnergy)
            /
            self.supercapCF
        )

        return float(np.clip(
            voltage,
            self.supercapMinV,
            self.supercapMaxV
        ))
        # NEW: Actual capacitor voltage corresponding to usable stored energy.


    def getObs(self):

        dataTick = min(self.curTick, self.ticksPerDay - 1)

        pvGen = self.dailyData["pvGen"][dataTick]

        baseDemand = self.dailyData["baseDemand"][dataTick]

        buyPrice = self.dailyData["buyPrice"][dataTick]

        sellPrice = self.dailyData["sellPrice"][dataTick]

        timeAngle = 2 * np.pi * dataTick / self.ticksPerDay

        timeSin = np.sin(timeAngle)

        timeCos = np.cos(timeAngle)

        totalDefEn = sum(dd["remainingEn"] 
                         for dd in self.dailyData["defDemandState"])

        dueSoonEn = 0.0

        for dd in self.dailyData["defDemandState"]:

            if (dd["remainingEn"] > 0):

                if (dataTick <= dd["endTick"] <= (dataTick + self.defDemandATicks)):

                    dueSoonEn += dd["remainingEn"]

        pvNorm = (pvGen / self.pvPowerMax)

        scSoc = self.supercapEn / self.maxSupercapEn

        baseDemandNorm = baseDemand / self.loadDemandMax

        buyNorm = buyPrice / self.maxPrice

        sellNorm = sellPrice / self.maxPrice

        totalDefNorm = totalDefEn / self.maxDefEn

        dueSoonNorm = dueSoonEn / self.maxDueSoon

        netBalance = pvGen - baseDemand

        maxBalance = max(
            self.pvPowerMax,
            self.loadDemandMax
        )

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

        capFreeNorm = (
            (self.maxSupercapEn - self.supercapEn)
            /
            self.maxSupercapEn
        )

        maxProjectedNetEn = self.pvPowerMax * self.tickDur * self.priceLookaheadTicks

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

            "curTick" :
                self.curTick,

            "supercapEn" :
                self.supercapEn,

            "supercapSoc" :
                (
                    self.supercapEn
                    /
                    self.maxSupercapEn
                ),

            "supercapFreeEn" :
                (
                    self.maxSupercapEn
                    -
                    self.supercapEn
                ),

            "supercapVoltage" :
                self.getSupercapVoltage(),

            "supercapPhysicalEn" :
                self.getSupercapPhysicalEnergy(),

            "supercapUsableEn" :
                self.supercapEn,

            "totalCost" :
                self.totalCost,

            "totalUnmetDemand" :
                self.totalUnmetDemand,

            "unmetDemandThisTick" :
                (
                    self.totalUnmetDemand
                    -
                    self.prevUnmetDemand
                ),

            "buyPrice" :
                self.dailyData["buyPrice"][dataTick],

            "sellPrice" :
                self.dailyData["sellPrice"][dataTick],

            "avgFutureBuy" :
                avgFutureBuy,

            "avgFutureSell" :
                avgFutureSell,

            "minFutureBuy" :
                minFutureBuy,

            "maxFutureSell" :
                maxFutureSell,

            "projectedFutureNetEn" :
                projectedFutureNetEn
        }

        return info


    def step(self, action):

        if (self.curTick >= self.ticksPerDay):

            obs = self.getObs()

            info = self.getInfo()

            return (obs, 0.0, True, False, info)

        gridAction = np.clip(action[0], -1.0, 1.0)

        scAction = np.clip(action[1], -1.0, 1.0)

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

        demandEn = baseDemand * self.tickDur

        netDemand = 0.0

        for dd in self.dailyData["defDemandState"]:

            if (dd["startTick"] <= self.curTick and dd["remainingEn"] > 0):

                if netDemand < 0:

                    served = min(abs(netDemand) * 0.1, dd["remainingEn"])
                    dd["remainingEn"] -= served

        defEnActive = 0.0

        for dd in self.dailyData["defDemandState"]:

            if dd["startTick"] <= self.curTick:

                defEnActive += dd["remainingEn"]

        demandEn = demandEn + defEnActive

        if scAction >= 0:

            scPowerWanted = scAction * self.maxScCharge

        else:

            scPowerWanted = scAction * self.maxScDischarge

        scEnergyDelta = scPowerWanted * self.tickDur

        scPenalty = 0.0

        if scEnergyDelta > 0:

            availableSpace = self.maxSupercapEn - self.supercapEn

            actualCharge = min(scEnergyDelta, availableSpace)

            if actualCharge < scEnergyDelta:

                scPenalty += self.penaltyCapOvUn

            self.supercapEn += actualCharge

            scFlow = -actualCharge

        else:

            actualDischarge = min(abs(scEnergyDelta), self.supercapEn)

            if actualDischarge < abs(scEnergyDelta):

                scPenalty += self.penaltyCapOvUn

            self.supercapEn -= actualDischarge

            scFlow = actualDischarge

        netDemand = demandEn - pvEn - scFlow

        arbitrageImportEn = 0.0

        arbitrageExportEn = 0.0

        boughtForStorage = False

        if self.shouldBuyForStorage(buyPrice, avgFutureBuy):

            capSpace = self.maxSupercapEn - self.supercapEn

            maxBuyStoreEn = self.gridBuyPowerMax * self.tickDur

            actualBuyStoreEn = min(
                maxBuyStoreEn,
                capSpace
            )

            self.supercapEn += actualBuyStoreEn

            arbitrageImportEn += actualBuyStoreEn

            boughtForStorage = actualBuyStoreEn > 0.0

        if (not boughtForStorage) and self.shouldSellFromStorage(
            sellPrice,
            avgFutureSell,
            projectedFutureNetEn
        ):

            reserveEnergy = self.minSellSoc * self.maxSupercapEn

            sellableEnergy = max(0.0, self.supercapEn - reserveEnergy)

            maxSellEn = self.gridSellPowerMax * self.tickDur

            actualSellEn = min(
                maxSellEn,
                sellableEnergy
            )

            self.supercapEn -= actualSellEn

            arbitrageExportEn += actualSellEn

        importedEn = arbitrageImportEn

        exportedEn = arbitrageExportEn

        if netDemand > 0:

            if gridAction >= 0:

                demandImportEn = gridAction * netDemand

                importedEn += demandImportEn

            else:

                demandImportEn = 0.0

            unmetDemand = max(0.0, netDemand - demandImportEn)

            self.totalUnmetDemand += unmetDemand

        else:

            surplus = abs(netDemand)

            if gridAction < 0:

                exportedEn += abs(gridAction) * surplus

        cost = importedEn * buyPrice

        profit = exportedEn * sellPrice

        self.totalCost += (cost - profit)

        unmetPenalty = (
            (
                self.totalUnmetDemand
                -
                self.prevUnmetDemand
            )
            * self.penaltyDemand
        )

        reward = (
            -cost
            + profit
            + unmetPenalty
            + scPenalty
        )

        if arbitrageImportEn > 0.0:
            reward += 0.5

        if arbitrageExportEn > 0.0:
            reward += 0.5

        defCompleted = 0.0

        for dd in self.dailyData["defDemandState"]:

            if dd["remainingEn"] <= 0 and not dd.get("served", False):

                defCompleted += 1.0

                dd["served"] = True

        reward += defCompleted * 5.0

        self.prevUnmetDemand = self.totalUnmetDemand

        self.curTick += 1

        terminated = (self.curTick >= self.ticksPerDay)

        obs = self.getObs()

        info = self.getInfo()

        info["importedEn"] = importedEn

        info["exportedEn"] = exportedEn

        info["arbitrageImportEn"] = arbitrageImportEn

        info["arbitrageExportEn"] = arbitrageExportEn

        info["costThisTick"] = cost

        info["profitThisTick"] = profit

        return (obs, reward, terminated, False, info)