import gymnasium as gym #RL Env
from gymnasium import spaces #Observation or action ranges
import torch as py #Training (fixed typo: pytorch -> torch)
import numpy as np
import random #seeded random gen


class SmartGridEnv(gym.Env):

    vBus = 15
    supercapCF = 0.46

    maxSupercapEn = (
        0.5 * supercapCF * (vBus ** 2)
    )

    loadDemandMax = 4 #Power in Watts
    pvPowerMax = 8 #Watts

    tickDur = 5 #seconds
    ticksPerDay = 60

    penaltyDemand = -100 #Heavy Penalty when load not sufficiently supplied
    penaltyCapOvUn = -10 #Lighter Penalty as its an inefficiency

    defDemandATicks = 30 #Deferable demand window ticks

    maxPrice = 1.0
    maxDefEn = 1.0 #Joules
    maxDueSoon = 1.0 #Joules

    maxScCharge = 2
    maxScDischarge = 2 #Placeholders


    def __init__(self):

        super().__init__()

        obsLow = np.array([
            -1.0, #Time Cosine
            -1.0, #Time Sine
             0.0, #Normalised PV Power
             0.0, #Super Capacitor State of charge
             0.0, #Normalised Base Demand
             0.0, #Normalised Buy Price
             0.0, #Normalised Sell Price
             0.0, #Normalised deferable demand
             0.0, #Normalised urgent demand
            -1.0, #Normalised net Power Balance
        ], dtype=np.float32)

        obsHigh = np.array(
            10 * [1.0], dtype=np.float32
        )

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

            buyPriceVal = self.seededRandom.uniform(0.2,1.0)

            sellPriceVal = self.seededRandom.uniform(0.05,0.5)

            self.dailyData["pvGen"].append(pvVal)
            self.dailyData["baseDemand"].append(baseDemandVal)
            self.dailyData["buyPrice"].append(buyPriceVal)
            self.dailyData["sellPrice"].append(sellPriceVal)

            # creates random flexible demand tasks each tick during reset &
            # assigns start/end window + energy requirement &
            # stores in defDemandState for later scheduling by agent
            # Soham's Shit
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


    def getObs(self):

        dataTick = min(self.curTick, self.ticksPerDay - 1)

        pvGen = self.dailyData["pvGen"][dataTick]

        baseDemand = self.dailyData["baseDemand"][dataTick]

        buyPrice = self.dailyData["buyPrice"][dataTick]

        sellPrice = self.dailyData["sellPrice"][dataTick]

        timeAngle = 2*np.pi*dataTick/self.ticksPerDay

        timeSin = np.sin(timeAngle)

        timeCos = np.cos(timeAngle)

        totalDefEn = sum(dd["remainingEn"] 
                         for dd in self.dailyData["defDemandState"])

        dueSoonEn = 0.0

        for dd in self.dailyData["defDemandState"]:

            if (dd["remainingEn"] > 0):

                if (dataTick <= dd["endTick"] <= (dataTick+self.defDemandATicks)):

                    dueSoonEn += dd["remainingEn"]

        pvNorm = (pvGen/self.pvPowerMax)

        scSoc = self.supercapEn/self.maxSupercapEn

        baseDemandNorm = baseDemand/self.loadDemandMax

        buyNorm = buyPrice/self.maxPrice

        sellNorm = sellPrice/self.maxPrice

        totalDefNorm = totalDefEn/self.maxDefEn

        dueSoonNorm = dueSoonEn/self.maxDueSoon

        netBalance = pvGen-baseDemand

        maxBalance = max(
            self.pvPowerMax,
            self.loadDemandMax
        )

        netNorm = netBalance/maxBalance

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
            netNorm
        ], dtype=np.float32)

        return np.clip(
            obs,
            self.observation_space.low,
            self.observation_space.high
        )


    def getInfo(self):

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

            "totalCost" :
                self.totalCost,

            "totalUnmetDemand" :
                self.totalUnmetDemand,

            "unmetDemandThisTick" :
                (
                    self.totalUnmetDemand
                    -
                    self.prevUnmetDemand
                )
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

        pvEn = pvGen * self.tickDur

        demandEn = baseDemand * self.tickDur

        # progresses deferred demand consumption over time
        # allows partial servicing when energy surplus exists
        # reduces remaining energy (remainingEn) based on available net surplus 

        netDemand = 0.0  # FIX: was undefined before loop

        for dd in self.dailyData["defDemandState"]:

            if (dd["startTick"] <= self.curTick and dd["remainingEn"] > 0):

                if netDemand < 0:

                    served = min(abs(netDemand) * 0.1, dd["remainingEn"])
                    dd["remainingEn"] -= served

        defEnActive = 0.0

        # adds active deferred demand into total demand load
        # increases system pressure when tasks become active
        # agent must account for flexible + fixed demand

        for dd in self.dailyData["defDemandState"]:

            if dd["startTick"] <= self.curTick:

                defEnActive += dd["remainingEn"]

        demandEn = demandEn + defEnActive  # FIXED indentation

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

        importedEn = 0.0
        exportedEn = 0.0

        if netDemand > 0:

            if gridAction >= 0:

                importedEn = gridAction * netDemand

            unmetDemand = max(0.0, netDemand - importedEn)

            self.totalUnmetDemand += unmetDemand

        else:

            surplus = abs(netDemand)

            if gridAction < 0:

                exportedEn = abs(gridAction) * surplus

        cost = importedEn * buyPrice

        profit = exportedEn * sellPrice

        self.totalCost += (cost - profit)

        unmetPenalty = (
            (
                self.totalUnmetDemand
                - self.prevUnmetDemand
            )
            * self.penaltyDemand
        )

        reward = (
            -cost
            + profit
            + unmetPenalty
            + scPenalty
        )

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

        return (obs, reward, terminated, False, info)
  