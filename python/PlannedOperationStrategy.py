from .Strategy import *
from .cusfam import *
from .Simon import *


class PlannedOperationStrategy(Strategy):

    beforeTime: float
    afterTime: float
    duration: float
    targetPower: float
    powerRatio: float
    changeTime: float
    endTime: float
    delTime: float
    time: float
    initialASI: float
    finalASI: float
    xeamp: float
    diffASI: float

    def __init__(self, s: Simon) -> None:
        super().__init__(s)
        self.xeamp = 1.0
        self.finalASI = 1.0
        self.reset()

    def reset(self):
        self.time = 0.0


    def setPowerSchedule(self, powerRatio: float, targetPower: float, delTime: float, duration: float, beforeTime: float = 2.0*3600, afterTime: float = 32.0*3600):
        self.delTime = delTime
        self.targetPower = targetPower
        self.duration = duration
        self.beforeTime = beforeTime
        self.afterTime = afterTime
        self.powerRatio = powerRatio / 3600
        self.changeTime = (1.0 - targetPower) / self.powerRatio
        self.endTime = self.beforeTime + self.changeTime * \
            2 + self.duration + self.afterTime

    def setXenonFactor(self, xeamp: float):
        self.xeamp = xeamp

    def setTargetASI(self, initialASI: float, lowPowerASI: float | None):
        self.time = 0.0

        self.initialASI = initialASI
        if lowPowerASI is None:
            self.finalASI = self.initialASI
        else:
            self.finalASI = lowPowerASI

        self.slopeASI = (self.finalASI - self.initialASI) / \
            (self.targetPower - 1.0)

    def next(self) -> bool:
        return self.time < self.endTime

    def runStep(self, stdopt: SteadyOption, result: SimonResult) -> float:
        self.time += self.delTime

        etime = self.beforeTime
        if self.time <= etime + 1.E-6:
            self.s.calculateStatic(stdopt)
            self.s.getResult(result)
            stdopt.ppm = result.ppm
            return self.time

        stdopt.xenon = XE_TR

        etime += self.changeTime
        if self.time <= etime + 1.E-6:
            stdopt.plevel -= self.powerRatio * self.delTime
            self.s.depleteByTime(XE_TR, SM_TR, self.delTime)
            self.s.calculateStatic(stdopt)
            self.s.getResult(result)
            targetASI = self.slopeASI * (stdopt.plevel - 1.0) + self.initialASI
            self.s.searchRodPositionDown(stdopt, targetASI, result)
            stdopt.ppm = result.ppm
            return self.time

        etime += self.duration
        if self.time <= etime + 1.E-6:
            self.s.depleteByTime(XE_TR, SM_TR, self.delTime)
            self.s.calculateStatic(stdopt)
            self.s.getResult(result)
            targetASI = self.slopeASI * (stdopt.plevel - 1.0) + self.initialASI
            self.s.searchRodPositionDown(stdopt, targetASI, result)
            stdopt.ppm = result.ppm
            return self.time

        etime += self.changeTime
        if self.time <= etime + 1.E-6:
            stdopt.plevel += self.powerRatio * self.delTime
            self.s.depleteByTime(XE_TR, SM_TR, self.delTime)
            self.s.calculateStatic(stdopt)
            self.s.getResult(result)
            targetASI = self.slopeASI * (stdopt.plevel - 1.0) + self.initialASI
            self.s.searchRodPositionUp(stdopt, targetASI, result)
            stdopt.ppm = result.ppm
            return self.time

        etime += self.afterTime
        if self.time <= etime + 1.E-6:
            self.s.depleteByTime(XE_TR, SM_TR, self.delTime)
            self.s.calculateStatic(stdopt)
            self.s.getResult(result)
            targetASI = self.slopeASI * (stdopt.plevel - 1.0) + self.initialASI
            self.s.searchRodPositionUp(stdopt, targetASI, result)
            stdopt.ppm = result.ppm
            return self.time
        else:
            self.s.depleteByTime(XE_TR, SM_TR, self.delTime)
            self.s.calculateStatic(stdopt)
            self.s.getResult(result)
            return self.time
