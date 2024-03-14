from .Strategy import *
from .cusfam import *
from .Simon import *


class XenonOscillationStrategy(Strategy):

    endTime: float
    delTime: float
    time: float
    targetASI: float
    xeamp: float
    diffASI: float

    def __init__(self, s: Simon) -> None:
        super().__init__(s)
        self.xeamp = 1.0
        self.diffASI = 0.01
        self.targetASI = 1.0
        self.reset()

    def reset(self):
        self.time = 0.0

    def setTime(self, endTime: float, delTime: float):
        self.endTime = endTime
        self.delTime = delTime

    def setTargetASI(self, targetASI: float):
        self.targetASI = targetASI

    def setXenonFactor(self, xeamp: float):
        self.xeamp = xeamp

    def setLimitOfASIDifference(self, diffASI: float):
        self.diffASI = diffASI

    def next(self) -> bool:
        return self.time < self.endTime

    def runStep(self, stdopt: SteadyOption, result: SimonResult) -> float:
        self.time += self.delTime
        self.s.depleteByTime(XE_TR, stdopt.samarium, self.delTime, self.xeamp)
        self.s.calculateStatic(stdopt)
        self.s.getResult(result)
        stdopt.ppm = result.ppm

        if result.asi < self.targetASI - self.diffASI:
            self.s.searchRodPositionDown(stdopt, self.targetASI, result)
            stdopt.ppm = result.ppm
        else:
            self.s.searchRodPositionUp(stdopt, self.targetASI, result)
            stdopt.ppm = result.ppm

        return self.time
