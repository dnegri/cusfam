from .Simon import *
from .ControlRod import *
from .cusfam import *


class SimonManager:
    __instance = None

    @classmethod
    def __getInstance(cls):
        return cls.__instance

    @classmethod
    def instance(cls, *args, **kargs):
        cls.__instance = cls(*args, **kargs)
        cls.instance = cls.__getInstance
        return cls.__instance

    def getSimon(burnup: float, stdopt: SteadyOption) -> Simon:
        s = Simon("/Users/jiyoon/codes/simon/main/run/S4C3/restart_files/S403NDR.SMG",
                  "/Users/jiyoon/codes/simon/main/run/S4C3/plant_files/PLUS7_V127.XS",
                  "/Users/jiyoon/codes/simon/main/run/S4C3/plant_files/PLUS7_V127.FF",
                  "/Users/jiyoon/codes/simon/main/run/S4C3/restart_files/S403NDR", 2)

        s.setBurnup(burnup, stdopt)

        rodP = ControlRod('P', s.g.core_height)
        rodA = ControlRod('A', s.g.core_height)
        rodB = ControlRod('B', s.g.core_height)
        rodR1 = ControlRod('R1', s.g.core_height)
        rodR2 = ControlRod('R2', s.g.core_height, rodR1,
                           (0.4 * s.g.core_height, 0.4 * s.g.core_height))
        rodR3 = ControlRod('R3', s.g.core_height, rodR2,
                           (0.4 * s.g.core_height, 0.4 * s.g.core_height))
        rodR4 = ControlRod('R4', s.g.core_height, rodR3,
                           (0.4 * s.g.core_height, 0.4 * s.g.core_height))
        rodR5 = ControlRod('R5', s.g.core_height, rodR4,
                           (0.4 * s.g.core_height, 0.6 * s.g.core_height))

        rodP.setPDIL([(1.00, 0.72*s.g.core_height),
                      (0.75, 0.40*s.g.core_height),
                      (0.50, 0.00)])

        rodR5.setPDIL([(1.00, 0.72*s.g.core_height),
                       (0.75, 0.40*s.g.core_height),
                       (0.50, 0.00)])

        rodR4.setPDIL([(0.50, 0.60*s.g.core_height),
                       (0.25, 0.00*s.g.core_height)])

        rodR3.setPDIL([(0.25, 0.60*s.g.core_height),
                       (0.175, 0.40*s.g.core_height),
                       (0.00, 0.40*s.g.core_height)])

        s.setControlRods([rodP, rodR5, rodR4, rodR3, rodR2, rodR1,  rodA, rodB], [
            rodP, rodR5, rodR4, rodR3], [[rodP], [rodR5, rodR4, rodR3]], rodR5)

        return s
