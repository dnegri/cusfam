from typing import Any

KEFF: Any
XE_TR: Any
XE_EQ: Any
SM_TR: Any
CBC: Any

class SimonGeometry:
    nz: Any
    nxya: Any
    kbc: Any
    kec: Any
    core_height: Any
    hz: Any
    pass


class SteadyOption:
    crit: int
    shpmtch: Any
    feedtf: Any
    feedtm: Any
    xenon: Any
    samarium: Any
    tin: Any
    eigvt: Any
    maxiter: Any
    epsiter: Any
    ppm: Any
    b10a: Any
    plevel: Any
    pass


class SimonResult:
    error: float
    eigv: float
    ppm: float
    fq: float
    fxy: float
    fr: float
    fz: float
    asi: float
    tf: float
    tm: float
    rod_pos: dict[str, float]
    pow2d: list[float]
    pow1d: list[float]

    def __init__(self, nxya, nz) -> None: ...


def init(smgFile, tsetFile, ffFile, nthreads) -> None: ...
def getGeometry(simonInstance,  sg) -> None: ...
def setBurnupPoints(simonInstance, burnups) -> None: ...
def setBurnup(simonInstance, dir_burnup, burnup,  option) -> None: ...
def saveSnapshot(id, simonInstance) -> None: ...
def loadSnapshot(id, simonInstance) -> None: ...
def calcStatic(simonInstance,  option) -> None: ...
def searchASI(simonInstance,  option, targetASI) -> None: ...
def resetASI(simonInstance) -> None: ...
def setPowerShape(simonInstance, hzshp, powshp) -> None: ...
def calcPinPower(simonInstance) -> None: ...
def getResult(simonInstance,  result) -> None: ...
def setRodPosition(simonInstance, rodid,  position) -> None: ...
def deplete(simonInstance, xe_option, sm_option, del_burnup) -> None: ...
def depleteByTime(simonInstance, xe_option, sm_option,
                  tsec, xeamp=1.0) -> None: ...


def depleteXeSm(simonInstance, xe_option, sm_option, tsec) -> None: ...
