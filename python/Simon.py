import re
import glob
from typing import overload
import copy
from cusfam import *
from PDIL import *
from ControlRod import *

class Simon:

    ASI_SEARCH_FAILED = -1.0
    ASI_SEARCH_ROD_INSERTION = -1
    ASI_SEARCH_ROD_WITHDRAWL = 1

    ERROR_NO = 0
    ERROR_REACH_PDIL = 10001
    ERROR_REACH_BOTTOM = 10002
    ERROR_REACH_TOP = 10003
    ERROR_REACH_BOTTOM = 10004
    ERROR_LIMIT_MOVE = 10005

    AO_EPS = 0.001

    nzf: int
    rods: dict[str, ControlRod]
    leadRod: ControlRod
    regRods: list[ControlRod]
    catRods: list[list[ControlRod]]
    elevation: list[float]
    g: SimonGeometry

    def __init__(self, smgFile: str, tsetFile: str, ffFile: str, rstFile: str, nthreads: int = 4):
        self.s = init(smgFile, tsetFile, ffFile, nthreads)
        self.g = SimonGeometry()
        getGeometry(self.s, self.g)
        self.rstFile = rstFile
        rstFiles = [f for f in glob.glob(rstFile+".FCE.*.SMR") if f != None]
        burnups: list[float] = []

        for f in rstFiles:
            burnups.append(float(re.search(r'FCE\.([0-9]+)\.SMR', f).group(1)))

        burnups.sort()

        self.setBurnupPoints(burnups)
        self.rods = dict()
        self.nzf = self.g.kec-self.g.kbc
        self.elevation = [0,] * self.nzf

        self.elevation[0] = self.g.hz[self.g.kbc]
        for k in range(1, self.nzf) :
            self.elevation[k] = self.elevation[k-1] + self.g.hz[self.g.kbc+k]

    def __del__(self):
        print(self.__class__.__name__+' is destructed')

    def setBurnupPoints(self, burnups: list[float]):
        setBurnupPoints(self.s, burnups)

    def setBurnup(self, burnup: float, stdOption: SteadyOption):
        setBurnup(self.s, self.rstFile, burnup, stdOption)

    def setControlRods(self, rods: list[ControlRod],regRods: list[ControlRod],  catRods: list[list[ControlRod]], leadRod: ControlRod):
        for r in rods:
            self.rods[r.id] = r
            setRodPosition(self.s, r.id, r.pos)

        self.leadRod = leadRod
        self.regRods = regRods
        self.catRods = catRods

    def findElevationLocation(self, pos: float) -> int :

        for k in range(len(self.elevation)):
            if pos <= self.elevation[k] : 
                return k

        return -1

    def calculateStatic(self, stdOption):
        calcStatic(self.s, stdOption)

    def calculatePinPower(self):
        calcPinPower(self.s)

    def deplete(self, xe_option, sm_option, del_burn):
        deplete(self.s, xe_option, sm_option, del_burn)

    def depleteByTime(self, xe_option, sm_option, tsec, xeamp=1.0):
        depleteByTime(self.s, xe_option, sm_option, tsec, xeamp)

    def depleteXeSm(self, xe_option, sm_option, tsec):
        depleteXeSm(self.s, xe_option, sm_option, tsec)

    def setRodPosition(self, rodid: str, position: float):
        self.rods[rodid].setPosition(position)
        setRodPosition(self.s, rodid, position)

    def setRodPositionsWithOverlap(self, rod: ControlRod, leadRodPosition: float):
        rod.setOverlappedPosition(leadRodPosition)
        for id in self.rods:
            setRodPosition(self.s, id, self.rods[id].pos)

    def setRodPositions(self, positions: dict[str, float]):
        for rodid in positions:
            self.setRodPosition(rodid, positions[rodid])

    def getResult(self, result: SimonResult):
        getResult(self.s, result)

    def getShutDownMargin(self, stdOption, positions: dict[str, float], stuck_rods):
        crit0 = stdOption.crit
        xe_bak = stdOption.xenon

        stdOption.crit = KEFF
        stdOption.xenon = XE_TR

        self.setRodPositions(positions)
        self.calculateStatic(stdOption)
        result = SimonResult(self.g.nxya, self.g.nz)
        self.getResult(result)
        out_eigv = result.eigv

        self.setRodPositions(['R', 'B', 'A', 'P'], [0, ] * 4)
        for stuck_rod in stuck_rods:
            self.setRodPosition(stuck_rod, self.g.core_height)

        self.calculateStatic(stdOption)
        self.getResult(result)
        in_eigv = result.eigv

        stdOption.crit = crit0
        stdOption.xenon = xe_bak

        sdm = out_eigv-in_eigv

        return sdm

    def searchRodPosition(self, stdOption: SteadyOption, targetASI: float, result: SimonResult):
        if result.asi < targetASI:
            self.searchRodPositionDown(stdOption, targetASI, result)
        else:
            self.searchRodPositionUp(stdOption, targetASI, result)


        return result

    def searchRodPositionUp(self, stdOption: SteadyOption, targetASI: float, result: SimonResult):
        result.error = self.ERROR_REACH_TOP
        # self.calculateStatic(stdOption)
        # self.getResult(result)

        if result.asi < targetASI: return

        crit0 = copy.copy(stdOption.crit)
        stdOption.crit = KEFF

        found = False

        for rods in reversed(self.catRods):
            for rod in rods:
                pos0 = rod.pos
                pdil = rod.getPDIL(stdOption.plevel)

                if pos0 >= self.g.core_height : continue

                # find starting elevation lower than pos
                k = self.findElevationLocation(rod.pos)-1
                # pos = self.elevation[k]
                pos = rod.pos

                # self.calculateStatic(stdOption)
                # self.getResult(result)

                prevPos = pos
                prevASI = result.asi
                
                while pos < self.g.core_height:
                    k = k + 1
                    prevPos = pos
                    prevASI = result.asi

                    if abs(self.elevation[k]-pos) < 1.0 : k = min(k+1, self.nzf-1)

                    pos = self.elevation[k]

                    if pos < pdil:
                        result.error = self.ERROR_REACH_PDIL
                        break

                    self.setRodPositionsWithOverlap(rod, pos)
                    self.calculateStatic(stdOption)
                    self.getResult(result)

                    if result.asi < targetASI + self.AO_EPS:
                        found = True
                        break

                    # if result.asi > prevASI:
                    #     break

                if found:
                    if abs(prevASI - result.asi) > self.AO_EPS : 
                        pos = (prevPos - pos) / (prevASI - result.asi + 1.E-6) * (targetASI - result.asi) + pos
                    pos = max(min(pos, self.g.core_height), pos0)
                    self.setRodPositionsWithOverlap(rod, pos)
                    self.calculateStatic(stdOption)
                    self.getResult(result)
                    result.error = self.ERROR_NO
                    break # for id

            if found : break

        stdOption.crit = crit0
        self.calculateStatic(stdOption)
        self.getResult(result)

        return result

    def searchRodPositionDown(self, stdOption: SteadyOption, targetASI: float, result: SimonResult):
        result.error = self.ERROR_REACH_BOTTOM
        # self.calculateStatic(stdOption)
        # self.getResult(result)

        if result.asi > targetASI: return

        crit0 = copy.copy(stdOption.crit)
        stdOption.crit = KEFF

        found = False

        for rods in self.catRods:
            for rod in rods:
                pdil = rod.getPDIL(stdOption.plevel)

                # find starting elevation lower than pos
                pos0 = rod.pos

                k = self.findElevationLocation(rod.pos)
                pos = rod.pos

                # self.calculateStatic(stdOption)
                # self.getResult(result)

                prevPos = pos
                prevASI = result.asi
                
                while pos > 0.0:
                    k = k - 1
                    prevPos = pos
                    prevASI = result.asi

                    if(abs(self.elevation[k]-pos) < 1.0) : k = max(k-1, 0)

                    pos = self.elevation[k]
                    if pos < pdil:
                        result.error = self.ERROR_REACH_PDIL
                        break

                    pos = round(pos)
                    self.setRodPositionsWithOverlap(rod, pos)
                    self.calculateStatic(stdOption)
                    self.getResult(result)

                    if result.asi > targetASI + self.AO_EPS:
                        found = True
                        break

                    # if result.asi < prevASI - self.AO_EPS:
                    #     pos = prevPos
                    #     result.asi = prevASI
                    #     self.setRodPositionsWithOverlap(rod, pos)
                    #     break

                    if k == 0:
                        result.error = self.ERROR_LIMIT_MOVE
                        break

                if found:
                    pos = (prevPos - pos) / (prevASI - result.asi + 1.E-6) * (targetASI - result.asi) + pos
                    pos = round(pos)
                    self.setRodPositionsWithOverlap(rod, pos)
                    self.calculateStatic(stdOption)
                    self.getResult(result)
                    result.error = self.ERROR_NO
                    break # for id

                if result.error == self.ERROR_LIMIT_MOVE : break

            if result.error != self.ERROR_NO :
                print(f'ERROR CODE : {result.error}');
                
            if found: break
            # if result.error == self.ERROR_LIMIT_MOVE : break


        if stdOption.crit != crit0:
            stdOption.crit = crit0
            self.calculateStatic(stdOption)
            self.getResult(result)

        return result
