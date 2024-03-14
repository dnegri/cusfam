import time
import csv
from itertools import islice
from random import random
from time import perf_counter
from cusfam import *
from Simon import *
from PDIL import *
from ControlRod import *


def generateVirtualSignal(s: Simon, endTime: float) -> list[tuple[float, float]]:

    signals: list[tuple[float, float]] = []

    dt: float = 60 * 15
    stepForHour: int = int(3600 / dt)

    stdOption: SteadyOption = SteadyOption()
    result: SimonResult = SimonResult(s.g.nxya, s.g.nz)

    stdOption.maxiter = 100
    stdOption.crit = CBC
    stdOption.feedtf = True
    stdOption.feedtm = True
    stdOption.xenon = XE_EQ
    stdOption.samarium = SM_TR
    stdOption.tin = 296.0
    stdOption.plevel = 1.00
    stdOption.epsiter = 3.0E-5

    time: float = 0.0

    s.calculateStatic(stdOption)
    s.getResult(result)
    signals.append((0.0, result.asi))

    # pertub control rod
    stdOption.xenon = XE_TR
    s.setRodPositionsWithOverlap('R5', 381.0*0.72)
    s.calculateStatic(stdOption)

    for i in range(stepForHour):
        s.depleteXeSm(XE_TR, SM_TR, dt)
        s.calculateStatic(stdOption)
        time += dt
        signals.append((time, result.asi))
        print(
            f'TIME : {time/3600:.2f} (HR), ASI : {result.asi:0.3f}, PPM : {result.ppm:.2f} KEFF : {result.eigv:.5f} ')

    s.setRodPositionsWithOverlap('R5', 381.0)
    s.calculateStatic(stdOption)

    while time < endTime:
        s.depleteXeSm(XE_TR, SM_TR, dt)
        s.calculateStatic(stdOption)
        s.getResult(result)
        time += dt
        signals.append((time, result.asi))
        print(
            f'TIME : {time/3600:.2f} (HR), ASI : {result.asi:0.3f}, PPM : {result.ppm:.2f} KEFF : {result.eigv:.5f} ')

    with open('signals.csv', 'w') as f:
        f = csv.writer(f)
        f.writerows(signals)

    return signals


def sample_static(s, std_option):
    s.calculateStatic(std_option)


def sample_deplete(s, std_option):

    burn_del = 1000.0
    burn_end = 1000.0
    burn_step = int(burn_end / burn_del)

    result = SimonResult(s.g.nxya, s.g.nz)

    start = time.time()
    for i in range(burn_step):
        s.calculateStatic(std_option)
        s.getResult(result)
        s.calculatePinPower()
        print("STEP : ", i, " -> PPM : ", result.ppm)
        s.deplete(XE_EQ, SM_TR, burn_del)

    end = time.time()
    print(end - start)
    print("END DEP")


def sample_dynxe(s, std_option):
    std_option.plevel = 0.9
    std_option.xenon = XE_TR
    sample_static(s, std_option)

    std_option.crit = KEFF
    for i in range(100):
        s.depleteXeSm(XE_TR, SM_TR, 3600)
        s.calculateStatic(std_option)
        s.getResult(result)
        print(
            f'PPM : {result.ppm:.2f} KEFF : {result.eigv:.5f} and ASI : {result.asi:0.3f}')

if __name__ == "__main__":

    # s = Simon("D:/codes/cusfam/main/run/ucn6/c12/UCN612ASBDEP.FCE.SMG",
    #          "D:/codes/cusfam/main/run/ucn6/db/UCN6_OPR1000_rev1.XS",
    #          "D:/codes/cusfam/main/run/ucn6/db/UCN6_OPR1000_rev1.FF",
    #          "D:/codes/cusfam/main/run/ucn6/c12/UCN612ASBDEP.FCE")

    # s = Simon("D:/codes/cusfam/main/run/ygn3/Y301ASBDEP.SMG",
    #          "D:/codes/cusfam/main/run/ygn3/KMYGN34C01_PLUS7_XSE.XS",
    #          "D:/codes/cusfam/main/run/ygn3/KMYGN34C01_PLUS7_XSE.FF",
    #          "D:/codes/cusfam/main/run/ygn3/Y301ASBDEP")

    # s = Simon("D:/work/corefollow/ygn3/c01/depl/rst/Y301ASBDEP.FCE.SMG",
    #           "../run/KMYGN34C01_PLUS7_XSE.XS",
    #           "D:/work/corefollow/ygn3/c01/depl/rst/Y301ASBDEP.FCE")

    s = Simon("../run/ygn3/Y312ASBDEP.SMG",
              "../run/ygn3/KMYGN34C01_PLUS7_XSE.XS",
              "../run/ygn3/KMYGN34C01_PLUS7_XSE.FF",
              "../run/ygn3/Y312ASBDEP", 1)

    std_option = SteadyOption()
    s.setBurnup(17000, std_option)

    exit(0)

    rodP = ControlRod('P', s.g.core_height)
    rodA = ControlRod('A', s.g.core_height)
    rodB = ControlRod('B', s.g.core_height)
    rodR1 = ControlRod('R1', s.g.core_height)
    rodR2 = ControlRod('R2', s.g.core_height, rodR1, (0.4 * s.g.core_height, 0.4 * s.g.core_height))
    rodR3 = ControlRod('R3', s.g.core_height, rodR2, (0.4 * s.g.core_height, 0.4 * s.g.core_height))
    rodR4 = ControlRod('R4', s.g.core_height, rodR3, (0.4 * s.g.core_height, 0.4 * s.g.core_height))
    rodR5 = ControlRod('R5', s.g.core_height, rodR4, (0.4 * s.g.core_height, 0.6 * s.g.core_height))

    rodR5.setPDIL([(1.00, 0.72*s.g.core_height),
                   (0.75, 0.40*s.g.core_height),
                   (0.50, 0.00)])

    rodR4.setPDIL([(0.50, 0.60*s.g.core_height),
                   (0.25, 0.00*s.g.core_height)])

    rodR3.setPDIL([(0.25, 0.60*s.g.core_height),
                   (0.175, 0.40*s.g.core_height),
                   (0.00, 0.40*s.g.core_height)])


    s.setControlRods([rodP, rodR5, rodR4, rodR3, rodR2, rodR1,  rodA, rodB], [rodP, rodR5, rodR4, rodR3], rodR5)

    std_option.maxiter = 100
    std_option.crit = CBC
    std_option.feedtf = True
    std_option.feedtm = True
    std_option.xenon = XE_EQ
    std_option.samarium = SM_TR
    std_option.tin = 290.0
    std_option.eigvt = 1.00000
    std_option.plevel = 1.00
    std_option.epsiter = 1.0E-5
    result = SimonResult(s.g.nxya, s.g.nz)

    s.calculateStatic(std_option)
    s.getResult(result)
    std_option.ppm = result.ppm
    
    # generateVirtualSignal(s, 150.0*3600.0)

    # find ESI for target power
    std_option.plevel = 0.7
    targetASI = result.asi

    for i in range(10) :
        result = s.searchRodPosition(std_option, targetASI)
        std_option.ppm = result.ppm
        print(f'Target ASI    : [{targetASI:.3f}]')
        print(*[s.rods[rod].pos for rod in s.rods], sep=', ')
        print(f'Resulting ASI : [{result.asi:.3f}]')
        print(f'Resulting CBC : [{result.ppm:.3f}]')
        print(f'ERROR Code    : [{result.error:5d}]')

    std_option.plevel = 1.0
    for i in range(10) :
        result = s.searchRodPositionUp(std_option, targetASI)
        std_option.ppm = result.ppm
        print(f'Target ASI    : [{targetASI:.3f}]')
        print(*[s.rods[rod].pos for rod in s.rods], sep=', ')
        print(f'Resulting ASI : [{result.asi:.3f}]')
        print(f'Resulting CBC : [{result.ppm:.3f}]')
        print(f'ERROR Code    : [{result.error:5d}]')

    exit(0)

    result = SimonResult(s.g.nxya, s.g.nz)
    # for i in range(0,len(bp)) :
# s.setRodPosition(['B', 'A', 'P'], [0.0, ] * 4, 0.0)
# s.setRodPosition(['R'], [0.0, ] * 4, 0.0)
# s.setRodPosition(['B'], [0.0, ] * 4, 0.0)
# s.setRodPosition(['A'], [0.0, ] * 4, 0.0)
# s.setRodPosition(['P'], [0.0, ] * 4, 0.0)
# s.setRodPosition1("B42", 381.0);
# s.setBurnup(0.0)
    s.calculateStatic(std_option)
    s.getResult(result)
    std_option.ppm = result.ppm
    print(
        f"ASI : {result.asi:.3f}, PPM : {result.ppm:.1f},  PPM : {result.ppm:.2f} KEFF : {result.eigv:.5f}")
    exit(0)

    std_option.crit = KEFF
    std_option.ppm = result.ppm
    std_option.feedtf = False
    std_option.feedtm = False
    std_option.xenon = XE_TR
    s.setRodPosition(['R', 'B', 'A', 'P'], [0.0, ] * 4, 0.0)
    s.setRodPosition1("B22", 381.0)

    s.calculateStatic(std_option)
    s.getResult(result)

    print(
        f"ASI : {result.asi:.3f}, PPM : {result.ppm:.1f},  PPM : {result.ppm:.2f} KEFF : {result.eigv:.5f}")

    s.setRodPosition(['R', 'B', 'A', 'P'], [0.0, ] * 4, 0.0)
    s.setRodPosition1("B42", 381.0)

    s.calculateStatic(std_option)
    s.getResult(result)
    print(
        f"ASI : {result.asi:.3f}, PPM : {result.ppm:.1f},  PPM : {result.ppm:.2f} KEFF : {result.eigv:.5f}")

    exit(0)

    std_option = SteadyOption()
    std_option.maxiter = 100
    std_option.crit = KEFF
    std_option.feedtf = False
    std_option.feedtm = False
    std_option.xenon = XE_NO
    std_option.tin = 292.22
    std_option.eigvt = 1.0
    std_option.ppm = 800
    std_option.plevel = 0.0
    std_option.epsiter = 1.0E-7

    # s.setRodPosition(['B', 'A', 'P'], [0.0, ] * 4, 0.0)
    s.setRodPosition(['R'], [0.0, ] * 4, 0.0)
    s.setRodPosition1("B42", 381.0)
    s.calculateStatic(std_option)
    s.getResult(result)

    exit(0)

    # std_option.xenon = XE_TR
    s.calculateStatic(std_option)
    s.getResult(result)

    std_option.ppm = result.ppm

    std_option.crit = KEFF
    std_option.feedtf = False
    std_option.feedtm = False
    std_option.xenon = XE_TR
    s.setRodPosition(['R', 'B', 'A', 'P'], [0.0, ] * 4, 0.0)
    s.calculateStatic(std_option)

    s.setRodPosition1("B22", 381.0)
    s.calculateStatic(std_option)
    s.getResult(result)
    print(
        f"ASI : {result.asi:.3f}, PPM : {result.ppm:.1f},  PPM : {result.ppm:.2f} KEFF : {result.eigv:.5f}")

    exit(0)

    s.setBurnup(0.0)
    sample_deplete(s, std_option)

    std_option.xenon = XE_EQ
    s.setRodPosition(['R', 'B', 'A', 'P'], [0, ] * 4, s.g.core_height)
    s.calculateStatic(std_option)
    s.calculatePinPower()
    s.getResult(result)
    print(
        f"ASI : {result.asi:.3f}, PPM : {result.ppm:.1f},  Fxy : {result.fxy:.3f}")

    std_option.xenon = XE_TR
    std_option.plevel = 0.0
    std_option.tin = 295.6
    s.setRodPosition1('R5', 72.0/100*381.0)
    s.calculateStatic(std_option)
    s.calculatePinPower()
    s.getResult(result)
    print(
        f"ASI : {result.asi:.3f}, PPM : {result.ppm:.1f},  Fxy : {result.fxy:.3f}")
    std_option.ppm = result.ppm
    for i in range(0, 30):
        s.depleteXeSm(XE_TR, SM_TR, 3600)
        s.calculateStatic(std_option)
        s.calculatePinPower()
        s.getResult(result)
        print(
            f"ASI : {result.asi:.3f}, PPM : {result.ppm:.1f}, Fxy : {result.fxy:.3f}")
        std_option.ppm = result.ppm

    exit(-1)

    # trip
    # std_option.crit = KEFF
    # std_option.plevel = 0.0
    std_option.xenon = XE_TR
    s.setRodPosition(['R5'], [0.0], 72.0)
    s.calculateStatic(std_option)
    s.getResult(result)
    print(f"ASI : {result.asi:.3f}, PPM : {result.ppm:.1f}")

    # xenon build-up for 30 hours
    std_option.xenon = XE_TR
    for i in range(100):
        s.depleteXeSm(XE_TR, SM_TR, 1800)
        s.calculateStatic(std_option)
        s.getResult(result)
        print(f"ASI : {result.asi:.3f}, PPM : {result.ppm:.1f}")

    # std_option.crit = CBC
    # ecp
    # s.setRodPosition1('P', 200.0)
    # position = sample_asisearch(s, std_option, -0.6, 0.0)
    # print(f"ASI : {result.asi:.3f}, PPM : {result.ppm:.1f}")

    # power ascension
    # rodids = ['R5', 'R4', 'R3']
    # overlaps = [0.0 * s.g.core_height, 0.6 * s.g.core_height, 1.2 * s.g.core_height]

    # target_asi = -0.1

    # for i in range(100) :
    #    std_option.plevel = i*0.01
    #    pdil = s.getPDIL(std_option.plevel*100)
    #    r5_pdil = pdil[0]

    #    result = s.searchRodPositionO(std_option, target_asi, rodids, overlaps, r5_pdil, preStepPos[1])

    #    r5_pos = result.rod_pos['R5']
    #    r4_pos = result.rod_pos['R4']
    #    print(f"ASI : {result.asi:.3f}, PPM : {result.ppm:.1f}, RODP : [{r5_pos:.1f}, {r4_pos:.1f}]")

    # sample_static(s_full, std_option)
    # s_full.getResult(result)
    # print(f'PPM : {result.ppm:.3f}')

    # std_option.ppm = result.ppm
    # std_option.xenon = XE_TR
    # sample_static(s, std_option)

    # s.getResult(result)
    # print(f'PPM : {result.ppm:.3f}')

    # print(f'Initial ASI [{result.asi:.3f}]')

    # std_option.plevel = 0.9
    # std_option.xenon = XE_TR
    # sample_static(s, std_option)

    # sample_dynxe(s, std_option)

    # sample_shutdown_margin(s, std_option, ['B41', 'B42'])
    # sample_shutdown_margin(s, std_option, ['B41', 'B42'])
    # position = 381.0

    # std_option.epsiter = 1.E-3

    # position = sample_asisearch(s, std_option,0.10, position)
    # position = sample_asisearch(s, std_option,0.15, position)
    # position = sample_asisearch(s, std_option,0.15, position)
    # position = sample_asisearch(s, std_option,0.16, position)

    # rodids = ['R5', 'R4', 'R3']
    # overlaps = [0, 0.4 * s.g.core_height, 0.7 * s.g.core_height]
    # r5_pdil = 0.72 * s.g.core_height
    # s.setRodPosition(rodids, overlaps, 381.0)
    # sample_static(s, std_option)
    # s.getResult(result)
    # print(f'PPM : {result.ppm:.3f}')
