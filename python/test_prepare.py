from Simon import *


def runPowerDown(powerRatio, ppmRatio, s, stdopt, targetASI, endTime, delTime, time, result) -> float:
    while time <= endTime:
        time += delTime
        stdopt.plevel -= powerRatio * delTime
        stdopt.ppm += ppmRatio * delTime
        s.depleteByTime(XE_TR, SM_TR, delTime)
        s.calculateStatic(stdopt)
        s.getResult(result)
        s.searchRodPositionDown(stdopt, targetASI, result)
        stdopt.ppm = result.ppm
        strRodPos = ' '.join(["{:.2f}".format(s.rods[rod].pos)
                             for rod in s.rods])
        print(f'> TRANSIENT :  ' + '{:6.2f}'.format(time/3600)+f' {stdopt.plevel:.2f} '+'{: .3f}'.format(
            result.asi)+'{: 6.1f}'.format(result.ppm)+f' {strRodPos} {result.error:5d}')

    return time


def runPowerUp(powerRatio, ppmRatio, s, stdopt, targetASI, endTime, delTime, time, result) -> float:
    while time <= endTime:
        time += delTime
        stdopt.plevel += powerRatio * delTime
        stdopt.ppm -= ppmRatio * delTime
        s.depleteByTime(XE_TR, SM_TR, delTime)
        s.calculateStatic(stdopt)
        s.getResult(result)
        s.searchRodPositionUp(stdopt, targetASI, result)
        stdopt.ppm = result.ppm
        strRodPos = ' '.join(["{:.2f}".format(s.rods[rod].pos)
                             for rod in s.rods])
        print(f'> TRANSIENT :  ' + '{:6.2f}'.format(time/3600)+f' {stdopt.plevel:.2f} '+'{: .3f}'.format(
            result.asi)+'{: 6.1f}'.format(result.ppm)+f' {strRodPos} {result.error:5d}')

    return time


def runUp(s, stdopt, targetASI, endTime, delTime, time, result) -> float:

    rodTime = 0.0
    prevASI = result.asi

    while time <= endTime:
        time += delTime
        s.depleteByTime(XE_TR, SM_TR, delTime)
        s.calculateStatic(stdopt)
        s.getResult(result)

        if rodTime == 0:
            if result.asi > prevASI and prevASI < targetASI and result.asi > targetASI:
                rodTime = time

        if rodTime != 0:
            if time < rodTime+7*3600:
                s.searchRodPositionUp(stdopt, targetASI, result)
            else:
                rodTime = 0.0

        prevASI = result.asi
        stdopt.ppm = result.ppm
        strRodPos = ' '.join(["{:.2f}".format(s.rods[rod].pos)
                             for rod in s.rods])
        print(f'> TRANSIENT :  ' + '{:6.2f}'.format(time/3600)+f' {stdopt.plevel:.2f} '+'{: .3f}'.format(
            result.asi)+'{: 6.1f}'.format(result.ppm)+f' {strRodPos} {result.error:5d}')

        if s.rods['P'].pos >= s.g.core_height:
            break

    return time


def run(direction, s, stdopt, targetASI, endTime, delTime, time, result) -> float:

    rodTime = 0.0
    while time <= endTime:
        time += delTime
        s.depleteByTime(XE_TR, SM_TR, delTime)
        s.calculateStatic(stdopt)
        s.getResult(result)

        if direction == 1:
            s.searchRodPositionUp(stdopt, targetASI, result)
        elif direction == -1:
            s.searchRodPositionDown(stdopt, targetASI, result)

        stdopt.ppm = result.ppm
        strRodPos = ' '.join(["{:.2f}".format(s.rods[rod].pos)
                             for rod in s.rods])
        print(f'> TRANSIENT :  ' + '{:6.2f}'.format(time/3600)+f' {stdopt.plevel:.2f} '+'{: .3f}'.format(
            result.asi)+'{: 6.1f}'.format(result.ppm)+f' {strRodPos} {result.error:5d}')

        if direction == 1 and s.rods['P'].pos >= s.g.core_height:
            break

    return time


def runUpDown(s, stdopt, targetASI, endTime, delTime, time, result) -> float:
    while time <= endTime:
        time += delTime
        s.depleteByTime(XE_TR, SM_TR, delTime)
        s.calculateStatic(stdopt)
        s.getResult(result)
        s.searchRodPosition(stdopt, targetASI, result)
        stdopt.ppm = result.ppm
        strRodPos = ' '.join(["{:.2f}".format(s.rods[rod].pos)
                             for rod in s.rods])
        print(f'> TRANSIENT :  ' + '{:6.2f}'.format(time/3600)+f' {stdopt.plevel:.2f} '+'{: .3f}'.format(
            result.asi)+'{: 6.1f}'.format(result.ppm)+f' {strRodPos} {result.error:5d}')

    return time


s = Simon("../run/S4C3/restart_files/S403NDR.SMG",
          "../run/S4C3/plant_files/PLUS7_V127.XS",
          "../run/S4C3/plant_files/PLUS7_V127.FF",
          "../run/S4C3/restart_files/S403NDR", 8)

std_option = SteadyOption()
s.setBurnup(15000, std_option)

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

rodP.setPDIL([(1.00, 0.72*s.g.core_height), (0.75, 0.72*s.g.core_height)])
rodR5.setPDIL([(1.00, 0.72*s.g.core_height),
               (0.75, 0.40*s.g.core_height),
               (0.50, 0.00)])

rodR4.setPDIL([(0.50, 0.60*s.g.core_height),
               (0.25, 0.00*s.g.core_height)])

rodR3.setPDIL([(0.25, 0.60*s.g.core_height),
               (0.175, 0.40*s.g.core_height),
               (0.00, 0.40*s.g.core_height)])


s.setControlRods([rodP, rodR5, rodR4, rodR3, rodR2, rodR1,  rodA, rodB], [
                 rodP, rodR5, rodR4, rodR3], [[rodP], [rodR5]], rodR5)
# s.setControlRods([rodP, rodR5, rodR4, rodR3, rodR2, rodR1,  rodA, rodB], [rodP, rodR5, rodR4, rodR3], [[rodP]], rodR5)
