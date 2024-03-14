from cusfam import *
import test_prepare as pre
import random

if __name__ == "__main__":

    s = pre.s
    stdopt = SteadyOption()
    stdopt.maxiter = 100
    stdopt.crit = CBC
    stdopt.feedtf = True
    stdopt.feedtm = True
    stdopt.xenon = XE_EQ
    stdopt.samarium = SM_TR
    stdopt.tin = 290.0
    stdopt.eigvt = 1.00000
    stdopt.plevel = 1.00
    stdopt.epsiter = 1.0E-5

    result = SimonResult(s.g.nxya, s.g.nz)
    s.calculateStatic(stdopt)
    s.getResult(result)
    stdopt.ppm = result.ppm

    targetASI = result.asi

    # xenon osci. control
    delTime = 15 * 60 # time step - 15min
    time = 0.0
    endTime = 43.*3600.
    stdopt.xenon = XE_TR
    while time <= endTime:
        time += delTime
        # dp = random.random()*0.01 - 0.005
        # stdopt.plevel = 1.0+dp
        s.depleteByTime(XE_TR, SM_TR, delTime)
        s.calculateStatic(stdopt)
        s.getResult(result)
        stdopt.ppm = result.ppm
        strRodPos = ' '.join(["{:.2f}".format(s.rods[rod].pos) for rod in s.rods])
        print(f'> TRANSIENT :  ' + '{:6.2f}'.format(time/3600)+f' {stdopt.plevel:.2f} '+'{: .3f}'.format(result.asi)+'{: 6.1f}'.format(result.ppm)+f' {strRodPos} {result.error:5d}')
    

    while time <= endTime:
        time += delTime
        s.depleteByTime(XE_TR, SM_TR, delTime)
        s.calculateStatic(stdopt)
        s.getResult(result)
        s.searchRodPositionUp(stdopt, targetASI, result)
        stdopt.ppm = result.ppm
        strRodPos = ' '.join(["{:.2f}".format(s.rods[rod].pos) for rod in s.rods])
        print(f'> TRANSIENT :  ' + '{:6.2f}'.format(time/3600)+f' {stdopt.plevel:.2f} '+'{: .3f}'.format(result.asi)+'{: 6.1f}'.format(result.ppm)+f' {strRodPos} {result.error:5d}')


    



