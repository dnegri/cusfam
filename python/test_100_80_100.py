from cusfam import *
import test_prepare as pre

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

    # calculate ESI for 100% power
    s.calculateStatic(stdopt)
    s.getResult(result)
    stdopt.ppm = result.ppm
    rodPos100 = {rod.id: rod.pos for rod in s.regRods}
    ppm100 = stdopt.ppm

    strRodPos = ' '.join(["{:.2f}".format(s.rods[rod].pos) for rod in s.rods])
    print(f'> STATIC    :  {0.0:.1f} {stdopt.plevel:.2f} {result.asi:.3f} {result.ppm:.3f} {strRodPos} {result.error:5d}')    


    targetASI = result.asi

    # find control rod position for 80% power and the ESI
    stdopt.plevel = 0.85
    s.calculateStatic(stdopt)
    s.getResult(result)
    stdopt.ppm = result.ppm
    
    for i in range(2) :
        s.searchRodPosition(stdopt, targetASI, result)
        stdopt.ppm = result.ppm
        print(f'Target ASI    : [{targetASI:.3f}]')
        print(*[s.rods[rod].pos for rod in s.rods], sep=', ')
        print(f'Resulting ASI : [{result.asi:.3f}]')
        print(f'Resulting CBC : [{result.ppm:.3f}]')
        print(f'ERROR Code    : [{result.error:5d}]')
    
    strRodPos = ' '.join(["{:.2f}".format(s.rods[rod].pos) for rod in s.rods])
    print(f'> STATIC    :  {0.0:.1f} {stdopt.plevel:.2f} {result.asi:.3f} {result.ppm:.3f} {strRodPos} {result.error:5d}')    

    rodPos80 = {rod.id: rod.pos for rod in s.regRods}
    ppm80 = result.ppm

    powerRatio = 0.03 / 3600 # 3 %/hour 
    powerChange = 0.15 # 20%
    changeTime = powerChange / powerRatio # simulation time ! 6.75 hr
    ppmRatio = (ppm80 - ppm100) / changeTime

    # return to initial state
    stdopt.plevel = 1.0
    s.setRodPositions(rodPos100)    
    s.calculateStatic(stdopt)
    s.getResult(result)
    stdopt.ppm = result.ppm

    stdopt.crit = CBC
    stdopt.xenon = XE_TR
    endTime = changeTime # simulation time
    delTime = 15 * 60 # time step - 15min
    time = 0.0

    time = pre.runPowerDown(powerRatio, ppmRatio, s, stdopt, targetASI, endTime, delTime, time, result)
    
    endTime += 100*3600. # cool time with 80% power
    time = pre.run(-1, s, stdopt, targetASI, endTime, delTime, time, result)

    endTime += powerChange / powerRatio # simulation time power ascension
    time = pre.runPowerUp(powerRatio, ppmRatio, s, stdopt, targetASI, endTime, delTime, time, result)

    endTime += powerChange / powerRatio
    time = pre.runPowerUp(0.0, ppmRatio, s, stdopt, targetASI, endTime, delTime, time, result)

    stdopt.crit = CBC
    endTime = 75*3600.0
    time = pre.run(1, s, stdopt, targetASI, endTime, delTime, time, result)
    
    while True :
        endTime = time+100*3600.0
        time = pre.run(1, s, stdopt, targetASI, endTime, delTime, time, result)

        endTime = time
        




    



