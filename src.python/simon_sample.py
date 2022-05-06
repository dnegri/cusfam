from itertools import islice
from random import random
from time import perf_counter
from cusfam import *
import time
from Simon import Simon

def sample_static(s, std_option) :
    s.calculateStatic(std_option)

def sample_deplete(s, std_option) :

    burn_del = 1000.0
    burn_end = 1000.0
    burn_step = int(burn_end / burn_del)
    
    result = SimonResult(s.g.nxya, s.g.nz)

    start = time.time()
    for i in range(burn_step) :
        s.calculateStatic(std_option)
        s.getResult(result)
        s.calculatePinPower()
        print("STEP : ", i, " -> PPM : ", result.ppm)
        s.deplete(XE_EQ, SM_TR, burn_del)

    end = time.time()
    print(end - start)
    print("END DEP")

def sample_dynxe(s, std_option) :
    std_option.plevel = 0.9
    std_option.xenon = XE_TR
    sample_static(s, std_option)

    std_option.crit = KEFF
    for i in range(100) :
        s.depleteXeSm(XE_TR, SM_TR, 3600)
        s.calculateStatic(std_option)
        s.getResult(result)
        print(f'KEFF : {result.eigv:.5f} and ASI : {result.asi:0.3f}')


def sample_asisearch(s, std_option, asi_target, position) :
    rodids = ['R5', 'R4', 'R3']
    overlaps = [0, 0.4 * s.g.core_height, 0.7 * s.g.core_height]
    r5_pdil = 0.0 # 0.72 * s.g.core_height
    result = s.searchRodPosition(std_option, asi_target, rodids, overlaps, r5_pdil, position)

    print(f'Target ASI : [{asi_target:.3f}]')
    print(f'Resulting ASI : [{result.asi:.3f}]')
    print(f'Resulting CBC : [{result.ppm:.3f}]')
    print(f'ERROR Code : [{result.error:5d}]')
    print('Rod positions')
    for rodid in rodids :
        print(f'{rodid:12s}  :  {result.rod_pos[rodid]:12.3f}')
    return result.rod_pos['R5']

def sample_shutdown_margin(s, std_option, stuck_rods) :
    rodids = ['R5', 'R4', 'R3']
    overlaps = [0, 0.4 * s.g.core_height, 0.7 * s.g.core_height]
    r5_pdil = 0.72 * s.g.core_height
    r5_pos = r5_pdil

    sdm = 0.0
    
    sdm = s.getShutDownMargin(std_option, rodids, overlaps, r5_pdil, r5_pos, stuck_rods)
    
    sdm = sdm*1.E5

    print(f'SHUTDOWN MARGIN : [{sdm:.2f} pcm]')


if __name__ == "__main__":
    #init("../run/geom.simon", "../run/KMYGN34C01_PLUS7_XSE.XS", "../run");
    #init("D:/work/corefollow/ygn3/c01/depl/rst/Y301ASBDEP.SMG",
    #"../run/KMYGN34C01_PLUS7_XSE.XS",
    #"D:/work/corefollow/ygn3/c01/depl/rst/Y301ASBDEP");
    
    #s = Simon("D:/codes/cusfam/main/run/ucn6/c12/UCN612ASBDEP.SMG", 
    #          "D:/codes/cusfam/main/run/ucn6/db/UCN6_OPR1000_rev1.XS", 
    #          "D:/codes/cusfam/main/run/ucn6/db/UCN6_OPR1000_rev1.FF", 
    #          "D:/codes/cusfam/main/run/ucn6/c12/UCN612ASBDEP")

    s = Simon("D:/codes/cusfam/main/run/ygn3/Y312ASBDEP.SMG", 
              "D:/codes/cusfam/main/run/ygn3/KMYGN34C01_PLUS7_XSE.XS", 
              "D:/codes/cusfam/main/run/ygn3/KMYGN34C01_PLUS7_XSE.FF", 
              "D:/codes/cusfam/main/run/ygn3/Y312ASBDEP")

    #s = Simon("D:/work/corefollow/ygn3/c01/depl/rst/Y301ASBDEP.FCE.SMG", "../run/KMYGN34C01_PLUS7_XSE.XS", "D:/work/corefollow/ygn3/c01/depl/rst/Y301ASBDEP.FCE")
    bp = [0.0, 50.0, 150.0, 500.0, 1000.0,
         2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0, 11000.0, 12000.0, 13000.0, 14000.0, 15000.0, 16000.0]
    s.setBurnupPoints(bp)

    std_option = SteadyOption()
    std_option.maxiter = 100
    std_option.crit = CBC
    std_option.feedtf = True
    std_option.feedtm = True
    std_option.xenon = XE_EQ
    std_option.tin = 292.22
    std_option.eigvt = 1.0
    std_option.ppm = 800
    std_option.plevel = 1.0
    std_option.epsiter = 1.0E-4

    result = SimonResult(s.g.nxya, s.g.nz)

    #std_option.xenon = XE_TR

    s.setBurnup(0.0)
    sample_deplete(s, std_option)
    
    std_option.xenon = XE_EQ
    s.setRodPosition(['R', 'B', 'A', 'P'], [0, ] * 4, s.g.core_height)
    s.calculateStatic(std_option)
    s.calculatePinPower()
    s.getResult(result)
    print(f"ASI : {result.asi:.3f}, PPM : {result.ppm:.1f},  Fxy : {result.fxy:.3f}")

    std_option.xenon = XE_TR
    std_option.plevel = 0.0
    std_option.tin = 295.6
    s.setRodPosition1('R5', 72.0/100*381.0)
    s.calculateStatic(std_option)
    s.calculatePinPower()
    s.getResult(result)
    print(f"ASI : {result.asi:.3f}, PPM : {result.ppm:.1f},  Fxy : {result.fxy:.3f}")
    std_option.ppm = result.ppm
    for i in range(0,30) :
        s.depleteXeSm(XE_TR, SM_TR, 3600)
        s.calculateStatic(std_option)
        s.calculatePinPower()
        s.getResult(result)
        print(f"ASI : {result.asi:.3f}, PPM : {result.ppm:.1f}, Fxy : {result.fxy:.3f}")
        std_option.ppm = result.ppm
    

    exit(-1)

    ## trip
    #std_option.crit = KEFF
    #std_option.plevel = 0.0
    std_option.xenon = XE_TR
    s.setRodPosition(['R5'], [0.0], 72.0)
    s.calculateStatic(std_option)
    s.getResult(result)
    print(f"ASI : {result.asi:.3f}, PPM : {result.ppm:.1f}")

    # xenon build-up for 30 hours
    std_option.xenon = XE_TR
    for i in range(100) :
        s.depleteXeSm(XE_TR, SM_TR, 1800)
        s.calculateStatic(std_option)
        s.getResult(result)
        print(f"ASI : {result.asi:.3f}, PPM : {result.ppm:.1f}")

    #std_option.crit = CBC
    ## ecp
    #s.setRodPosition1('P', 200.0)
    #position = sample_asisearch(s, std_option, -0.6, 0.0)
    #print(f"ASI : {result.asi:.3f}, PPM : {result.ppm:.1f}")
    
    ## power ascension
    #rodids = ['R5', 'R4', 'R3']
    #overlaps = [0.0 * s.g.core_height, 0.6 * s.g.core_height, 1.2 * s.g.core_height]

    #target_asi = -0.1

    #for i in range(100) :
    #    std_option.plevel = i*0.01
    #    pdil = s.getPDIL(std_option.plevel*100)
    #    r5_pdil = pdil[0]

    #    result = s.searchRodPositionO(std_option, target_asi, rodids, overlaps, r5_pdil, preStepPos[1])
        
    #    r5_pos = result.rod_pos['R5']
    #    r4_pos = result.rod_pos['R4']
    #    print(f"ASI : {result.asi:.3f}, PPM : {result.ppm:.1f}, RODP : [{r5_pos:.1f}, {r4_pos:.1f}]")


    #sample_static(s_full, std_option)
    #s_full.getResult(result)
    #print(f'PPM : {result.ppm:.3f}')

    #std_option.ppm = result.ppm
    #std_option.xenon = XE_TR
    #sample_static(s, std_option)

    #s.getResult(result)
    #print(f'PPM : {result.ppm:.3f}')

    #print(f'Initial ASI [{result.asi:.3f}]')


    #std_option.plevel = 0.9
    #std_option.xenon = XE_TR
    #sample_static(s, std_option)

    #sample_dynxe(s, std_option)    

    #sample_shutdown_margin(s, std_option, ['B41', 'B42'])
    #sample_shutdown_margin(s, std_option, ['B41', 'B42'])
    #position = 381.0

    #std_option.epsiter = 1.E-3

    #position = sample_asisearch(s, std_option,0.10, position)
    #position = sample_asisearch(s, std_option,0.15, position)
    #position = sample_asisearch(s, std_option,0.15, position)
    #position = sample_asisearch(s, std_option,0.16, position)

    #rodids = ['R5', 'R4', 'R3']
    #overlaps = [0, 0.4 * s.g.core_height, 0.7 * s.g.core_height]
    #r5_pdil = 0.72 * s.g.core_height
    #s.setRodPosition(rodids, overlaps, 381.0)
    #sample_static(s, std_option)
    #s.getResult(result)
    #print(f'PPM : {result.ppm:.3f}')


    
