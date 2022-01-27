import math
from itertools import islice
from random import random
from time import perf_counter
from cusfam import *
import time
from Simon import Simon
import os
import multiprocessing
from threading import Thread
from PyQt5.QtCore import QPointF, pyqtSignal, QThread, QObject
import Definitions as df


class CalculationManager(QObject):

    progress = pyqtSignal()
    finished = pyqtSignal()

    def __init__(self, queue, parent=None):
        super(CalculationManager, self).__init__(parent)
        # TODO SGH, Make Routines
        #self.s = Simon("C:/Users/geonho/Documents/SVN/SIMON/SIMON_CODE/rst/Y301ASBDEP.SMG", "C:/Users/geonho/Documents/SVN/SIMON/SIMON_CODE/run/KMYGN34C01_PLUS7_XSE.XS", "C:/Users/geonho/Documents/SVN/SIMON/SIMON_CODE/rst/Y301ASBDEP")
        self.s = Simon("C:/Users/geonho/Documents/SVN/SIMON/SIMON_CODE_v2/rst/Y301ASBDEP.SMG", 
                       "C:/Users/geonho/Documents/SVN/SIMON/SIMON_CODE_v2/run/KMYGN34C01_PLUS7_XSE.XS", 
                       "C:/Users/geonho/Documents/SVN/SIMON/SIMON_CODE_v2/rst/Y301ASBDEP")

        self.s_full = Simon("C:/Users/geonho/Documents/SVN/SIMON/SIMON_CODE_v2/rst/Y301ASBDEP.FCE.SMG", 
                            "C:/Users/geonho/Documents/SVN/SIMON/SIMON_CODE_v2/run/KMYGN34C01_PLUS7_XSE.XS", 
                            "C:/Users/geonho/Documents/SVN/SIMON/SIMON_CODE_v2/rst/Y301ASBDEP.FCE")

        #self.s = Simon("D:/codes/knf_simon1/trunk/rst/Y301ASBDEP.SMG",
        #               "D:/codes/knf_simon1/trunk/run/KMYGN34C01_PLUS7_XSE.XS",
        #                "D:/codes/knf_simon1/trunk/rst/Y301ASBDEP")
        #self.s_full = Simon("D:/codes/knf_simon1/trunk/rst/Y301ASBDEP.FCE.SMG",
        #               "D:/codes/knf_simon1/trunk/run/KMYGN34C01_PLUS7_XSE.XS",
        #                "D:/codes/knf_simon1/trunk/rst/Y301ASBDEP.FCE")
        self.calcOption = None
        self.outputArray = []
        self.restartFlag = False
        self.queue = queue
        
    def load(self):
        while True:
            #time.sleep(1)
            self.queue.get()
            if(self.calcOption==df.CalcOpt_ASI):
                self.startShutdown()
                self.calcOption = None
            elif(self.calcOption==df.CalcOpt_SDM):
                self.startSDM()
                self.calcOption = None
            elif(self.calcOption==df.CalcOpt_Lifetime):
                self.startLifeTime()
                self.calcOption = None
            elif(self.calcOption==df.CalcOpt_ECP):
                self.startECP()

    def setSDMVariables(self,calcOpt,bp,stuckRod, mode_index):
        self.calcOption = calcOpt
        self.bp = bp
        self.stuckRodPos = stuckRod
        self.mode_index = mode_index

    def setShutdownVariables(self,calcOpt,bp,eig,target_ASI,pArray):
        self.calcOption = calcOpt
        self.bp = bp
        self.eig = eig
        self.target_ASI = target_ASI
        self.pArray = pArray
    
    def setLifeTimeVariable(self,calcOpt,power,target_eig,pArray):
        self.calcOption = calcOpt
        self.power = power
        self.eig = target_eig
        self.pArray = pArray
        
    def setECPVariables(self,calcOpt, targetOpt, inputArray):
        self.calcOption = calcOpt
        self.targetOpt = targetOpt
        self.inputArray = inputArray        

    def startShutdown(self):
        start = time.time()
        self.tableDatasetFlag = True
        self.outputFlag = True
        # Todo SGH, Remove tmp later

        self.outputArray = [ [ 0.0, 0.0, 381.0, 381.0, 381.0 , 381.0] ]
        self.progress.emit()

        self.s.setBurnup(self.bp)

        std_option = SteadyOption()
        std_option.maxiter = 100
        std_option.crit = CBC
        std_option.feedtf = True
        std_option.feedtm = True
        std_option.xenon = XE_EQ
        std_option.tin = 295.8
        std_option.eigvt = self.eig
        std_option.ppm = 800.0
        nStep = len(self.pArray)

        for iStep in range(nStep):
            std_option.plevel = self.pArray[iStep][1]/100.0

            result = SimonResult()
            self.s.calculateStatic(std_option)
            #self.sample_static(std_option)
            self.s.getResult(result);
            std_option.ppm = result.ppm
            std_option.xenon = XE_TR

            print(f'Initial AO [{result.ao:.3f}]')
            #sample_deplete(s, std_option);
            self.asisearch(std_option,self.target_ASI, iStep)

        end = time.time()
        print(end - start)
        
        tmp = self.outputArray.pop(0)
        self.finished.emit()
        # return self.outputArray

    def asisearch(self,std_option, target_ASI, iStep) :
        # TODO, Add P Control Rod
        rodids = [ 'R5', 'R4', 'R3'];
        overlaps = [ 0.0 * self.s.g.core_height, 0.6 * self.s.g.core_height, 1.2 * self.s.g.core_height]
        r5_pdil = 0.0
        # r5_pdil = 0.72 * self.s.g.core_height
        if(len(self.outputArray)!=0):
            preStepPos = [self.outputArray[-1][2], self.outputArray[-1][3], self.outputArray[-1][4]]
        else:
            preStepPos = [ 381.0, 381.0, 381.0 ]

        ao_target = -target_ASI
        position = self.s.g.core_height
        result = self.s.searchRodPosition(std_option, ao_target, rodids, overlaps, r5_pdil, position, preStepPos)

        unitArray = []
        unitArray.append(-result.ao)
        unitArray.append(result.ppm)
        print(f'Target AO : [{ao_target:.3f}]')
        print(f'Resulting AO : [{result.ao:.3f}]')
        print(f'Resulting CBC : [{result.ppm:.3f}]')
        print(f'ERROR Code : [{result.error:5d}]')
        print('Rod positions')

        if(len(result.rod_pos)!=0):
            for rodid in rodids :
                print(f'{rodid:12s}  :  {result.rod_pos[rodid]:12.3f}')
                unitArray.append(result.rod_pos[rodid])
            #unitArray.append(190.5)
        else:
            unitArray.append(190.5)
            unitArray.append(190.5)
            unitArray.append(190.5)

        # TODO Make Bank P
        unitArray.append(381.0)
        self.outputArray.append(unitArray)
        self.progress.emit()

    def setRestartCalculation(self,rodChangeHist, rodPosInput):
        self.rodChangeHist = rodChangeHist
        self.rodPosInput = rodPosInput
        self.restartFlag = True

    def startSDM(self):
        if self.mode_index > 0:
            self.s_full.setBurnup(self.bp)

            std_option = SteadyOption()
            std_option.maxiter = 100
            std_option.crit = CBC
            std_option.feedtf = True
            std_option.feedtm = True
            std_option.xenon = XE_EQ
            std_option.tin = 295.8
            std_option.eigvt = 1.0
            std_option.ppm = 800.0
            std_option.plevel = 1.0

            self.s_full.setRodPosition(['R', 'B', 'A', 'P'], [0] * 4, 0.0)

            for stuck_rod in self.stuckRodPos:
                self.s_full.setRodPosition1(stuck_rod, self.s_full.g.core_height)

            result = SimonResult()
            self.s_full.calculateStatic(std_option)
            self.s_full.getResult(result)
            #std_option.ppm = result.ppm

            self.s_full.setRodPosition(['R','B','A','P'], [0] * 4, self.s_full.g.core_height)

            print(f'SHUTDOWN Boron : [{result.ppm:.2f} pcm]')
        else:
            self.s_full.setBurnup(self.bp)

            std_option = SteadyOption()
            std_option.maxiter = 100
            std_option.crit = CBC
            std_option.feedtf = True
            std_option.feedtm = True
            std_option.xenon = XE_EQ
            std_option.tin = 295.8
            std_option.eigvt = 1.0
            std_option.ppm = 800.0
            std_option.plevel = 1.0

            result = SimonResult()

            self.s_full.setRodPosition(['R','B','A','P'], [0] * 4, self.s_full.g.core_height)
            #sample_static(self.s, std_option)
            self.s_full.calculateStatic(std_option)
            self.s_full.getResult(result);
            std_option.ppm = result.ppm
            std_option.xenon = XE_TR

            rodids = ['R5', 'R4', 'R3']
            overlaps = [0, 0.4 * self.s_full.g.core_height, 0.7 * self.s_full.g.core_height]
            r5_pdil = 0.72 * self.s_full.g.core_height
            r5_pos = r5_pdil

            sdm = self.s_full.getShutDownMargin(std_option, rodids, overlaps, r5_pdil, r5_pos, self.stuckRodPos)
            sdm = sdm

            self.s_full.setRodPosition(['R','B','A','P'], [0] * 4, self.s_full.g.core_height)

            print(f'SHUTDOWN MARGIN : [{sdm:.2f} pcm]')



    def startSDMM345(self):
        
        self.s_full.setBurnup(self.bp)

    #def startLifeTime(self):

    #    std_option = SteadyOption()
    #    std_option.maxiter = 100
    #    std_option.crit = CBC
    #    std_option.feedtf = True
    #    std_option.feedtm = True
    #    std_option.xenon = XE_EQ
    #    std_option.tin = 295.8
    #    std_option.eigvt = self.eig
    #    std_option.ppm = 800.0
    #    std_option.plevel = 1.0

    #    self.s.setBurnup(0)

    #    result = SimonResult()
    #    self.s.calculateStatic(std_option)
    #    self.s.getResult(result);
    #    std_option.ppm = result.ppm

    #    burn_del = 10000.0
    #    burn_end = 20000.0
    #    burn_step = math.ceil(burn_end/burn_del) + 1

    #    #keff_new = eig
    #    #keff_old = eig

    #    # 01. Calculation Expected MOC Step

    #    #for iStep in range(burn_step):
    #    #s.calculateStatic(std_option)
    #    oldPPM = result.ppm
    #    i=1
    #    print("STEP : ", i, "		 PPM : ", oldPPM)
    #    self.s.setBurnup(10000.0)
    #    self.s.calculateStatic(std_option)
    #    #self.s.deplete(XE_EQ, SM_TR, burn_del)
    #    self.s.getResult(result)
    #    newPPM = result.ppm
    #    i=2
    #    print("STEP : ", i, "		 PPM : ", newPPM)

    #    self.s.setBurnup(13650.0)
    #    self.s.calculateStatic(std_option)
    #    #self.s.deplete(XE_EQ, SM_TR, burn_del)
    #    self.s.getResult(result)
    #    newPPM = result.ppm
    #    i=3
    #    print("STEP : ", i, "		 PPM : ", newPPM)
    #    pass
    def startLifeTime(self):

        std_option = SteadyOption()
        std_option.maxiter = 100
        std_option.crit = CBC
        #std_option.crit = KEFF
        std_option.feedtf = True
        std_option.feedtm = True
        std_option.xenon = XE_EQ
        std_option.tin = 295.8
        std_option.eigvt = self.eig
        std_option.ppm = 10.0
        std_option.plevel = 1.0

        self.s.setBurnup(0)

        result = SimonResult()
        self.s.calculateStatic(std_option)
        self.s.getResult(result);
        std_option.ppm = result.ppm

        burn_del = 10000.0
        burn_end = 20000.0
        burn_step = math.ceil(burn_end/burn_del) + 1

        #keff_new = eig
        #keff_old = eig

        # 01. Calculation Expected MOC Step

        #for iStep in range(burn_step):
        #s.calculateStatic(std_option)
        oldPPM = result.ppm
        i=1
        print("STEP : ", i, "		 PPM : ", oldPPM)
        self.s.setBurnup(10000.0)
        self.s.calculateStatic(std_option)
        #self.s.deplete(XE_EQ, SM_TR, burn_del)
        self.s.getResult(result)
        newPPM = result.ppm
        i=2
        print("STEP : ", i, "		 PPM : ", newPPM)

        self.s.setBurnup(13650.0)
        self.s.calculateStatic(std_option)
        #self.s.deplete(XE_EQ, SM_TR, burn_del)
        self.s.getResult(result)
        newPPM = result.ppm
        i=3
        print("STEP : ", i, "		 PPM : ", newPPM)

        self.s.setBurnup(13550.0)
        self.s.calculateStatic(std_option)
        #self.s.deplete(XE_EQ, SM_TR, burn_del)
        self.s.getResult(result)
        newPPM = result.ppm
        i=4
        print("STEP : ", i, "		 PPM : ", newPPM)
        pass

    def startECP(self):
        #01. Set ECP Dataset
        if(self.targetOpt==df.select_Boron):
            [ pw, bp, mtc, ppm, P_Pos, r5Pos, r4Pos, shutdown_P_Pos, shutdown_r5Pos, shutdown_r4Pos ] = self.inputArray
        elif(self.targetOpt==df.select_RodPos):
            [ pw, bp, mtc, ppm, P_Pos, r5Pos, r4Pos, shutdown_ppm ] = self.inputArray

        #02. Calculate Conditions before shutdown
        std_option = SteadyOption()
        std_option.maxiter = 100
        std_option.crit = CBC
        #std_option.crit = KEFF
        std_option.feedtf = True
        std_option.feedtm = True
        std_option.xenon = XE_EQ
        #std_option.tin = mtc
        std_option.tin = 295.8
        std_option.eigvt = 0.967130
        std_option.ppm = 800
        std_option.plevel = pw/100.0
        rod_IDS = [ "R5", "R4" ]
        rod_pos = [ r5Pos, r4Pos ]
        for idx in range(len(rod_IDS)):
            self.s.setRodPosition1(rod_IDS[idx],rod_pos[idx])

        #03. Calculate Conditions after shutdown
        
        self.s.setBurnup(13650.0)
        result = SimonResult()
        self.s.calculateStatic(std_option)
        self.s.getResult(result);
        std_option.eigvt = result.eigv
        out_ppm_old = result.ppm
        k_pre = result.eigv
        if(self.targetOpt==df.select_Boron):
            shutdown_rod_pos = [ shutdown_r5Pos, shutdown_r4Pos ]
            for idx in range(len(rod_IDS)):
                self.s.setRodPosition1(rod_IDS[idx],shutdown_rod_pos[idx])
            result = SimonResult()
            self.s.calculateStatic(std_option)
            self.s.getResult(result);
            out_ppm = result.ppm
            print(out_ppm)
        elif(self.targetOpt==df.select_RodPos):
            overlaps = [ 0.0 * self.s.g.core_height, 0.6 * self.s.g.core_height, 1.2 * self.s.g.core_height]
            r5_pdil = 0.0
            position = self.s.g.core_height
            std_option.ppm = shutdown_ppm
            result = self.s.searchRodPosition(std_option, k_pre, rod_IDS, overlaps, r5_pdil, position, rod_pos)
            print(result.eigv)