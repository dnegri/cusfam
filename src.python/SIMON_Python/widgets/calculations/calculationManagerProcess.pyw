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
from PyQt5.QtCore import QPointF, pyqtSignal, QThread, QObject, QProcess
import Definitions as df
import numpy as np
import glob
import constants as cs
import utils as ut

import multiprocessing as mp
import threading as td

R5_95_65 = [249.936, 5.2832]
R5_65_50 = [91.44, 6.096]

R4_95_65 = [381.0, 2.032]
R4_65_50 = [320.04, 6.096]
R4_50_42 = [228.6, 9.525]
R4_42_20 = [152.4, 6.92727]

R3_42_20 = [381.0, 9.0054]
R3_20_0 = [182.88, 1.524]

decay_table = [0.5, 0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
               100, 100, 100, 100, 100, 100, 100]

class CalculationManager(QThread):

    update_working = pyqtSignal(str)
    finished_working = pyqtSignal(str)

    def __init__(self, queue, parent=None):
        super(CalculationManager, self).__init__(parent)

        self.updated = "updated"
        self.finished = "finished"

        self.user = None

        self.calcOption = None
        self.outputArray = []
        self.queue = queue

        self.put_queue = mp.Queue(maxsize=10)
        self.get_queue = mp.Queue(maxsize=10)

        self.axial_position = []
        self.initialized = False
        self.plant_name = ""
        self.cycle_name = ""
        self.cycle_burnup_values = []
        self.is_started = False

        self.agents = []
        self.kbc = 1
        self.kec = -1

        # Output initialize
        self.sdm2_ppm = 0.0
        self.sdm2_temp = 0.0
        self.sdm2_rsdm = 0.0

        self.sdm1_srw = 0.0
        self.sdm1_sdm = 0.0
        self.sdm1_defect = 0.0
        self.sdm1_n1w = 0.0
        self.sdm1_temp = 0.0
        self.sdm1_rsdm = 0.0

        self.out_ppm_old = []
        self.out_ppm = []
        self.rodPos = []
        self.ecpP1d = []

        self.shutdown_output = []
        self.reoperation_output = []

    def check_files(self, user):

        # Format Alert Plant Letter can be more than two but must be defined in cs.DEFINED_PLANTS
        # Plant number is only one number 3 or 4 not 04
        # Cycle number can be more than one
        #print(user.restart_file)
        if not user.restart_file:
            return False

        plant_name_length = ut.get_string_length(user.restart_file[:4])

        if plant_name_length == 0 \
                or not user.restart_file[:plant_name_length] in cs.DEFINED_PLANTS\
                or ut.get_int_value(user.restart_file[plant_name_length+1:plant_name_length+4]) == -1:
            return False

        if len(glob.glob(user.restart_directory + user.restart_file + "*.SMG")) == 0:
            return False
        if len(glob.glob(user.plant_directory + user.plant_file + "*.XS",)) == 0:
            return False
        if len(glob.glob(user.restart_directory + user.restart_file+ "*.SMR")) == 0:
            return False
        if len(glob.glob(user.restart_directory + user.restart_file + ".FCE" +"*.SMG")) == 0:
            return False
        if len(glob.glob(user.restart_directory + user.restart_file+".FCE"+ "*.SMR")) == 0:
            return False

        return True

    def set_names(self, user):

        self.user = user

        #Format Alert Plant Letter can be
        if not user.plant_file:
            return

        if len(user.plant_file) == 0:
            return

        if not user.restart_file:
            return

        if len(user.restart_file) == 0:
            return

        plant_name_length = ut.get_string_length(user.restart_file[:4])
        cycle_number = ut.get_int_value(user.restart_file[plant_name_length+1:plant_name_length+4])

        self.plant_name = cs.DEFINED_PLANTS[user.restart_file[:plant_name_length]]+\
                          user.restart_file[plant_name_length:plant_name_length+1]
        self.cycle_name = str(cycle_number)

        self.cycle_burnup_values = []
        try:
            for name in glob.glob(user.restart_directory + user.restart_file + "*.SMR"):
                if int(name.split(cs.RESTART_FILE_BURNUP_SEPERATOR)[cs.RESTART_FILE_BURNUP_INDEX]) \
                        not in self.cycle_burnup_values:
                    self.cycle_burnup_values.append(
                        int(name.split(cs.RESTART_FILE_BURNUP_SEPERATOR)[cs.RESTART_FILE_BURNUP_INDEX]))
        except ValueError:
            raise("Contact Admin!!!!.\nFile Naming Error, Check Restart File Name with Admin")

        self.cycle_burnup_values.sort()

    def load(self, user):

        if not user.plant_file:
            return

        if len(user.plant_file) == 0:
            return

        if not user.restart_file:
            return

        if len(user.restart_file) == 0:
            return

        self.set_names(user)

        if self.initialized:
            self.setKillOutput()
            return

        self.initialized = True
        self.is_started = False

        self.agents = []

        files = [user.restart_directory+user.restart_file+".SMG",
                   user.plant_directory+user.plant_file+".XS",
                    user.restart_directory+user.restart_file,
                 user.restart_directory + user.restart_file + ".FCE" + ".SMG",
                 user.plant_directory + user.plant_file + ".XS",
                 user.restart_directory + user.restart_file + ".FCE"
                 ]

        for thread_id in range(1):
            self.agents.append(ASTRAProcess(self.put_queue, self.get_queue, files, self.cycle_burnup_values))

        for agent in self.agents:
            agent.start()

        self.axial_position = []
        self.calcOption = df.CalcOpt_HEIGHT
        self.queue.put([df.CalcOpt_HEIGHT,])

    def run(self):

        while True:
            options = self.queue.get()
            is_successful = "Unsuccessful"

            self.put_queue.put(options)
            calculation_option = options[0]
            if calculation_option == df.CalcOpt_ASI:
                self.setShutdownVariables(options[0], options[1], options[2], options[3], options[4])
            elif calculation_option == df.CalcOpt_ASI_RESTART:
                self.setShutdownVariables(options[0], options[1], options[2], options[3], options[4])
            elif calculation_option == df.CalcOpt_SDM:
                self.setSDMVariables(options[0], options[1], options[2], options[3], options[4])
            elif calculation_option == df.CalcOpt_RO:
                self.setReopreationVariable(options[0], options[1], options[2])
            elif calculation_option == df.CalcOpt_RO_RESTART:
                self.setReopreationVariable(options[0], options[1], options[2])
            elif calculation_option == df.CalcOpt_RECENT:
                self.setRecentCalculationVariable(options[0], options[1])
            elif calculation_option == df.CalcOpt_ECP:
                self.setECPVariables(options[0], options[1], options[2])

            self.is_started = True
            while True:

                outputs = self.get_queue.get()

                calculation_option = outputs[0]
                calculation_result = outputs[1]
                calculation_error = outputs[2]

                if calculation_option==df.CalcOpt_ASI:
                    self.shutdown_output = outputs[3]
                elif calculation_option==df.CalcOpt_ASI_RESTART:
                    self.shutdown_output = outputs[3]
                elif calculation_option==df.CalcOpt_RO:
                    self.reoperation_output = outputs[3]
                elif calculation_option==df.CalcOpt_RO_RESTART:
                    self.reoperation_output = outputs[3]
                elif calculation_option==df.CalcOpt_SDM:
                    if len(outputs) == 6:
                        self.setSDM345Output(outputs[3:])
                    else:
                        self.setSDM12Output(outputs[3:])
                elif calculation_option==df.CalcOpt_RECENT:
                    self.setRecentCalculationVariable(options[0], options[1])
                    is_successful = "Successful"
                    self.finished = True
                elif calculation_option == df.CalcOpt_ECP:
                    self.setECPOutput(outputs[3:])
                elif calculation_option == df.CalcOpt_HEIGHT:
                    self.axial_position = outputs[3]
                    self.kbc = outputs[4]
                    self.kec = outputs[5]
                elif calculation_option == df.CalcOpt_KILL:
                    self.setKillOutput()

                if not calculation_error:
                    is_successful = "Successful"
                else:
                    is_successful = "Unsuccessful"

                if calculation_result == self.finished or calculation_error:
                    break

                self.update_working.emit("True")

            self.finished_working.emit(is_successful)

            self.is_started = False


    def setRecentCalculationVariable(self,calcOpt,row):
        self.calcOption = calcOpt
        self.row = row

    def setSDMVariables(self,calcOpt,bp, stuckRod, mode_index, rsdm):
        self.calcOption = calcOpt
        self.bp = bp
        self.stuckRodPos = stuckRod
        self.mode_index = mode_index
        self.rsdm = rsdm

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

    def setReopreationVariable(self,calcOpt, targetASI, pArray):
        self.calcOption = calcOpt
        self.target_ASI = targetASI
        self.pArray = pArray

    def setECPVariables(self,calcOpt, targetOpt, inputArray):
        self.calcOption = calcOpt
        self.targetOpt  = targetOpt
        self.inputArray = inputArray

    def setSDM345Output(self, outputs):
        self.sdm2_ppm = outputs[0]
        self.sdm2_temp = outputs[1]
        self.sdm2_rsdm = outputs[2]

    def setSDM12Output(self, outputs):

        self.sdm1_srw = outputs[0]
        self.sdm1_sdm = outputs[1]
        self.sdm1_defect = outputs[2]
        self.sdm1_n1w = outputs[3]
        self.sdm1_temp = outputs[4]
        self.sdm1_rsdm = outputs[5]

    def setECPOutput(self, outputs):
        self.out_ppm_old = outputs[0]
        self.out_ppm = outputs[1]
        self.rodPos = outputs[2]
        self.ecpP1d = outputs[3]

    def setKillOutput(self, is_restart=False):
        user = self.user
        if len(self.agents) > 0:
            self.agents[0].kill()
        self.agents = []
        time.sleep(1)
        # self.get_queue = mp.Queue(maxsize=1)
        # self.put_queue = mp.Queue(maxsize=1)
        files = [user.restart_directory + user.restart_file + ".SMG",
                 user.plant_directory + user.plant_file + ".XS",
                 user.restart_directory + user.restart_file,
                 user.restart_directory + user.restart_file + ".FCE" + ".SMG",
                 user.plant_directory + user.plant_file + ".XS",
                 user.restart_directory + user.restart_file + ".FCE"
                 ]

        for thread_id in range(1):
            self.agents.append(
                ASTRAProcess(self.put_queue, self.get_queue, files, self.cycle_burnup_values))

        for agent in self.agents:
            agent.start()


    def restartProcess(self):
        self.get_queue.put([df.CalcOpt_KILL, self.finished, True])


#class ASTRAProcess(mp.Process):
class ASTRAProcess(td.Thread):

    def __init__(self, get_queue, put_queue, files, cycle_burnup_values):
        #mp.Process.__init__(self)
        td.Thread.__init__(self)

        self.updated = "updated"
        self.finished = "finished"

        self.files = files
        self.s = None
        self.s_full = None
        self.get_queue = get_queue
        self.put_queue = put_queue
        self.cycle_burnup_values = cycle_burnup_values
        self.eoc_cycle_burnup_value = cycle_burnup_values[-1]
        self.outputArray = []

    def run(self):
        files = self.files

        print("load start")
        print("load start0")
        self.s = Simon(files[0],
                       files[1],
                       files[2])
        print("load start1")
        self.s.setBurnupPoints(self.cycle_burnup_values)
        print("load start")
        self.s_full = Simon(files[3],
                            files[4],
                            files[5])
        self.s_full.setBurnupPoints(self.cycle_burnup_values)
        print("load ready")

        while True:

            options = self.get_queue.get()
            calculation_option = options[0]

            self.outputArray = []

            if calculation_option == df.CalcOpt_ASI:
                self.setShutdownVariables(options[0], options[1], options[2], options[3], options[4])
                self.startShutdown()
            if calculation_option == df.CalcOpt_ASI_RESTART:
                self.setShutdownVariables(options[0], options[1], options[2], options[3], options[4])
                self.startShutdownRestart()
            elif calculation_option == df.CalcOpt_SDM:
                self.setSDMVariables(options[0], options[1], options[2], options[3], options[4])
                self.startSDM()
            elif calculation_option == df.CalcOpt_RO:
                self.setReopreationVariable(options[0], options[1], options[2])
                self.startReoperation()
            elif calculation_option == df.CalcOpt_RO_RESTART:
                self.setReopreationVariable(options[0], options[1], options[2])
                self.startReoperationRestart()
            elif calculation_option == df.CalcOpt_RECENT:
                self.setRecentCalculationVariable(options[0], options[1])
            elif calculation_option == df.CalcOpt_ECP:
                self.setECPVariables(options[0], options[1], options[2])
                self.startECP()
            elif calculation_option == df.CalcOpt_HEIGHT:
                self.getAxialHeight()

    def setRecentCalculationVariable(self, calcOpt, row):
        self.calcOption = calcOpt
        self.row = row
        self.put_queue.put([self.calcOption, True, row])

    def setSDMVariables(self, calcOpt, bp, stuckRod, mode_index, rsdm):
        self.calcOption = calcOpt
        self.bp = bp
        self.stuckRodPos = stuckRod
        self.mode_index = mode_index
        self.rsdm = rsdm

    def setShutdownVariables(self, calcOpt, bp, eig, target_ASI, pArray):
        self.calcOption = calcOpt
        self.bp = bp
        self.eig = eig
        self.target_ASI = target_ASI
        self.pArray = pArray

    def setLifeTimeVariable(self, calcOpt, power, target_eig, pArray):
        self.calcOption = calcOpt
        self.power = power
        self.eig = target_eig
        self.pArray = pArray

    def setReopreationVariable(self, calcOpt, targetASI, pArray):
        self.calcOption = calcOpt
        self.target_ASI = targetASI
        self.pArray = pArray

    def setECPVariables(self, calcOpt, targetOpt, inputArray):
        self.calcOption = calcOpt
        self.targetOpt = targetOpt
        self.inputArray = inputArray

    def startShutdown(self):
        self.tableDatasetFlag = True
        self.outputFlag = True

        self.outputArray = []
        # self.update_working.emit("True")

        self.s.setBurnup(self.bp)

        std_option = SteadyOption()
        std_option.maxiter = 100
        std_option.crit = CBC
        std_option.feedtf = True
        std_option.feedtm = True
        std_option.xenon = XE_EQ
        std_option.tin = df.inlet_temperature
        std_option.eigvt = self.eig
        std_option.ppm = 800.0
        std_option.plevel = 1.0

        nStep = len(self.pArray)

        self.s.setRodPosition(['R', 'B', 'A', 'P'], [0, ] * 4, self.s.g.core_height)

        result = SimonResult(self.s.g.nxya, self.s.g.nz)
        self.s.calculateStatic(std_option)
        self.s.getResult(result)

        std_option.ppm = result.ppm
        print("p1", result.ppm)

        for iStep in range(nStep):
            std_option.plevel = self.pArray[iStep][1] / 100.0

            std_option.xenon = XE_TR
            std_option.crit = CBC

            self.s.calculateStatic(std_option)
            self.s.getResult(result)
            std_option.ppm = result.ppm
            print("p2", result.ppm)

            print("power", std_option.plevel, result.ppm)

            p_position = 381.0
            self.asisearch(std_option, self.target_ASI, iStep, p_position)
            self.put_queue.put([df.CalcOpt_ASI, self.updated, False, self.outputArray])

        self.put_queue.put([df.CalcOpt_ASI, self.finished, False, self.outputArray])

    def startShutdownRestart(self):

        self.outputArray = []

        self.s.setBurnup(self.bp)

        std_option = SteadyOption()
        std_option.maxiter = 100
        std_option.crit = CBC
        std_option.feedtf = True
        std_option.feedtm = True
        std_option.xenon = XE_EQ
        std_option.tin = df.inlet_temperature
        std_option.eigvt = self.eig
        std_option.ppm = 800.0
        std_option.plevel = 1.0
        rodIds = ['P', 'R5', 'R4', 'R3']
        nStep = len(self.pArray)

        self.s.setRodPosition(['R', 'B', 'A', 'P'], [0, ] * 4, self.s.g.core_height)
        result = SimonResult(self.s.g.nxya, self.s.g.nz)
        self.s.calculateStatic(std_option)
        self.s.getResult(result)

        std_option.ppm = result.ppm

        for iStep in range(nStep):
            # print(self.pArray[iStep][1] / 100.0, self.pArray[iStep][df.asi_i_bp:])
            std_option.plevel = self.pArray[iStep][1] / 100.0
            self.s.setRodPosition(['R', 'B', 'A', 'P'], [0, ] * 4, self.s.g.core_height)
            result = SimonResult(self.s.g.nxya, self.s.g.nz)
            self.s.calculateStatic(std_option)
            self.s.getResult(result)

            std_option.ppm = result.ppm

            for rod_index, rodId in enumerate(rodIds):
                self.s.setRodPosition1(rodId, self.pArray[iStep][df.asi_i_bp + rod_index])

            result = SimonResult(self.s.g.nxya, self.s.g.nz)
            self.s.calculateStatic(std_option)
            self.s.getResult(result)

            self.outputArray.append([result.asi, result.ppm, ]
                                    + self.pArray[iStep][df.asi_i_bp:]
                                    + [result.pow1d, self.convert_quarter_to_full(result.pow2d, df.OPR1000_MIDDLE,
                                                                                  df.OPR1000_QUAR_TO_FULL)])

            self.put_queue.put([df.CalcOpt_ASI_RESTART, self.updated, False, self.outputArray])

        self.put_queue.put([df.CalcOpt_ASI_RESTART, self.finished, False, self.outputArray])

    def startReoperation(self):

        self.outputArray = []

        self.s.setBurnup(self.pArray[0][df.asi_i_burnup])

        std_option = SteadyOption()
        std_option.maxiter = 100
        std_option.crit = CBC
        std_option.feedtf = True
        std_option.feedtm = True
        std_option.xenon = XE_EQ
        std_option.tin = df.inlet_temperature
        std_option.ppm = 800.0
        std_option.eigvt = self.pArray[0][3]
        std_option.plevel = 1.0

        self.s.setRodPosition(['R', 'B', 'A', 'P'], [0, ] * 4, self.s.g.core_height)

        result = SimonResult(self.s.g.nxya, self.s.g.nz)
        self.s.calculateStatic(std_option)
        self.s.getResult(result)

        std_option.ppm = result.ppm
        std_option.xenon = XE_TR
        std_option.plevel = self.pArray[0][df.asi_i_power] / 100.0

        self.s.setRodPosition(['P', 'R5', 'R4', 'R3'], [0] * 4, 0)

        result = SimonResult(self.s.g.nxya, self.s.g.nz)
        self.s.calculateStatic(std_option)
        self.s.getResult(result)

        self.outputArray.append([result.asi, result.ppm] +
                                self.pArray[0][df.asi_i_bp:] +
                                [result.pow1d, self.convert_quarter_to_full(result.pow2d, df.OPR1000_MIDDLE,
                                                                            df.OPR1000_QUAR_TO_FULL)]+[1,])
        std_option.ppm = result.ppm

        #decay time
        std_option.plevel = 0.0
        std_option.crit = KEFF

        self.decay(self.pArray[1][df.asi_i_time], std_option)

        std_option.plevel = self.pArray[1][df.asi_i_power] / 100.0
        std_option.crit = CBC

        self.s.setRodPosition(['P', 'R5', 'R4', 'R3'], [0] * 4, 0)

        result = SimonResult(self.s.g.nxya, self.s.g.nz)
        self.s.calculateStatic(std_option)
        self.s.getResult(result)
        self.outputArray.append([result.asi, result.ppm] +
                                self.pArray[1][df.asi_i_bp:] +
                                [result.pow1d, self.convert_quarter_to_full(result.pow2d, df.OPR1000_MIDDLE,
                                                                            df.OPR1000_QUAR_TO_FULL)]+[1,])

        std_option.plevel = self.pArray[2][df.asi_i_power] / 100.0
        ids = ['P', 'R5', 'R4', 'R3']
        for i, rodId in enumerate(ids):
            self.s.setRodPosition1(rodId, self.pArray[2][df.asi_i_bp + i])

        result = SimonResult(self.s.g.nxya, self.s.g.nz)
        self.s.calculateStatic(std_option)
        self.s.getResult(result)

        self.outputArray.append([result.asi, result.ppm] +
                                self.pArray[2][df.asi_i_bp:] +
                                [result.pow1d, self.convert_quarter_to_full(result.pow2d, df.OPR1000_MIDDLE,
                                                                            df.OPR1000_QUAR_TO_FULL)]+[1,])

        nStep = len(self.pArray)

        for iStep in range(3, nStep):
            std_option.plevel = self.pArray[iStep][df.asi_i_power] / 100.0

            std_option.xenon = XE_TR
            std_option.crit = CBC

            self.s.calculateStatic(std_option)
            self.s.getResult(result)
            std_option.ppm = result.ppm

            p_position = self.pArray[2][df.asi_i_bp]

            self.asisearchO(std_option, self.target_ASI, iStep, p_position)
            self.put_queue.put([df.CalcOpt_RO, self.updated, False, self.outputArray])

        self.put_queue.put([df.CalcOpt_RO, self.finished, False, self.outputArray])

    def startReoperationRestart(self):

        self.outputArray = []

        self.s.setBurnup(self.pArray[0][df.asi_i_burnup])

        std_option = SteadyOption()
        std_option.maxiter = 100
        std_option.crit = CBC
        std_option.feedtf = True
        std_option.feedtm = True
        std_option.xenon = XE_EQ
        std_option.tin = df.inlet_temperature
        std_option.ppm = 800.0
        std_option.eigvt = self.pArray[0][3]
        std_option.plevel = 1.0

        self.s.setRodPosition(['R', 'B', 'A', 'P'], [0, ] * 4, self.s.g.core_height)

        result = SimonResult(self.s.g.nxya, self.s.g.nz)
        self.s.calculateStatic(std_option)
        self.s.getResult(result)

        std_option.ppm = result.ppm
        std_option.xenon = XE_TR
        std_option.plevel = self.pArray[0][df.asi_i_power] / 100.0

        self.s.setRodPosition(['P', 'R5', 'R4', 'R3'], [0] * 4, 0)

        result = SimonResult(self.s.g.nxya, self.s.g.nz)
        self.s.calculateStatic(std_option)
        self.s.getResult(result)

        self.outputArray.append([result.asi, result.ppm] +
                                self.pArray[0][df.asi_i_bp:] +
                                [result.pow1d, self.convert_quarter_to_full(result.pow2d, df.OPR1000_MIDDLE,
                                                                            df.OPR1000_QUAR_TO_FULL)])
        std_option.ppm = result.ppm

        #decay
        std_option.plevel = 0.0
        std_option.crit = KEFF

        self.decay(self.pArray[1][df.asi_i_time], std_option)

        std_option.plevel = self.pArray[1][df.asi_i_power] / 100.0
        std_option.crit = CBC

        self.s.setRodPosition(['P', 'R5', 'R4', 'R3'], [0] * 4, 0)

        result = SimonResult(self.s.g.nxya, self.s.g.nz)
        self.s.calculateStatic(std_option)
        self.s.getResult(result)
        self.outputArray.append([result.asi, result.ppm] +
                                self.pArray[1][df.asi_i_bp:] +
                                [result.pow1d, self.convert_quarter_to_full(result.pow2d, df.OPR1000_MIDDLE,
                                                                            df.OPR1000_QUAR_TO_FULL)])

        std_option.plevel = self.pArray[2][df.asi_i_power] / 100.0
        ids = ['P', 'R5', 'R4', 'R3']
        for i, rodId in enumerate(ids):
            self.s.setRodPosition1(rodId, self.pArray[2][df.asi_i_bp + i])

        result = SimonResult(self.s.g.nxya, self.s.g.nz)
        self.s.calculateStatic(std_option)
        self.s.getResult(result)

        self.outputArray.append([result.asi, result.ppm] +
                                self.pArray[2][df.asi_i_bp:] +
                                [result.pow1d, self.convert_quarter_to_full(result.pow2d, df.OPR1000_MIDDLE,
                                                                            df.OPR1000_QUAR_TO_FULL)])

        rodIds = ['P', 'R5', 'R4', 'R3']
        nStep = len(self.pArray)

        for iStep in range(3, nStep):
            std_option.plevel = self.pArray[iStep][df.asi_i_power] / 100.0

            self.s.setRodPosition(['R', 'B', 'A', 'P'], [0, ] * 4, self.s.g.core_height)
            result = SimonResult(self.s.g.nxya, self.s.g.nz)
            self.s.calculateStatic(std_option)
            self.s.getResult(result)

            std_option.ppm = result.ppm

            for rod_index, rodId in enumerate(rodIds):
                self.s.setRodPosition1(rodId, self.pArray[iStep][df.asi_i_bp + rod_index])

            result = SimonResult(self.s.g.nxya, self.s.g.nz)
            self.s.calculateStatic(std_option)
            self.s.getResult(result)
            #
            self.outputArray.append([result.asi, result.ppm, ]
                                    + self.pArray[iStep][df.asi_i_bp:]
                                    + [result.pow1d, self.convert_quarter_to_full(result.pow2d, df.OPR1000_MIDDLE,
                                                                                  df.OPR1000_QUAR_TO_FULL)])

            self.put_queue.put([df.CalcOpt_RO_RESTART, self.updated, False, self.outputArray])
            # self.update_working.emit("True")

        self.put_queue.put([df.CalcOpt_RO_RESTART, self.finished, False, self.outputArray])

    def asisearch(self, std_option, target_ASI, iStep, rodP_position):

        rodids = ['P', 'R5', 'R4', 'R3'];
        overlaps = [0.0 * self.s.g.core_height, 0.6 * self.s.g.core_height, 1.2 * self.s.g.core_height]
        r5_pdil = 0.0
        # r5_pdil = 0.72 * self.s.g.core_height
        if len(self.outputArray) != 0:
            preStepPos = [self.outputArray[-1][2], self.outputArray[-1][3], self.outputArray[-1][4], self.outputArray[-1][5]]
        else:
            preStepPos = [381, 381.0, 381.0, 381.0]

        # result = SimonResult(self.s.g.nxya, self.s.g.nz)
        # std_option.crit = CBC
        # self.s.calculateStatic(std_option)
        # self.s.getResult(result)
        #
        # print("rod RR0", result.ppm, std_option.plevel)
        #
        # std_option.crit = KEFF
        # result = SimonResult(self.s.g.nxya, self.s.g.nz)
        # number_of_interval = 999
        # with open('asi{}.txt'.format(number_of_interval), 'w') as f:
        #     for i in range(number_of_interval):
        #         self.s.depleteXeSm(XE_TR, SM_TR, 100*3600/number_of_interval)
        #         self.s.calculateStatic(std_option)
        #         self.s.getResult(result)
        #         print("RR01", 100*3600/number_of_interval*(i+1), result.asi)
        #         f.write("{},{}\n".format(100*3600/number_of_interval*(i+1),result.asi))
        #
        # std_option.crit = CBC
        # self.s.calculateStatic(std_option)
        # self.s.getResult(result)
        #
        # print("rod RR1", result.ppm, std_option.plevel)
        std_option.crit = CBC
        if preStepPos[0] > 381*0.45:
            result = self.s.searchRodPositionO(std_option, target_ASI, rodids[:1], [0.0,], 0.0, preStepPos[0])

            rod_pos = result.rod_pos
            rod_pos['R5'] = preStepPos[1]
            rod_pos['R4'] = preStepPos[2]
            rod_pos['R3'] = preStepPos[3]

            if rod_pos['P'] == preStepPos[0]:
                ao_target = -target_ASI
                position = self.s.g.core_height
                result = self.s.searchRodPosition(std_option, ao_target, rodids[1:], overlaps, r5_pdil, position,
                                                  preStepPos[1:])

                rod_pos = result.rod_pos
                rod_pos['P'] = preStepPos[0]
                print("rod R", result.ppm, std_option.plevel)
            else:
                print("rod P", result.ppm, std_option.plevel)

        else:

            ao_target = -target_ASI
            position = self.s.g.core_height
            result = self.s.searchRodPosition(std_option, ao_target, rodids[1:], overlaps, r5_pdil, position, preStepPos[1:])

            rod_pos = result.rod_pos
            rod_pos['P'] = preStepPos[0]

            print("rod RR2", result.ppm, std_option.plevel)

        #print("power", std_option.plevel)
        unitArray = []
        unitArray.append(result.asi)
        unitArray.append(result.ppm)

        #unitArray.append(rodP_position)
        if (len(rod_pos) != 0):
            for rodid in rodids:
                # print(f'{rodid:12s}  :  {result.rod_pos[rodid]:12.3f}')
                unitArray.append(rod_pos[rodid])
            # unitArray.append(190.5)
        else:
            unitArray.append(190.5)
            unitArray.append(190.5)
            unitArray.append(190.5)

        unitArray.append(result.pow1d)

        unitArray.append(self.convert_quarter_to_full(result.pow2d, df.OPR1000_MIDDLE, df.OPR1000_QUAR_TO_FULL))

        self.outputArray.append(unitArray)

        # if iStep == 0:
        #     self.outputArray.pop(0)
        #self.update_working.emit("Update")

    def asisearchO(self, std_option, target_ASI, iStep, rodP_position):

        rodids = ['R5', 'R4', 'R3']
        overlaps = [0.0 * self.s.g.core_height, 0.6 * self.s.g.core_height, 1.2 * self.s.g.core_height]
        counter_strike = 0

        while True:
            counter_strike += 1

            # XESM Depletion ydnam

            std_option.crit = KEFF
            self.s.depleteXeSm(XE_TR, SM_TR,
                               (self.pArray[iStep][df.asi_i_time] - self.pArray[iStep - 1][df.asi_i_time]) * 3600)
            self.s.calculateStatic(std_option)
            # self.s.getResult(result)
            #print(f'KEFF : {result.eigv:.5f} and ASI : {result.asi:0.3f}')

            std_option.crit = CBC
            pdil = self.getPDIL(std_option.plevel*100)
            r5_pdil = pdil[0]
            if len(self.outputArray) != 0:
                preStepPos = [rodP_position, self.outputArray[-1][3], self.outputArray[-1][4], self.outputArray[-1][5], ]
            else:
                preStepPos = [rodP_position, 381.0, 381.0, 381.0, ]

            position = self.s.g.core_height
            result = self.s.searchRodPositionO(std_option, target_ASI, rodids, overlaps, r5_pdil, preStepPos[1])
            rod_pos_r5 = result.rod_pos
            is_P = False
            if result.asi >= target_ASI:
                result = self.s.searchRodPositionO(std_option, target_ASI, ['P',], [0.0,], 0.0, preStepPos[0])
                rod_pos_r5['P'] = result.rod_pos['P']
                is_P = True

            # XESM Depletion ydnam

            # if not (std_option.plevel >= .18 and (result.asi < -0.27 or result.asi > 0.27)):

            unitArray = []
            unitArray.append(result.asi)
            unitArray.append(result.ppm)

            if is_P:
                unitArray.append(rod_pos_r5['P'])
            else:
                unitArray.append(rodP_position)

            if len(result.rod_pos) != 0:
                for rodid in rodids:
                    unitArray.append(rod_pos_r5[rodid])
            else:
                unitArray.append(190.5)
                unitArray.append(190.5)
                unitArray.append(190.5)

            unitArray.append(result.pow1d)

            unitArray.append(self.convert_quarter_to_full(result.pow2d, df.OPR1000_MIDDLE, df.OPR1000_QUAR_TO_FULL))
            unitArray.append(counter_strike)

            self.outputArray.append(unitArray)
            break


    def startSDM(self):

        self.s_full.setBurnup(self.bp)

        std_option = SteadyOption()
        std_option.maxiter = 100
        std_option.crit = CBC
        std_option.feedtf = True
        std_option.feedtm = True
        std_option.xenon = XE_EQ
        std_option.tin = df.inlet_temperature
        std_option.eigvt = 1.0
        std_option.ppm = 800.0
        std_option.plevel = 1.0

        result = SimonResult(self.s_full.g.nxya, self.s_full.g.nz)

        self.s_full.setRodPosition(['R', 'B', 'A', 'P'], [0] * 4, self.s_full.g.core_height)
        self.s_full.calculateStatic(std_option)
        self.s_full.getResult(result)

        if self.mode_index > 0:

            std_option = SteadyOption()
            std_option.maxiter = 100
            std_option.crit = CBC
            std_option.feedtf = True
            std_option.feedtm = True
            std_option.xenon = XE_TR
            std_option.tin = df.inlet_temperature
            std_option.eigvt = 1 / (1 / (1.0007 + (1.00239 - 1.0007) * self.bp / self.eoc_cycle_burnup_value)
                                    + df.sdm_astra_uncertainty/100
                                    + 0.010)
            std_option.plevel = 0.0

            result = SimonResult(self.s_full.g.nxya, self.s_full.g.nz)

            self.s_full.setRodPosition(['R', 'B', 'A', 'P'], [0] * 4, 0)

            for stuck_rod in self.stuckRodPos:
                self.s_full.setRodPosition1(stuck_rod, self.s_full.g.core_height)

            self.s_full.calculateStatic(std_option)
            self.s_full.getResult(result)

            self.sdm2_ppm = result.ppm
            self.sdm2_temp = df.inlet_temperature
            self.sdm2_rsdm = self.rsdm

            self.put_queue.put([df.CalcOpt_SDM, self.finished, False,
                                self.sdm2_ppm,
                                self.sdm2_temp,
                                self.sdm2_rsdm, ])
            print(f'SHUTDOWN Boron : [{result.ppm:.2f} pcm]')
        else:

            std_option.ppm = result.ppm
            std_option.xenon = XE_TR

            rodids = ['R5', 'R4', 'R3']
            overlaps = [0, 0.4 * self.s_full.g.core_height, 0.7 * self.s_full.g.core_height]
            r5_pdil = 0.72 * self.s_full.g.core_height
            r5_pos = r5_pdil

            sdm, n1w, defect, srw = self.s_full.getShutDownMargin(std_option, rodids, overlaps, r5_pdil, r5_pos,
                                                                  self.stuckRodPos)
            sdm = sdm

            self.sdm1_srw = srw / 1000
            self.sdm1_sdm = sdm / 1000
            self.sdm1_defect = defect / 1000
            self.sdm1_n1w = n1w / 1000
            self.sdm1_temp = df.inlet_temperature
            self.sdm1_rsdm = self.rsdm

            self.put_queue.put([df.CalcOpt_SDM, self.finished, False,
                                self.sdm1_srw,
                                self.sdm1_sdm,
                                self.sdm1_defect,
                                self.sdm1_n1w,
                                self.sdm1_temp,
                                self.sdm1_rsdm])

            print(f'SHUTDOWN MARGIN : [{sdm:.2f} pcm]')

    def startECP(self):

        # 01. Set ECP Dataset
        if self.targetOpt == df.select_Boron:
            [self.pw, self.bp, self.mtc, eigen, P_Pos, r5Pos, r4Pos, shutdown_P_Pos, shutdown_r5Pos,
             shutdown_r4Pos, self.delta_time] = self.inputArray
        elif self.targetOpt == df.select_RodPos:
            [self.pw, self.bp, self.mtc, eigen, P_Pos, r5Pos, r4Pos, target_ppm, self.delta_time] = self.inputArray
        #
        self.out_ppm_old = []
        self.out_ppm = []
        self.rodPos = []
        self.p1d = []

        r3Pos = 381.0

        # 02. Calculate Conditions before shutdown
        std_option = SteadyOption()
        std_option.maxiter = 100
        std_option.crit = CBC
        # std_option.crit = KEFF
        std_option.feedtf = True
        std_option.feedtm = True
        std_option.xenon = XE_TR
        std_option.tin = self.mtc
        std_option.eigvt = eigen
        std_option.ppm = 800
        std_option.plevel = self.pw / 100.0

        self.s.setRodPosition(['R', 'B', 'A', 'P'], [0, ] * 4, self.s.g.core_height)

        self.s.setBurnup(self.bp)
        result = SimonResult(self.s.g.nxya, self.s.g.nz)
        self.s.calculateStatic(std_option)
        self.s.getResult(result)

        self.out_ppm_old.append(result.ppm)
        self.out_ppm.append(result.ppm)
        self.rodPos.append([self.s.g.core_height,]*3)
        self.p1d.append(result.pow1d)

        self.put_queue.put([df.CalcOpt_ECP, self.updated, False, self.out_ppm_old, self.out_ppm, self.rodPos, self.p1d])

        rod_IDS = ["R5", "R4", "R3"]
        rod_pos = [0.0, 0.0, 0.0]
        for idx in range(len(rod_IDS)):
            self.s.setRodPosition1(rod_IDS[idx], rod_pos[idx])

        rod_P = 'P'
        self.s.setRodPosition1(rod_P, 0.0)
        rod_IDS_OUT = ["P", "R5", "R4"]

        # 03. Calculate Conditions after shutdown

        self.s.setBurnup(self.bp)
        result = SimonResult(self.s.g.nxya, self.s.g.nz)
        self.s.calculateStatic(std_option)
        self.s.getResult(result)

        # print("ppm", result.ppm)

        self.out_ppm_old.append(result.ppm)
        self.out_ppm.append(result.ppm)
        self.rodPos.append(rod_pos)
        self.p1d.append(result.pow1d)

        # print(result.eigv)
        self.put_queue.put([df.CalcOpt_ECP, self.updated, False, self.out_ppm_old, self.out_ppm, self.rodPos, self.p1d])

        deltime = 0

        for iStep in range(len(decay_table)+1):
            # std_option.eigvt = result.eigv
            self.out_ppm_old.append(result.ppm)
            self.s.setRodPosition(['R', 'B', 'A', 'P'], [0, ] * 4, 0)

            std_option.plevel = 0.0
            std_option.crit = KEFF

            decay_len = decay_table[iStep]
            deltime += decay_len

            if deltime > self.delta_time:
                decay_len = deltime - self.delta_time

            self.s.depleteXeSm(XE_TR, SM_TR, decay_len*3600)
            self.s.calculateStatic(std_option)
            self.s.getResult(result)
            print(f'KEFF : {result.eigv:.5f} and ASI : {result.asi:0.3f}')

            std_option.crit = CBC
            self.s.setRodPosition(['R', 'B', 'A', 'P'], [0, ] * 4, 381.0)
            if self.targetOpt == df.select_Boron:
                std_option.xenon = XE_TR
                shutdown_rod_pos = [shutdown_P_Pos, shutdown_r5Pos, shutdown_r4Pos, r3Pos]
                for idx in range(len(rod_IDS_OUT)):
                    self.s.setRodPosition1(rod_IDS_OUT[idx], shutdown_rod_pos[idx])
                # result = SimonResult(self.s.g.nxya, self.s.g.nz)
                self.s.calculateStatic(std_option)
                self.s.getResult(result)

                self.out_ppm.append(result.ppm)
                self.rodPos.append([shutdown_P_Pos, shutdown_r5Pos, shutdown_r4Pos])
                self.p1d.append(result.pow1d)

            else:
                overlaps = [0.0 * self.s.g.core_height, 0.6 * self.s.g.core_height, 1.2 * self.s.g.core_height]
                r5_pdil = 0.0
                std_option.xenon = XE_TR
                self.s.setRodPosition(['R', 'B', 'A', 'P'], [0, ] * 4, 381.0)
                result = SimonResult(self.s.g.nxya, self.s.g.nz)
                self.s.calculateStatic(std_option)
                self.s.getResult(result)
                position = self.s.g.core_height
                result = self.s.search_ECP_RodPosition(std_option, target_ppm, rod_IDS, overlaps, r5_pdil, position,
                                                       rod_pos)
                rodPos_temp = []
                # print(rod_pos)
                for rodid in rod_IDS_OUT:
                    # print(f'{rodid:12s}  :  {result.rod_pos[rodid]:12.3f}')
                    rodPos_temp.append(result.rod_pos[rodid])

                self.out_ppm.append(result.ppm)
                self.rodPos.append(rodPos_temp)
                self.p1d.append(result.pow1d)

            self.put_queue.put([df.CalcOpt_ECP, self.updated, False, self.out_ppm_old, self.out_ppm, self.rodPos, self.p1d])
            if deltime >= self.delta_time:
                break

        self.put_queue.put([df.CalcOpt_ECP, self.finished, False, self.out_ppm_old, self.out_ppm, self.rodPos, self.p1d])

    def decay(self, end_time, std_option):

        del_times = []
        decay_time = 0
        for decay_index in range(len(decay_table)):

            if decay_time <= end_time <= decay_time+decay_table[decay_index]:
                break
            else:
                del_times.append(decay_table[decay_index])
            decay_time_del = decay_table[decay_index]
            decay_time += decay_time_del


        sum_all = np.sum(del_times)
        if end_time - sum_all > 0:
            del_times.append(end_time-sum_all)

        result = SimonResult(self.s.g.nxya, self.s.g.nz)
        results = []
        for deltime in del_times:
            self.s.depleteXeSm(XE_TR, SM_TR, deltime * 1 * 3600)
            self.s.calculateStatic(std_option)
            self.s.getResult(result)
            print(f'KEFF : {result.eigv:.5f} and ASI : {result.asi:0.3f}')
            results.append(result)

        return results

    def getAxialHeight(self):
        axial_position = []
        current_height = 0

        for height_index in range(self.s.g.kbc, self.s.g.kec):
            height = self.s.g.hz[height_index]
            axial_position.append((current_height + current_height + height) / 2)
            current_height += height

        self.put_queue.put([df.CalcOpt_HEIGHT,self.finished, False, axial_position, self.s.g.kbc, self.s.g.kec])

    def convert_quarter_to_full(self, quarter, middle, quart_full_index):

        full = np.zeros((middle * 2 + 1, middle * 2 + 1))
        for quarter_index, row_col in enumerate(quart_full_index):
            row = row_col[0]
            col = row_col[1]
            full[row + middle, col + middle] = quarter[quarter_index]
            full[col + middle, -row + middle] = quarter[quarter_index]
            full[-row + middle, -col + middle] = quarter[quarter_index]
            full[-col + middle, row + middle] = quarter[quarter_index]

        return full

    def getPDIL(self, power):
        if 100 >= power > 95:
            rod_position = [R5_95_65[0], R4_95_65[0], R3_42_20[0]]
        elif 95 >= power > 65:
            power_increment = 95-power
            rod_position = [R5_95_65[0]-power_increment*R5_95_65[1],
                            R4_95_65[0]-power_increment*R4_95_65[1],
                            R3_42_20[0]]
        elif 65 >= power > 50:
            power_increment = 65-power
            rod_position = [R5_65_50[0]-power_increment*R5_65_50[1],
                            R4_65_50[0]-power_increment*R4_65_50[1],
                            R3_42_20[0]]
        elif 50 >= power > 42:
            power_increment = 50-power
            rod_position = [0,
                            R4_50_42[0]-power_increment*R4_50_42[1],
                            R3_42_20[0]]
        elif 42 >= power > 20:
            power_increment = 50-power
            rod_position = [0,
                            R4_42_20[0]-power_increment*R4_42_20[1],
                            R3_42_20[0]-power_increment*R3_42_20[1]]
        elif 20 >= power >= 0:
            power_increment = 20 - power
            rod_position = [0,
                            0,
                            R3_20_0[0] - power_increment * R3_20_0[1]]
        else:
            rod_position = [381,
                            381,
                            381]
            #print("Power out of range {}".format(power))
        return rod_position
