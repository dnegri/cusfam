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


decay_table = [0.5, 0.5, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
               100, 100, 100, 100, 100, 100, 100]

initial_bps = [0.0, 50.0, 150.0, 500.0, 1000.0,]

SUCC = "Successful"
UNSUCC = "Unsuccessful"

class CalculationManager(QThread):

    SUCC = SUCC
    UNSUCC = UNSUCC

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

        self.results = OutputProcessManager(self.get_queue, self.update_working, self.finished_working)
        self.results.start()

        self.restart_file = ""
        self.plant_file = ""

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
        if len(glob.glob(user.plant_directory + user.plant_file+ "*.FF")) == 0:
            return False

        return True

    def set_names(self, user):

        self.user = user

        if not user.plant_file:
            return

        if len(user.plant_file) == 0:
            return

        if not user.restart_file:
            return

        if len(user.restart_file) == 0:
            return

        self.put_queue = mp.Queue(maxsize=10)
        # self.get_queue = mp.Queue(maxsize=10)

        # self.results.get_queue = self.get_queue
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
                 user.restart_directory + user.restart_file + ".FCE",
                 user.plant_directory + user.plant_file + ".FF",
                 ]

        self.restart_file = user.restart_file
        self.plant_file = user.plant_file

        for thread_id in range(1):
            self.agents.append(ASTRAProcess(self.put_queue, self.get_queue, files, self.cycle_burnup_values))

        for agent in self.agents:
            agent.start()

        self.axial_position = []
        self.calcOption = df.CalcOpt_HEIGHT
        self.queue.put([df.CalcOpt_HEIGHT,])

    def run(self):

        while True:
            # print("waiting")
            options = self.queue.get()
            is_successful = "Unsuccessful"
            # print("killed yo")
            calculation_option = options[0]
            if calculation_option == df.CalcOpt_ASI:
                self.setShutdownVariables(options[0], options[1], options[2], options[3], options[4])
            elif calculation_option == df.CalcOpt_ASI_RESTART:
                self.setShutdownVariables(options[0], options[1], options[2], options[3], options[4])
            elif calculation_option == df.CalcOpt_SDM:
                self.setSDMVariables(options[0], options[1], options[2], options[3], options[4], options[5])
            elif calculation_option == df.CalcOpt_RO:
                self.setReopreationVariable(options[0], options[1], options[2], options[3])
            elif calculation_option == df.CalcOpt_RO_RESTART:
                self.setReopreationVariable(options[0], options[1], options[2], options[3])
            elif calculation_option == df.CalcOpt_RECENT:
                self.setRecentCalculationVariable(options[0], options[1])
            elif calculation_option == df.CalcOpt_ECP:
                self.setECPVariables(options[0], options[1], options[2], options[3])
            elif calculation_option == df.CalcOpt_KILL:
                self.setKillOutput()
                self.finished_working.emit(is_successful)

            if calculation_option != df.CalcOpt_KILL:
                self.put_queue.put(options)
            #
            # while True:
            #
            #     outputs = self.get_queue.get()
            #
            #     calculation_option = outputs[0]
            #     calculation_result = outputs[1]
            #     calculation_error = outputs[2]
            #
            #     if calculation_option==df.CalcOpt_ASI:
            #         self.shutdown_output = outputs[3]
            #     elif calculation_option==df.CalcOpt_ASI_RESTART:
            #         self.shutdown_output = outputs[3]
            #     elif calculation_option==df.CalcOpt_RO:
            #         self.reoperation_output = outputs[3]
            #     elif calculation_option==df.CalcOpt_RO_RESTART:
            #         self.reoperation_output = outputs[3]
            #     elif calculation_option==df.CalcOpt_SDM:
            #         if len(outputs) == 6:
            #             self.setSDM345Output(outputs[3:])
            #         else:
            #             self.setSDM12Output(outputs[3:])
            #     elif calculation_option==df.CalcOpt_RECENT:
            #         self.finished = True
            #     elif calculation_option == df.CalcOpt_ECP:
            #         self.setECPOutput(outputs[3:])
            #     elif calculation_option == df.CalcOpt_HEIGHT:
            #         self.axial_position = outputs[3]
            #         self.kbc = outputs[4]
            #         self.kec = outputs[5]
            #     elif calculation_option == df.CalcOpt_KILL:
            #         self.setKillOutput()
            #
            #     if not calculation_error:
            #         is_successful = "Successful"
            #     else:
            #         is_successful = "Unsuccessful"
            #
            #     if calculation_result == self.finished or calculation_error:
            #         break
            #
            #     self.update_working.emit("True")
            #
            # self.finished_working.emit(is_successful)
            #
            # self.is_started = False


    def setRecentCalculationVariable(self,calcOpt,row):
        self.calcOption = calcOpt
        self.row = row

    def setSDMVariables(self,calcOpt,bp, stuckRod, mode_index, rsdm, pArray):
        self.calcOption = calcOpt
        self.bp = bp
        self.stuckRodPos = stuckRod
        self.mode_index = mode_index
        self.rsdm = rsdm
        self.pArray = pArray

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

    def setReopreationVariable(self,calcOpt, targetASI, pArray, snap_length):
        self.calcOption = calcOpt
        self.target_ASI = targetASI
        self.pArray = pArray
        self.snap_length = snap_length

    def setECPVariables(self,calcOpt, targetOpt, inputArray, pArray):
        self.calcOption = calcOpt
        self.targetOpt  = targetOpt
        self.inputArray = inputArray
        self.pArray = pArray

    def setKillPocessCondition(self, user):
        print(self.restart_file, user.restart_file, self.plant_file, user.plant_file)
        if self.restart_file != "":
            if self.restart_file != user.restart_file or self.plant_file != user.plant_file:
                self.setKillOutput()

    def setKillOutput(self, is_restart=False):
        user = self.user

        if len(self.agents) > 0:
            self.agents[0].kill()
        self.agents = []
        time.sleep(1)
        # self.get_queue = mp.Queue(maxsize=10)
        self.put_queue = mp.Queue(maxsize=10)

        files = [user.restart_directory + user.restart_file + ".SMG",
                 user.plant_directory + user.plant_file + ".XS",
                 user.restart_directory + user.restart_file,
                 user.restart_directory + user.restart_file + ".FCE" + ".SMG",
                 user.plant_directory + user.plant_file + ".XS",
                 user.restart_directory + user.restart_file + ".FCE",
                 user.plant_directory + user.plant_file + ".FF",
                 ]
        print(files)
        self.restart_file = user.restart_file
        self.plant_file = user.plant_file

        for thread_id in range(1):
            self.agents.append(
                ASTRAProcess(self.put_queue, self.get_queue, files, self.cycle_burnup_values))

        for agent in self.agents:
            agent.start()

    def restartProcess(self):
        self.queue.put([df.CalcOpt_KILL, self.finished, True])


class OutputProcessManager(td.Thread):

    def __init__(self, get_queue, update_working, finished_working):
        td.Thread.__init__(self)

        self.updated = "updated"
        self.finished = "finished"

        self.get_queue = get_queue

        self.axial_position = []
        self.initialized = False
        self.plant_name = ""
        self.cycle_name = ""
        self.cycle_burnup_values = []
        self.is_started = False

        self.kbc = 1
        self.kec = -1
        self.row = 0

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
        self.ecpPeak = []

        self.shutdown_output = []
        self.reoperation_output = []
        self.ecp_output = []

        self.update_working = update_working
        self.finished_working = finished_working

    def run(self):

        while True:
            outputs = self.get_queue.get()

            calculation_option = outputs[0]
            calculation_result = outputs[1]
            calculation_error = outputs[2]

            if calculation_option == df.CalcOpt_ASI:
                self.shutdown_output = outputs[3]
            elif calculation_option == df.CalcOpt_ASI_RESTART:
                self.shutdown_output = outputs[3]
            elif calculation_option == df.CalcOpt_RO:
                self.reoperation_output = outputs[3]
            elif calculation_option == df.CalcOpt_RO_RESTART:
                self.reoperation_output = outputs[3]
            elif calculation_option == df.CalcOpt_SDM:
                if len(outputs) == 6:
                    self.setSDM345Output(outputs[3:])
                else:
                    self.setSDM12Output(outputs[3:])
            elif calculation_option == df.CalcOpt_ECP:
                self.ecp_output = outputs[3]
            elif calculation_option == df.CalcOpt_HEIGHT:
                self.axial_position = outputs[3]
                self.kbc = outputs[4]
                self.kec = outputs[5]
            elif calculation_option == df.CalcOpt_RECENT:
                self.row = outputs[3]

            if not calculation_error:
                is_successful = SUCC
            else:
                is_successful = UNSUCC

            self.calcOption = outputs[0]
            if calculation_result == self.finished or calculation_error:
                self.finished_working.emit(is_successful)
            else:
                self.update_working.emit("True")

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
        self.ecpPeak = outputs[4]
        self.ecpReact = outputs[5]

class ASTRAProcess(mp.Process):
# class ASTRAProcess(td.Thread):

    def __init__(self, get_queue, put_queue, files, cycle_burnup_values):
        mp.Process.__init__(self)
        # td.Thread.__init__(self)

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
        self.calcOption = df.CalcOpt_KILL

    def run(self):
        files = self.files

        print("start")
        self.s = Simon(files[0],
                       files[1],
                       files[6],
                       files[2])
        self.s.setBurnupPoints(self.cycle_burnup_values)
        self.s_full = Simon(files[3],
                            files[4],
                            files[6],
                            files[5])
        self.s_full.setBurnupPoints(self.cycle_burnup_values)
        print("end")

        self.put_queue.put([df.CalcOpt_INIT, self.finished, False])

        while True:

            options = self.get_queue.get()
            calculation_option = options[0]

            self.outputArray = []

            if calculation_option == df.CalcOpt_ASI:
                self.setShutdownVariables(options[0], options[1], options[2], options[3], options[4])
                std_option = self.startSnapshot(self.s, options[5])
                self.startShutdown(std_option, options[5])
            if calculation_option == df.CalcOpt_ASI_RESTART:
                self.setShutdownVariables(options[0], options[1], options[2], options[3], options[4])
                std_option = self.startSnapshot(self.s, options[5])
                self.startShutdownRestart(std_option, options[5])
            elif calculation_option == df.CalcOpt_SDM:
                self.setSDMVariables(options[0], options[1], options[2], options[3], options[4], options[5])
                std_option = self.startSnapshot(self.s_full, len(options[5]))
                self.startSDM(std_option)
            elif calculation_option == df.CalcOpt_RO:
                self.setReopreationVariable(options[0], options[1], options[2], options[3])
                std_option = self.startSnapshot(self.s, options[3])
                self.startReoperation(std_option, options[3])
            elif calculation_option == df.CalcOpt_RO_RESTART:
                self.setReopreationVariable(options[0], options[1], options[2], options[3])
                std_option = self.startSnapshot(self.s, options[3])
                self.startReoperationRestart(std_option, options[3])
            elif calculation_option == df.CalcOpt_RECENT:
                self.setRecentCalculationVariable(options[0], options[1])
            elif calculation_option == df.CalcOpt_ECP:
                self.setECPVariables(options[0], options[1], options[2], options[3])
                std_option = self.startSnapshot(self.s, len(options[3]))
                self.startECP(std_option)
            elif calculation_option == df.CalcOpt_HEIGHT:
                self.getAxialHeight()

    def setRecentCalculationVariable(self, calcOpt, row):
        self.calcOption = calcOpt
        self.row = row
        self.put_queue.put([self.calcOption, self.finished, False, row])

    def setSDMVariables(self, calcOpt, bp, stuckRod, mode_index, rsdm, pArray):
        self.calcOption = calcOpt
        self.bp = bp
        self.stuckRodPos = stuckRod
        self.mode_index = mode_index
        self.rsdm = rsdm
        self.pArray = pArray

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

    def setReopreationVariable(self, calcOpt, targetASI, pArray, ro_snap_length):
        self.calcOption = calcOpt
        self.target_ASI = targetASI
        self.pArray = pArray
        self.ro_snap_length = ro_snap_length

    def setECPVariables(self, calcOpt, targetOpt, inputArray, pArray):
        self.calcOption = calcOpt
        self.targetOpt = targetOpt
        self.inputArray = inputArray
        self.pArray = pArray

    def startSnapshot(self, simon_process, end_depletion_index, ):

        simon_process.setBurnup(self.pArray[0][df.asi_i_burnup])

        std_option = SteadyOption()
        std_option.maxiter = 100
        std_option.crit = CBC
        std_option.feedtf = True
        std_option.feedtm = True
        std_option.xenon = XE_EQ
        std_option.tin = df.inlet_temperature
        # std_option.tin = 292.22
        std_option.ppm = 800.0
        # std_option.ppm = 1286.4
        std_option.eigvt = self.pArray[0][df.asi_i_keff]
        std_option.plevel = 1.0
        std_option.epsiter = 5.0E-5

        simon_process.setRodPosition(['R', 'B', 'A', 'P'], [0, ] * 4, simon_process.g.core_height)

        result = SimonResult(simon_process.g.nxya, simon_process.g.nz)
        simon_process.calculateStatic(std_option)
        simon_process.getResult(result)

        rst_burnup = 0

        for burnup in self.cycle_burnup_values:
            if self.pArray[0][df.asi_i_burnup] - 1000.0 < burnup:
                rst_burnup = burnup
                break

        current_delta_burnup = self.pArray[0][df.asi_i_burnup]-rst_burnup

        std_option.ppm = result.ppm
        std_option.xenon = XE_TR
        std_option.plevel = self.pArray[0][df.asi_i_power] / 100.0

        if len(self.pArray[0]) >= df.asi_i_bp:
            simon_process.setRodPosition(['R', 'B', 'A', 'P'], [0, ] * 4, simon_process.g.core_height)
            ids = ['P', 'R5', 'R4', 'R3']
            for i, rodId in enumerate(ids):
                simon_process.setRodPosition1(rodId, self.pArray[0][df.asi_i_bp + i])
        else:
            simon_process.setRodPosition(['R', 'B', 'A', 'P'], [0, ] * 4, simon_process.g.core_height)

        if current_delta_burnup > 0:
            simon_process.deplete(XE_EQ, SM_TR, current_delta_burnup)
            simon_process.calculateStatic(std_option)
            simon_process.getResult(result)
            std_option.ppm = result.ppm

        for i in range(1, end_depletion_index):
            current_delta_burnup = self.pArray[i][df.asi_i_burnup] - self.pArray[i-1][df.asi_i_burnup]
            std_option.ppm = result.ppm
            std_option.plevel = self.pArray[i][df.asi_i_power] / 100.0

            ids = ['P', 'R5', 'R4', 'R3']

            for k, rodId in enumerate(ids):
                simon_process.setRodPosition1(rodId, self.pArray[i][df.asi_i_bp + k])

            simon_process.deplete(XE_EQ, SM_TR, current_delta_burnup)
            simon_process.calculateStatic(std_option)
            simon_process.getResult(result)
            std_option.ppm = result.ppm

            self.put_queue.put([df.CalcOpt_CECOR, self.updated, False])

        return std_option

    def startShutdown(self, std_option, end_depletion_index):

        self.outputArray = []

        std_option.maxiter = 100
        std_option.crit = CBC
        std_option.feedtf = True
        std_option.feedtm = True
        std_option.xenon = XE_EQ
        std_option.tin = df.inlet_temperature
        std_option.eigvt = self.eig

        result = SimonResult(self.s.g.nxya, self.s.g.nz)
        self.s.calculateStatic(std_option)
        self.s.getResult(result)

        std_option.ppm = result.ppm

        nStep = len(self.pArray)

        for iStep in range(end_depletion_index, nStep):
            std_option.plevel = self.pArray[iStep][df.asi_i_power] / 100.0

            std_option.xenon = XE_TR
            std_option.crit = CBC

            self.s.calculateStatic(std_option)
            self.s.getResult(result)
            std_option.ppm = result.ppm

            p_position = 381.0
            self.asisearch(std_option, self.target_ASI, iStep, p_position)
            self.put_queue.put([df.CalcOpt_ASI, self.updated, False, self.outputArray])

        self.put_queue.put([df.CalcOpt_ASI, self.finished, False, self.outputArray])

    def startShutdownRestart(self, std_option, end_depletion_index):

        self.outputArray = []

        # self.s.setBurnup(self.bp)

        # std_option = SteadyOption()
        std_option.maxiter = 100
        std_option.crit = CBC
        std_option.feedtf = True
        std_option.feedtm = True
        std_option.xenon = XE_EQ
        std_option.tin = df.inlet_temperature
        std_option.eigvt = self.eig
        #std_option.ppm = 800.0
        std_option.plevel = 1.0
        rodIds = ['P', 'R5', 'R4', 'R3']
        nStep = len(self.pArray)

        self.s.setRodPosition(['R', 'B', 'A', 'P'], [0, ] * 4, self.s.g.core_height)
        result = SimonResult(self.s.g.nxya, self.s.g.nz)
        self.s.calculateStatic(std_option)
        self.s.getResult(result)

        std_option.ppm = result.ppm

        for iStep in range(end_depletion_index, nStep):
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

    def startReoperation(self, std_option, end_depletion_index):

        self.outputArray = []

        # self.s.setBurnup(self.pArray[end_depletion_index][df.asi_i_burnup])

        # std_option = SteadyOption()
        std_option.maxiter = 100
        std_option.crit = CBC
        std_option.feedtf = True
        std_option.feedtm = True
        std_option.xenon = XE_EQ
        std_option.tin = df.inlet_temperature
        std_option.ppm = 1247.920166015625
        std_option.eigvt = self.pArray[end_depletion_index][3]
        std_option.plevel = 1.0

        # self.s.setRodPosition(['R', 'B', 'A', 'P'], [0, ] * 4, self.s.g.core_height)
        #
        # result = SimonResult(self.s.g.nxya, self.s.g.nz)
        # self.s.calculateStatic(std_option)
        # self.s.getResult(result)

        std_option.plevel = self.pArray[end_depletion_index][df.asi_i_power] / 100.0

        self.s.setRodPosition(['P', 'R5', 'R4', 'R3'], [0] * 4, 0)

        result = SimonResult(self.s.g.nxya, self.s.g.nz)
        self.s.calculateStatic(std_option)
        self.s.calculatePinPower()
        self.s.getResult(result)
        result.fr, result.fq = 0, 0

        self.outputArray.append([result.asi, result.ppm] + [result.fr, result.fxy, result.fq] +
                                self.pArray[end_depletion_index][df.asi_i_bp:] +
                                [result.pow1d, self.convert_quarter_to_full(result.pow2d, df.OPR1000_MIDDLE,
                                                                            df.OPR1000_QUAR_TO_FULL)]+[0,std_option.plevel, 0.0])
        std_option.ppm = result.ppm

        #decay time
        std_option.plevel = 0.0
        std_option.crit = KEFF
        std_option.feedtf = False
        std_option.feedtm = False

        self.decay(self.pArray[end_depletion_index+1][df.asi_i_time], std_option)

        std_option.plevel = self.pArray[end_depletion_index+1][df.asi_i_power] / 100.0
        std_option.crit = CBC

        self.s.setRodPosition(['P', 'R5', 'R4', 'R3'], [0] * 4, 0)

        result = SimonResult(self.s.g.nxya, self.s.g.nz)
        self.s.calculateStatic(std_option)
        self.s.calculatePinPower()
        self.s.getResult(result)
        result.fr, result.fq = 0, 0

        self.outputArray.append([result.asi, result.ppm] + [result.fr, result.fxy, result.fq] +
                                self.pArray[end_depletion_index+1][df.asi_i_bp:] +
                                [result.pow1d, self.convert_quarter_to_full(result.pow2d, df.OPR1000_MIDDLE,
                                                                            df.OPR1000_QUAR_TO_FULL)]+[self.pArray[end_depletion_index+1][df.asi_i_time],std_option.plevel, 0.0])

        std_option.plevel = self.pArray[end_depletion_index+2][df.asi_i_power] / 100.0
        ids = ['P', 'R5', 'R4', 'R3']
        for i, rodId in enumerate(ids):
            self.s.setRodPosition1(rodId, self.pArray[end_depletion_index+2][df.asi_i_bp + i])

        result = SimonResult(self.s.g.nxya, self.s.g.nz)
        self.s.calculateStatic(std_option)
        self.s.calculatePinPower()
        self.s.getResult(result)
        result.fr, result.fq = 0, 0

        self.outputArray.append([result.asi, result.ppm] + [result.fr, result.fxy, result.fq] +
                                self.pArray[end_depletion_index+2][df.asi_i_bp:] +
                                [result.pow1d, self.convert_quarter_to_full(result.pow2d, df.OPR1000_MIDDLE,
                                                                            df.OPR1000_QUAR_TO_FULL)]+[0,std_option.plevel, 0.0])

        nStep = len(self.pArray)

        std_option.feedtf = True
        std_option.feedtm = True
        for iStep in range(end_depletion_index+3, nStep):
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

    def startReoperationRestart(self, std_option, end_depletion_index):

        self.outputArray = []

        # self.s.setBurnup(self.pArray[0][df.asi_i_burnup])

        # std_option = SteadyOption()
        std_option.maxiter = 100
        std_option.crit = CBC
        std_option.feedtf = True
        std_option.feedtm = True
        std_option.xenon = XE_EQ
        std_option.tin = df.inlet_temperature
        # std_option.ppm = 800.0
        std_option.eigvt = self.pArray[end_depletion_index][3]
        std_option.plevel = 1.0

        self.s.setRodPosition(['R', 'B', 'A', 'P'], [0, ] * 4, self.s.g.core_height)

        result = SimonResult(self.s.g.nxya, self.s.g.nz)
        self.s.calculateStatic(std_option)
        self.s.getResult(result)

        std_option.ppm = result.ppm
        std_option.xenon = XE_TR
        std_option.plevel = self.pArray[end_depletion_index][df.asi_i_power] / 100.0

        self.s.setRodPosition(['P', 'R5', 'R4', 'R3'], [0] * 4, 0)

        result = SimonResult(self.s.g.nxya, self.s.g.nz)
        self.s.calculateStatic(std_option)
        self.s.getResult(result)

        self.outputArray.append([result.asi, result.ppm] +
                                self.pArray[end_depletion_index][df.asi_i_bp:] +
                                [result.pow1d, self.convert_quarter_to_full(result.pow2d, df.OPR1000_MIDDLE,
                                                                            df.OPR1000_QUAR_TO_FULL)])
        std_option.ppm = result.ppm

        #decay
        std_option.plevel = 0.0
        std_option.crit = KEFF

        self.decay(self.pArray[end_depletion_index+1][df.asi_i_time], std_option)

        std_option.plevel = self.pArray[end_depletion_index+1][df.asi_i_power] / 100.0
        std_option.crit = CBC

        self.s.setRodPosition(['P', 'R5', 'R4', 'R3'], [0] * 4, 0)

        result = SimonResult(self.s.g.nxya, self.s.g.nz)
        self.s.calculateStatic(std_option)
        self.s.getResult(result)
        self.outputArray.append([result.asi, result.ppm] +
                                self.pArray[end_depletion_index+1][df.asi_i_bp:] +
                                [result.pow1d, self.convert_quarter_to_full(result.pow2d, df.OPR1000_MIDDLE,
                                                                            df.OPR1000_QUAR_TO_FULL)])

        std_option.plevel = self.pArray[end_depletion_index+2][df.asi_i_power] / 100.0
        ids = ['P', 'R5', 'R4', 'R3']
        for i, rodId in enumerate(ids):
            self.s.setRodPosition1(rodId, self.pArray[end_depletion_index+2][df.asi_i_bp + i])

        result = SimonResult(self.s.g.nxya, self.s.g.nz)
        self.s.calculateStatic(std_option)
        self.s.getResult(result)

        self.outputArray.append([result.asi, result.ppm] +
                                self.pArray[end_depletion_index+2][df.asi_i_bp:] +
                                [result.pow1d, self.convert_quarter_to_full(result.pow2d, df.OPR1000_MIDDLE,
                                                                            df.OPR1000_QUAR_TO_FULL)])

        rodIds = ['P', 'R5', 'R4', 'R3']
        nStep = len(self.pArray)

        for iStep in range(end_depletion_index+3, nStep):
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

        if len(self.outputArray) != 0:
            preStepPos = [self.outputArray[-1][df.asi_o_bp], self.outputArray[-1][df.asi_o_b5], self.outputArray[-1][df.asi_o_b4], self.outputArray[-1][df.asi_o_b3]]
        else:
            preStepPos = [381, 381.0, 381.0, 381.0]

        std_option.crit = KEFF
        # result = SimonResult(self.s.g.nxya, self.s.g.nz)

        if self.pArray[iStep][df.asi_i_time] - self.pArray[iStep - 1][df.asi_i_time] > 0:
            delta_time = self.pArray[iStep][df.asi_i_time] - self.pArray[iStep - 1][df.asi_i_time]
            delta_interval = 0.5
            start_time = 0
            while start_time < delta_time:
                new_delta = delta_interval
                if start_time + delta_interval > delta_time:
                    new_delta = delta_time - start_time
                self.s.depleteXeSm(XE_TR, SM_TR, new_delta * 3600)
                self.s.calculateStatic(std_option)
                # self.s.getResult(result)
                start_time += new_delta
        else:
            delta_time = 0
        std_option.crit = CBC
        if preStepPos[0] > 381*0.52:
            result = self.s.searchRodPositionO(std_option, target_ASI, rodids[:1], [0.0,], 0.0, preStepPos[0])

            rod_pos = result.rod_pos
            rod_pos['R5'] = preStepPos[1]
            rod_pos['R4'] = preStepPos[2]
            rod_pos['R3'] = preStepPos[3]

        else:

            ao_target = -target_ASI
            position = self.s.g.core_height
            result = self.s.searchRodPosition(std_option, ao_target, rodids[1:], overlaps, r5_pdil, position, preStepPos[1:])

            rod_pos = result.rod_pos
            rod_pos['P'] = preStepPos[0]

        result.fr, result.fq = 0, 0
        unitArray = []
        unitArray.append(result.asi)
        unitArray.append(result.ppm)
        unitArray.append(result.fr)
        unitArray.append(result.fxy)
        unitArray.append(result.fq)

        if (len(rod_pos) != 0):
            for rodid in rodids:
                unitArray.append(rod_pos[rodid])
        else:
            unitArray.append(190.5)
            unitArray.append(190.5)
            unitArray.append(190.5)

        unitArray.append(result.pow1d)

        unitArray.append(self.convert_quarter_to_full(result.pow2d, df.OPR1000_MIDDLE, df.OPR1000_QUAR_TO_FULL))

        unitArray.append(delta_time)
        unitArray.append(std_option.plevel)
        unitArray.append(0.0)

        self.outputArray.append(unitArray)

        # if iStep == 0:
        #     self.outputArray.pop(0)
        #self.update_working.emit("Update")

    def asisearchO(self, std_option, target_ASI, iStep, rodP_position):

        rodids = ['R5', 'R4', 'R3']
        overlaps = [0.0 * self.s.g.core_height, 0.6 * self.s.g.core_height, 1.2 * self.s.g.core_height]
        counter_strike = 0

        for _ in range(30):
            counter_strike += 1
            print("hello", _, counter_strike)
            # XESM Depletion ydnam

            if self.pArray[iStep][df.asi_i_time] - self.pArray[iStep - 1][df.asi_i_time] > 0:
                delta_time = self.pArray[iStep][df.asi_i_time] - self.pArray[iStep - 1][df.asi_i_time]
                delta_interval = 0.5
                start_time = 0
                while start_time < delta_time:
                    new_delta = delta_interval
                    if start_time + delta_interval > delta_time:
                        new_delta = delta_time - start_time
                    self.s.depleteXeSm(XE_TR, SM_TR, new_delta * 3600)
                    self.s.calculateStatic(std_option)
                    # self.s.getResult(result)
                    start_time += new_delta
            else:
                delta_time = 0

            std_option.crit = CBC
            pdil = ut.getPDIL(std_option.plevel*100)
            r5_pdil = pdil[0]
            if len(self.outputArray) != 0:
                preStepPos = [self.outputArray[-1][df.asi_o_bp], self.outputArray[-1][df.asi_o_b5], self.outputArray[-1][df.asi_o_b4], self.outputArray[-1][df.asi_o_b3], ]
            else:
                preStepPos = [rodP_position, 381.0, 381.0, 381.0, ]

            is_P = False
            if preStepPos[0] < 381.0:
                result = self.s.searchRodPositionO(std_option, target_ASI, ['P', ], [0.0, ], 0.0, preStepPos[0])
                rod_pos_r5 = result.rod_pos
                rod_pos_r5['P'] = result.rod_pos['P']
                is_P = True
            else:
                result = self.s.searchRodPositionO(std_option, target_ASI, rodids, overlaps, r5_pdil, preStepPos[1])
                rod_pos_r5 = result.rod_pos
            # XESM Depletion ydnam
            is_found = False
            if std_option.plevel >= .18:
                if -0.27 < result.asi < 0.27:
                    is_found = True

            else:
                if -0.60 < result.asi < 0.60:
                    is_found = True

            if is_found:
                result.fr, result.fq = 0, 0
                unitArray = []
                unitArray.append(result.asi)
                unitArray.append(result.ppm)
                unitArray.append(result.fr)
                unitArray.append(result.fxy)
                unitArray.append(result.fq)

                if is_P:
                    unitArray.append(rod_pos_r5['P'])
                    unitArray.append(self.outputArray[-1][df.asi_o_b5])
                    unitArray.append(self.outputArray[-1][df.asi_o_b4])
                    unitArray.append(self.outputArray[-1][df.asi_o_b3])
                else:
                    unitArray.append(preStepPos[0])
                    for rodid in rodids:
                        unitArray.append(rod_pos_r5[rodid])

                unitArray.append(result.pow1d)

                unitArray.append(self.convert_quarter_to_full(result.pow2d, df.OPR1000_MIDDLE, df.OPR1000_QUAR_TO_FULL))
                unitArray.append(counter_strike*(delta_time))
                unitArray.append(std_option.plevel)
                unitArray.append(0.0)

                self.outputArray.append(unitArray)
                break

    def startSDM(self, std_option):

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

    def startECP(self, std_option):

        self.outputArray = []

        # 01. Set ECP Dataset
        if self.targetOpt == df.select_Boron:
            [self.pw, self.bp, self.mtc, eigen, P_Pos, r5Pos, r4Pos, shutdown_P_Pos, shutdown_r5Pos,
             shutdown_r4Pos, self.delta_time] = self.inputArray
        elif self.targetOpt == df.select_RodPos:
            [self.pw, self.bp, self.mtc, eigen, P_Pos, r5Pos, r4Pos, target_ppm, self.delta_time] = self.inputArray
        #

        r3Pos = 381.0

        # 02. Calculate Conditions before shutdown
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

        # self.s.setBurnup(self.bp)
        result = SimonResult(self.s.g.nxya, self.s.g.nz)
        self.s.calculateStatic(std_option)
        self.s.calculatePinPower()
        self.s.getResult(result)

        result.fr, result.fq = 0, 0
        self.outputArray.append([result.asi, result.ppm] + [result.fr, result.fxy, result.fq] +
                                [self.s.g.core_height,]*4 +
                                [result.pow1d, self.convert_quarter_to_full(result.pow2d, df.OPR1000_MIDDLE,
                                                                            df.OPR1000_QUAR_TO_FULL)]+[0,std_option.plevel, 0.0])

        rod_IDS = ["R5", "R4", "R3"]
        rod_pos = [0.0, 0.0, 0.0]
        for idx in range(len(rod_IDS)):
            self.s.setRodPosition1(rod_IDS[idx], rod_pos[idx])

        rod_P = 'P'
        self.s.setRodPosition1(rod_P, 0.0)
        rod_IDS_OUT = ["P", "R5", "R4"]

        # 03. Calculate Conditions after shutdown

        #self.s.setBurnup(self.bp)
        result = SimonResult(self.s.g.nxya, self.s.g.nz)
        self.s.calculateStatic(std_option)
        self.s.calculatePinPower()
        self.s.getResult(result)

        result.fr, result.fq = 0, 0
        self.outputArray.append([result.asi, result.ppm] + [result.fr, result.fxy, result.fq] +
                                [0.0]*4 +
                                [result.pow1d, self.convert_quarter_to_full(result.pow2d, df.OPR1000_MIDDLE,
                                                                            df.OPR1000_QUAR_TO_FULL)]+[0,std_option.plevel, 0.0])


        deltime = 0

        rod_IDS = ["R5", "R4", "R3"]
        rod_pos = [381.0, 381.0, 0.0, ]

        std_option.feedtf = False
        std_option.feedtm = False

        for iStep in range(len(decay_table)+1):
            self.s.setRodPosition(['R', 'B', 'A', 'P'], [0, ] * 4, 0)

            std_option.plevel = 0.0
            std_option.crit = KEFF

            decay_len = decay_table[iStep]
            deltime += decay_len

            if deltime > self.delta_time:
                decay_len = deltime - self.delta_time

            self.s.depleteXeSm(XE_TR, SM_TR, decay_len*3600)
            # self.s.calculateStatic(std_option)
            # self.s.getResult(result)

            std_option.crit = CBC
            self.s.setRodPosition(['R', 'B', 'A', 'P'], [0, ] * 4, 381.0)
            if self.targetOpt == df.select_Boron:
                std_option.xenon = XE_TR
                shutdown_rod_pos = [shutdown_P_Pos, shutdown_r5Pos, shutdown_r4Pos, r3Pos]
                for idx in range(len(rod_IDS_OUT)):
                    self.s.setRodPosition1(rod_IDS_OUT[idx], shutdown_rod_pos[idx])
                # result = SimonResult(self.s.g.nxya, self.s.g.nz)
                self.s.calculateStatic(std_option)
                self.s.calculatePinPower()
                self.s.getResult(result)
                rodPos_temp = shutdown_rod_pos
                reactivity = 0.0

            else:
                overlaps = [0.0 * self.s.g.core_height, 0.6 * self.s.g.core_height, 1.2 * self.s.g.core_height]
                r5_pdil = 0.0
                std_option.xenon = XE_TR

                self.s.setRodPosition(['R', 'B', 'A', 'P'], [0, ] * 4, 381.0)
                result = SimonResult(self.s.g.nxya, self.s.g.nz)
                self.s.calculateStatic(std_option)
                self.s.getResult(result)
                position = self.s.g.core_height
                result = self.s.search_ECP_RodPosition_React(std_option, target_ppm, rod_IDS, overlaps, r5_pdil, position,
                                                       rod_pos)

                rod_pos = [result.rod_pos[rod_IDS[0]], result.rod_pos[rod_IDS[1]], result.rod_pos[rod_IDS[2]]]
                rodPos_temp = []
                for rodid in rod_IDS_OUT:
                    rodPos_temp.append(result.rod_pos[rodid])
                reactivity = round((1 - (1 / result.eigv)) * 100000)
                rodPos_temp.append(381.0)

            result.fr, result.fq = 0, 0
            self.outputArray.append([result.asi, result.ppm] + [result.fr, result.fxy, result.fq] +
                                    rodPos_temp +
                                    [result.pow1d, self.convert_quarter_to_full(result.pow2d, df.OPR1000_MIDDLE,
                                                                                df.OPR1000_QUAR_TO_FULL)] + [decay_len,
                                                                                                             std_option.plevel,
                                                                                                             reactivity])

            if deltime >= self.delta_time:
                break
            else:
                self.put_queue.put([df.CalcOpt_ECP, self.updated, False, self.outputArray])

        self.put_queue.put([df.CalcOpt_ECP, self.finished, False, self.outputArray])

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

