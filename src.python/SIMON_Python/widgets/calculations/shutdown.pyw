from PyQt5.QtCore import QPointF
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QSize, pyqtSlot, pyqtSignal

from model import *
import datetime

import Definitions as df

import constants as cs

from widgets.calculations.calculation_widget import CalculationWidget
from widgets.calculations.IO_table import table_IO_widget
import widgets.utils.PyShutdownTableWidget as table01
import widgets.utils.PyRodPositionSplineChart as unitSplineChart
from widgets.output.axial.axial_plot import AxialWidget
from widgets.output.radial.radial_graph import RadialWidget
import math
from widgets.output.trend.trends_graph import trendWidget
from widgets.output.trend.trends02_graph import trend02Widget
from widgets.utils.Map_Quarter import Ui_unitWidget_OPR1000_quarter as opr
from widgets.utils.Map_Quarter import Ui_unitWidget_APR1400_quarter as apr

#from ui_unitWidget_IO import output_table_widget
# import tmp0001
from widgets.utils.PySaveMessageBox import PySaveMessageBox

from itertools import islice
from random import random
from time import perf_counter
# from cusfam import *
import time
import os
import multiprocessing
from threading import Thread

import numpy as np

from PyQt5.QtCore import QPointF, QThread

from widgets.utils.splash_screen import SplashScreen
from widgets.utils.PySaveMessageBox import PySaveMessageBox, QMessageBoxWithStyle

_POINTER_A_ = "self.ui.SD_DB"
_NUM_PRINTOUT_ = 5
_MINIMUM_ITEM_COUNT = 20

pointers = {

    "SD_InputSelect": "calculation_type",
    "pushButton_InputModel": "snapshot_text",
    "SD_Input01": "ndr_burnup",
    "SD_Input02": "ndr_target_keff",
    "SD_Input03": "ndr_power_ratio",
    "SD_Input04": "ndr_power_asi",

}


class Shutdown_Widget(CalculationWidget):

    def __init__(self, db, ui, calManager, queue, message_ui):

        super().__init__(db, ui, None, SD_Input, pointers, calManager, queue, message_ui)

        self.input_pointer = "self.current_calculation.asi_input"

        self.RodPosUnit = df.RodPosUnit_cm
        # 01. Setting Initial Shutdown Input Setting
        # 01-1 Input Array Format
        self.inputArray = []
        # 01-2 Table Array Format
        self.tableArray = []
        self.outputArray = []
        self.snapshotArray = []

        # Input Dataset for SD_TableWidget Rod Position
        self.recalculationIndex = -1
        self.nStep = 0
        self.nSnapshot = 0
        self.fixedTime_flag = True
        self.ASI_flag = True
        self.CBC_flag = True
        self.Power_flag = True
        self.loadInputData = None

        # 03. Insert Table

        # self.tableItem = ["Time\n(hour)", "Burnup\n(MWD/MTU)", "Power\n(%)"  , "Keff"        , "ASI",
        #                   "Boron\n(ppm)", "Bank P\n(cm)"     , "Bank 5\n(cm)", "Bank 4\n(cm)", "Bank 3\n(cm)", ]
        # self.tableItemFormat = ["%.1f","%.1f","%.2f","%.5f","%.3f",
        #                         "%.1f","%.1f","%.1f","%.1f","%.1f"]
        self.tableItem = ["Time\n(hour)", "Power\n(%)"  ,
                          "ASI",          "Boron\n(ppm)", "Fr", "Fxy", "Fq",
                          "Bank P\n(cm)"     , "Bank 5\n(cm)", "Bank 4\n(cm)", "Bank 3\n(cm)", ]
        self.tableItemFormat = ["%.1f","%.2f",
                                "%.3f","%.1f","%.3f","%.3f","%.3f",
                                "%.1f","%.1f","%.1f","%.1f"]
        self.SD_TableWidget = table01.ShutdownTableWidget(self.ui.frame_SD_OutputWidget, self.tableItem, self.tableItemFormat)
        self.IO_table = table_IO_widget()
        # 03-1 Hide Dataset
        self.SD_TableWidget.hide()
        self.ui.LabelSub_SD02.hide()
        self.ui.SD_Input02.hide()
        # self.layoutTableButton = self.SD_TableWidget.returnButtonLayout()

        # self.gridLayout_SD_TABLE = QtWidgets.QGridLayout()
        # self.gridLayout_SD_TABLE.setObjectName("gridLayout_SD_TABLEWIDGET_DEFINED")
        #self.ui.gridLayout_SD_TABLE.addWidget(self.SD_TableWidget, 0, 0, 1, 1)
        #self.ui.gridLayout_SD_TABLE.addLayout(self.layoutTableButton, 1, 0, 1, 1)
        #self.ui.gridlayout_SD_TableWidget.addLayout(self.gridLayout_SD_TABLE, 0, 0, 1, 1)

        #self.ui.gridlayout_SD_TableWidget.addLayout(layoutTableButton,0,0,2,1)

        # [self.unitButton01, self.unitButton02, self.unitButton03,] = self.SD_TableWidget.returnTableButton()

        self.unitChart = None
        self.unitChart02 = None
        self.radialWidget = None
        self.radialWidget_opr1000 = None
        self.radialWidget_apr1400 = None
        self.axialWidget = None

        self.map_opr1000 = None
        self.map_opr1000_frame = None
        self.map_opr1000_grid = None
        self.map_apr1400 = None
        self.map_apr1400_frame = None
        self.map_apr1400_grid = None

        self.addOutput()


        # 07. Setting Widget Interactions
        self.buttonGroupInput = QtWidgets.QButtonGroup()
        self.reductionGroupInput = QtWidgets.QButtonGroup()
        # self.settingAutoExclusive()
        self.settingLinkAction()

        # 08. Setting Input Type
        self.ui.SD_InputSelect.setCurrentIndex(df._INPUT_TYPE_USER_)
        self.inputType = df._INPUT_TYPE_USER_
        self.ui.LabelSub_Selete01.setVisible(False)
        self.ui.pushButton_InputModel.setVisible(False)

        #self.ui.LabelSub_Selete01.hide()

        # self.load()
        self.delete_current_calculation()

        self.ui.SD_run_button.setText("Run")
        self.ui.SD_run_button.setStyleSheet(df.styleSheet_Run)
        self.ui.SD_run_button.setDisabled(False)



        # sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        # # sizePolicy.setHorizontalStretch(0)
        # # sizePolicy.setVerticalStretch(200)
        # sizePolicy.setHeightForWidth(True)
        # self.ui.SD_InputLP_frame.setSizePolicy(sizePolicy)
        # #sizePolicy.setHeightForWidth(self.ui.SD_InputLP_frame.sizePolicy().hasHeightForWidth())
        # #self.ui.SD_InputLP_frame.setSizePolicy(sizePolicy)
    def linkInputModule(self,module):
        self.loadInputData = module

    def load(self, a_calculation=None):

        self.load_input(a_calculation)
        self.load_output(a_calculation)
        self.load_snapshot(a_calculation)

        self.ui.SD_Input01.setMaximum(self.calcManager.cycle_burnup_values[-1] + 1000)

        # if len(self.inputArray) == 0:
            # self.ui.SD_run_button.setText("Create Scenario")
            # self.ui.SD_run_button.setStyleSheet(df.styleSheet_Create_Scenarios)
            # self.ui.SD_run_button.setDisabled(False)
        # else:
        #     self.ui.SD_run_button.setText("Run")
        #     self.ui.SD_run_button.setStyleSheet(df.styleSheet_Run)
        #     self.ui.SD_run_button.setDisabled(False)

    def settingLinkAction(self):
        self.ui.pushButton_InputModel.clicked['bool'].connect(self.readModel)

        self.ui.SD_run_button.clicked['bool'].connect(self.start_calc)
        self.ui.SD_IO_button.clicked['bool'].connect(self.open_IO_table)
        self.ui.SD_save_button.clicked['bool'].connect(self.start_save)

        # self.unitButton01.clicked['bool'].connect(self.clickSaveAsExcel)
        # self.unitButton02.clicked['bool'].connect(self.resetPositionData)

        #self.unitChart.return_x_pos.connect(self.get_xxxx_pos)

        # self.SD_TableWidget.itemSelectionChanged.connect(self.cell_changed)
        # self.unitButton03.clicked['bool'].connect(self.clearOuptut)

        self.ui.SD_InputSelect.currentIndexChanged['int'].connect(self.changeInputType)
        # self.ui.tabWidget_Shutdown.currentChanged['int'].connect(self.changeTabCurrent)

        self.unitChart.canvas.mpl_connect('pick_event', self.clickEvent01)
        self.unitChart02.canvas.mpl_connect('pick_event', self.clickEvent02)

        # self.unitChart.canvas.mpl_connect('button_press_event', self.clickEvent01)
        # self.unitChart02.canvas.mpl_connect('button_press_event', self.clickEvent02)

        #self.ui.

    def open_IO_table(self):
        self.IO_table.open_IO_table()

    def changeTabCurrent(self,idx):
        if(idx==1):
            tmp = min(self.ui.SD_InputLP_Dframe.height(), self.ui.SD_InputLP_Dframe.width())
            self.ui.SD_InputLP_frame.setMaximumSize(QSize(tmp, tmp))

    def changeInputType(self,idx):
        self.inputType = idx
        if(idx==df._INPUT_TYPE_USER_):
            self.ui.LabelSub_Selete01.setVisible(False)
            self.ui.pushButton_InputModel.setVisible(False)
        elif(idx==df._INPUT_TYPE_SNAPSHOT_):
            self.ui.LabelSub_Selete01.setText("Snapshot Setup")
            self.ui.pushButton_InputModel.setText("Load Data")
            self.ui.pushButton_InputModel.setStyleSheet(df.styleSheet_Create_Scenarios)
            self.ui.LabelSub_Selete01.setVisible(True)
            self.ui.pushButton_InputModel.setVisible(True)
        elif(idx==df._INPUT_TYPE_FILE_CSV_):
            self.ui.LabelSub_Selete01.setText("Read File")
            self.ui.pushButton_InputModel.setText("Load CSV")
            self.ui.LabelSub_Selete01.setVisible(True)
            self.ui.pushButton_InputModel.setVisible(True)
        elif(idx==df._INPUT_TYPE_FILE_EXCEL_):
            self.ui.LabelSub_Selete01.setText("Read File")
            self.ui.pushButton_InputModel.setText("Load Excel")
            self.ui.LabelSub_Selete01.setVisible(True)
            self.ui.pushButton_InputModel.setVisible(True)

    def readModel(self):
        if(self.inputType==df._INPUT_TYPE_SNAPSHOT_):
            for key_plant in cs.DEFINED_PLANTS.keys():
                if cs.DEFINED_PLANTS[key_plant] in self.calcManager.plant_name:
                    plant_name = key_plant+self.calcManager.plant_name[len(cs.DEFINED_PLANTS[key_plant]):]

            cecor_message = self.loadInputData.read_cecor_output(self.current_calculation.user, plant_name, self.calcManager.cycle_name)

            if len(cecor_message) > 0:
                QtWidgets.qApp.processEvents()
                msgBox = QMessageBoxWithStyle(self.get_ui_component())
                msgBox.setWindowTitle(cs.UNABLE_CECOR_OUTPUT)
                msgBox.setText(cecor_message)
                msgBox.setStandardButtons(QMessageBox.Ok)
                msgBox.setCustomStyle()
                result = msgBox.exec_()
                return

            self.loadInputData.readSnapshotData()
            [self.nSnapshot, inputArray] = self.loadInputData.returnSnapshot()
            self.snapshotArray = inputArray

            if len(inputArray) > 0:
                burnup_text = "B:{:d}".format(int(inputArray[-1][1]))
                self.ui.pushButton_InputModel.setText(burnup_text)
                self.ui.pushButton_InputModel.setStyleSheet(df.styleSheet_Run)
                self.current_calculation.sd_input.snapshot_text = burnup_text
                self.ui.SD_Input01.setValue(inputArray[-1][1])
                self.save_snapshot()

        elif(self.inputType==df._INPUT_TYPE_FILE_CSV_):


            status = self.loadInputData.openCSV()

            self.tableArray = []
            if(status==False):
                return
            [ self.nStep, self.targetASI,  self.inputArray ] = self.loadInputData.returnCSV()
            self.initBU = self.inputArray[0][2]
            self.targetEigen = self.inputArray[0][3]
            rdcPerHour = abs((self.inputArray[1][1]-self.inputArray[0][1])/(self.inputArray[1][0]-self.inputArray[0][0]))
            self.ui.SD_Input01.setValue(self.initBU)
            self.ui.SD_Input02.setValue(self.targetEigen)
            self.ui.SD_Input03.setValue(rdcPerHour)
            self.ui.SD_Input04.setValue(self.targetASI)
            # if len(self.inputArray) == 0:
            #     self.ui.SD_run_button.setText("Create Scenario")
            #     self.ui.SD_run_button.setStyleSheet(df.styleSheet_Create_Scenarios)
            #     self.ui.SD_run_button.setDisabled(False)
            # else:
            #     self.ui.SD_run_button.setText("Run")
            #     self.ui.SD_run_button.setStyleSheet(df.styleSheet_Run)
            #     self.ui.SD_run_button.setDisabled(False)


            for idx in range(self.nStep):
                unitArray = []
                unitArray.append([True,False,False])
                unitArray.append([self.inputArray[idx][0],self.inputArray[idx][1]])
                unitArray.append([0.0,0.0,0.0,0.0,0.0])
                unitArray.append([0.0,0.0,0.0,0.0])
                self.tableArray.append(unitArray)
            self.SD_TableWidget.add_input_array(self.tableArray)
            self.IO_table.IO_TableWidget.add_input_array(self.tableArray)
            #self.SD_TableWidget.addInputArray(self.inputArray)
            self.last_table_created = datetime.datetime.now()

    def resizeRadialWidget(self,size):
        self.map_opr1000_frame.setMaximumSize(QSize(size, size))
        self.map_apr1400_frame.setMaximumSize(QSize(size, size))

    def get_input(self, calculation_object):
        return calculation_object.sd_input

    def get_output(self, calculation_object):
        return calculation_object.sd_output

    def get_manager_output(self):
        return self.calcManager.results.shutdown_output

    def set_manager_output(self, output):
        self.calcManager.results.shutdown_output = output

    def create_default_input_output(self, user):
        now = datetime.datetime.now()
        sd_input = SD_Input.create()
        sd_output = SD_Output.create()
        SD_calculation = Calculations.create(user=user,
                                             calculation_type=cs.CALCULATION_SD,
                                             created_date=now,
                                             modified_date=now,
                                             sd_input=sd_input,
                                             sd_output=sd_output
                                             )
        return SD_calculation, sd_input, sd_output

    def setSuccecciveInput(self):

        self.clearOutput()
        #01. Initialize
        #01-1. U    ser Input Case: Initialize
        self.inputArray = []
        addTime = 0.0
        # if(self.inputType != df._INPUT_TYPE_SNAPSHOT_):
        #
        # else:
        #     addTime = self.inputArray[-1][0]

        # Initialize
        self.inputArray = []

        initBU = self.ui.SD_Input01.value()
        targetEigen = self.ui.SD_Input02.value()
        targetASI = self.ui.SD_Input04.value()

        self.initBU = initBU
        self.targetEigen = targetEigen
        self.targetASI = targetASI

        rdcPerHour = self.ui.SD_Input03.value()
        EOF_Power = 0.0  # self.ui.rdc04.value()
        self.rdcPerHour = rdcPerHour
        # TODO, make Except loop
        if (rdcPerHour == 0.0):
            print("Error!")
            return

        if rdcPerHour > 3.0:
            powerPerTimes = []
            start_power = 0.0
            while start_power < rdcPerHour:
                delta_power = 3.0
                if delta_power + start_power > rdcPerHour:
                    delta_power = rdcPerHour - start_power
                powerPerTimes.append(delta_power)
                start_power += delta_power
        else:
            powerPerTimes = [rdcPerHour,]

        powers = []
        currentPower = 100.0
        while currentPower > EOF_Power:
            for power in powerPerTimes:
                currentPower -= power
                if currentPower < EOF_Power:
                    currentPower = EOF_Power
                powers.append(currentPower)

        nStep = len(powers)

        self.recalculationIndex = nStep

        #self.unitChart.adjustTime(overlap_time)
        # update Power Variation Per Hour and power increase flag
        # ( Power Ascention Mode == True, Power Reduction Mode == False)
        self.unitChart.updateRdcPerHour(rdcPerHour,False)
        self.unitChart.resizeMaxTimeAxes(100/rdcPerHour)
        self.unitChart02.resizeMaxTimeAxes(100/rdcPerHour)

        for i in range(nStep//len(powerPerTimes)):
            start_power = -power
            for j, power in enumerate(powerPerTimes):
                start_power += power
                time = start_power/rdcPerHour + i
                unitArray = [ time, powers[i*len(powerPerTimes)+j], initBU, targetEigen ]
                self.inputArray.append(unitArray)

        self.nStep = nStep
        # if(self.inputType == df._INPUT_TYPE_SNAPSHOT_):
        #     self.SD_TableWidget.addSnapshotInputArray(self.nSnapshot,self.inputArray)
        # else:
        self.SD_TableWidget.addInputArray(self.inputArray)
        self.IO_table.IO_TableWidget.addInputArray(self.inputArray)
        self.last_table_created = datetime.datetime.now()

    def clickSaveAsExcel(self):
        is_succ = self.SD_TableWidget.clickSaveAsExcel()
        if not is_succ:
            msgBox = QMessageBoxWithStyle(self.get_ui_component())
            msgBox.setWindowTitle("Output not found")
            msgBox.setText("Output not found to save to excel")
            msgBox.setStandardButtons(QMessageBox.Ok)
            msgBox.setCustomStyle()
            result = msgBox.exec_()

    def resetPositionData(self):
        self.SD_TableWidget.resetPositionData()
        self.IO_table.IO_TableWidget.resetPositionData()

    ##################################
    ## Shutdown Calculation Routine ##
    ##################################
    # def startCalc(self):
    #    if(multiprocessing.cpu_count()>1):
    #        self.multiprocessingFlag = True
    #        self.thread001 = Thread(target=self.ASI_Calculation, args=())
    #        self.thread001.start()
    #    else:
    #        self.multiprocessingFlag = False
    #        self.ASI_Calculation()

    def start_calc(self):

        super().start_calc()

        if self.unitChart:
            self.unitChart.clearData()

        if self.SD_TableWidget:
            self.SD_TableWidget.clearOutputArray()
            self.IO_table.IO_TableWidget.clearOutputArray()

        self.current_calculation.sd_output.success = False

        self.setSuccecciveInput()
        self.load()
        calcOpt, initBU, targetEigen, targetASI, pArray, snap_length, error = self.get_calculation_input()

        #self.unitChart.axisX.setMax(len(pArray))
        if not error and calcOpt != df.CalcOpt_KILL:
            self.start_calculation_message = SplashScreen()
            self.start_calculation_message.killed.connect(self.killManagerProcess)
            self.start_calculation_message.init_progress(len(pArray), 500)

            # self.ui.SD_run_button.setText(cs.RUN_BUTTON_RUNNING)
            # self.ui.SD_run_button.setDisabled(True)
            self.SD_TableWidget.last_update = 0
            self.IO_table.IO_TableWidget.last_update = 0
            # self.calcManager.setShutdownVariables(calcOpt, self.initBU, self.targetEigen, self.targetASI, pArray)
            self.SD_TableWidget.clearOutputArray()
            self.IO_table.IO_TableWidget.clearOutputArray()
            self.queue.put((calcOpt, initBU, targetEigen, targetASI, pArray, snap_length))

    def finished(self, is_success):

        self.is_run_selected = False

        if self.start_calculation_message:
            self.start_calculation_message.close()

        #self.definePointData(self.calcManager.outputArray)
        self.SD_TableWidget.selectRow(0)
        self.IO_table.IO_TableWidget.selectRow(0)

        self.ui.SD_run_button.setText("Run")
        self.ui.SD_run_button.setDisabled(False)
        # pd2d = []
        pd1d = [-1]
        # rp = []
        # r5 = []
        # r4 = []
        # r3 = []
        for row in range(len(self.calcManager.results.shutdown_output)):
            # pd2d.append(self.calcManager.results.shutdown_output[row][df.asi_o_p2d])
            if self.calcManager.results.shutdown_output[row][df.asi_o_power] > 0:
                pd1d.append(max(self.calcManager.results.shutdown_output[row][df.asi_o_p1d][self.calcManager.results.kbc:self.calcManager.results.kec]))

            # pd1d.append(max(self.calcManager.results.shutdown_output[row][df.asi_o_p1d][self.calcManager.results.kbc:self.calcManager.results.kec]))

            # rp.append(self.calcManager.results.shutdown_output[row][df.asi_o_bp])
            # r5.append(self.calcManager.results.shutdown_output[row][df.asi_o_b5])
            # r4.append(self.calcManager.results.shutdown_output[row][df.asi_o_b4])
            # r3.append(self.calcManager.results.shutdown_output[row][df.asi_o_b3])

        self.axialWidget.setMaximumPower(max(pd1d))

        if is_success == self.calcManager.SUCC:
            self.save_output(self.calcManager.results.shutdown_output)
            self.showOutput()

        # self.cell_changed()

    def showOutput(self):
        if self.start_calculation_message:
            self.start_calculation_message.progress()

        if len(self.calcManager.results.shutdown_output) > 0:

            # last_update = self.SD_TableWidget.last_update
            # last_update = self.IO_table.IO_TableWidget.last_update
            last_update = 0

            self.SD_TableWidget.appendOutputTable(self.calcManager.results.shutdown_output[last_update:], last_update)
            self.IO_table.IO_TableWidget.appendOutputTable(self.calcManager.results.shutdown_output[last_update:], last_update)
            self.appendPointData(self.calcManager.results.shutdown_output, 0)

            # outputs = self.calcManager.results.shutdown_output
            # pd2d = outputs[-1][df.asi_o_p2d]
            # pd1d = outputs[-1][df.asi_o_p1d]
            # self.axialWidget.drawAxial(pd1d[self.calcManager.results.kbc:self.calcManager.results.kec],
            #                            self.calcManager.results.axial_position)
            # self.radialWidget.slot_astra_data(pd2d)


    def get_calculation_input(self):
        calcOpt = df.CalcOpt_ASI
        if self.SD_TableWidget.checkModified():
            msgBox = QMessageBoxWithStyle(self.get_ui_component())
            msgBox.setWindowTitle("Rod Modification Detected")
            msgBox.setText("Rod: Rod Position Search (Target ASI)\n"
                           "CBC: Boron Search with (Target Rod Position)")
            msgBox.addButton(QPushButton('Rod Search'), QMessageBox.YesRole)
            msgBox.addButton(QPushButton('CBC Search'), QMessageBox.NoRole)
            msgBox.addButton(QPushButton('Cancel'), QMessageBox.NoRole)
            # msgBox.setWindowFlag(QtCore.Qt.FramelessWindowHint)
            msgBox.setCustomStyle()
            result = msgBox.exec_()
            if result == 1:
                calcOpt = df.CalcOpt_ASI_RESTART
            elif result == 2:
                calcOpt = df.CalcOpt_KILL

        if calcOpt == df.CalcOpt_ASI_RESTART:
            _, initBU, targetEigen, targetASI, pArray, snap_length, error = self.getInputRestart()
        else:
            _, initBU, targetEigen, targetASI, pArray, snap_length, error = self.getInput()

        return calcOpt, initBU, targetEigen, targetASI, pArray, snap_length, error

    def getInput(self):

        pArray = []

        snap_length = len(self.snapshotArray)

        if self.ui.SD_InputSelect.currentIndex() == 1:
            for value in self.snapshotArray:
                pArray.append([value[0], value[2], value[1], value[3]]+ [0, 0, 0] +value[4:])

        print("shutdown pArray", pArray)

        for iStep in range(self.nStep):
            pArray.append(self.inputArray[iStep])
        error = self.check_calculation_input()
        return df.CalcOpt_ASI, self.initBU, self.targetEigen, self.targetASI, pArray, snap_length, error

    def getInputRestart(self):

        rod_array = self.SD_TableWidget.getRodValues()
        if self.nStep != len(rod_array):
            raise ValueError("Program Error length mismatch shutdown")
        pArray = []

        snap_length = len(self.snapshotArray)

        for value in self.snapshotArray:
            pArray.append([value[0], value[2], value[1]]+value[3:])

        for iStep in range(self.nStep):
            pArray.append(self.inputArray[iStep] + [0,0,]+rod_array[iStep])
        error = self.check_calculation_input()
        return df.CalcOpt_ASI_RESTART, self.initBU, self.targetEigen, self.targetASI, pArray, snap_length, error

    def check_calculation_input(self):

        if self.ui.SD_Input01.value() >= self.calcManager.cycle_burnup_values[-1]+1000:
            msgBox = QMessageBoxWithStyle(self.get_ui_component())
            msgBox.setWindowTitle("Burnup Out of Range")
            msgBox.setText("{}MWD/MTU excedes EOC Cycle Burnup({} MWD/MTU)\n"
                           "Put Cycle Burnup less than {}MWD/MTU".format(self.ui.SD_Input01.value(),
                                                            self.calcManager.cycle_burnup_values[-1],
                                                            self.calcManager.cycle_burnup_values[-1]+1000))
            msgBox.setStandardButtons(QMessageBox.Ok)
            msgBox.setCustomStyle()
            #msgBox.setWindowFlag(QtCore.Qt.FramelessWindowHint)
            result = msgBox.exec_()
            # if result == QMessageBox.Cancel:
            return True
        return False

    def definePointData(self, outputArray):
        posR5 = []
        posR4 = []
        posR3 = []
        posCBC = []

        for iStep in range(self.nStep):
            # dt = (1.0 - self.debugData01) / 0.03
            posR5.append(QPointF(self.inputArray[iStep][0], outputArray[iStep][df.asi_o_b5]))
            posR4.append(QPointF(self.inputArray[iStep][0], outputArray[iStep][df.asi_o_b4]))
            posR3.append(QPointF(self.inputArray[iStep][0], outputArray[iStep][df.asi_o_b3]))
            posCBC.append(QPointF(self.inputArray[iStep][0], outputArray[iStep][df.asi_o_boron]))

        rodPos = [posR5, posR4, posR3]

        #self.unitChart.replaceRodPosition(3, rodPos, posCBC)

    def appendPointData(self, outputArray, start):

        # self.unitChartClass.clear()
        num = len(outputArray)

        if num == 0:
            return
        try:

            time = []
            power = []

            posP = []
            posR5 = []
            posR4 = []
            posR3 = []
            posASI = []
            posCBC = []

            pos_Fxy = []
            pos_Fr = []
            pos_Fq = []
            current_time = 0
            asi_band_time = 0
            for i in range(len(outputArray)):
                posP.append(outputArray[i][df.asi_o_bp])
                posR5.append(outputArray[i][df.asi_o_b5])
                posR4.append(outputArray[i][df.asi_o_b4])
                posR3.append(outputArray[i][df.asi_o_b3])
                posASI.append(outputArray[i][df.asi_o_asi])
                posCBC.append(outputArray[i][df.asi_o_boron])
                pos_Fxy.append(outputArray[i][df.asi_o_fxy])
                pos_Fr.append(-1.0)#outputArray[i][df.asi_o_fr])
                #pos_Fr.append(outputArray[i][df.asi_o_fr])
                pos_Fq.append(-1.0)
                current_time += outputArray[i][df.asi_o_time]
                time.append(current_time)
                power.append(outputArray[i][df.asi_o_power]*100)
                if power[-1] < 20.0:
                    asi_band_time += outputArray[i][df.asi_o_time]
                #pos_Fq.append(outputArray[i][df.asi_o_fq])

            self.unitChart.insertTime(time)
            self.unitChart02.insertTime(time)

            #rodPos = [posP, posR5, posR4, posR3, ]
            posOpt = [ posASI, posCBC, power ]
            posOpt02 = [ posASI, posCBC, power  ]

            self.unitChart.updateRdcPerHour(self.rdcPerHour,False, asi_band_time)
            self.unitChart.insertDataSet(time, posP, posR5, posR4, posOpt )
            self.unitChart02.insertDataSet(time, pos_Fxy, pos_Fr, pos_Fq, posOpt02 )
        except:
            pass
        #self.unitChartClass.appendRodPosition(len(rodPos), rodPos, posC

    def empty_output(self):

        sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.ui.SD_widgetChart.sizePolicy().hasHeightForWidth())
        self.ui.SD_widgetChart.setSizePolicy(sizePolicy5)

        sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.ui.SD_WidgetRadial.sizePolicy().hasHeightForWidth())
        self.ui.SD_WidgetRadial.setSizePolicy(sizePolicy5)

        sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.ui.SD_WidgetAxial.sizePolicy().hasHeightForWidth())
        self.ui.SD_WidgetAxial.setSizePolicy(sizePolicy5)

    def addOutput(self):

        if not self.unitChart:

            lay = self.ui.grid_SD_frameChart

            self.unitChart = trendWidget(self.fixedTime_flag, self.ASI_flag, False, True)#self.CBC_flag, self.Power_flag)#self.ui.SD_widgetChart)

            sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            sizePolicy5.setHorizontalStretch(10)
            sizePolicy5.setVerticalStretch(10)
            #sizePolicy5.set
            #sizePolicy5.setHeightForWidth(self.ui.SD_widgetChart.sizePolicy().hasHeightForWidth())
            self.unitChart.setSizePolicy(sizePolicy5)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(self.unitChart, 0, 0, 1, 1)

        #


        if not self.unitChart02:

            lay02 = self.ui.grid_SD_frameChart02
            # lay = QtWidgets.QVBoxLayout(self.ui.SD_widgetChart)
            # lay.setContentsMargins(0, 0, 0, 0)
            self.unitChart02 = trend02Widget(self.fixedTime_flag, False, True, False)#self.ui.SD_widgetChart)
            #unitChartView = self.unitChart.returnChart()
            sizePolicy6 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            sizePolicy6.setHorizontalStretch(10)
            sizePolicy6.setVerticalStretch(10)
            #sizePolicy5.set
            #sizePolicy5.setHeightForWidth(self.ui.SD_widgetChart.sizePolicy().hasHeightForWidth())
            self.unitChart02.setSizePolicy(sizePolicy6)
            lay02.setContentsMargins(0, 0, 0, 0)
            #self.unitChart.insertDataSet()

            # a = QWidget()
            # self.aa = tmp0001.Ui_TEST_WIDGET()
            # self.aa.setupUi(a)
            #lay.addWidget(a, 0, 0, 1, 1)

            lay02.addWidget(self.unitChart02, 0, 0, 1, 1)

        if not self.radialWidget:
            # 04. Insert Radial Chart
            # sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            # sizePolicy5.setHorizontalStretch(6)
            # sizePolicy5.setVerticalStretch(10)
            # sizePolicy5.setHeightForWidth(self.ui.SD_WidgetRadial.sizePolicy().hasHeightForWidth())
            # self.ui.SD_WidgetRadial.setSizePolicy(sizePolicy5)
            # layR = QtWidgets.QVBoxLayout(self.ui.SD_WidgetRadial)
            # layR.setContentsMargins(0, 0, 0, 0)
            # self.radialWidget = RadialWidget()
            # layR.addWidget(self.radialWidget)
            # self.SD_InputLP_frame = QtWidgets.QFrame(self.SD_InputLP_Dframe)
            # self.SD_InputLP_frame = opr1000map(self.SD_InputLP_frame)

            #self.ui.SD_InputLPgrid.removeWidget()



            self.map_opr1000 = opr(self.ui.SD_InputLP_Dframe,self.ui.gridLayout_SD_InputLP_Dframe)
            self.map_opr1000_frame , self.map_opr1000_grid = self.map_opr1000.return_opr_frame()
            self.ui.gridLayout_SD_InputLP_Dframe.addWidget(self.map_opr1000_frame , 0, 0, 1, 1)
            self.radialWidget_opr1000 = RadialWidget(self.map_opr1000_frame , self.map_opr1000_grid, df.type_opr1000)
            # self.map_opr1000_frame.hide()

            self.map_apr1400 = apr(self.ui.SD_InputLP_Dframe,self.ui.gridLayout_SD_InputLP_Dframe)
            self.map_apr1400_frame, self.map_apr1400_grid = self.map_apr1400.return_apr_frame()
            self.ui.gridLayout_SD_InputLP_Dframe.addWidget(self.map_apr1400_frame, 0, 0, 1, 1)
            self.radialWidget_apr1400 = RadialWidget(self.map_apr1400_frame,self.map_apr1400_grid, df.type_apr1400)
            self.map_apr1400_frame.hide()

            self.radialWidget = self.radialWidget_opr1000

        if not self.axialWidget:
            pass
            # 05. Insert Axial Chart
            sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            sizePolicy5.setHorizontalStretch(6)
            sizePolicy5.setVerticalStretch(10)
            #sizePolicy5.setHeightForWidth(self.ui.SD_WidgetAxial.sizePolicy().hasHeightForWidth())
            self.ui.SD_WidgetAxial.setSizePolicy(sizePolicy5)
            #layA = self.ui.SD_WidgetAxial
            layA = QtWidgets.QVBoxLayout(self.ui.SD_WidgetAxial)
            layA.setContentsMargins(0, 0, 0, 0)
            self.axialWidget = AxialWidget()
            layA.addWidget(self.axialWidget)#0,0,1,1)
            #self.SD_WidgetAxial.setMaximumSize(QtCore.QSize(200, 16777215))
            #self.axialWidget



    #####################
    #### TEST ROUTINE ###
    #####################

    def clearOutput(self):
        self.SD_TableWidget.clearOutputArray()
        self.SD_TableWidget.clearOutputRodArray()
        self.SD_TableWidget.last_update = 0
        self.unitChart.clearData()
        self.unitChart02.clearData()