from PyQt5.QtCore import QPointF
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtWidgets import *

from model import *
import datetime

import Definitions as df

import constants as cs

from widgets.calculations.calculation_widget import CalculationWidget

import widgets.utils.PyShutdownTableWidget as table01
import widgets.utils.PyRodPositionSplineChart as unitSplineChart
from widgets.output.axial.axial_plot import AxialWidget
from widgets.output.radial.radial_graph import RadialWidget
import math
from widgets.output.trend.trends_graph import trendWidget
# import tmp0001
from widgets.utils.PySaveMessageBox import PySaveMessageBox

from itertools import islice
from random import random
from time import perf_counter
from cusfam import *
import time
import os
import multiprocessing
from threading import Thread

import numpy as np

from PyQt5.QtCore import QPointF, QThread

from widgets.utils.splash_screen import SplashScreen

_POINTER_A_ = "self.ui.SD_DB"
_NUM_PRINTOUT_ = 5
_MINIMUM_ITEM_COUNT = 20

pointers = {

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
        self.inputArray = []
        self.outputArray = []

        # Input Dataset for SD_TableWidget Rod Position
        self.recalculationIndex = -1
        self.nStep = 0
        self.fixedTime_flag = True
        self.ASI_flag = True
        self.CBC_flag = True
        self.Power_flag = True

        # 03. Insert Table
        self.tableItem = ["Time\n(hour)", "Power\n(%)", "Burnup\n(MWD/MTU)", "Keff", "ASI", "Boron\n(ppm)",
                           "Bank P\n(cm)", "Bank 5\n(cm)", "Bank 4\n(cm)", "Bank 3\n(cm)", ]
        self.SD_TableWidget = table01.ShutdownTableWidget(self.ui.frame_SD_OutputWidget, self.tableItem)
        self.layoutTableButton = self.SD_TableWidget.returnButtonLayout()

        # self.gridLayout_SD_TABLE = QtWidgets.QGridLayout()
        # self.gridLayout_SD_TABLE.setObjectName("gridLayout_SD_TABLEWIDGET_DEFINED")
        self.ui.gridLayout_SD_TABLE.addWidget(self.SD_TableWidget, 0, 0, 1, 1)
        self.ui.gridLayout_SD_TABLE.addLayout(self.layoutTableButton, 1, 0, 1, 1)
        #self.ui.gridlayout_SD_TableWidget.addLayout(self.gridLayout_SD_TABLE, 0, 0, 1, 1)

        #self.ui.gridlayout_SD_TableWidget.addLayout(layoutTableButton,0,0,2,1)

        [self.unitButton01, self.unitButton02, self.unitButton03,] = self.SD_TableWidget.returnTableButton()

        self.unitChart = None
        self.radialWidget = None
        self.axialWidget = None
        self.addOutput()

        # 07. Setting Widget Interactions
        self.buttonGroupInput = QtWidgets.QButtonGroup()
        self.reductionGroupInput = QtWidgets.QButtonGroup()
        # self.settingAutoExclusive()
        self.settingLinkAction()

        # self.load()

    def load(self):
        if len(self.inputArray) == 0:
            self.ui.SD_run_button.setText("Create Scenario")
            self.ui.SD_run_button.setStyleSheet(df.styleSheet_Create_Scenarios)
            self.ui.SD_run_button.setDisabled(False)
        else:
            self.ui.SD_run_button.setText("Run")
            self.ui.SD_run_button.setStyleSheet(df.styleSheet_Run)
            self.ui.SD_run_button.setDisabled(False)

        self.load_input()

    def set_all_component_values(self, ecp_input):
        super().set_all_component_values(ecp_input)

        # self.ui.LabelSub_SD02.hide()
        # self.ui.SD_Input02.hide()

    def index_changed(self, key):
        super().index_changed(key)
        if self.current_calculation and self.last_table_created:
            if self.current_calculation.modified_date > self.last_table_created:
                self.ui.SD_run_button.setText("Create Scenario")
                self.ui.SD_run_button.setStyleSheet(df.styleSheet_Create_Scenarios)
                self.ui.SD_run_button.setDisabled(False)

    def value_changed(self, key):
        super().value_changed(key)
        if self.current_calculation and self.last_table_created:
            if self.current_calculation.modified_date > self.last_table_created:
                self.ui.SD_run_button.setText("Create Scenario")
                self.ui.SD_run_button.setStyleSheet(df.styleSheet_Create_Scenarios)
                self.ui.SD_run_button.setDisabled(False)

    def settingLinkAction(self):
        self.ui.SD_run_button.clicked['bool'].connect(self.start_calc)
        self.ui.SD_save_button.clicked['bool'].connect(self.start_save)

        self.unitButton01.clicked['bool'].connect(self.clickSaveAsExcel)
        self.unitButton02.clicked['bool'].connect(self.resetPositionData)

        self.SD_TableWidget.itemSelectionChanged.connect(self.cell_changed)
        self.unitButton03.clicked['bool'].connect(self.clearOuptut)

    def get_input(self, calculation_object):
        return calculation_object.sd_input

    def get_default_input(self, user):
        now = datetime.datetime.now()

        SD_input = SD_Input.create(
            ndr_burnup=0,
            ndr_target_keff=1.0,
            ndr_power_ratio=3.0,
            ndr_power_asi=0.10,
        )

        SD_calculation = Calculations.create(user=user,
                                             calculation_type=cs.CALCULATION_SD,
                                             created_date=now,
                                             modified_date=now,
                                             sd_input=SD_input
                                             )
        return SD_calculation, SD_input

    def setSuccecciveInput(self):
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

        # TODO, make Except loop
        if (rdcPerHour == 0.0):
            print("Error!")
            return

        nStep = math.ceil((100.0 - EOF_Power) / rdcPerHour + 1.0)

        self.recalculationIndex = nStep

        #self.unitChart.adjustTime(overlap_time)
        # update Power Variation Per Hour and power increase flag
        # ( Power Ascention Mode == True, Power Reduction Mode == False)
        self.unitChart.updateRdcPerHour(rdcPerHour,False)

        for i in range(nStep - 1):
            time = 1.0 * i
            power = 100.0 - i * rdcPerHour
            unitArray = [time, power, initBU, targetEigen]
            self.inputArray.append(unitArray)

        time = 100.0 / rdcPerHour
        power = 0.0
        unitArray = [time, power, initBU, targetEigen]
        self.inputArray.append(unitArray)
        self.nStep = nStep
        self.SD_TableWidget.addInputArray(self.inputArray)
        self.last_table_created = datetime.datetime.now()

    def clickSaveAsExcel(self):
        is_succ = self.SD_TableWidget.clickSaveAsExcel()
        if not is_succ:
            msgBox = QMessageBox(self.get_ui_component())
            msgBox.setWindowTitle("Output not found")
            msgBox.setText("Output not found to save to excel")
            msgBox.setStandardButtons(QMessageBox.Ok)
            result = msgBox.exec_()

    def resetPositionData(self):
        self.SD_TableWidget.resetPositionData()

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

        if self.ui.SD_run_button.text() == cs.RUN_BUTTON_CREATE_SCENARIO:
            self.setSuccecciveInput()
            self.load()
        else:
            calcOpt, initBU, targetEigen, targetASI, pArray, error = self.get_calculation_input()

            #self.unitChart.axisX.setMax(len(pArray))
            if not error and calcOpt != df.CalcOpt_KILL:
                self.start_calculation_message = SplashScreen()
                self.start_calculation_message.killed.connect(self.killManagerProcess)
                self.start_calculation_message.init_progress(len(pArray), 500)

                self.ui.SD_run_button.setText(cs.RUN_BUTTON_RUNNING)
                self.ui.SD_run_button.setDisabled(True)
                self.SD_TableWidget.last_update = 0
                # self.calcManager.setShutdownVariables(calcOpt, self.initBU, self.targetEigen, self.targetASI, pArray)
                self.SD_TableWidget.clearOutputArray()
                self.queue.put((calcOpt, initBU, targetEigen, targetASI, pArray))

    def finished(self):

        if self.start_calculation_message:
            self.start_calculation_message.close()

        #self.definePointData(self.calcManager.outputArray)
        self.SD_TableWidget.selectRow(0)

        self.ui.SD_run_button.setText("Run")
        self.ui.SD_run_button.setDisabled(False)
        # self.cell_changed()

    def showOutput(self):

        self.start_calculation_message.progress()
        last_update = self.SD_TableWidget.last_update
        self.SD_TableWidget.appendOutputTable(self.calcManager.shutdown_output[last_update:], last_update)
        self.appendPointData(self.calcManager.shutdown_output, 0)
        # pd2d = self.calcManager.outputArray[-1][-1]
        # pd1d = self.calcManager.outputArray[-1][-2]
        # self.axialWidget.drawAxial(pd1d, self.calcManager.axial_position)
        # self.radialWidget.slot_astra_data(pd2d)

    def cell_changed(self):
        model_index = self.SD_TableWidget.selectedIndexes()
        if len(model_index) > 0:
            row = model_index[-1].row()
            if row < len(self.calcManager.shutdown_output):
                pd2d = self.calcManager.shutdown_output[row][-1]
                pd1d = self.calcManager.shutdown_output[row][-2]

                p = self.calcManager.shutdown_output[row][2]
                r5 = self.calcManager.shutdown_output[row][3]
                r4 = self.calcManager.shutdown_output[row][4]
                r3 = self.calcManager.shutdown_output[row][5]

                data = {' P': p, 'R5': r5, 'R4': r4, 'R3': r3}

                self.axialWidget.drawBar(data)

                power = self.SD_TableWidget.InputArray[row][1]
                if power == 0:
                    self.axialWidget.clearAxial()
                    self.radialWidget.clear_data()
                else:
                    self.axialWidget.drawAxial(pd1d[self.calcManager.kbc:self.calcManager.kec],
                                               self.calcManager.axial_position)
                    self.radialWidget.slot_astra_data(pd2d)

    def get_calculation_input(self):
        calcOpt = df.CalcOpt_ASI
        if self.SD_TableWidget.checkModified():
            msgBox = QMessageBox(self.get_ui_component())
            msgBox.setWindowTitle("Rod Modification Detected")
            msgBox.setText("Rod: Rod Position Search (Target ASI)\n"
                           "CBC: Boron Search with (Target Rod Position)")
            msgBox.addButton(QPushButton('Rod Search'), QMessageBox.YesRole)
            msgBox.addButton(QPushButton('CBC Search'), QMessageBox.NoRole)
            msgBox.addButton(QPushButton('Cancel'), QMessageBox.NoRole)
            # msgBox.setWindowFlag(QtCore.Qt.FramelessWindowHint)
            result = msgBox.exec_()
            if result == 1:
                calcOpt = df.CalcOpt_ASI_RESTART
            elif result == 2:
                calcOpt = df.CalcOpt_KILL

        if calcOpt == df.CalcOpt_ASI_RESTART:
            _, initBU, targetEigen, targetASI, pArray, error = self.getInputRestart()
        else:
            _, initBU, targetEigen, targetASI, pArray, error = self.getInput()

        return calcOpt, initBU, targetEigen, targetASI, pArray, error

    def getInput(self):

        pArray = []
        for iStep in range(self.nStep):
            pArray.append(self.inputArray[iStep])
        error = self.check_calculation_input()
        return df.CalcOpt_ASI, self.initBU, self.targetEigen, self.targetASI, pArray, error

    def getInputRestart(self):

        rod_array = self.SD_TableWidget.getRodValues()
        if self.nStep != len(rod_array):
            raise ValueError("Program Error length mismatch shutdown")
        pArray = []
        for iStep in range(self.nStep):
            pArray.append(self.inputArray[iStep]+ [0,0,]+rod_array[iStep])
        error = self.check_calculation_input()
        return df.CalcOpt_ASI_RESTART, self.initBU, self.targetEigen, self.targetASI, pArray, error

    def check_calculation_input(self):

        if self.ui.SD_Input01.value() >= self.calcManager.cycle_burnup_values[-1]:
            msgBox = QMessageBox(self.get_ui_component())
            msgBox.setWindowTitle("Burnup Out of Range")
            msgBox.setText("{}MWD/MTU excedes EOC Cycle Burnup({} MWD/MTU)\n"
                           "Put Cycle Burnup less than {}MWD/MTU".format(self.ui.SD_Input01.value(),
                                                            self.calcManager.cycle_burnup_values[-1],
                                                            self.calcManager.cycle_burnup_values[-1]))
            msgBox.setStandardButtons(QMessageBox.Ok)
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
            posR5.append(QPointF(self.inputArray[iStep][0], outputArray[iStep][2]))
            posR4.append(QPointF(self.inputArray[iStep][0], outputArray[iStep][3]))
            posR3.append(QPointF(self.inputArray[iStep][0], outputArray[iStep][4]))
            posCBC.append(QPointF(self.inputArray[iStep][0], outputArray[iStep][0]))

        rodPos = [posR5, posR4, posR3]

        #self.unitChart.replaceRodPosition(3, rodPos, posCBC)

    def appendPointData(self, outputArray, start):

        # self.unitChartClass.clear()
        num = len(outputArray)
        time = []
        power = []
        for idx in range(num):
            time.append(self.inputArray[idx][0])
            power.append(self.inputArray[idx][1])

        posP = []
        posR5 = []
        posR4 = []
        posR3 = []
        posASI = []
        posCBC = []
        for i in range(len(outputArray)):
            posP.append(outputArray[i][2])
            posR5.append(outputArray[i][3])
            posR4.append(outputArray[i][4])
            posR3.append(outputArray[i][5])
            posASI.append(outputArray[i][0])
            posCBC.append(outputArray[i][1])
            # posP.append(QPointF(start+i, outputArray[i][2]))
            # posR5.append(QPointF(start+i, outputArray[i][3]))
            # posR4.append(QPointF(start+i, outputArray[i][4]))
            # posR3.append(QPointF(start+i, outputArray[i][5]))
            # posCBC.append(QPointF(start+i, outputArray[i][0]))

        #rodPos = [posP, posR5, posR4, posR3, ]
        posOpt = [ posASI, posCBC, power ]
        self.unitChart.insertDataSet(time, posP, posR5, posR4, posOpt )
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
            # 02. Insert Spline Chart
            #self.unitChart = unitSplineChart.UnitSplineChart(self.RodPosUnit)
            #self.unitChart = trendWidget(self.ui.SD_widgetChart)#self.RodPosUnit)
            #self.


            lay = self.ui.grid_SD_frameChart
            # lay = QtWidgets.QVBoxLayout(self.ui.SD_widgetChart)
            # lay.setContentsMargins(0, 0, 0, 0)
            self.unitChart = trendWidget(self.fixedTime_flag, self.ASI_flag, self.CBC_flag, self.Power_flag)#self.ui.SD_widgetChart)
            #unitChartView = self.unitChart.returnChart()
            sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            sizePolicy5.setHorizontalStretch(10)
            sizePolicy5.setVerticalStretch(10)
            #sizePolicy5.set
            #sizePolicy5.setHeightForWidth(self.ui.SD_widgetChart.sizePolicy().hasHeightForWidth())
            self.unitChart.setSizePolicy(sizePolicy5)
            lay.setContentsMargins(0, 0, 0, 0)
            #self.unitChart.insertDataSet()

            # a = QWidget()
            # self.aa = tmp0001.Ui_TEST_WIDGET()
            # self.aa.setupUi(a)
            #lay.addWidget(a, 0, 0, 1, 1)

            lay.addWidget(self.unitChart, 0, 0, 1, 1)
            #lay.addLayout(self.layoutTableButton, 1, 0, 1, 1)

            # sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            # sizePolicy5.setHorizontalStretch(10)
            # sizePolicy5.setVerticalStretch(10)
            # sizePolicy5.setHeightForWidth(self.ui.SD_widgetChart.sizePolicy().hasHeightForWidth())
            # self.ui.SD_widgetChart.setSizePolicy(sizePolicy5)

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

            self.radialWidget = RadialWidget(self.ui.SD_InputLP_frame, self.ui.SD_InputLPgrid)

        if not self.axialWidget:
            # 05. Insert Axial Chart
            sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            sizePolicy5.setHorizontalStretch(6)
            sizePolicy5.setVerticalStretch(10)
            sizePolicy5.setHeightForWidth(self.ui.SD_WidgetAxial.sizePolicy().hasHeightForWidth())
            self.ui.SD_WidgetAxial.setSizePolicy(sizePolicy5)
            layA = QtWidgets.QVBoxLayout(self.ui.SD_WidgetAxial)
            layA.setContentsMargins(0, 0, 0, 0)
            self.axialWidget = AxialWidget()
            layA.addWidget(self.axialWidget)

    #####################
    #### TEST ROUTINE ###
    #####################

    def clearOuptut(self):
        self.SD_TableWidget.clearOutputArray()
        self.SD_TableWidget.clearOutputRodArray()