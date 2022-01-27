
from PyQt5.QtCore import (QCoreApplication, QPropertyAnimation, QDate, QDateTime, QMetaObject, QObject, QPoint, QRect, QSize, QTime, QUrl, Qt, QEvent, pyqtSlot)
from PyQt5.QtCore import QPointF
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *

from model import *
import datetime

import Definitions as df

import constants as cs

from widgets.calculations.calculation_widget import CalculationWidget

import widgets.utils.PyShutdownTableWidget as table01
from widgets.output.trend.trends_graph import trendWidget
import widgets.utils.PyRodPositionSplineChartECP as unitSplineChart
import widgets.calculations.calculationManager as calcManager
from widgets.output.radial.radial_graph import RadialWidget
from widgets.output.axial.axial_plot import AxialWidget
import math

from widgets.utils.PySaveMessageBox import PySaveMessageBox

from itertools import islice
from random import random
from time import perf_counter
from cusfam import *
import datetime

from PyQt5.QtCore import QPointF, QThread

from widgets.utils.splash_screen import SplashScreen
import numpy as np
import widgets.calculations.calculationManagerProcess as CMP

pointers = {

    #"ECP_Input01": "bs_ndr_date_time",
    "ECP_Input01_date": "bs_ndr_date",
    "ECP_Input01_time": "bs_ndr_time",
    "ECP_Input03": "bs_ndr_power",
    "ECP_Input04": "bs_ndr_burnup",

    #"ECP_Input12": "as_ndr_date_time",
    "ECP_Input12": "as_ndr_delta_time",
    "ECP_Input13": "as_ndr_boron_concentration",

    "ECP_Input14": "as_ndr_bank_position_P",
    "ECP_Input15": "as_ndr_bank_position_5",
    "ECP_Input16": "as_ndr_bank_position_4",

}

_POINTER_A_ = "self.ui.ECP_DB"
_NUM_PRINTOUT_ = 5
_MINIMUM_ITEM_COUNT = 40


class ECPWidget(CalculationWidget):

    def __init__(self, db, ui, calManager, queue, message_ui):

        super().__init__(db, ui, None, ECP_Input, pointers, calManager, queue, message_ui)

        self.input_pointer = "self.current_calculation.ecp_input"

        self.inputArray = []

        # 01. Setting Initial ECP Control Input Data
        self.posBeforeBankP = 381.0
        self.posBeforeBank5 = 381.0
        self.posBeforeBank4 = 381.0
        self.posBeforeBank3 = 381.0

        self.posAfterBankP = 381.0
        self.posAfterBank5 = 381.0
        self.posAfterBank4 = 381.0
        self.posAfterBank3 = 381.0

        self.fixedTime_flag = False
        self.ASI_flag = False
        self.CBC_flag = True
        self.Power_flag = False

        self.ECP_CalcOpt = df.select_none
        self.ECP_TargetOpt = df.select_none
        self.flag_RodPosAfterShutdown = False

        self.RodPosUnitBefore = df.RodPosUnit_cm
        self.unitChangeFlagBefore = False
        self.initSnapshotFlagBefore = False

        self.unitChangeFlag = False
        self.RodPosUnitAfter = df.RodPosUnit_cm
        self.initSnapshotFlag = False

        # 03. Insert Table
        self.tableItem = ["Time\n(hour)","Power\n(%)","Burnup\n(MWD/MTU)","Keff","Boron\n(ppm)",
                           "Bank P\n(cm)", "Bank 5\n(cm)","Bank 4\n(cm)",]
        self.ECP_TableWidget = table01.ShutdownTableWidget(self.ui.frame_ECP_TableWidget, self.tableItem, 4, 1, 3)
        layoutTableButton = self.ECP_TableWidget.returnButtonLayout()
        self.ui.gridlayout_ECP_TableWidget.addWidget(self.ECP_TableWidget, 0, 0, 1, 1)
        self.ui.gridlayout_ECP_TableWidget.addLayout(layoutTableButton,1,0,1,1)
        [ self.unitButton01, self.unitButton02, self.unitButton03 ] = self.ECP_TableWidget.returnTableButton()

        # 03. Setting Widget Interactions
        self.buttonGroupInput = QtWidgets.QButtonGroup()
        self.buttonGroupTarget = QtWidgets.QButtonGroup()
        # TODO SGH, MAKE CALC. OPT.
        #self.settingAutoExclusive()
        self.settingLinkAction()

        # 04. Setting Initial UI widget
        #self.ui.ECP_Main02.hide()
        #self.ui.ECP_Main03.hide()

        # 05. Insert ChartView to chartWidget
        # sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        # sizePolicy5.setHorizontalStretch(6)
        # sizePolicy5.setVerticalStretch(10)
        # sizePolicy5.setHeightForWidth(self.ui.ECP_widgetChart.sizePolicy().hasHeightForWidth())
        # self.ui.ECP_widgetChart.setSizePolicy(sizePolicy5)


        # self.RodPosUnit = df.RodPosUnit_cm
        # self.unitChartClass = unitSplineChart.UnitSplineChart(self.RodPosUnit)
        #
        # sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        # sizePolicy5.setHorizontalStretch(10)
        # sizePolicy5.setVerticalStretch(10)
        # sizePolicy5.setHeightForWidth(self.ui.ECP_widgetChart.sizePolicy().hasHeightForWidth())
        # self.ui.ECP_widgetChart.setSizePolicy(sizePolicy5)
        #
        # lay = QtWidgets.QVBoxLayout(self.ui.ECP_widgetChart)
        # lay.setContentsMargins(0, 0, 0, 0)
        # unitChartView = self.unitChartClass.returnChart()
        # lay.addWidget(unitChartView)
        # self.unitChartClass.axisX.setMax(10)
        # self.unitChartClass.axisY_CBC.setMax(1000)
        # self.current_index = 0

        #lay = QtWidgets.QVBoxLayout(self.ui.ECP_widgetChart)
        lay = self.ui.grid_ECP_frameChart
        self.unitChart = trendWidget(self.fixedTime_flag,self.ASI_flag,self.CBC_flag,self.Power_flag)
        sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy5.setHorizontalStretch(10)
        sizePolicy5.setVerticalStretch(10)
        self.unitChart.setSizePolicy(sizePolicy5)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.unitChart, 0, 0, 1, 1)
        # self.load()


    def load(self):
        self.load_input()
        if len(self.inputArray) == 0:
            self.ui.ECP_run_button.setText("Create Scenario")
            self.ui.ECP_run_button.setStyleSheet(df.styleSheet_Create_Scenarios)
        else:
            self.ui.ECP_run_button.setText("Run")
            self.ui.ECP_run_button.setStyleSheet(df.styleSheet_Run)

        #self.load_input()

    def set_all_component_values(self, ecp_input):
        super().set_all_component_values(ecp_input)

        if ecp_input.search_type == 0:
            self.ui.ECP_Input00.setCurrentIndex(0)
            self.settingTargetOpt()
            #self.ui.ECP_CalcTarget01.setChecked(True)
        else:
            self.ui.ECP_Input00.setCurrentIndex(1)
            self.settingTargetOpt()
            #self.ui.ECP_CalcTarget02.setChecked(True)

        # self.ui.LabelSub_ECP03.hide()
        # self.ui.ECP_Input03.hide()


    def index_changed(self, key):
        super().index_changed(key)
        if self.current_calculation and self.last_table_created:
            if self.current_calculation.modified_date > self.last_table_created:
                self.ui.ECP_run_button.setText("Create Scenario")
                self.ui.ECP_run_button.setStyleSheet(df.styleSheet_Create_Scenarios)

    def value_changed(self, key):
        super().value_changed(key)
        if self.current_calculation and self.last_table_created:
            if self.current_calculation.modified_date > self.last_table_created:
                self.ui.ECP_run_button.setText("Create Scenario")
                self.ui.ECP_run_button.setStyleSheet(df.styleSheet_Create_Scenarios)

    def date_time_changed(self, key):
        super().date_time_changed(key)
        if self.current_calculation and self.last_table_created:
            if self.current_calculation.modified_date > self.last_table_created:
                self.ui.ECP_run_button.setText("Create Scenario")
                self.ui.ECP_run_button.setStyleSheet(df.styleSheet_Create_Scenarios)

    def settingAutoExclusive(self):
        self.buttonGroupTarget.addButton(self.ui.ECP_CalcTarget01)
        self.buttonGroupTarget.addButton(self.ui.ECP_CalcTarget02)

    def settingLinkAction(self):

        #self.ui.ECP_CalcTarget01.toggled['bool'].connect(self.settingTargetOpt)
        #self.ui.ECP_CalcTarget02.toggled['bool'].connect(self.settingTargetOpt)
        self.ui.ECP_Input00.currentIndexChanged['int'].connect(self.settingTargetOpt)

        self.ui.ECP_save_button.clicked['bool'].connect(self.start_save)
        self.ui.ECP_run_button.clicked['bool'].connect(self.start_calc)

        self.unitButton01.clicked['bool'].connect(self.clickSaveAsExcel)
        self.unitButton02.clicked['bool'].connect(self.resetPositionData)
        self.unitButton03.clicked['bool'].connect(self.clearOuptut)

    def settingTargetOpt(self):
        search_type = 0
        self.ui.ECP_run_button.setText("Create Scenario")
        self.ui.ECP_run_button.setStyleSheet(df.styleSheet_Create_Scenarios)
        tmp = self.ui.ECP_Input00.currentIndex()
        if self.ui.ECP_Input00.currentIndex() == 0:
        #if self.ui.ECP_CalcTarget01.isChecked():
            self.ECP_TargetOpt = df.select_Boron
            # show Reactor Condition After Shutdown Input Frame
            #self.ui.ECP_Main02.show()
            #self.ui.ECP_Main03.show()
            # show Rod Position Input Widget
            self.ui.LabelSub_ECP13.hide()
            self.ui.ECP_Input13.hide()
            self.ui.LabelSub_ECP14.show()
            self.ui.ECP_Input14.show()
            self.ui.LabelSub_ECP15.show()
            self.ui.ECP_Input15.show()
            self.ui.LabelSub_ECP16.show()
            self.ui.ECP_Input16.show()
            # self.ui.label_ECP_Report13.hide()
            # self.ui.labelECP013.hide()
            # self.ui.label_ECP_Report14.hide()
            # self.ui.labelECP014.hide()
            # self.ui.label_ECP_Report15.hide()
            # self.ui.labelECP015.hide()
            # hide Boron Concentration Input Widget
            # self.ui.LabelSub_ECP13.hide()
            # self.ui.ECP_Input13.hide()
            # self.ui.label_ECP_Report12.show()
            # self.ui.labelECP012.show()

            # self.BeforeRodPosChangedEvent()
            # self.AfterRodPosChangedEvent()

        elif self.ui.ECP_Input00.currentIndex() == 1:
        #elif self.ui.ECP_CalcTarget02.isChecked():
            self.ECP_TargetOpt = df.select_RodPos
            # show Reactor Condition After Shutdown Input Frame
            #self.ui.ECP_Main02.show()
            #self.ui.ECP_Main03.show()
            # Hide Rod Position Input
            self.ui.LabelSub_ECP13.show()
            self.ui.ECP_Input13.show()
            self.ui.LabelSub_ECP14.hide()
            self.ui.ECP_Input14.hide()
            self.ui.LabelSub_ECP15.hide()
            self.ui.ECP_Input15.hide()
            self.ui.LabelSub_ECP16.hide()
            self.ui.ECP_Input16.hide()
            # self.ui.label_ECP_Report13.show()
            # self.ui.labelECP013.show()
            # self.ui.label_ECP_Report14.show()
            # self.ui.labelECP014.show()
            # self.ui.label_ECP_Report15.show()
            # self.ui.labelECP015.show()
            # Show Boron Concentration Input Widget
            # self.ui.LabelSub_ECP13.show()
            # self.ui.ECP_Input13.show()
            # self.ui.label_ECP_Report12.hide()
            # self.ui.labelECP012.hide()

            #self.ui.ECP_Main07.show()
            search_type = 1

            # self.BeforeRodPosChangedEvent()

        if self.current_calculation:
            current_input = self.get_input(self.current_calculation)
            current_input.search_type = search_type


    def setSuccecciveInput(self):
        # Initialize
        self.inputArray = []

        startT    = self.ui.ECP_Input01_time.time()
        startD    = self.ui.ECP_Input01_date.date()

        deltaTime = self.ui.ECP_Input12.value()
        second_criticality = deltaTime * 3600.0
        #check input
        if not 0 < second_criticality <= 100*3600:
            msgBox = QMessageBox(self.get_ui_component())
            msgBox.setText("Delta time should be in between 0 and 100 hour\nYour delta time is {}"
                           .format(second_criticality//3600))
            msgBox.setStandardButtons(QMessageBox.Ok)
            result = msgBox.exec_()
            return

        pw = self.ui.ECP_Input03.value()
        bp = self.ui.ECP_Input04.value()
        eigen = 1.0

        self.inputArray.append([0, pw, bp, eigen])
        output_array = []

        deltime = 0
        time_index = 0
        while deltime < second_criticality//3600.0:
            self.inputArray.append([deltime, 0.0, bp, eigen])
            if len(CMP.decay_table) == time_index:
                break
            deltime += CMP.decay_table[time_index]
            time_index += 1

        if deltime >= second_criticality//3600.0:
            self.inputArray.append([second_criticality//3600.0, 0.0, bp, eigen])


        timeArray = []
        for idx in range(len(self.inputArray)):
            timeArray.append(self.inputArray[idx][0])

        self.unitChart.insertTime(timeArray)

        output_array.append([381.0, 381.0, 381.0])
        output_array.append([0.0, 0.0, 0.0])

        for i in range(len(self.inputArray)-2):
            if self.ECP_TargetOpt==df.select_Boron:
                self.shutdown_P_Pos = self.ui.ECP_Input14.value()
                self.shutdown_r5Pos = self.ui.ECP_Input15.value()
                self.shutdown_r4Pos = self.ui.ECP_Input16.value()
                an_output = [self.shutdown_P_Pos, self.shutdown_r5Pos, self.shutdown_r4Pos]
            else:
                self.shutdown_ppm = self.ui.ECP_Input13.value()
                an_output = [self.shutdown_ppm]
            output_array.append(an_output)

        self.ECP_TableWidget.addInputArray(self.inputArray)
        self.ECP_TableWidget.makeOutputTable(output_array)

    def get_input(self, calculation_object):
        return calculation_object.ecp_input


    def get_default_input(self, user):
        now = datetime.datetime.now()
        ecp_input = ECP_Input.create(
            search_type=0,

            #bs_ndr_date_time=now,
            bs_ndr_date=now,
            bs_ndr_time=now,
            bs_ndr_power=0,
            bs_ndr_burnup=0,
            bs_ndr_average_temperature=0,
            bs_ndr_target_eigen = 1.0,
            bs_ndr_bank_position_P=381.0,
            bs_ndr_bank_position_5=381.0,
            bs_ndr_bank_position_4=381.0,
            #as_ndr_date_time=now,
            as_ndr_delta_time=0.0,
            as_ndr_boron_concentration=0,
            as_ndr_bank_position_P=381.0,
            as_ndr_bank_position_5=381.0,
            as_ndr_bank_position_4=381.0,
        )

        ecp_calculation = Calculations.create(user=user,
                                              calculation_type=cs.CALCULATION_ECP,
                                              created_date=now,
                                              modified_date=now,
                                              ecp_input=ecp_input
                                              )
        return ecp_calculation, ecp_input

    def start_calc(self):
        super().start_calc()

        if self.ui.ECP_run_button.text() == cs.RUN_BUTTON_CREATE_SCENARIO:
            self.setSuccecciveInput()
            self.load()
        else:
            #self.setSuccecciveInput()
            # Set Input Variable
            self.startD = self.ui.ECP_Input01_date.date()
            self.startT = self.ui.ECP_Input01_time.time()
            self.startTime = QDateTime()
            self.startTime.setDate(self.startD)
            self.startTime.setTime(self.startT)
            #self.startTime = self.ui.ECP_Input01.dateTime()
            self.deltaTime = self.ui.ECP_Input12.value()
            self.endTime = self.startTime.addSecs(self.deltaTime * 3600.0)
            #self.endTime = self.ui.ECP_Input12.dateTime()
            self.pw = self.ui.ECP_Input03.value()
            self.bp = self.ui.ECP_Input04.value()
            self.mtc = df.inlet_temperature
            self.eigen = 1.0
            self.P_Pos = 381.0
            self.r5Pos = 381.0
            self.r4Pos = 381.0
            inputArray = [ self.pw, self.bp, self.mtc, self.eigen, self.P_Pos, self.r5Pos, self.r4Pos]
            if self.ECP_TargetOpt==df.select_Boron:
                self.shutdown_P_Pos = self.ui.ECP_Input14.value()
                self.shutdown_r5Pos = self.ui.ECP_Input15.value()
                self.shutdown_r4Pos = self.ui.ECP_Input16.value()
                inputArray.append(self.shutdown_P_Pos)
                inputArray.append(self.shutdown_r5Pos)
                inputArray.append(self.shutdown_r4Pos)
            elif self.ECP_TargetOpt==df.select_RodPos:
                self.shutdown_ppm = self.ui.ECP_Input13.value()
                inputArray.append(self.shutdown_ppm)
            inputArray.append(self.deltaTime)

            error = self.check_calculation_input()

            if not error:
                self.cal_start = True
                self.current_index = 0
                self.start_calculation_message = SplashScreen()
                self.start_calculation_message.killed.connect(self.killManagerProcess)
                self.start_calculation_message.init_progress(10, 500)

                self.ui.ECP_run_button.setText("Running....")
                self.ui.ECP_run_button.setDisabled(True)
                self.ECP_TableWidget.last_update = 0
                self.ECP_TableWidget.clearOutputArray()

                self.queue.put((df.CalcOpt_ECP, self.ECP_TargetOpt, inputArray))

    def check_calculation_input(self):

        if self.ui.ECP_Input04.value() >= self.calcManager.cycle_burnup_values[-1]:
            msgBox = QMessageBox(self.get_ui_component())
            msgBox.setWindowTitle("Burnup Out of Range")
            msgBox.setText("{}MWD/MTU excedes EOC Cycle Burnup({} MWD/MTU)\n"
                           "Put Cycle Burnup less than {}MWD/MTU".format(self.ui.ECP_Input04.value(),
                                                            self.calcManager.cycle_burnup_values[-1],
                                                            self.calcManager.cycle_burnup_values[-1]))
            msgBox.setStandardButtons(QMessageBox.Ok)
            #msgBox.setWindowFlag(QtCore.Qt.FramelessWindowHint)
            result = msgBox.exec_()
            # if result == QMessageBox.Cancel:
            return True
        return False

    def ECP_output_index_changed(self):
        index = self.ui.ECP_Output1.currentIndex()
        if len(self.calcManager.out_ppm) > index:
            self.show_output_values(index)

            p = 381.0
            r5 = self.calcManager.rodPos[index][0]
            r4 = self.calcManager.rodPos[index][1]
            r3 = self.calcManager.rodPos[index][2]

            data = {' P': p, 'R5': r5, 'R4': r4, 'R3': r3}
            self.axialWidget.drawBar(data)
            # print(data)

            pd1d = self.calcManager.ecpP1d[index]
            self.axialWidget.drawAxial(pd1d[self.calcManager.kbc:self.calcManager.kec],
                                       self.calcManager.axial_position)

    def showOutput(self):
        self.start_calculation_message.progress()
        if len(self.calcManager.out_ppm) > 0:
            self.show_output_values(len(self.calcManager.out_ppm) -1)

    def finished(self):
        if self.start_calculation_message:
            self.start_calculation_message.close()

        self.ui.ECP_run_button.setText("Run")
        self.ui.ECP_run_button.setDisabled(False)

        # if len(self.calcManager.out_ppm) > 0:
        #     self.show_output_values()
        # self.ECP_output_index_changed()

    def show_output_values(self, show_index=0):
        # startTime_Day  = self.startTime.date()
        # startTime_Time = self.startTime.time()
        # startTime_STR = "{:4d}-{:02d}-{:02d} {:02d}:{:02d}".format(startTime_Day.year(),startTime_Day.month(),startTime_Day.day(),startTime_Time.hour(),startTime_Time.minute())
        # endTime_Day  = self.endTime.date()
        # endTime_Time = self.endTime.time()
        # endTime_STR = "{:4d}-{:02d}-{:02d} {:02d}:{:02d}".format(endTime_Day.year(),endTime_Day.month(),endTime_Day.day(),endTime_Time.hour(),endTime_Time.minute())

        if self.ECP_TargetOpt==df.select_Boron:

            [self.pw, self.bp, self.mtc, eigen, P_Pos, r5Pos, r4Pos, shutdown_P_Pos, shutdown_r5Pos,
             shutdown_r4Pos, deltaTime] = self.calcManager.inputArray
            # self.unitChartClass.axisY_CBC.setMax(max(1200, np.max(self.calcManager.out_ppm)+100))
            # self.unitChartClass.axisY_CBC.setMin(max(0, np.max(self.calcManager.out_ppm)+100-1000))
            #self.unitChart.insertTime()axisX.setMax(deltaTime)
            output_array = []
            for i in range(0, len(self.calcManager.out_ppm)):
                output_array.append([self.calcManager.out_ppm[i],
                                     self.calcManager.rodPos[i][0],
                                     self.calcManager.rodPos[i][1],
                                     self.calcManager.rodPos[i][2],])
            self.appendPointData(output_array, 0)
            self.ECP_TableWidget.makeOutputTable(output_array)

        if self.ECP_TargetOpt==df.select_RodPos:

            [ self.pw, self.bp, self.mtc, eigen, P_Pos, r5Pos, r4Pos, target_ppm, deltaTime] = self.calcManager.inputArray

            # self.unitChartClass.axisY_CBC.setMax(max(1200, np.max(self.calcManager.out_ppm)+100))
            # self.unitChartClass.axisY_CBC.setMin(max(0, np.max(self.calcManager.out_ppm)+100-1000))
            #self.unitChart.axisX.setMax(target_ppm + 100)
            #self.unitChart.axisX.setMax(deltaTime)

            output_array1 = []
            for i in range(0, len(self.calcManager.out_ppm)):
                output_array1.append([self.calcManager.out_ppm[i],
                                     self.calcManager.rodPos[i][0],
                                     self.calcManager.rodPos[i][1],
                                     self.calcManager.rodPos[i][2],
                                     ])
            self.appendPointData(output_array1, 0)
            self.ECP_TableWidget.makeOutputTable(output_array1)

    def appendPointData(self, outputArray, start):

        posP = []
        posR5 = []
        posR4 = []
        posCBC = []
        for i in range(len(outputArray)):
            posP.append(outputArray[i][1])
            posR5.append(outputArray[i][2])
            posR4.append(outputArray[i][3])
            posCBC.append(outputArray[i][0])
        # self.unitChartClass.clear()
        #
        # posP = []
        # posR5 = []
        # posR4 = []
        # posR3 = []
        # posCBC = []
        # for i in range(len(outputArray)):
        #     posP.append(QPointF(start+i, outputArray[i][1]))
        #     posR5.append(QPointF(start+i, outputArray[i][2]))
        #     posR4.append(QPointF(start+i, outputArray[i][3]))
        #     posCBC.append(QPointF(start+i, outputArray[i][0]))
        #
        # rodPos = [posP, posR5, posR4, ]
        #
        # self.unitChartClass.appendRodPosition(len(rodPos), rodPos, posCBC)
        posOpt = [ posCBC ]
        self.unitChart.insert_ECP_DataSet( posP, posR5, posR4, posOpt)

    def clickSaveAsExcel(self):
        self.ECP_TableWidget.clickSaveAsExcel()

    def resetPositionData(self):
        self.ECP_TableWidget.resetPositionData()

    def clearOuptut(self):
        self.ECP_TableWidget.clearOutputArray()
        self.ECP_TableWidget.clearOutputRodArray()


    # def setSuccecciveInput(self):
    #     # Initialize
    #     self.inputArray = []
    #
    #     initBU = self.ui.ECP_Input04.value()
    #     targetEigen = self.ui.ECP_Input06.value()
    #
    #     timeInterval = 3.0
    #     core_power = self.ui.ECP_Input03.value()
    #
    #     nStep = 10
    #     for i in range(nStep):
    #         time = timeInterval * i
    #         power = core_power
    #         unitArray = [time, power, initBU, targetEigen]
    #         self.inputArray.append(unitArray)
    #
    #     self.nStep = nStep
    #     self.ECP_TableWidget.addInputArray(self.inputArray)
    #     self.last_table_created = datetime.datetime.now()

