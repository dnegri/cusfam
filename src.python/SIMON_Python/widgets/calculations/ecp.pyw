
from PyQt5.QtCore import (QCoreApplication, QPropertyAnimation, QDate, QDateTime, QMetaObject, QObject, QPoint, QRect, QSize, QTime, QUrl, Qt, QEvent, pyqtSlot)
from PyQt5.QtCore import QPointF
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *

from model import *
import datetime

import Definitions as df

import constants as cs

from widgets.calculations.calculation_widget import CalculationWidget
from widgets.calculations.IO_table import table_IO_widget

import widgets.utils.PyShutdownTableWidget as table01
from widgets.output.trend.trends_graph import trendWidget
from widgets.output.trend.trends02_graph import trend02Widget
import widgets.utils.PyRodPositionSplineChartECP as unitSplineChart
import widgets.calculations.calculationManager as calcManager
from widgets.output.radial.radial_graph import RadialWidget
from widgets.output.axial.axial_plot import AxialWidget
from widgets.utils.Map_Quarter import Ui_unitWidget_OPR1000_quarter as opr
from widgets.utils.Map_Quarter import Ui_unitWidget_APR1400_quarter as apr
import math

from widgets.utils.PySaveMessageBox import PySaveMessageBox, QMessageBoxWithStyle

from itertools import islice
from random import random
from time import perf_counter
# from cusfam import *
import datetime

from PyQt5.QtCore import QPointF, QThread

from widgets.utils.splash_screen import SplashScreen
import numpy as np
import widgets.calculations.calculationManagerProcess as CMP

pointers = {

    #"ECP_Input01": "bs_ndr_date_time",
    "ECP_InputSelect": "calculation_type",
    "pushButton_InputModel": "snapshot_text",
    #"ECP_Input01_date": "bs_ndr_date",
    #"ECP_Input01_time": "bs_ndr_time",
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
        self.snapshotArray = []

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
        # self.tableItem = ["Time\n(hour)","Power\n(%)","Burnup\n(MWD/MTU)","Keff","Boron\n(ppm)",
        #                    "Bank P\n(cm)", "Bank 5\n(cm)","Bank 4\n(cm)",]
        # self.tableItemFormat = ["%.1f","%.2f","%.1f","%.5f","%.3f",
        #                         "%.1f","%.1f","%.1f"]
        self.tableItem = ["Time\n(hour)", "Power\n(%)"  ,
                          "ASI",          "Boron\n(ppm)/\nReactivity", "Fr", "Fxy", "Fq",
                          "Bank P\n(cm)" ,"Bank 5\n(cm)", "Bank 4\n(cm)", "Bank 3\n(cm)", ]
        self.tableItemFormat = ["%.1f","%.2f",
                                "%.3f","%.1f","%.3f","%.3f","%.3f",
                                "%.1f","%.1f","%.1f","%.1f"]
        #self.ECP_TableWidget = table01.ShutdownTableWidget(self.ui.frame_ECP_TableWidget, self.tableItem, self.tableItemFormat, 4, 1, 3)
        #self.ECP_TableWidget = table01.ShutdownTableWidget(self.ui.frame_ECP_TableWidget, self.tableItem, self.tableItemFormat)
        self.ECP_TableWidget = table01.ShutdownTableWidget(self.ui.ECP_frameChart, self.tableItem, self.tableItemFormat)
        self.IO_table = table_IO_widget(ECP_Flag=True)
        layoutTableButton = self.ECP_TableWidget.returnButtonLayout()
        self.ui.gridlayout_ECP_TableWidget.addWidget(self.ECP_TableWidget, 0, 0, 1, 1)
        self.ui.gridlayout_ECP_TableWidget.addLayout(layoutTableButton,1,0,1,1)
        self.ECP_TableWidget.hide()
        #[ self.unitButton01, self.unitButton02, self.unitButton03 ] = self.ECP_TableWidget.returnTableButton()

        # 03. Setting Widget Interactions
        self.buttonGroupInput = QtWidgets.QButtonGroup()
        self.buttonGroupTarget = QtWidgets.QButtonGroup()
        # TODO SGH, MAKE CALC. OPT.
        #self.settingAutoExclusive()

        # 04. trend, radial, axial graph add
        self.unitChart = None
        self.unitChart02 = None
        self.radialWidget = None
        self.axialWidget = None

        self.map_opr1000 = None
        self.map_opr1000_frame = None
        self.map_opr1000_grid = None
        self.map_apr1400 = None
        self.map_apr1400_frame = None
        self.map_apr1400_grid = None

        self.addOutput()


        self.settingLinkAction()
        #
        # # 04. Setting Initial UI widget
        # #self.ui.ECP_Main02.hide()
        # #self.ui.ECP_Main03.hide()
        #
        # # 05. Insert ChartView to chartWidget
        # # sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        # # sizePolicy5.setHorizontalStretch(6)
        # # sizePolicy5.setVerticalStretch(10)
        # # sizePolicy5.setHeightForWidth(self.ui.ECP_widgetChart.sizePolicy().hasHeightForWidth())
        # # self.ui.ECP_widgetChart.setSizePolicy(sizePolicy5)
        #
        #
        # # self.RodPosUnit = df.RodPosUnit_cm
        # # self.unitChartClass = unitSplineChart.UnitSplineChart(self.RodPosUnit)
        # #
        # # sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        # # sizePolicy5.setHorizontalStretch(10)
        # # sizePolicy5.setVerticalStretch(10)
        # # sizePolicy5.setHeightForWidth(self.ui.ECP_widgetChart.sizePolicy().hasHeightForWidth())
        # # self.ui.ECP_widgetChart.setSizePolicy(sizePolicy5)
        # #
        # # lay = QtWidgets.QVBoxLayout(self.ui.ECP_widgetChart)
        # # lay.setContentsMargins(0, 0, 0, 0)
        # # unitChartView = self.unitChartClass.returnChart()
        # # lay.addWidget(unitChartView)
        # # self.unitChartClass.axisX.setMax(10)
        # # self.unitChartClass.axisY_CBC.setMax(1000)
        # # self.current_index = 0
        #
        # #lay = QtWidgets.QVBoxLayout(self.ui.ECP_widgetChart)
        # # lay = self.ui.grid_ECP_frameChart
        # # self.unitChart = trendWidget(self.fixedTime_flag,self.ASI_flag,self.CBC_flag,self.Power_flag)
        # # sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        # # sizePolicy5.setHorizontalStretch(10)
        # # sizePolicy5.setVerticalStretch(10)
        # # self.unitChart.setSizePolicy(sizePolicy5)
        # # lay.setContentsMargins(0, 0, 0, 0)
        # # lay.addWidget(self.unitChart, 0, 0, 1, 1)
        # # # self.load()

        # 05. Setting Input Type
        self.loadInputData = None
        self.ui.ECP_InputSelect.setCurrentIndex(df._INPUT_TYPE_USER_)
        self.inputType = df._INPUT_TYPE_USER_
        self.ui.LabelSub_Selete01.setVisible(False)
        self.ui.pushButton_InputModel.setVisible(False)

        self.ui.ECP_Input03.setMaximum(110.0)
        self.ui.ECP_Input12.setMaximum(300.0)

        self.delete_current_calculation()

    def linkInputModule(self, module):
        self.loadInputData = module

    def load(self, a_calculation=None):
        self.load_input(a_calculation)
        self.load_output(a_calculation)
        self.load_snapshot(a_calculation)

        # if len(self.inputArray) == 0:
        #     self.ui.ECP_run_button.setText("Create Scenario")
        #     self.ui.ECP_run_button.setStyleSheet(df.styleSheet_Create_Scenarios)
        # else:
        #     self.ui.ECP_run_button.setText("Run")
        #     self.ui.ECP_run_button.setStyleSheet(df.styleSheet_Run)

        self.ui.ECP_Input04.setMaximum(self.calcManager.cycle_burnup_values[-1] + 1000)

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
        # if self.current_calculation and self.last_table_created:
        #     if self.current_calculation.modified_date > self.last_table_created:
        #         self.ui.ECP_run_button.setText("Create Scenario")
        #         self.ui.ECP_run_button.setStyleSheet(df.styleSheet_Create_Scenarios)

    def value_changed(self, key):
        super().value_changed(key)
        # if self.current_calculation and self.last_table_created:
        #     if self.current_calculation.modified_date > self.last_table_created:
        #         self.ui.ECP_run_button.setText("Create Scenario")
        #         self.ui.ECP_run_button.setStyleSheet(df.styleSheet_Create_Scenarios)

    def date_time_changed(self, key):
        super().date_time_changed(key)
        # if self.current_calculation and self.last_table_created:
        #     if self.current_calculation.modified_date > self.last_table_created:
        #         self.ui.ECP_run_button.setText("Create Scenario")
        #         self.ui.ECP_run_button.setStyleSheet(df.styleSheet_Create_Scenarios)

    def settingAutoExclusive(self):
        self.buttonGroupTarget.addButton(self.ui.ECP_CalcTarget01)
        self.buttonGroupTarget.addButton(self.ui.ECP_CalcTarget02)

    def settingLinkAction(self):
        self.ui.pushButton_InputModel.clicked['bool'].connect(self.readModel)
        self.ui.ECP_InputSelect.currentIndexChanged['int'].connect(self.changeInputType)
        #self.ui.ECP_CalcTarget01.toggled['bool'].connect(self.settingTargetOpt)
        #self.ui.ECP_CalcTarget02.toggled['bool'].connect(self.settingTargetOpt)
        self.ui.ECP_Input00.currentIndexChanged['int'].connect(self.settingTargetOpt)

        self.ui.ECP_save_button.clicked['bool'].connect(self.start_save)
        self.ui.ECP_IO_button.clicked['bool'].connect(self.open_IO_table)
        self.ui.ECP_run_button.clicked['bool'].connect(self.start_calc)

        # self.unitButton01.clicked['bool'].connect(self.clickSaveAsExcel)
        # self.unitButton02.clicked['bool'].connect(self.resetPositionData)
        # self.unitButton03.clicked['bool'].connect(self.clearOuptut)

        self.unitChart.canvas.mpl_connect('pick_event', self.clickEvent01)
        self.unitChart02.canvas.mpl_connect('pick_event', self.clickEvent02)
    def open_IO_table(self):
        self.IO_table.open_IO_table()
    def changeInputType(self,idx):
        self.inputType = idx
        if(idx==df._INPUT_TYPE_USER_):
            self.ui.LabelSub_Selete01.setVisible(False)
            self.ui.pushButton_InputModel.setVisible(False)
        elif(idx==df._INPUT_TYPE_SNAPSHOT_):
            self.ui.LabelSub_Selete01.setText("Snapshot Setup")
            self.ui.pushButton_InputModel.setText("Load Data")
            self.ui.LabelSub_Selete01.setVisible(True)
            self.ui.pushButton_InputModel.setVisible(True)
            self.ui.pushButton_InputModel.setStyleSheet(df.styleSheet_Create_Scenarios)
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
                self.current_calculation.ecp_input.snapshot_text = burnup_text
                self.ui.ECP_Input04.setValue(inputArray[-1][1])
                self.save_snapshot()

        elif(self.inputType==df._INPUT_TYPE_FILE_CSV_):
            pass
            # status = self.loadInputData.openCSV()
            # if(status==False):
            #     return
            # [ self.nStep, self.targetASI,  self.inputArray ] = self.loadInputData.returnCSV()
            # self.initBU = self.inputArray[0][2]
            # self.targetEigen = self.inputArray[0][3]
            # rdcPerHour = abs((self.inputArray[1][1]-self.inputArray[0][1])/(self.inputArray[1][0]-self.inputArray[0][0]))
            # self.ui.SD_Input01.setValue(self.initBU)
            # self.ui.SD_Input02.setValue(self.targetEigen)
            # self.ui.SD_Input03.setValue(rdcPerHour)
            # self.ui.SD_Input04.setValue(self.targetASI)
            # if len(self.inputArray) == 0:
            #     self.ui.SD_run_button.setText("Create Scenario")
            #     self.ui.SD_run_button.setStyleSheet(df.styleSheet_Create_Scenarios)
            #     self.ui.SD_run_button.setDisabled(False)
            # else:
            #     self.ui.SD_run_button.setText("Run")
            #     self.ui.SD_run_button.setStyleSheet(df.styleSheet_Run)
            #     self.ui.SD_run_button.setDisabled(False)
            # self.SD_TableWidget.addInputArray(self.inputArray)
            # self.last_table_created = datetime.datetime.now()

    def settingTargetOpt(self):
        search_type = 0
        # self.ui.ECP_run_button.setText("Create Scenario")
        # self.ui.ECP_run_button.setStyleSheet(df.styleSheet_Create_Scenarios)
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
        self.clearOutput()

        self.inputArray = []

        # startT    = self.ui.ECP_Input01_time.time()
        # startD    = self.ui.ECP_Input01_date.date()

        deltaTime = self.ui.ECP_Input12.value()
        second_criticality = deltaTime * 3600.0
        #check input
        if not 0 < second_criticality <= 300*3600:
            msgBox = QMessageBoxWithStyle(self.get_ui_component())
            msgBox.setText("Delta time should be in between 0 and 300 hour\nYour delta time is {}"
                           .format(second_criticality//3600))
            msgBox.setStandardButtons(QMessageBox.Ok)
            msgBox.setCustomStyle()
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

        self.time_index = time_index

        if deltime >= second_criticality//3600.0:
            self.inputArray.append([second_criticality//3600.0, 0.0, bp, eigen])


        timeArray = []
        for idx in range(len(self.inputArray)):
            timeArray.append(self.inputArray[idx][0])

        self.unitChart.insertTime(timeArray)
        self.unitChart02.insertTime(timeArray)

        output_array.append([0.0, 0.0, 0.0, 0.0, 0.0, 381.0, 381.0, 381.0, 381.0])
        output_array.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        for i in range(len(self.inputArray)-2):
            if self.ECP_TargetOpt==df.select_Boron:
                self.shutdown_P_Pos = self.ui.ECP_Input14.value()
                self.shutdown_r5Pos = self.ui.ECP_Input15.value()
                self.shutdown_r4Pos = self.ui.ECP_Input16.value()
                an_output = [0.0, 0.0, 0.0, 0.0, 0.0, self.shutdown_P_Pos, self.shutdown_r5Pos, self.shutdown_r4Pos, 381.0]
            else:
                self.shutdown_ppm = self.ui.ECP_Input13.value()
                an_output = [0.0, self.shutdown_ppm, 0.0, 0.0, 0.0, 381.0, 381.0, 381.0, 381.0]
            output_array.append(an_output)

        self.ECP_TableWidget.addInputArray(self.inputArray)
        self.IO_table.IO_TableWidget.addInputArray(self.inputArray)
        self.ECP_TableWidget.makeOutputTable(output_array)
        self.IO_table.IO_TableWidget.makeOutputTable(output_array)
        # self.startD = self.ui.ECP_Input01_date.date()
        # self.startT = self.ui.ECP_Input01_time.time()
        # self.startTime = QDateTime()
        # self.startTime.setDate(self.startD)
        # self.startTime.setTime(self.startT)
        # self.startTime = self.ui.ECP_Input01.dateTime()
        self.deltaTime = self.ui.ECP_Input12.value()
        # self.endTime = self.startTime.addSecs(self.deltaTime * 3600.0)
        # self.endTime = self.ui.ECP_Input12.dateTime()
        self.pw = self.ui.ECP_Input03.value()
        self.bp = self.ui.ECP_Input04.value()
        self.mtc = df.inlet_temperature
        self.eigen = 1.0
        self.P_Pos = 381.0
        self.r5Pos = 381.0
        self.r4Pos = 381.0
        self.inputArray = [self.pw, self.bp, self.mtc, self.eigen, self.P_Pos, self.r5Pos, self.r4Pos]
        if self.ECP_TargetOpt == df.select_Boron:
            self.shutdown_P_Pos = self.ui.ECP_Input14.value()
            self.shutdown_r5Pos = self.ui.ECP_Input15.value()
            self.shutdown_r4Pos = self.ui.ECP_Input16.value()
            self.inputArray.append(self.shutdown_P_Pos)
            self.inputArray.append(self.shutdown_r5Pos)
            self.inputArray.append(self.shutdown_r4Pos)
        elif self.ECP_TargetOpt == df.select_RodPos:
            self.shutdown_ppm = self.ui.ECP_Input13.value()
            self.inputArray.append(self.shutdown_ppm)
        self.inputArray.append(self.deltaTime)

    def resizeRadialWidget(self,size):
        self.map_opr1000_frame.setMaximumSize(QSize(size, size))
        self.map_apr1400_frame.setMaximumSize(QSize(size, size))

    def get_input(self, calculation_object):
        return calculation_object.ecp_input

    def get_output(self, calculation_object):
        return calculation_object.ecp_output

    def get_manager_output(self):
        return self.calcManager.results.ecp_output

    def set_manager_output(self, output):
        self.calcManager.results.ecp_output = output

    def create_default_input_output(self, user):
        now = datetime.datetime.now()
        ecp_input = ECP_Input.create(
            bs_ndr_date=now,
            bs_ndr_time=now,
        )
        ecp_output = ECP_Output.create()

        ecp_calculation = Calculations.create(user=user,
                                              calculation_type=cs.CALCULATION_ECP,
                                              created_date=now,
                                              modified_date=now,
                                              ecp_input=ecp_input,
                                              ecp_output=ecp_output
                                              )

        return ecp_calculation, ecp_input, ecp_output

    def start_calc(self):
        super().start_calc()


        # if self.ui.ECP_run_button.text() == cs.RUN_BUTTON_CREATE_SCENARIO:
        #     self.setSuccecciveInput()
        #     self.load()
        # else:
        self.setSuccecciveInput()
        self.load()
        #self.setSuccecciveInput()
        # Set Input Variable
        # self.startD = self.ui.ECP_Input01_date.date()
        # self.startT = self.ui.ECP_Input01_time.time()
        # self.startTime = QDateTime()
        # self.startTime.setDate(self.startD)
        # self.startTime.setTime(self.startT)
        #self.startTime = self.ui.ECP_Input01.dateTime()
        # self.endTime = self.startTime.addSecs(self.deltaTime * 3600.0)
        #self.endTime = self.ui.ECP_Input12.dateTime()

        self.deltaTime = self.ui.ECP_Input12.value()
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

        pArray = []

        if self.ui.ECP_InputSelect.currentIndex() == 1:
            for value in self.snapshotArray:
                pArray.append([value[0], value[2], value[1], value[3]] + [0, 0, 0] + value[4:])

        if len(pArray) == 0:
            pArray.append([0, 100, self.bp, self.eigen, 0, 0, 0,])

        error = self.check_calculation_input()

        if not error:
            self.start_calculation_message = SplashScreen()
            self.start_calculation_message.killed.connect(self.killManagerProcess)
            # print("timedex",self.time_index+2)
            self.start_calculation_message.init_progress(self.time_index+2, 500)
            self.cal_start = True
            self.current_index = 0

            # self.ui.ECP_run_button.setText("Running....")
            # self.ui.ECP_run_button.setDisabled(True)
            self.ECP_TableWidget.last_update = 0
            self.IO_table.IO_TableWidget.last_update = 0
            self.ECP_TableWidget.clearOutputArray()
            self.IO_table.IO_TableWidget.clearOutputArray()

            self.queue.put((df.CalcOpt_ECP, self.ECP_TargetOpt, inputArray, pArray))

    def check_calculation_input(self):

        if self.ui.ECP_Input04.value() >= self.calcManager.cycle_burnup_values[-1]+1000:
            msgBox = QMessageBoxWithStyle(self.get_ui_component())
            msgBox.setWindowTitle("Burnup Out of Range")
            msgBox.setText("{}MWD/MTU excedes EOC Cycle Burnup({} MWD/MTU)\n"
                           "Put Cycle Burnup less than {}MWD/MTU".format(self.ui.ECP_Input04.value(),
                                                            self.calcManager.cycle_burnup_values[-1],
                                                            self.calcManager.cycle_burnup_values[-1]+1000))
            msgBox.setStandardButtons(QMessageBox.Ok)
            msgBox.setCustomStyle()
            #msgBox.setWindowFlag(QtCore.Qt.FramelessWindowHint)
            result = msgBox.exec_()
            # if result == QMessageBox.Cancel:
            return True
        return False

    def ECP_output_index_changed(self):
        index = self.ui.ECP_Output1.currentIndex()
        if len(self.calcManager.results.out_ppm) > index:
            self.show_output_values(index)

            p = 381.0
            r5 = self.calcManager.results.rodPos[index][0]
            r4 = self.calcManager.results.rodPos[index][1]
            r3 = self.calcManager.results.rodPos[index][2]

            data = {' P': p, 'R5': r5, 'R4': r4, 'R3': r3}
            self.axialWidget.drawBar(data)
            # print(data)

            pd1d = self.calcManager.results.ecpP1d[index]
            self.axialWidget.drawAxial(pd1d[self.calcManager.results.kbc:self.calcManager.results.kec],
                                       self.calcManager.results.axial_position)

    def showOutput(self):
        if self.start_calculation_message:
            self.start_calculation_message.progress()

        # if len(self.calcManager.results.out_ppm) > 0:
        #     self.show_output_values(len(self.calcManager.results.out_ppm) -1)

        #self.appendPointData(self.calcManager.outputArray[last_update:], last_update)
        if len(self.calcManager.results.ecp_output) > 0:
            self.appendPointData(self.calcManager.results.ecp_output, 0)

            # last_update = self.ECP_TableWidget.last_update
            # last_update = self.IO_table.IO_TableWidget.last_update
            last_update = 0
            self.ECP_TableWidget.appendOutputTable(self.calcManager.results.ecp_output[last_update:], last_update)
            self.IO_table.IO_TableWidget.appendOutputTable(self.calcManager.results.ecp_output[last_update:],
                                                           last_update)
            # outputs = self.calcManager.results.ecp_output
            # pd2d = outputs[-1][df.asi_o_p2d]
            # pd1d = outputs[-1][df.asi_o_p1d]
            # self.axialWidget.drawAxial(pd1d[self.calcManager.results.kbc:self.calcManager.results.kec],
            #                            self.calcManager.results.axial_position)
            # self.radialWidget.slot_astra_data(pd2d)

    def finished(self, is_success):

        self.is_run_selected = False

        if self.start_calculation_message:
            self.start_calculation_message.close()

        pd1d = [-1]
        # rp = []
        # r5 = []
        # r4 = []
        # r3 = []
        for row in range(len(self.calcManager.results.ecp_output)):
            # pd2d.append(self.calcManager.results.shutdown_output[row][df.asi_o_p2d])
            if self.calcManager.results.ecp_output[row][df.asi_o_power] > 0:
                pd1d.append(max(self.calcManager.results.ecp_output[row][df.asi_o_p1d][self.calcManager.results.kbc:self.calcManager.results.kec]))

            # rp.append(self.calcManager.results.shutdown_output[row][df.asi_o_bp])
            # r5.append(self.calcManager.results.shutdown_output[row][df.asi_o_b5])
            # r4.append(self.calcManager.results.shutdown_output[row][df.asi_o_b4])
            # r3.append(self.calcManager.results.shutdown_output[row][df.asi_o_b3])

        self.axialWidget.setMaximumPower(max(pd1d))

        # self.ui.ECP_run_button.setText("Run")
        # self.ui.ECP_run_button.setDisabled(False)

        # if len(self.calcManager.out_ppm) > 0:
        #     self.show_output_values()
        # self.ECP_output_index_changed()
        if is_success == self.calcManager.SUCC:
            self.save_output(self.calcManager.results.ecp_output)
            self.showOutput()
        else:
            ecp_output = ECP_Output.create(
                success=False,
                table="",
            )

            self.current_calculation.ecp_output = ecp_output
            self.current_calculation.ecp_output.save()
            self.current_calculation.save()


    def appendPointData(self, outputArray, start):

        # self.unitChartClass.clear()
        num = len(outputArray)

        if num == 0:
            return

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

        pos_Fxy = []
        pos_Fr = []
        pos_Fq = []
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
            if power[-1] < 20.0:
                asi_band_time += outputArray[i][df.asi_o_time]
            #pos_Fq.append(outputArray[i][df.asi_o_fq])


        #rodPos = [posP, posR5, posR4, posR3, ]
        posOpt = [ posASI, posCBC, power ]
        posOpt02 = [ posASI, posCBC, power  ]
        self.unitChart.insertDataSet(time, posP, posR5, posR4, posOpt )
        self.unitChart02.insertDataSet(time, pos_Fxy, pos_Fr, pos_Fq, posOpt02 )
        self.unitChart.updateRdcPerHour(self.rdcPerHour,False, asi_band_time)

    def appendPointData(self, outputArray, start):

        posP = []
        posR5 = []
        posR4 = []
        posCBC = []
        posReact = []
        time = []
        #asi_o_asi indicates reactivity here
        current_time = 0
        for i in range(len(outputArray)):
            posCBC.append(outputArray[i][df.asi_o_boron])
            posReact.append(outputArray[i][df.asi_o_reactivity])
            posP.append(outputArray[i][df.asi_o_bp])
            posR5.append(outputArray[i][df.asi_o_b5])
            posR4.append(outputArray[i][df.asi_o_b4])
            # posR4.append(outputArray[i][df.asi_o_b3])
            current_time += outputArray[i][df.asi_o_time]
            time.append(current_time)
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

        self.unitChart.insertTime(time)
        self.unitChart02.insertTime(time)

        posOpt = [ posCBC,]
        self.unitChart.insert_ECP_DataSet(time, posP, posR5, posR4, posOpt)
        posOpt1 = [ posReact ,]
        self.unitChart02.insert_ECP_DataSet(time, posP, posR5, posR4, posOpt1)

    def clickSaveAsExcel(self):
        self.ECP_TableWidget.clickSaveAsExcel()

    def resetPositionData(self):
        self.ECP_TableWidget.resetPositionData()

    def addOutput(self):
        if not self.unitChart:
            lay = self.ui.grid_ECP_frameChart
            # self.unitChart = trendWidget(self.fixedTime_flag, self.ASI_flag, self.CBC_flag, self.Power_flag)
            self.unitChart = trendWidget(self.fixedTime_flag, self.ASI_flag, self.CBC_flag, self.Power_flag)
            sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            sizePolicy5.setHorizontalStretch(10)
            sizePolicy5.setVerticalStretch(10)
            self.unitChart.setSizePolicy(sizePolicy5)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(self.unitChart, 0, 0, 1, 1)

        if not self.unitChart02:

            lay02 = self.ui.grid_ECP_frameChart02
            self.unitChart02 = trendWidget(self.fixedTime_flag, False, False, False, True)#self.ui.SD_widgetChart)
            sizePolicy6 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            sizePolicy6.setHorizontalStretch(10)
            sizePolicy6.setVerticalStretch(10)
            self.unitChart02.setSizePolicy(sizePolicy6)
            lay02.setContentsMargins(0, 0, 0, 0)
            lay02.addWidget(self.unitChart02, 0, 0, 1, 1)
        if not self.radialWidget:

            self.map_opr1000 = opr(self.ui.ECP_InputLP_Dframe,self.ui.gridLayout_ECP_InputLP_Dframe)
            self.map_opr1000_frame , self.map_opr1000_grid = self.map_opr1000.return_opr_frame()
            self.ui.gridLayout_ECP_InputLP_Dframe.addWidget(self.map_opr1000_frame , 0, 0, 1, 1)
            self.radialWidget = RadialWidget(self.map_opr1000_frame , self.map_opr1000_grid, df.type_opr1000)
            #self.map_opr1000_frame.hide()
            self.map_apr1400 = apr(self.ui.ECP_InputLP_Dframe,self.ui.gridLayout_ECP_InputLP_Dframe)
            self.map_apr1400_frame, self.map_apr1400_grid = self.map_apr1400.return_apr_frame()
            self.ui.gridLayout_ECP_InputLP_Dframe.addWidget(self.map_apr1400_frame, 0, 0, 1, 1)
            self.radialWidget02 = RadialWidget(self.map_apr1400_frame,self.map_apr1400_grid, df.type_apr1400)
            self.map_apr1400_frame.hide()

        if not self.axialWidget:
            # 05. Insert Axial Chart
            sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            sizePolicy5.setHorizontalStretch(6)
            sizePolicy5.setVerticalStretch(10)
            #sizePolicy5.setHeightForWidth(self.ui.RO_WidgetAxial.sizePolicy().hasHeightForWidth())
            self.ui.ECP_WidgetAxial.setSizePolicy(sizePolicy5)
            layA = QtWidgets.QVBoxLayout(self.ui.ECP_WidgetAxial)
            layA.setContentsMargins(0,0,0,0)
            self.axialWidget = AxialWidget()
            layA.addWidget(self.axialWidget)

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

    def clearOutput(self):
        self.ECP_TableWidget.clearOutputArray()
        self.ECP_TableWidget.clearOutputRodArray()
        self.ECP_TableWidget.last_update = 0
        self.unitChart.clearData()
        self.unitChart02.clearData()