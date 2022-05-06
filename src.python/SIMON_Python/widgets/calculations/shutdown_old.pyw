import sys
import platform
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import (QCoreApplication, QPropertyAnimation, QDate, QDateTime, QMetaObject, QObject, QPointF, QRect, QSize, QTime, QUrl, Qt, QEvent, Slot)
from PyQt5.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont, QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter, QPixmap, QRadialGradient)
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
import PyQt5.QtCharts as QtCharts
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from model import *
import datetime
import os
import pandas as pd
import glob
from ui_main_rev18 import Ui_MainWindow
import Definitions as df
import widgets.utils.PyUnitButton as ts

import constants as cs
import utils as ut

from widgets.calculations.calculation_widget import CalculationWidget

import widgets.utils.PyRodPositionBarChart as unitChart
import widgets.utils.PyRodPositionSplineChart as unitSplineChart
import widgets.utils.PyTableDoubleSpinBox as unitDoubleBox
from widgets.output.axial.axial_plot import AxialWidget
from widgets.output.radial.radial_graph import RadialWidget
import math
import widgets.utils.PyStartButton as startB
from PyQt5.QtWidgets import QStyledItemDelegate

from widgets.utils.PySaveMessageBox import PySaveMessageBox

from itertools import islice
from random import random
from time import perf_counter
from cusfam import *
import time
from Simon import Simon
import os
import multiprocessing
from threading import Thread

_POINTER_A_ = "self.ui.SD_DB"
_NUM_PRINTOUT_ = 5
_MINIMUM_ITEM_COUNT = 40

pointers = {

    #"ASI_Snapshot": "ss_snapshot_file",
    #"rdc01": "ndr_burnup",
    #"rdc02": "ndr_power"#,
    # "ASI_RodPos05": "ndr_bank_position_5",
    # "ASI_RodPos05_Unit": "ndr_bank_type",
    # "ASI_RodPos04": "ndr_bank_position_4",
    # "ASI_RodPos03": "ndr_bank_position_3",
    # "ASI_RodPos_P": "ndr_bank_position_P",

}

pointers = {

    "SD_Snapshot": "ss_snapshot_file",
    "SD_Input01": "ndr_burnup",
    "SD_Input02": "ndr_target_keff",
    "SD_Input03": "ndr_target_keff",

}


class Shutdown_Widget(CalculationWidget):

    def __init__(self, db, ui):

        super().__init__(db, ui, ui.SD_DB, SD_Input, pointers)

        self.input_pointer = "self.current_calculation.asi_input"

        # 01. Setting Initial ASI Control Input Data
        self.SD_PosBank5 = QPointF(0.0, 0.0)
        self.SD_PosBank4 = QPointF(0.5, 0.4*381.0)
        self.SD_PosBank3 = QPointF(2.0, 0.7*381.0)
        self.SD_PosBankP = QPointF(3.0, 101.0)
        self.RodPosUnit = df.RodPosUnit_cm
        self.SD_CalcOpt = df.select_none
        self.unitChangeFlag = False
        self.initSnapshotFlag = False
        self.inputArray = []
        # Input Dataset for SD_TableWidget Rod Position
        self.calc_rodPosBox = []
        self.calc_rodPos = []
        self.tableDatasetFlag = False
        self.rodPosChangedFlag = False
        self.outputFlag = False
        self.rodPosChangedHistory = []

        # 02. Setting Initial Rod Position Bar Chart
        self.unitChartClass = unitSplineChart.UnitSplineChart(self.RodPosUnit)

        # 03. Setting Widget Interactions
        self.buttonGroupInput = QtWidgets.QButtonGroup()
        self.reductionGroupInput = QtWidgets.QButtonGroup()
        self.settingAutoExclusive()
        self.settingLinkAction()


        # 04. Setting Initial UI widget
        self.ui.SD_Main03.hide()
        self.ui.SD_tableWidget.horizontalHeader().setVisible(True)
        self.ui.SD_tableWidget.verticalHeader().setVisible(True)

        #self.ui.tableWidgetFA.horizontalHeader().setMinimumHeight(80)
        #self.ui.tableWidgetFA.horizontalHeader().setMaximumHeight(80)
        #self.ui.tableWidgetFA.horizontalHeader().setSectionsMovable(True)
        #self.ui.tableWidgetFA.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # Setting horizontalHeader click event
        #self.proxy = QSortFilterProxyModel(self)
        #self.model = QtGui.QStandardItemModel(self)

        # 05. Insert ChartView to chartWidget
        self.nSet = 1
        lay = QtWidgets.QVBoxLayout(self.ui.SD_widgetChart)
        lay.setContentsMargins(0,0,0,0)
        unitChartView = self.unitChartClass.returnChart()
        lay.addWidget(unitChartView)
        width = []
        for column in range(self.ui.SD_tableWidget.horizontalHeader().count()):
            self.ui.SD_tableWidget.horizontalHeader().setSectionResizeMode(column,QHeaderView.ResizeToContents)
            width.append(self.ui.SD_tableWidget.horizontalHeader().sectionSize(column))        
   
        wfactor = self.ui.SD_tableWidget.horizontalHeader().width() / sum(width)
        for column in range(self.ui.SD_tableWidget.horizontalHeader().count()):
            self.ui.SD_tableWidget.horizontalHeader().setSectionResizeMode(column, QHeaderView.Stretch)
            self.ui.SD_tableWidget.horizontalHeader().resizeSection(column, width[column]*wfactor)
        
        #self.ui.SD_tableWidget.resizeColumnsToContents()
        pass
            
        layR = QtWidgets.QVBoxLayout(self.ui.SD_WidgetRadial)
        layR.setContentsMargins(0,0,0,0)
        self.RadialWidget = RadialWidget()
        layR.addWidget(self.RadialWidget)



        layA = QtWidgets.QVBoxLayout(self.ui.SD_WidgetAxial)
        layA.setContentsMargins(0,0,0,0)
        self.AxialWidget = AxialWidget()
        #self.AxialWidget.setStyle()
        layA.addWidget(self.AxialWidget)
        # 06. link signal and slot for start calculation
        #self.makestartbutton(self.startfunc,super().savefunc, self.ui.sd_main05_calcbutton, self.ui.sd_control_calcbutton_grid)

        self.ui.SD_tabWidget.setCurrentIndex(0)
        self.ui.SD_tabWidget.setTabEnabled(1,False)
        self.ui.SD_tabWidget.setTabVisible(1,False)
        self.ui.SD_tabInput.hide()

        # 07. Make Successic Input Setup for Power Reduction Calculation Option
        #tmp = QStyledItemDelegate()
        #tmp.createEditor(self.ui.SD_tableWidget,)

        #self.setTableSetting()
        #self.ui.SD_tableWidget.
        self.s = Simon("C:/Users/geonho/Documents/SVN/SIMON/SIMON_CODE/rst/Y301ASBDEP.SMG", "C:/Users/geonho/Documents/SVN/SIMON/SIMON_CODE/run/KMYGN34C01_PLUS7_XSE.XS", "C:/Users/geonho/Documents/SVN/SIMON/SIMON_CODE/rst/Y301ASBDEP")

    def load(self):
        pass
        #PLANT AND RESTART FILES
        # self.ui.SD_PlantName.clear()
        # self.ui.SD_PlantCycle.clear()
        #
        # query = LoginUser.get(LoginUser.username == cs.ADMIN_USER)
        # user = query.login_user
        # plants, errors = ut.getPlantFiles(user)
        # for plant in plants:
        #     self.ui.SD_PlantName.addItem(plant)
        # self.ui.SD_PlantName.setCurrentText(user.plant_file)
        #
        # restarts, errors = ut.getRestartFiles(user)
        # for restart in restarts:
        #     self.ui.SD_PlantCycle.addItem(restart)
        # self.ui.SD_PlantCycle.setCurrentText(user.restart_file)

        self.load_input()
        self.load_recent_calculation()

    def set_all_component_values(self, SD_input):

        if SD_input.calculation_type == 0:
            self.ui.SD_InpOpt2_NDR.setChecked(True)
        else:
            self.ui.SD_InpOpt1_Snapshot.setChecked(True)

        #Set values to components
        # for key in pointers.keys():
        #
        #     component = eval("self.ui.{}".format(key))
        #     value = eval("SD_input.{}".format(pointers[key]))
        #
        #     if pointers[key]:
        #         if isinstance(component, QComboBox):
        #             component.setCurrentText(value)
        #         else:
        #             component.setValue(float(value))

    def settingAutoExclusive(self):
        self.buttonGroupInput.addButton(self.ui.SD_InpOpt1_Snapshot)
        self.buttonGroupInput.addButton(self.ui.SD_InpOpt2_NDR)

        self.reductionGroupInput.addButton(self.ui.SD_RdcOpt01)
        self.reductionGroupInput.addButton(self.ui.SD_RdcOpt02)
        #self.reductionGroupInput.addButton(self.ui.SD_Reduction003)

    def settingLinkAction(self):
        pass
        self.ui.SD_InpOpt1_Snapshot.clicked['bool'].connect(self.settingInputOpt)
        self.ui.SD_InpOpt2_NDR.clicked['bool'].connect(self.settingInputOpt)

        self.ui.SD_InpOpt1_Snapshot.clicked['bool'].connect(self.updateBankPosSnapshot)
        self.ui.SD_InpOpt2_NDR.clicked['bool'].connect(self.updateBankPosUserInput)

        self.ui.SD_RDC_apply.clicked['bool'].connect(self.setSuccecciveInput)
        self.ui.SD_run_button.clicked['bool'].connect(self.startCalc)
        #self.ui.SD_run_button.clicked['bool'].connect(self.startCalcTest)

        self.ui.SD_tableWidget_button01.clicked['bool'].connect(self.clickSaveAsExcel)
        self.ui.SD_tableWidget_button02.clicked['bool'].connect(self.resetPositionData)
        self.ui.SD_tableWidget_button03.clicked['bool'].connect(self.ASI_ReCalculation)



        # self.ui.SD_tableWidget.cellClicked.connect(self.cell_click)
        # self.ui.SD_tableWidget.cellChanged['int','int'].connect(self.cell_change)
        # self.ui.SD_RodPos05.valueChanged['double'].connect(self.rodPosChangedEvent)
        # self.ui.SD_RodPos04.valueChanged['double'].connect(self.rodPosChangedEvent)
        # self.ui.SD_RodPos03.valueChanged['double'].connect(self.rodPosChangedEvent)
        # self.ui.SD_RodPos_P.valueChanged['double'].connect(self.rodPosChangedEvent)

        # self.ui.SD_RodPos05_Unit.currentIndexChanged['int'].connect(self.changeRodPosUnit)

    def settingInputOpt(self):
        calculatin_type = 0
        if (self.ui.SD_InpOpt2_NDR.isChecked()):
            self.SD_CalcOpt = df.select_NDR
            self.ui.SD_Main03.show()
        elif (self.ui.SD_InpOpt1_Snapshot.isChecked()):
            self.SD_CalcOpt = df.select_snapshot
            self.ui.SD_Main03.hide()
            calculatin_type = 1

        if self.current_calculation:
            current_input = self.get_input(self.current_calculation)
            current_input.calculation_type = calculatin_type
            current_input.save()

    def rodPosChangedEvent(self):
        if (self.unitChangeFlag == False and self.initSnapshotFlag == False):
            self.updateBankPosUserInput()
        else:
            pass

    def updateBankPosSnapshot(self):
        # TODO, Read Rod Position from Snapshot
        self.initSnapshotFlag = True
        # self.SD_PosBank5 = 50.0 * df.convertRodPosUnit[0][self.RodPosUnit]
        # self.SD_PosBank4 = 50.0 * df.convertRodPosUnit[0][self.RodPosUnit]
        # self.SD_PosBank3 = 50.0 * df.convertRodPosUnit[0][self.RodPosUnit]
        # self.SD_PosBankP = 50.0 * df.convertRodPosUnit[0][self.RodPosUnit]
        # self.ui.SD_RodPos05.setValue(self.SD_PosBank5)
        # self.ui.SD_RodPos04.setValue(self.SD_PosBank4)
        # self.ui.SD_RodPos03.setValue(self.SD_PosBank3)
        # self.ui.SD_RodPos_P.setValue(self.SD_PosBankP)
        # self.replaceRodPosition()
        self.initSnapshotFlag = False

    def updateBankPosUserInput(self):
        pass
        # self.SD_PosBank5 = self.ui.SD_RodPos05.value()
        # self.SD_PosBank4 = self.ui.SD_RodPos04.value()
        # self.SD_PosBank3 = self.ui.SD_RodPos03.value()
        # self.SD_PosBankP = self.ui.SD_RodPos_P.value()
        #self.replaceRodPosition()

    def replaceRodPosition(self):
        self.unitChartClass.replaceRodPosition(self.SD_PosBank5, self.SD_PosBank4, self.SD_PosBank3,
                                               self.SD_PosBankP, self.RodPosUnit)

        if self.load_rod_tab and self.ui.SD_tabWidget.currentIndex() != 2:
            self.ui.SD_tabWidget.setCurrentIndex(2)

    def changeRodPosUnit(self, index):
        self.unitChangeFlag = True
        self.rodPosChangedEvent()
        text = "   " + self.ui.SD_RodPos05_Unit.currentText()
        self.ui.SD_RodPos04_Unit.setText(text)
        self.ui.SD_RodPos03_Unit.setText(text)
        self.ui.SD_RodPos_P_Unit.setText(text)

        currentOpt = self.RodPosUnit
        self.SD_PosBank5 = self.SD_PosBank5 * df.convertRodPosUnit[currentOpt][index]
        self.SD_PosBank4 = self.SD_PosBank4 * df.convertRodPosUnit[currentOpt][index]
        self.SD_PosBank3 = self.SD_PosBank3 * df.convertRodPosUnit[currentOpt][index]
        self.SD_PosBankP = self.SD_PosBankP * df.convertRodPosUnit[currentOpt][index]

        self.ui.SD_RodPos05.setValue(self.SD_PosBank5)
        self.ui.SD_RodPos04.setValue(self.SD_PosBank4)
        self.ui.SD_RodPos03.setValue(self.SD_PosBank3)
        self.ui.SD_RodPos_P.setValue(self.SD_PosBankP)

        self.RodPosUnit = index
        self.replaceRodPosition()
        self.unitChangeFlag = False

    def makeStartButton(self, startFunc, saveFunc,  frame, grid):
        pass
        #self.ui.SD_save_button.clicked['bool'].connect(saveFunc)
        #self.ui.SD_run_button.clicked['bool'].connect(startFunc)
        """
        self.SD_StartButton = startB.startButton(df.CalcOpt_SD, startFunc, frame)

        self.SD_StartButton.setObjectName(u"SD_Control_StartButton")
        sizePolicy3 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy3.setHeightForWidth(self.SD_StartButton.sizePolicy().hasHeightForWidth())
        self.SD_StartButton.setSizePolicy(sizePolicy3)
        self.SD_StartButton.setMinimumSize(QSize(200, 40))
        self.SD_StartButton.setMaximumSize(QSize(16777215, 40))
        font2 = QFont()
        font2.setFamily(u"Segoe UI")
        font2.setPointSize(14)
        font2.setBold(False)
        font2.setWeight(50)
        self.SD_StartButton.setFont(font2)
        self.SD_StartButton.setStyleSheet(u"QPushButton {\n"
                                                 "	border: 2px solid rgb(52, 59, 72);\n"
                                                 "	border-radius: 5px;	\n"
                                                 "	background-color: rgb(85, 170, 255);\n"
                                                 "}\n"
                                                 "QPushButton:hover {\n"
                                                 "	background-color: rgb(72, 144, 216);\n"
                                                 "	border: 2px solid rgb(61, 70, 86);\n"
                                                 "}\n"
                                                 "QPushButton:pressed {	\n"
                                                 "	background-color: rgb(52, 59, 72);\n"
                                                 "	border: 2px solid rgb(43, 50, 61);\n"
                                                 "}")

        self.SD_SaveButton = startB.startButton(df.CalcOpt_SD, saveFunc, frame)

        self.SD_SaveButton.setObjectName(u"SD_Control_SaveButton")
        sizePolicy3 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy3.setHeightForWidth(self.SD_SaveButton.sizePolicy().hasHeightForWidth())
        self.SD_SaveButton.setSizePolicy(sizePolicy3)
        self.SD_SaveButton.setMinimumSize(QSize(200, 40))
        self.SD_SaveButton.setMaximumSize(QSize(16777215, 40))
        font2 = QFont()
        font2.setFamily(u"Segoe UI")
        font2.setPointSize(14)
        font2.setBold(False)
        font2.setWeight(50)
        self.SD_SaveButton.setFont(font2)
        self.SD_SaveButton.setStyleSheet(u"QPushButton {\n"
                                                 "	border: 2px solid rgb(52, 59, 72);\n"
                                                 "	border-radius: 5px;	\n"
                                                 "	background-color: rgb(52, 59, 72);\n"
                                                 "}\n"
                                                 "QPushButton:hover {\n"
                                                 "	background-color: rgb(85, 170, 255);\n"
                                                 "	border: 2px solid rgb(61, 70, 86);\n"
                                                 "}\n"
                                                 "QPushButton:pressed {	\n"
                                                 "	background-color: rgb(72, 144, 216);\n"
                                                 "	border: 2px solid rgb(43, 50, 61);\n"
                                                 "}")

        grid.addWidget(self.SD_SaveButton, 1, 0, 1, 1)
        grid.addWidget(self.SD_StartButton, 1, 1, 1, 1)
        self.SD_SaveButton.setText(QCoreApplication.translate("MainWindow", u"Save", None))
        self.SD_StartButton.setText(QCoreApplication.translate("MainWindow", u"Run", None))
        """

    def get_input(self, calculation_object):
        return calculation_object.sd_input

    def getDefaultInput(self, user):
        now = datetime.datetime.now()

        SD_input = SD_Input.create(
            calculation_type=0,
            search_type=0,
            ss_snapshot_file="",
            ndr_burnup=0,
            ndr_target_keff=0,
            ndr_power=0,
        )

        SD_calculation = Calculations.create(user=user,
                                              calculation_type=cs.CALCULATION_SD,
                                              created_date=now,
                                              modified_date=now,
                                              sd_input=SD_input
                                              )
        return SD_calculation, SD_input

    def setSuccecciveInput(self):
        # Initialize and reset Dataset for TableWidget
        self.ui.SD_tableWidget.setRowCount(0)
        self.inputArray = []
        self.calc_rodPos = []
        self.calc_rodPosBox = []
        self.tableDatasetFlag = False

        initBU     = self.ui.SD_Input01.value()
        targetEigen = self.ui.SD_Input02.value()
        targetASI = self.ui.SD_Input04.value()

        self.initBU     = initBU
        self.targetEigen = targetEigen
        self.targetASI = targetASI

        rdcPerHour = self.ui.SD_Input03.value()
        EOF_Power  = 0.0 #self.ui.rdc04.value()

        # TODO, make Except loop
        if(rdcPerHour==0.0):
            print("Error!")
            return

        nStep = math.ceil(( 100.0 - EOF_Power ) / rdcPerHour + 1.0)
        #print(nStep)

        # Initial Settings for SD_TableWidget
        self.ui.SD_tableWidget.setRowCount(nStep)
        self.setTableItemSetting()
        self.nInput = nStep

        for i in range(nStep-1):
            time = 1.0 * i
            power = 100.0 - i * rdcPerHour
            unitArray = [ time, power, initBU, targetEigen ]
            self.inputArray.append(unitArray)

        time = 100.0 / rdcPerHour
        power = 0.0
        unitArray = [ time, power, initBU, targetEigen]
        self.inputArray.append(unitArray)
        _translate = QtCore.QCoreApplication.translate
        for iStep in range(nStep-1):
            for iRow in range(4):
                item = QtWidgets.QTableWidgetItem()
                font = QtGui.QFont()
                font.setPointSize(11)
                font.setBold(False)
                font.setWeight(75)
                item.setFont(font)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                if (iRow == 0):
                    text = "%.1f" % self.inputArray[iStep][iRow]
                elif (iRow == 1):
                    text = "%.2f" % self.inputArray[iStep][iRow]
                elif (iRow == 2):
                    text = "%.1f" % self.inputArray[iStep][iRow]
                elif (iRow == 3):
                    text = "%.5f" % self.inputArray[iStep][iRow]
                item.setText(_translate("Form", text))
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                self.ui.SD_tableWidget.setItem(iStep, iRow, item)

        for iRow in range(4):
            item = QtWidgets.QTableWidgetItem()
            font = QtGui.QFont()
            font.setPointSize(11)
            font.setBold(False)
            font.setWeight(75)
            item.setFont(font)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            if (iRow == 0):
                text = "%.3f" % self.inputArray[nStep-1][iRow]
            elif (iRow == 1):
                text = "%.2f" % self.inputArray[nStep-1][iRow]
            elif (iRow == 2):
                text = "%.1f" % self.inputArray[nStep-1][iRow]
            elif (iRow == 3):
                text = "%.5f" % self.inputArray[nStep-1][iRow]
            item.setText(_translate("Form", text))
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.ui.SD_tableWidget.setItem(nStep-1, iRow, item)


    def startCalc(self):
        if(multiprocessing.cpu_count()>1):
            self.multiprocessingFlag = True
            self.thread001 = Thread(target=self.ASI_Calculation, args=())
            self.thread001.start()
        else:
            self.multiprocessingFlag = False
            self.ASI_Calculation()


    def ASI_Calculation(self):
        self.tableDatasetFlag = True
        self.outputFlag = True

        start = time.time()
        #del self.s
        #self.s = Simon("C:/Users/geonho/Documents/SVN/SIMON/SIMON_CODE/rst/Y301ASBDEP.SMG", "C:/Users/geonho/Documents/SVN/SIMON/SIMON_CODE/run/KMYGN34C01_PLUS7_XSE.XS", "C:/Users/geonho/Documents/SVN/SIMON/SIMON_CODE/rst/Y301ASBDEP")
        self.outputArray = []
        bp = self.initBU
        eig = self.targetEigen

        self.s.setBurnup(bp)

        std_option = SteadyOption()
        std_option.maxiter = 100
        std_option.crit = CBC
        std_option.feedtf = True
        std_option.feedtm = True
        std_option.xenon = XE_EQ
        std_option.tin = 295.8
        std_option.eigvt = eig
        std_option.ppm = 800.0

        for iStep in range(self.nInput):
            std_option.plevel = self.inputArray[iStep][1]/100.0

            result = SimonResult()

            self.sample_static(std_option)
            self.s.getResult(result);
            std_option.ppm = result.ppm
            std_option.xenon = XE_TR

            print(f'Initial AO [{result.ao:.3f}]')
            #sample_deplete(s, std_option);
            self.asisearch(std_option, iStep);
        end = time.time()
        print(end - start)

        self.makeOutputTable()

        self.definePointData()




    def asisearch(self,std_option, iStep) :
        # rodids = ['R5', 'R4', 'R3'];
        rodids = [ 'R5', 'R4', 'R3'];
        overlaps = [ 0.0 * self.s.g.core_height, 0.4 * self.s.g.core_height, 0.7 * self.s.g.core_height]
        r5_pdil = 0.0
        #r5_pdil = 0.72 * self.s.g.core_height
        ao_target = -self.targetASI
        position = self.s.g.core_height
        result = self.s.searchRodPosition(std_option, ao_target, rodids, overlaps, r5_pdil, position)

        #pre_pos = [ 381.0, 381.0, 381.0]
        #if(iStep!=0):
        #    for i in range(3):
        #        pre_pos[i] = self.outputArray[iStep-1][i+2]
        #result = self.s.searchRodPosition(std_option, ao_target, rodids, pre_pos, overlaps, r5_pdil, position)

        unitArray = []
        unitArray.append(-result.ao)
        unitArray.append(result.ppm)
        print(f'Target AO : [{ao_target:.3f}]')
        print(f'Resulting AO : [{result.ao:.3f}]')
        print(f'Resulting CBC : [{result.ppm:.3f}]')
        print(f'ERROR Code : [{result.error:5d}]')
        print('Rod positions')
        for rodid in rodids :
            print(f'{rodid:12s}  :  {result.rod_pos[rodid]:12.3f}')
            unitArray.append(result.rod_pos[rodid])

        # TODO Make Bank P
        unitArray.append(381.0)
        self.outputArray.append(unitArray)


    def sample_static(self, std_option):
        self.s.calculateStatic(std_option)

    def makeOutputTable(self):
        _translate = QtCore.QCoreApplication.translate
        nStep = self.nInput
        for iStep in range(nStep):
            for iColumn in range(2):
                item = QtWidgets.QTableWidgetItem()
                font = QtGui.QFont()
                font.setPointSize(11)
                font.setBold(False)
                font.setWeight(75)
                item.setFont(font)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                text = "%.3f" % self.outputArray[iStep][iColumn]
                item.setText(_translate("Form", text))
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                self.ui.SD_tableWidget.setItem(iStep, iColumn+4, item)
        for iRow in range(nStep):
            for iColumn in range(4):
                self.calc_rodPosBox[iRow][iColumn].show()
                self.calc_rodPosBox[iRow][iColumn].setValue(self.outputArray[iRow][iColumn+2])
                self.calc_rodPos[iRow][iColumn] = self.outputArray[iRow][iColumn+2]

    def definePointData(self):
        posR5 = []
        posR4 = []
        posR3 = []
        posCBC= []
        
        for iStep in range(self.nInput):
            #dt = (1.0 - self.debugData01) / 0.03
            posR5.append(QPointF(self.inputArray[iStep][0],self.outputArray[iStep][2]))
            posR4.append(QPointF(self.inputArray[iStep][0],self.outputArray[iStep][3]))
            posR3.append(QPointF(self.inputArray[iStep][0],self.outputArray[iStep][4]))
            posCBC.append(QPointF(self.inputArray[iStep][0],self.outputArray[iStep][0]))

        rodPos = [posR5,posR4,posR3]

        self.unitChartClass.replaceRodPosition(3,rodPos,posCBC)



    def setTableItemSetting(self):
        self.rodPosSpinBox = []
        nRow = self.ui.SD_tableWidget.rowCount()
        nColumn = self.ui.SD_tableWidget.columnCount()

        for iRow in range(nRow):
            for iColumn in range(nColumn-4):
                tmp = QTableWidgetItem()
                tmp.setFlags(tmp.flags() ^ QtCore.Qt.ItemIsEditable)
                tmp.setTextAlignment(Qt.AlignCenter)
                self.ui.SD_tableWidget.setItem(iRow,iColumn,tmp)

        for iRow in range(nRow):
            unitRodPos = []
            unitData = []
            for iColumn in range(4):
                unitBox = self.makeUnitBox(iRow,iColumn)
                unitRodPos.append(unitBox)
                unitData.append(unitBox.value())

            self.calc_rodPosBox.append(unitRodPos)
            self.calc_rodPos.append(unitData)



    def makeUnitBox(self,iRow,iColumn):
        nColumn = 4
        subID = ["_P","_5","_4","_3"]

        # Set Font for DoubleSpinBox
        font = QFont()
        font.setFamily(u"Segoe UI")
        font.setBold(True)
        font.setPointSize(10)

        doubleSpinBoxWidget = QtWidgets.QWidget()

        doubleSpinBox = unitDoubleBox.tableDoubleSpinBox()
        doubleSpinBox.saveBoxPosition(iRow,iColumn)
        #doubleSpinBox = QtWidgets.QDoubleSpinBox()
        doubleSpinBox.setObjectName(u"SD_rodPos_%03d%s" %(iRow+1,subID[iColumn]))

        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(doubleSpinBox.sizePolicy().hasHeightForWidth())
        doubleSpinBox.setSizePolicy((sizePolicy))
        doubleSpinBox.setMinimumSize(QSize(0, 0))
        doubleSpinBox.setMaximumSize(QSize(16777215, 16777215))
        doubleSpinBox.setAlignment(Qt.AlignCenter)
        doubleSpinBox.setFont(font)
        #doubleSpinBox.setStyleSheet(u"padding: 3px;")
        doubleSpinBox.setButtonSymbols(QAbstractSpinBox.NoButtons)
        doubleSpinBox.setProperty("showGroupSeparator", True)
        doubleSpinBox.setDecimals(2)
        doubleSpinBox.setMinimum(0.000000000000000)
        doubleSpinBox.setMaximum(381.000000000000)
        doubleSpinBox.setSingleStep(1.000000000000000)
        doubleSpinBox.setStepType(QAbstractSpinBox.DefaultStepType)
        doubleSpinBox.setValue(381.000000000000000)

        layout = QHBoxLayout(doubleSpinBoxWidget)
        layout.addWidget(doubleSpinBox)
        layout.setAlignment(QtCore.Qt.AlignCenter)
        layout.setContentsMargins(0,0,0,0)

        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)

        # (a,b) = doubleSpinBox.returnBoxPosition()
        # print(a,b)
        self.ui.SD_tableWidget.setCellWidget(iRow,iColumn+6,doubleSpinBoxWidget)

        # doubleSpinBox.valueChanged['double'].connect(lambda: self.checkChangedCondition(iRow,iColumn))
        doubleSpinBox.editingFinished.connect(lambda: self.checkChangedCondition(iRow,iColumn))
        doubleSpinBox.hide()

        return doubleSpinBox

    def checkChangedCondition(self,iRow,iColumn):

        tmp = round(self.calc_rodPosBox[iRow][iColumn].value(),2)
        tmp2 = round(self.calc_rodPos[iRow][iColumn],2)
        print("return!",iRow, iColumn,tmp, tmp2)

        if (tmp !=tmp2):
            insertFlag = True
            unitRodPos = [iRow,iColumn]
            for idx in self.rodPosChangedHistory:
                if(idx==unitRodPos):
                    insertFlag = False
                    break
            if(insertFlag==True):
                self.rodPosChangedHistory.append(unitRodPos)
                self.rodPosChangedHistory.sort()

            # Change Font Color
            palette = QtGui.QPalette()
            palette.setColor(QtGui.QPalette.Text, QColor(0xFF,0x99,0x33))
            #self.calc_rodPosBox[iRow][iColumn].setPalette(palette)
            self.calc_rodPosBox[iRow][iColumn].setStyleSheet(u"color: rgb(255,51,51);")
            self.calc_rodPosBox[iRow][iColumn].update()

        else:
            popupFlag = False
            unitRodPos = [iRow,iColumn]
            for idx in self.rodPosChangedHistory:
                if(idx==unitRodPos):
                    popupFlag = True
                    break
            if(popupFlag==True):
                self.rodPosChangedHistory.remove(unitRodPos)

            palette = QtGui.QPalette()
            palette.setColor(QtGui.QPalette.Text, QColor(0xFF,0x99,0x33))
            #self.calc_rodPosBox[iRow][iColumn].setPalette(palette)
            self.calc_rodPosBox[iRow][iColumn].setStyleSheet(u"color: rgb(210,210,210);")
            self.calc_rodPosBox[iRow][iColumn].update()

            print("color didn't changed!")

    def resetPositionData(self):
        if(len(self.rodPosChangedHistory)==0):
            if(self.tableDatasetFlag==True):
                print("Error! Calculation dataset didn't changed!")
            else:
                print("Error! There is No Calculation DataSet")
            return
        
        for iRow in range(len(self.calc_rodPos)):
            for iColumn in range(4):
                value = self.calc_rodPos[iRow][iColumn]
                self.calc_rodPosBox[iRow][iColumn].setValue(value)
                self.calc_rodPosBox[iRow][iColumn].setStyleSheet(u"color: rgb(210,210,210);")
                self.calc_rodPosBox[iRow][iColumn].update()

        self.rodPosChangedHistory = []


    def ASI_ReCalculation(self):
        if(len(self.rodPosChangedHistory)==0):
            if(self.tableDatasetFlag==True):
                print("Error! Calculation dataset didn't changed!")
            else:
                print("Error! There is No Calculation DataSet")
            return

        self.tableDatasetFlag = True

        iRestart = self.rodPosChangedHistory[0][0]

        for idx in range(self.nInput-iRestart):
            self.outputArray.pop(-1)

        start = time.time()

        bp = self.initBU
        eig = self.targetEigen

        self.s.setBurnup(bp)

        std_option = SteadyOption()
        std_option.maxiter = 100
        std_option.crit = CBC
        std_option.feedtf = True
        std_option.feedtm = True
        std_option.xenon = XE_EQ
        std_option.tin = 295.8
        std_option.eigvt = eig
        std_option.ppm = 800.0


        std_option.plevel = self.inputArray[iRestart][1]/100.0
        result = SimonResult()
        self.sample_static(std_option)
        self.s.getResult(result);
        std_option.ppm = result.ppm
        std_option.xenon = XE_TR

        self.ASI_SearchGivenPosition(std_option, iRestart);

        for iStep in range(iRestart+1,self.nInput):
            std_option.plevel = self.inputArray[iStep][1]/100.0

            result = SimonResult()

            self.sample_static(std_option)
            self.s.getResult(result);
            std_option.ppm = result.ppm
            std_option.xenon = XE_TR

            print(f'Initial AO [{result.ao:.3f}]')

            self.asisearch(std_option, iStep);
        end = time.time()
        print(end - start)

        self.makeOutputTable()

        self.definePointData()


    def ASI_SearchGivenPosition(self,std_option, iStep) :
        rodids = ['R5', 'R4', 'R3'];
        # rodids = ['P', 'R5', 'R4'];
        pos = []
        overlaps = [ 0.0 * self.s.g.core_height, 0.4 * self.s.g.core_height, 0.7 * self.s.g.core_height] #, 1.0 * self.s.g.core_height]
        # r5_pdil = 0.72 * s.g.core_height
        ao_target = -self.targetASI

        for iColumn in range(3):
            pos.append(self.calc_rodPosBox[iStep][iColumn].value())
        result = self.s.searchASI(std_option, rodids, pos)

        unitArray = []
        unitArray.append(-result.ao)
        unitArray.append(result.ppm)
        print(f'Target AO : [{ao_target:.3f}]')
        print(f'Resulting AO : [{result.ao:.3f}]')
        print(f'Resulting CBC : [{result.ppm:.3f}]')
        print(f'ERROR Code : [{result.error:5d}]')
        print('Rod positions')
        for rodid in rodids :
            print(f'{rodid:12s}  :  {result.rod_pos[rodid]:12.3f}')
            unitArray.append(result.rod_pos[rodid])

        unitArray.append(381.0)
        self.outputArray.append(unitArray)








    ## TEST ROUTINE
    #def startCalcTest(self):
    #    #del self.s
    #    #self.s = Simon("C:/Users/geonho/Documents/SVN/SIMON/SIMON_CODE/rst/Y301ASBDEP.SMG", "C:/Users/geonho/Documents/SVN/SIMON/SIMON_CODE/run/KMYGN34C01_PLUS7_XSE.XS", "C:/Users/geonho/Documents/SVN/SIMON/SIMON_CODE/rst/Y301ASBDEP")
    #    self.outputFlag = True
    #    self.debugData01 = [ 1.000, 0.970, 0.940, 0.910, 0.880,
    #                          0.850, 0.820, 0.790, 0.760, 0.730,
    #                          0.700, 0.670, 0.640, 0.610, 0.580,
    #                          0.550, 0.520, 0.490, 0.460, 0.430,
    #                          0.400, 0.370, 0.340, 0.310, 0.300,
    #                          0.270, 0.240, 0.210, 0.180, 0.150,
    #                          0.120, 0.090, 0.060, 0.030, 0.000 ]
    #    #self.debugData02 =  [  [  100.0, 100.0, 100.0 ],
    #    #                    [  100.0, 100.0, 100.0 ],
    #    #                    [   89.0, 100.0, 100.0 ],
    #    #                    [   82.0, 100.0, 100.0 ],
    #    #                    [   73.0, 100.0, 100.0 ],
    #    #                    [   64.0, 100.0, 100.0 ],
    #    #                    [   51.0, 100.0, 100.0 ],
    #    #                    [   50.0,  90.0, 100.0 ],
    #    #                    [   50.0,  85.0, 100.0 ],
    #    #                    [   50.0,  81.0, 100.0 ],
    #    #                    [   50.0,  77.0, 100.0 ],
    #    #                    [   50.0,  69.0, 100.0 ],
    #    #                    [   50.0,  65.0, 100.0 ],
    #    #                    [   50.0,  62.0, 100.0 ],
    #    #                    [   50.0,  59.0,  95.0 ],
    #    #                    [   50.0,  58.0,  91.0 ],
    #    #                    [   50.0,  57.0,  89.0 ],
    #    #                    [   50.0,  56.0,  87.0 ],
    #    #                    [   50.0,  55.0,  85.0 ],
    #    #                    [   50.0,  54.0,  82.0 ],
    #    #                    [   50.0,  53.0,  80.0 ],
    #    #                    [   50.0,  52.0,  77.0 ],
    #    #                    [   50.0,  51.0,  74.0 ],
    #    #                    [   50.0,  50.0,  72.0 ],
    #    #                    [   50.0,  50.0,  71.0 ],
    #    #                    [   50.0,  50.0,  70.0 ],
    #    #                    [   50.0,  50.0,  70.0 ],
    #    #                    [   50.0,  50.0,  70.0 ],
    #    #                    [   50.0,  50.0,  70.0 ],
    #    #                    [   50.0,  50.0,  65.0 ],
    #    #                    [   50.0,  50.0,  65.0 ],
    #    #                    [   50.0,  50.0,  65.0 ],
    #    #                    [   50.0,  50.0,  60.0 ],
    #    #                    [   50.0,  50.0,  60.0 ],
    #    #                    [   50.0,  50.0,  60.0 ] ]

    #    #self.debugData02 =  [  [  50.0, 50.0, 50.0 ],
    #    #                    [   50.0,  50.0,  50.0 ],
    #    #                    [   50.0,  50.0,  50.0 ],
    #    #                    [   50.0,  50.0,  50.0 ],
    #    #                    [   50.0,  50.0,  50.0 ],
    #    #                    [   50.0,  50.0,  50.0 ],
    #    #                    [   50.0,  50.0,  50.0 ],
    #    #                    [   50.0,  50.0,  50.0 ],
    #    #                    [   50.0,  50.0,  50.0 ],
    #    #                    [   50.0,  50.0,  50.0 ],
    #    #                    [   50.0,  50.0,  50.0 ],
    #    #                    [   50.0,  50.0,  50.0 ],
    #    #                    [   50.0,  50.0,  50.0 ],
    #    #                    [   50.0,  50.0,  50.0 ],
    #    #                    [   50.0,  50.0,  50.0 ],
    #    #                    [   50.0,  50.0,  50.0 ],
    #    #                    [   50.0,  50.0,  50.0 ],
    #    #                    [   50.0,  50.0,  50.0 ],
    #    #                    [   50.0,  50.0,  50.0 ],
    #    #                    [   50.0,  50.0,  50.0 ],
    #    #                    [   50.0,  50.0,  50.0 ],
    #    #                    [   50.0,  50.0,  50.0 ],
    #    #                    [   50.0,  50.0,  50.0 ],
    #    #                    [   50.0,  50.0,  50.0 ],
    #    #                    [   50.0,  50.0,  50.0 ],
    #    #                    [   50.0,  50.0,  50.0 ],
    #    #                    [   50.0,  50.0,  50.0 ],
    #    #                    [   50.0,  50.0,  50.0 ],
    #    #                    [   50.0,  50.0,  50.0 ],
    #    #                    [   50.0,  50.0,  50.0 ],
    #    #                    [   50.0,  50.0,  50.0 ],
    #    #                    [   50.0,  50.0,  50.0 ],
    #    #                    [   50.0,  50.0,  50.0 ],
    #    #                    [   50.0,  50.0,  50.0 ],
    #    #                    [   50.0,  50.0,  50.0 ] ]
    #    self.debugData02 =  [  [  100.0, 100.0, 100.0 ],
    #                        [  100.0, 100.0, 100.0 ],
    #                        [  100.0, 100.0, 100.0 ],
    #                        [  100.0, 100.0, 100.0 ],
    #                        [  100.0, 100.0, 100.0 ],
    #                        [  100.0, 100.0, 100.0 ],
    #                        [  100.0, 100.0, 100.0 ],
    #                        [  100.0, 100.0, 100.0 ],
    #                        [  100.0, 100.0, 100.0 ],
    #                        [  100.0, 100.0, 100.0 ],
    #                        [  100.0, 100.0, 100.0 ],
    #                        [  100.0, 100.0, 100.0 ],
    #                        [  100.0, 100.0, 100.0 ],
    #                        [  100.0, 100.0, 100.0 ],
    #                        [  100.0, 100.0, 100.0 ],
    #                        [  100.0, 100.0, 100.0 ],
    #                        [  100.0, 100.0, 100.0 ],
    #                        [  100.0, 100.0, 100.0 ],
    #                        [  100.0, 100.0, 100.0 ],
    #                        [  100.0, 100.0, 100.0 ],
    #                        [  100.0, 100.0, 100.0 ],
    #                        [  100.0, 100.0, 100.0 ],
    #                        [  100.0, 100.0, 100.0 ],
    #                        [  100.0, 100.0, 100.0 ],
    #                        [  100.0, 100.0, 100.0 ],
    #                        [  100.0, 100.0, 100.0 ],
    #                        [  100.0, 100.0, 100.0 ],
    #                        [  100.0, 100.0, 100.0 ],
    #                        [  100.0, 100.0, 100.0 ],
    #                        [  100.0, 100.0, 100.0 ],
    #                        [  100.0, 100.0, 100.0 ],
    #                        [  100.0, 100.0, 100.0 ],
    #                        [  100.0, 100.0, 100.0 ],
    #                        [  100.0, 100.0, 100.0 ],
    #                        [  100.0, 100.0, 100.0 ] ]
    #    start = time.time()
    #    self.outputArray = []
    #    bp = 12000.0
    #    eig = 1.003385

    #    self.s.setBurnup(bp)
    #    self.setSuccecciveInputTest()

    #    std_option = SteadyOption()
    #    std_option.maxiter = 100
    #    std_option.crit = CBC
    #    std_option.feedtf = True
    #    std_option.feedtm = True
    #    std_option.xenon = XE_EQ
    #    std_option.tin = 295.8
    #    std_option.eigvt = eig
    #    std_option.ppm = 800.0

    #    nStep = len(self.debugData01)

    #    for iStep in range(nStep):
    #        std_option.plevel = self.debugData01[iStep]

    #        result = SimonResult()

    #        self.sample_static(std_option)
    #        self.s.getResult(result);
    #        std_option.ppm = result.ppm
    #        std_option.xenon = XE_TR

    #        print(f'Initial AO [{result.ao:.3f}]')
    #        #sample_deplete(s, std_option);
    #        self.sample_asisearchTest(std_option, iStep);
    #    end = time.time()
    #    print(end - start)

    #    # 01. 
    #    self.makeOutputTable()
    #    # 02. Draw 
    #    # 02.1 Define QPointF Plot Data
    #    self.definePointData()

    #def setSuccecciveInputTest(self):
    #    # Initialize and reset Dataset for TableWidget
    #    self.ui.SD_tableWidget.setRowCount(0)
    #    self.inputArray = []
    #    self.calc_rodPos = []
    #    self.calc_rodPosBox = []
    #    self.tableDatasetFlag = False

    #    initBU     = 12000.0 #self.ui.SD_Input01.value()
    #    targetEigen = 1.003385 #self.ui.SD_Input02.value()
    #    targetASI = 0.01 #self.ui.SD_Input04.value()

    #    self.initBU     = initBU
    #    self.targetEigen = targetEigen
    #    self.targetASI = targetASI

    #    rdcPerHour = 3.0 #self.ui.SD_Input03.value()
    #    EOF_Power  = 0.0 #self.ui.rdc04.value()


    #    nStep = len(self.debugData01)

    #    # Initial Settings for SD_TableWidget
    #    self.ui.SD_tableWidget.setRowCount(nStep)
    #    self.setTableItemSetting()
    #    self.nInput = nStep

    #    for i in range(nStep):
    #        time = (100.0 - 100.0*self.debugData01[i]) / rdcPerHour
    #        power = self.debugData01[i]*100.0
    #        unitArray = [ time, power, initBU, targetEigen ]
    #        self.inputArray.append(unitArray)

    #    _translate = QtCore.QCoreApplication.translate
    #    for iStep in range(nStep-1):
    #        for iRow in range(4):
    #            item = QtWidgets.QTableWidgetItem()
    #            font = QtGui.QFont()
    #            font.setPointSize(11)
    #            font.setBold(False)
    #            font.setWeight(75)
    #            item.setFont(font)
    #            item.setTextAlignment(QtCore.Qt.AlignCenter)
    #            if (iRow == 0):
    #                text = "%.1f" % self.inputArray[iStep][iRow]
    #            elif (iRow == 1):
    #                text = "%.2f" % self.inputArray[iStep][iRow]
    #            elif (iRow == 2):
    #                text = "%.1f" % self.inputArray[iStep][iRow]
    #            elif (iRow == 3):
    #                text = "%.5f" % self.inputArray[iStep][iRow]
    #            item.setText(_translate("Form", text))
    #            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
    #            self.ui.SD_tableWidget.setItem(iStep, iRow, item)

    #    for iRow in range(4):
    #        item = QtWidgets.QTableWidgetItem()
    #        font = QtGui.QFont()
    #        font.setPointSize(11)
    #        font.setBold(False)
    #        font.setWeight(75)
    #        item.setFont(font)
    #        item.setTextAlignment(QtCore.Qt.AlignCenter)
    #        if (iRow == 0):
    #            text = "%.3f" % self.inputArray[nStep-1][iRow]
    #        elif (iRow == 1):
    #            text = "%.2f" % self.inputArray[nStep-1][iRow]
    #        elif (iRow == 2):
    #            text = "%.1f" % self.inputArray[nStep-1][iRow]
    #        elif (iRow == 3):
    #            text = "%.5f" % self.inputArray[nStep-1][iRow]
    #        item.setText(_translate("Form", text))
    #        item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
    #        self.ui.SD_tableWidget.setItem(nStep-1, iRow, item)

    #def sample_asisearchTest(self,std_option, iStep) :
    #    rodids = ['R5', 'R4', 'R3'];
    #    # rodids = ['P', 'R5', 'R4'];
    #    overlaps = [ 0.0 * self.s.g.core_height, 0.4 * self.s.g.core_height, 0.7 * self.s.g.core_height]
    #    r5_pdil = 0.0 # 0.72 * s.g.core_height
    #    ao_target = -0.1
    #    # position = self.s.g.core_height
    #    pos = []
    #    for i in range(3):
    #        pos.append(self.debugData02[iStep][i]*3.81)
    #    result = self.s.searchASI(std_option, rodids, pos)

    #    unitArray = []
    #    unitArray.append(-result.ao)
    #    unitArray.append(result.ppm)
    #    print(f'Target AO : [{ao_target:.3f}]')
    #    print(f'Resulting AO : [{result.ao:.3f}]')
    #    print(f'Resulting CBC : [{result.ppm:.3f}]')
    #    print(f'ERROR Code : [{result.error:5d}]')
    #    print('Rod positions')
    #    for rodid in rodids :
    #        print(f'{rodid:12s}  :  {result.rod_pos[rodid]:12.3f}')
    #        unitArray.append(result.rod_pos[rodid])

    #    unitArray.append(381.0)
    #    self.outputArray.append(unitArray)

    #def definePointDataTest(self):
    #    posR5 = []
    #    posR4 = []
    #    posR3 = []
    #    posCBC= []
    #    nStep = len(self.debugData01)
    #    for iStep in range(nStep):
    #        dt = (1.0 - self.debugData01[iStep]) / 0.03
    #        posR5.append(QPointF(dt,self.outputArray[iStep][1]))
    #        posR4.append(QPointF(dt,self.outputArray[iStep][2]))
    #        posR3.append(QPointF(dt,self.outputArray[iStep][3]))
    #        posCBC.append(QPointF(dt,self.outputArray[iStep][-1]))

    #    rodPos = [posR5,posR4,posR3]

    #    self.unitChartClass.replaceRodPosition(3,rodPos,posCBC)

    #def makeOutputTableTest(self):
    #    _translate = QtCore.QCoreApplication.translate
    #    nStep = len(self.debugData01)
    #    self.ui.SD_tableWidget.setRowCount(nStep)
    #    for iStep in range(nStep):
    #        for iRow in range(6):
    #            item = QtWidgets.QTableWidgetItem()
    #            font = QtGui.QFont()
    #            font.setPointSize(11)
    #            font.setBold(False)
    #            font.setWeight(75)
    #            item.setFont(font)
    #            item.setTextAlignment(QtCore.Qt.AlignCenter)
    #            if (iRow == 0):
    #                text = "%.3f" % self.outputArray[iStep][iRow]
    #            elif (iRow == 4):
    #                text = "%.3f" % self.outputArray[iStep][iRow]
    #            elif (iRow == 5):
    #                text = "%.3f" % self.outputArray[iStep][iRow]
    #            else: # (iRow == 2):
    #                text = "%.2f" % self.outputArray[iStep][iRow]
    #            #elif (iRow == 3):
    #                #text = "%.2f" % self.outputArray[iStep][iRow]
    #            item.setText(_translate("Form", text))
    #            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
    #            self.ui.SD_tableWidget.setItem(iStep, iRow+4, item)

    def clickSaveAsExcel(self):
        pass
        # nRow = len(self.fullFA)
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(None, 'Export FA datas as Excel', ".", 'Microsoft Excel WorkSheet(*.xlsx);;'
                                                                                             'Microsoft Excel 97-2003 WorkSheet(*.xls)')
        if fn != '':

            base_dir = os.getcwd()
            xlxs_dir = os.path.join(base_dir, fn)
            writer = pd.ExcelWriter(xlxs_dir,engine="xlsxwriter")
            
            columnHeader = ["Time\n(Hour)",
                            "Relative\nPower",
                            "Cycle\nBurnup\n(MWD/MTU)",
                            "Keff",
                            "ASI",
                            "Boron\n(ppm)",
                            "Bank 5\nPosition(cm)",
                            "Bank 4\nPosition(cm)",
                            "Bank 3\nPosition(cm)"]
            #df2 = pd.DataFrame(excelDataSet,columns=columnHeader)
            #df2.to_excel(writer,
            #             sheet_name='FA_Inventory_Data',
            #             na_rep='NaN',
            #             header= False,
            #             index = False,
            #             startrow=1,
            #             startcol=1)
            #(max_row,max_col) = df2.shape
            workbook  = writer.book
            worksheet = workbook.add_worksheet()#s['FA_Inventory_Data']
            #worksheet.autofilter(0,0,max_row,max_col)

            headerFormat = workbook.add_format({'bold':True,
                                                'text_wrap': True,
                                                'align':'center',
                                                'valign':'vcenter',
                                                'fg_color':'#93cddd',
                                                'border':1})

            #format00 = workbook.add_format({'align':'center','border':1})
            format01 = workbook.add_format({'align':'center','border':1,'num_format':'0.0'})
            format02 = workbook.add_format({'align':'center','border':1,'num_format':'0.00'})
            format03 = workbook.add_format({'align':'center','border':1,'num_format':'0.000'})
            format05 = workbook.add_format({'align':'center','border':1,'num_format':'0.00000'})

            formatArray = [ format01,format02,format01,format05,format03,format03,format02,format02,format02]

            for rowIdx in range(len(columnHeader)):
                worksheet.write(0,rowIdx+1,columnHeader[rowIdx],headerFormat)

            for colIdx in range(len(self.inputArray)):
                for rowIdx in range(4):
                    worksheet.write(colIdx+1,rowIdx+1,self.inputArray[colIdx][rowIdx],formatArray[rowIdx])

            if (self.outputFlag ==False):
                for colIdx in range(len(self.inputArray)):
                    for rowIdx in range(4,9):
                        worksheet.write(colIdx+1,rowIdx+1,'',formatArray[rowIdx])
            else:
                for colIdx in range(len(self.inputArray)):
                    for rowIdx in range(4,9):
                        worksheet.write(colIdx+1,rowIdx+1,self.outputArray[colIdx][rowIdx-4],formatArray[rowIdx])
            worksheet.set_column('B:J',12)
            writer.close()