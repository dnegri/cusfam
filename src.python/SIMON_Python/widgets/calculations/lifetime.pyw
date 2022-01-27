#import sys
#import platform
#from PyQt5 import QtCore, QtGui, QtWidgets
#from PyQt5.QtCore import (QCoreApplication, QPropertyAnimation, QDate, QDateTime, QMetaObject, QObject, QPoint, QRect, QSize, QTime, QUrl, Qt, QEvent)
#from PyQt5.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont, QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter, QPixmap, QRadialGradient)
#from PyQt5 import QtWidgets
#from PyQt5.QtWidgets import QApplication, QMainWindow
#import sys
#import PyQt5.QtChart as QtCharts
#from PyQt5.QtCore import Qt
#from PyQt5.QtWidgets import *
#from model import *
#import datetime
#import os
#import glob
#from ui_main_rev18 import Ui_MainWindow
#import Definitions as df
#import widgets.utils.PyUnitButton as ts

#import constants as cs
#import utils as ut

#from widgets.calculations.calculation_widget import CalculationWidget
#import widgets.utils.PyShutdownTableWidget as table01
#import widgets.utils.PyRodPositionSplineChart as unitSplineChart

#from widgets.output.axial.axial_plot import AxialWidget
#from widgets.output.radial.radial_graph import RadialWidget

#import widgets.utils.PyRodPositionBarChart as unitChart
#import widgets.utils.PyStartButton as startB

from PyQt5.QtCore import QPointF
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QSizePolicy

from model import *
import datetime

import Definitions as df

import constants as cs

from widgets.calculations.calculation_widget import CalculationWidget

import widgets.utils.PyShutdownTableWidget as table01
import widgets.utils.PyRodPositionSplineChart as unitSplineChart
import widgets.calculations.calculationManager as calcManager
from widgets.output.axial.axial_plot import AxialWidget
from widgets.output.radial.radial_graph import RadialWidget
import math

from widgets.utils.PySaveMessageBox import PySaveMessageBox

from itertools import islice
from random import random
from time import perf_counter
from cusfam import *
import time
import os
import multiprocessing
from threading import Thread

from PyQt5.QtCore import QPointF, QThread

pointers_old = {

    # "Lifetime_Snapshot": "ss_snapshot_file",
    # "Lifetime_Input01": "ndr_burnup",
    # "Lifetime_Input02": "ndr_power",
    # "Lifetime_Input03_A": "ndr_stopping_criterion",
    # "Lifetime_Input03_B": "ndr_stopping_criterion_type",
    # "Lifetime_Input04_A": "ndr_depletion_interval",
    # "Lifetime_Input04_B": "ndr_depletion_interval_type",
    # "Lifetime_RodPos05": "ndr_bank_position_5",
    # "Lifetime_RodPos05_Unit": "ndr_bank_type",
    # "Lifetime_RodPos04": "ndr_bank_position_4",
    # "Lifetime_RodPos03": "ndr_bank_position_3",
    # "Lifetime_RodPos_P": "ndr_bank_position_P",

}

pointers = {
    #
    # "Lifetime_Snapshot": "ss_snapshot_file",
    # "LF_Input01": "ndr_power",
    # "LF_Input02": "ndr_burnup",
    # "LF_Input03": "ndr_stopping_criterion",
    # "LF_Input04": "ndr_depletion_interval",
    # "LF_Input05": "ndr_bank_position_P",
    # "LF_Input06": "ndr_bank_position_5",
    # "LF_Input07": "ndr_bank_position_4",
    # "LF_Input08": "ndr_bank_position_3",

}

class LifetimeWidget(CalculationWidget):

    def __init__(self, db, ui, calManager):

        super().__init__(db, ui, None, LifetimeInput, pointers, calManager)

        self.input_pointer = "self.current_calculation.lifetime_input"

        # 01. Setting Initial Lifetime Control Input Data
        self.Lifetime_PosBank5 = 0.0
        self.Lifetime_PosBank4 = 0.0
        self.Lifetime_PosBank3 = 0.0
        self.Lifetime_PosBankP = 0.0

        self.RodPosUnit = df.RodPosUnit_cm
        self.lifeTimeCalcOpt = df.select_none
        self.unitChangeFlag = False
        self.initSnapshotFlag = False
        self.inputArray = []
        # Input Dataset for SD_TableWidget Rod Position
        self.calc_rodPosBox = []
        self.calc_rodPos = []
        self.recalculationIndex = -1
        self.nStep = 0
        self.outputArray = []
        # 02. Setting Initial Rod Position Bar Chart
        # self.unitChartClass = unitChart.UnitBarChart(self.Lifetime_PosBank5,
        #                                              self.Lifetime_PosBank4,
        #                                              self.Lifetime_PosBank3,
        #                                              self.Lifetime_PosBankP,
        #                                              self.RodPosUnit)

        # 03. Insert Table
        self.tableItem = ["Time\n(hour)","Power\n(%)","Burnup\n(MWD/MTU)","Keff","ASI","Boron\n(ppm)","Bank P\n(cm)","Bank 5\n(cm)","Bank 4\n(cm)","Bank 3\n(cm)"]
        self.LifeTime_TableWidget = table01.ShutdownTableWidget(self.ui.frame_Lifetime_TableWidget, self.tableItem)
        layoutTableButton = self.LifeTime_TableWidget.returnButtonLayout()
        self.ui.gridlayout_Lifetime_TableWidget.addWidget(self.LifeTime_TableWidget, 0, 0, 1, 1)
        self.ui.gridlayout_Lifetime_TableWidget.addLayout(layoutTableButton,1,0,1,1)
        [ self.unitButton01, self.unitButton02, self.unitButton03 ] = self.LifeTime_TableWidget.returnTableButton()

        self.unitChartClass = None
        self.RadialWidget = None
        self.AxialWidget = None
        self.addOutput()

        # 03. Setting Widget Interactions
        self.buttonGroupInput = QtWidgets.QButtonGroup()
        self.settingAutoExclusive()
        self.settingLinkAction()

        # 04. Setting Initial UI widget
        #self.ui.LF_Main02.hide()

        # 05. Insert ChartView to chartWidget
        # self.nSet = 0
        # lay = QtWidgets.QVBoxLayout(self.ui.Lifetime_widgetChart)
        # lay.setContentsMargins(0,0,0,0)
        # unitChartView = self.unitChartClass.returnChart()
        # lay.addWidget(unitChartView)

        # 06. Link Signal and Slot for Start Calculation
        self.makeStartButton(self.startFunc,super().saveFunc, self.ui.LF_Main05, self.ui.Lifetime_CalcButton_Grid)

        self.operationArray = []
        #self.ui.Lifetime_tabWidget.setCurrentIndex(0)

    def load(self):
        #PLANT AND RESTART FILES
        # self.ui.Lifetime_PlantName.clear()
        # self.ui.Lifetime_PlantCycle.clear()
        #
        # query = LoginUser.get(LoginUser.username == cs.ADMIN_USER)
        # user = query.login_user
        # plants, errors = ut.getPlantFiles(user)
        # for plant in plants:
        #     self.ui.Lifetime_PlantName.addItem(plant)
        # self.ui.Lifetime_PlantName.setCurrentText(user.plant_file)
        #
        # restarts, errors = ut.getRestartFiles(user)
        # for restart in restarts:
        #     self.ui.Lifetime_PlantCycle.addItem(restart)
        # self.ui.Lifetime_PlantCycle.setCurrentText(user.restart_file)

        if len(self.operationArray) == 0:
            self.ui.Lifetime_run_button.setText("Create Scenario")
        else:
            self.ui.Lifetime_run_button.setText("Run")

        #self.load_input()
        #self.load_recent_calculation()

        #self.ui.ECP_Main05_tabWidget.setCurrentIndex(0)

    def set_all_component_values(self, lifetime_input):
        # if lifetime_input.calculation_type == 0:
        #     self.ui.Lifetime_InpOpt2_NDR.setChecked(True)
        # else:
        #     self.ui.Lifetime_InpOpt1_Snapshot.setChecked(True)

        for key in pointers.keys():
            component = eval("self.ui.{}".format(key))
            value = eval("lifetime_input.{}".format(pointers[key]))

            if pointers[key]:
                if isinstance(component, QComboBox):
                    component.setCurrentText(value)
                else:
                    component.setValue(float(value))

    def get_input(self, calculation_object):
        return calculation_object.lifetime_input

    def getDefaultInput(self, user):
        now = datetime.datetime.now()
        lifetime_input = LifetimeInput.create(
            calculation_type=0,
            ss_snapshot_file="",
            ndr_burnup=0,
            ndr_power=0,
            ndr_stopping_criterion=0,
            ndr_depletion_interval=0,
            ndr_bank_position_5=0,
            ndr_bank_position_4=0,
            ndr_bank_position_3=0,
            ndr_bank_position_P=0,
        )

        lifetime_calculation = Calculations.create(user=user,
                                                   calculation_type=cs.CALCULATION_LIFETIME,
                                                   created_date=now,
                                                   modified_date=now,
                                                   lifetime_input=lifetime_input)
        return lifetime_calculation, lifetime_input

    def settingAutoExclusive(self):
        #self.buttonGroupInput.addButton(self.ui.Lifetime_InpOpt1_Snapshot)
        #self.buttonGroupInput.addButton(self.ui.Lifetime_InpOpt2_NDR)
        pass

    def settingLinkAction(self):
        # self.ui.Lifetime_InpOpt1_Snapshot.clicked['bool'].connect(self.settingInputOpt)
        # self.ui.Lifetime_InpOpt2_NDR.clicked['bool'].connect(self.settingInputOpt)

        # self.ui.Lifetime_InpOpt1_Snapshot.clicked['bool'].connect(self.updateBankPosSnapshot)
        # self.ui.Lifetime_InpOpt2_NDR.clicked['bool'].connect(self.updateBankPosUserInput)
        self.ui.LF_Input05.valueChanged['double'].connect(self.rodPosChangedEvent)
        self.ui.LF_Input06.valueChanged['double'].connect(self.rodPosChangedEvent)
        self.ui.LF_Input07.valueChanged['double'].connect(self.rodPosChangedEvent)
        self.ui.LF_Input08.valueChanged['double'].connect(self.rodPosChangedEvent)

        #self.ui.LF_Input05.currentIndexChanged['int'].connect(self.changeRodPosUnit)

        # self.ui.Lifetime_StartButton.clicked['bool'].connect(self.startCalc)

    def settingInputOpt(self):
        calculatin_type = 0
        # if (self.ui.Lifetime_InpOpt2_NDR.isChecked()):
        #     self.lifeTimeCalcOpt = df.select_NDR
        #     self.ui.LF_Main02.show()
        # elif (self.ui.Lifetime_InpOpt1_Snapshot.isChecked()):
        #     self.lifeTimeCalcOpt = df.select_snapshot
        #     self.ui.LF_Main02.hide()
        #     calculatin_type = 1

        if self.current_calculation:
            current_input = self.get_input(self.current_calculation)
            current_input.calculation_type = calculatin_type
            current_input.save()

    def rodPosChangedEvent(self):
        if (self.unitChangeFlag == False and self.initSnapshotFlag == False):
            pass
            #self.updateBankPosUserInput()
        else:
            pass

    def updateBankPosSnapshot(self):
        # TODO, Read Rod Position from Snapshot
        self.initSnapshotFlag = True
        self.Lifetime_PosBank5 = 50.0 * df.convertRodPosUnit[0][self.RodPosUnit]
        self.Lifetime_PosBank4 = 50.0 * df.convertRodPosUnit[0][self.RodPosUnit]
        self.Lifetime_PosBank3 = 50.0 * df.convertRodPosUnit[0][self.RodPosUnit]
        self.Lifetime_PosBankP = 50.0 * df.convertRodPosUnit[0][self.RodPosUnit]
        self.ui.LF_Input05.setValue(self.Lifetime_PosBank5)
        self.ui.LF_Input06.setValue(self.Lifetime_PosBank4)
        self.ui.LF_Input07.setValue(self.Lifetime_PosBank3)
        self.ui.LF_Input08.setValue(self.Lifetime_PosBankP)
        self.replaceRodPosition()
        self.initSnapshotFlag = False

    def updateBankPosUserInput(self):
        self.Lifetime_PosBank5 = self.ui.LF_Input05.value()
        self.Lifetime_PosBank4 = self.ui.LF_Input06.value()
        self.Lifetime_PosBank3 = self.ui.LF_Input07.value()
        self.Lifetime_PosBankP = self.ui.LF_Input08.value()
        self.replaceRodPosition()

    def replaceRodPosition(self):
        self.unitChartClass.replaceRodPosition(self.Lifetime_PosBank5, self.Lifetime_PosBank4,
                                               self.Lifetime_PosBank3, self.Lifetime_PosBankP, self.RodPosUnit)

        if self.load_rod_tab and self.ui.Lifetime_tabWidget.currentIndex() != 2:
            self.ui.Lifetime_tabWidget.setCurrentIndex(2)

    def changeRodPosUnit(self, index):
        self.unitChangeFlag = True
        self.rodPosChangedEvent()
        # text = "   " + self.ui.Lifetime_RodPos05_Unit.currentText()
        # self.ui.Lifetime_RodPos04_Unit.setText(text)
        # self.ui.Lifetime_RodPos03_Unit.setText(text)
        # self.ui.Lifetime_RodPos_P_Unit.setText(text)

        currentOpt = self.RodPosUnit
        self.Lifetime_PosBank5 = self.Lifetime_PosBank5 * df.convertRodPosUnit[currentOpt][index]
        self.Lifetime_PosBank4 = self.Lifetime_PosBank4 * df.convertRodPosUnit[currentOpt][index]
        self.Lifetime_PosBank3 = self.Lifetime_PosBank3 * df.convertRodPosUnit[currentOpt][index]
        self.Lifetime_PosBankP = self.Lifetime_PosBankP * df.convertRodPosUnit[currentOpt][index]

        self.ui.LF_Input05.setValue(self.Lifetime_PosBank5)
        self.ui.LF_Input06.setValue(self.Lifetime_PosBank4)
        self.ui.LF_Input07.setValue(self.Lifetime_PosBank3)
        self.ui.LF_Input08.setValue(self.Lifetime_PosBankP)
        self.RodPosUnit = index
        self.replaceRodPosition()
        self.unitChangeFlag = False

    def makeStartButton(self, startFunc, saveFunc, frame, grid):

        self.ui.Lifetime_save_button.clicked['bool'].connect(saveFunc)
        self.ui.Lifetime_run_button.clicked['bool'].connect(startFunc)
        """
        self.Lifetime_StartButton = startB.startButton(df.CalcOpt_Lifetime, startFunc, frame)

        self.Lifetime_StartButton.setObjectName(u"Lifetime_StartButton")
        sizePolicy3 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy3.setHeightForWidth(self.Lifetime_StartButton.sizePolicy().hasHeightForWidth())
        self.Lifetime_StartButton.setSizePolicy(sizePolicy3)
        self.Lifetime_StartButton.setMinimumSize(QSize(200, 40))
        self.Lifetime_StartButton.setMaximumSize(QSize(16777215, 40))
        font2 = QFont()
        font2.setFamily(u"Segoe UI")
        font2.setPointSize(14)
        font2.setBold(False)
        font2.setWeight(50)
        self.Lifetime_StartButton.setFont(font2)
        self.Lifetime_StartButton.setStyleSheet(u"QPushButton {\n"
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
        self.Lifetime_SaveButton = startB.startButton(df.CalcOpt_Lifetime, saveFunc, frame)

        self.Lifetime_SaveButton.setObjectName(u"Lifetime_SaveButton")
        sizePolicy3 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy3.setHeightForWidth(self.Lifetime_SaveButton.sizePolicy().hasHeightForWidth())
        self.Lifetime_SaveButton.setSizePolicy(sizePolicy3)
        self.Lifetime_SaveButton.setMinimumSize(QSize(200, 40))
        self.Lifetime_SaveButton.setMaximumSize(QSize(16777215, 40))
        font2 = QFont()
        font2.setFamily(u"Segoe UI")
        font2.setPointSize(14)
        font2.setBold(False)
        font2.setWeight(50)
        self.Lifetime_SaveButton.setFont(font2)
        self.Lifetime_SaveButton.setStyleSheet(u"QPushButton {\n"
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

        grid.addWidget(self.Lifetime_SaveButton, 1, 0, 1, 1)
        self.Lifetime_SaveButton.setText(QCoreApplication.translate("MainWindow", u"Save", None))

        grid.addWidget(self.Lifetime_StartButton, 1, 1, 1, 1)
        self.Lifetime_StartButton.setText(QCoreApplication.translate("MainWindow", u"Run", None))

        """


    def addReOperationInput(self):
        pass
        #self.LifeTime_TableWidget.setRowCount(0)
        #self.operationArray = []
        #self.operationStep = 0
        #self.operationArray.append("hello")


        self.inputArray = []
        self.calc_rodPos = []
        self.calc_rodPosBox = []
        self.tableDatasetFlag = False

        self.corePower = self.ui.LF_Input01.value()
        self.bp        = self.ui.LF_Input02.value()
        self.targetEig = self.ui.LF_Input03.value()
        self.burnDel   = self.ui.LF_Input04.value()
        self.rodPos_P  = self.ui.LF_Input05.value()
        self.rodPos_5  = self.ui.LF_Input06.value()
        self.rodPos_4  = self.ui.LF_Input07.value()
        self.rodPos_3  = self.ui.LF_Input08.value()
        self.operationArray.append("Tmp")
        burn_step = int(self.bp / self.burnDel)

        for idx in range(burn_step-1):
            #unitArray
            pass
        #initBU     = self.ui.SD_Input01.value()
        #targetEigen = self.ui.SD_Input02.value()
        #targetASI = self.ui.SD_Input04.value()

        #self.initBU     = initBU
        #self.targetEigen = targetEigen
        #self.targetASI = targetASI

        #rdcPerHour = self.ui.SD_Input03.value()
        #EOF_Power  = 0.0 #self.ui.rdc04.value()

        ## TODO, make Except loop
        #if(rdcPerHour==0.0):
        #    print("Error!")
        #    return

        #nStep = math.ceil(( 100.0 - EOF_Power ) / rdcPerHour + 1.0)

        #self.recalculationIndex = nStep

        #for i in range(nStep-1):
        #    time = 1.0 * i
        #    power = 100.0 - i * rdcPerHour
        #    unitArray = [ time, power, initBU, targetEigen ]
        #    self.inputArray.append(unitArray)

        #time = 100.0 / rdcPerHour
        #power = 0.0
        #unitArray = [ time, power, initBU, targetEigen]
        #self.inputArray.append(unitArray)
        #self.nStep = nStep
        #self.SD_TableWidget.addInputArray(self.inputArray)


    def startFunc(self):

        if len(self.operationArray) == 0:
            self.addReOperationInput()
            self.load()
        else:
            pArray = []
            for iStep in range(self.nStep):
                pArray.append(self.inputArray[iStep])
            self.thread = QThread()
            self.calcManager.setLifeTimeVariable(df.CalcOpt_Lifetime, self.corePower, self.targetEig, pArray)

            self.calcManager.moveToThread(self.thread)

            self.thread.started.connect(self.calcManager.load)
            #self.thread.finished.connect(self.__finished)
            
            self.calcManager.finished.connect(self.thread.quit)
            #self.calcManager.progress.connect(self.showOutput)

            #self.outputArray = self.calcManager.startShutdown()
            self.thread.start()
            #self.addOutput()
        pass

    def empty_output(self):

        sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.ui.Lifetime_widgetChart.sizePolicy().hasHeightForWidth())
        self.ui.Lifetime_widgetChart.setSizePolicy(sizePolicy5)

        sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.ui.Lifetime_WidgetRadial.sizePolicy().hasHeightForWidth())
        self.ui.Lifetime_WidgetRadial.setSizePolicy(sizePolicy5)

        sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.ui.Lifetime_WidgetAxial.sizePolicy().hasHeightForWidth())
        self.ui.Lifetime_WidgetAxial.setSizePolicy(sizePolicy5)

    def addOutput(self):

        if not self.unitChartClass:
            # 02. Insert Spline Chart
            self.unitChartClass = unitSplineChart.UnitSplineChart(self.RodPosUnit)

            sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            sizePolicy5.setHorizontalStretch(10)
            sizePolicy5.setVerticalStretch(10)
            sizePolicy5.setHeightForWidth(self.ui.Lifetime_widgetChart.sizePolicy().hasHeightForWidth())
            self.ui.Lifetime_widgetChart.setSizePolicy(sizePolicy5)

            lay = QtWidgets.QVBoxLayout(self.ui.Lifetime_widgetChart)
            lay.setContentsMargins(0, 0, 0, 0)
            unitChartView = self.unitChartClass.returnChart()
            lay.addWidget(unitChartView)

        if not self.RadialWidget:
            # 04. Insert Radial Chart
            sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            sizePolicy5.setHorizontalStretch(6)
            sizePolicy5.setVerticalStretch(10)
            sizePolicy5.setHeightForWidth(self.ui.Lifetime_WidgetRadial.sizePolicy().hasHeightForWidth())
            self.ui.Lifetime_WidgetRadial.setSizePolicy(sizePolicy5)
            layR = QtWidgets.QVBoxLayout(self.ui.Lifetime_WidgetRadial)
            layR.setContentsMargins(0, 0, 0, 0)
            self.radialWidget = RadialWidget()
            layR.addWidget(self.radialWidget)

        if not self.AxialWidget:
            # 05. Insert Axial Chart
            sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            sizePolicy5.setHorizontalStretch(6)
            sizePolicy5.setVerticalStretch(10)
            sizePolicy5.setHeightForWidth(self.ui.Lifetime_WidgetAxial.sizePolicy().hasHeightForWidth())
            self.ui.Lifetime_WidgetAxial.setSizePolicy(sizePolicy5)
            layA = QtWidgets.QVBoxLayout(self.ui.Lifetime_WidgetAxial)
            layA.setContentsMargins(0, 0, 0, 0)
            self.axialWidget = AxialWidget()
            layA.addWidget(self.axialWidget)
