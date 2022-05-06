import sys
import platform
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import (QCoreApplication, QPropertyAnimation, QDate, QDateTime, QMetaObject, QObject, QPoint, QRect, QSize, QTime, QUrl, Qt, QEvent)
from PyQt5.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont, QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter, QPixmap, QRadialGradient)
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
import PyQt5.QtChart as QtCharts
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from model import *
import datetime
import os
import glob
from ui_main_rev18 import Ui_MainWindow
import Definitions as df
import widgets.utils.PyUnitButton as ts

import constants as cs
import utils as ut

from widgets.calculations.calculation_widget import CalculationWidget

import widgets.utils.PyRodPositionBarChart as unitChart
import widgets.utils.PyStartButton as startB

from widgets.utils.PySaveMessageBox import PySaveMessageBox

pointers = {

     "CoastDown_Snapshot": "ss_snapshot_file",
     "CD_Input01": "ndr_burnup",
     "CD_Input02": "ndr_target_asi",
     "CD_Input03": "ndr_stopping_criterion",
     "CD_Input04": "ndr_depletion_interval",
     "CD_Input05": "ndr_bank_position_P",
     "CD_Input06": "ndr_bank_position_5",
     "CD_Input07": "ndr_bank_position_4",
     "CD_Input08": "ndr_bank_position_3",

}


_POINTER_A_ = "self.ui.Coastdown_DB"
_NUM_PRINTOUT_ = 5
_MINIMUM_ITEM_COUNT = 40

class CoastdownWidget(CalculationWidget):

    def __init__(self, db, ui):

        super().__init__(db, ui, ui.Coastdown_DB, CoastDownInput, pointers)

        self.input_pointer = "self.current_calculation.coastdown_input"

        # 01. Setting Initial Coastdown Control Input Data
        self.Coastdown_PosBank5 = 0.0
        self.Coastdown_PosBank4 = 0.0
        self.Coastdown_PosBank3 = 0.0
        self.Coastdown_PosBankP = 0.0
        self.RodPosUnit = df.RodPosUnit_cm
        self.coastdownCalcOpt = df.select_none
        self.unitChangeFlag = False
        self.initSnapshotFlag = False

        # 02. Setting Initial Rod Position Bar Chart
        self.unitChartClass = unitChart.UnitBarChart(self.Coastdown_PosBank5,
                                                     self.Coastdown_PosBank4,
                                                     self.Coastdown_PosBank3,
                                                     self.Coastdown_PosBankP,
                                                     self.RodPosUnit)

        # 03. Setting Widget Interactions
        self.buttonGroupInput = QtWidgets.QButtonGroup()
        self.buttonGroupTarget = QtWidgets.QButtonGroup()
        self.settingAutoExclusive()
        self.settingLinkAction()

        # 04. Setting Initial UI widget
        self.ui.CD_Main03.hide()
        self.ui.LabelSub_CD02.hide()
        self.ui.CD_Input02.hide()

        # 05. Insert ChartView to chartWidget
        self.nSet = 0
        lay = QtWidgets.QVBoxLayout(self.ui.CoastDown_widgetChart)
        lay.setContentsMargins(0,0,0,0)
        unitChartView = self.unitChartClass.returnChart()
        lay.addWidget(unitChartView)

        # 06. Link Signal and Slot for Start Calculation
        self.makeStartButton(self.startFunc, super().saveFunc, self.ui.CoastDown_Main07, self.ui.Coastdown_CalcButton_Grid)
        #self.ui.Coastdown_DB.selectionChanged.connect

        self.ui.CoastDown_tabWidget.setCurrentIndex(0)

    def load(self):
        #PLANT AND RESTART FILES
        # self.ui.CoastDown_PlantCycle.clear()
        # self.ui.CoastDown_PlantName.clear()
        #
        # query = LoginUser.get(LoginUser.username == cs.ADMIN_USER)
        # user = query.login_user
        # plants, errors = ut.getPlantFiles(user)
        # for plant in plants:
        #     self.ui.CoastDown_PlantName.addItem(plant)
        # self.ui.CoastDown_PlantName.setCurrentText(user.plant_file)
        #
        # restarts, errors = ut.getRestartFiles(user)
        # for restart in restarts:
        #     self.ui.CoastDown_PlantCycle.addItem(restart)
        # self.ui.CoastDown_PlantCycle.setCurrentText(user.restart_file)

        self.load_input()
        self.load_recent_calculation()

    def set_all_component_values(self, coastdown_input):

        if coastdown_input.calculation_type == 0:
            self.ui.CoastDown_InpOpt2_NDR.setChecked(True)
        else:
            self.ui.CoastDown_InpOpt1_Snapshot.setChecked(True)

        if coastdown_input.search_type == 0:
            self.ui.CoastDown_CalcTarget01.setChecked(True)
        else:
            self.ui.CoastDown_CalcTarget02.setChecked(True)

        #Set values to components
        for key in pointers.keys():

            component = eval("self.ui.{}".format(key))
            value = eval("coastdown_input.{}".format(pointers[key]))
            if pointers[key]:
                if isinstance(component, QComboBox):
                    component.setCurrentText(value)
                else:
                    component.setValue(float(value))

        # self.ui.

    def settingAutoExclusive(self):
        self.buttonGroupInput.addButton(self.ui.CoastDown_InpOpt1_Snapshot)
        self.buttonGroupInput.addButton(self.ui.CoastDown_InpOpt2_NDR)
        self.buttonGroupTarget.addButton(self.ui.CoastDown_CalcTarget01)
        self.buttonGroupTarget.addButton(self.ui.CoastDown_CalcTarget02)

    def settingLinkAction(self):
        self.ui.CoastDown_InpOpt1_Snapshot.clicked['bool'].connect(self.settingInputOpt)
        self.ui.CoastDown_InpOpt2_NDR.clicked['bool'].connect(self.settingInputOpt)
        #
        self.ui.CoastDown_CalcTarget01.clicked['bool'].connect(self.settingTargetOpt)
        self.ui.CoastDown_CalcTarget02.clicked['bool'].connect(self.settingTargetOpt)

        self.ui.CoastDown_InpOpt1_Snapshot.clicked['bool'].connect(self.updateBankPosSnapshot)
        self.ui.CoastDown_InpOpt2_NDR.clicked['bool'].connect(self.updateBankPosUserInput)
        self.ui.CD_Input05.valueChanged['double'].connect(self.rodPosChangedEvent)
        self.ui.CD_Input06.valueChanged['double'].connect(self.rodPosChangedEvent)
        self.ui.CD_Input07.valueChanged['double'].connect(self.rodPosChangedEvent)
        self.ui.CD_Input08.valueChanged['double'].connect(self.rodPosChangedEvent)

        #self.ui.CoastDown_RodPos05_Unit.currentIndexChanged['int'].connect(self.changeRodPosUnit)

    def settingInputOpt(self):
        calculatin_type = 0

        if (self.ui.CoastDown_InpOpt2_NDR.isChecked()):
            self.coastdownCalcOpt = df.select_NDR
            self.ui.CD_Main03.show()
        elif (self.ui.CoastDown_InpOpt1_Snapshot.isChecked()):
            self.coastdownCalcOpt = df.select_snapshot
            self.ui.CD_Main03.hide()
            calculatin_type = 1

        if self.current_calculation:
            current_input = self.get_input(self.current_calculation)
            current_input.calculation_type = calculatin_type
            current_input.save()

    def settingTargetOpt(self):
        search_type = 0
        if (self.ui.CoastDown_CalcTarget01.isChecked()):
            self.ui.LabelSub_CD02.show()
            self.ui.CD_Input02.show()
        elif (self.ui.CoastDown_CalcTarget02.isChecked()):
            self.ui.LabelSub_CD02.hide()
            self.ui.CD_Input02.hide()
            search_type = 1

        if self.current_calculation:
            current_input = self.get_input(self.current_calculation)
            current_input.search_type = search_type
            current_input.save()

    def rodPosChangedEvent(self):
        if (self.unitChangeFlag == False and self.initSnapshotFlag == False):
            self.updateBankPosUserInput()
        else:
            pass

    def updateBankPosSnapshot(self):
        # TODO, Read Rod Position from Snapshot
        self.initSnapshotFlag = True
        self.Coastdown_PosBank5 = 50.0 * df.convertRodPosUnit[0][self.RodPosUnit]
        self.Coastdown_PosBank4 = 50.0 * df.convertRodPosUnit[0][self.RodPosUnit]
        self.Coastdown_PosBank3 = 50.0 * df.convertRodPosUnit[0][self.RodPosUnit]
        self.Coastdown_PosBankP = 50.0 * df.convertRodPosUnit[0][self.RodPosUnit]
        self.ui.CD_Input05.setValue(self.Coastdown_PosBank5)
        self.ui.CD_Input06.setValue(self.Coastdown_PosBank4)
        self.ui.CD_Input07.setValue(self.Coastdown_PosBank3)
        self.ui.CD_Input08.setValue(self.Coastdown_PosBankP)
        self.replaceRodPosition()
        self.initSnapshotFlag = False

    def updateBankPosUserInput(self):
        self.Coastdown_PosBank5 = self.ui.CD_Input05.value()
        self.Coastdown_PosBank4 = self.ui.CD_Input06.value()
        self.Coastdown_PosBank3 = self.ui.CD_Input07.value()
        self.Coastdown_PosBankP = self.ui.CD_Input08.value()
        self.replaceRodPosition()

    def replaceRodPosition(self):
        self.unitChartClass.replaceRodPosition(self.Coastdown_PosBank5, self.Coastdown_PosBank4,
                                               self.Coastdown_PosBank3, self.Coastdown_PosBankP, self.RodPosUnit)

        if self.load_rod_tab and self.ui.CoastDown_tabWidget.currentIndex() != 2:
            self.ui.CoastDown_tabWidget.setCurrentIndex(2)

    def changeRodPosUnit(self, index):
        self.unitChangeFlag = True
        self.rodPosChangedEvent()
        #text = "   " + self.ui.CoastDown_RodPos05_Unit.currentText()
        #self.ui.CoastDown_RodPos04_Unit.setText(text)
        #self.ui.CoastDown_RodPos03_Unit.setText(text)
        #self.ui.CoastDown_RodPos_P_Unit.setText(text)

        currentOpt = self.RodPosUnit
        self.Coastdown_PosBank5 = self.Coastdown_PosBank5 * df.convertRodPosUnit[currentOpt][index]
        self.Coastdown_PosBank4 = self.Coastdown_PosBank4 * df.convertRodPosUnit[currentOpt][index]
        self.Coastdown_PosBank3 = self.Coastdown_PosBank3 * df.convertRodPosUnit[currentOpt][index]
        self.Coastdown_PosBankP = self.Coastdown_PosBankP * df.convertRodPosUnit[currentOpt][index]

        self.ui.CoastDown_RodPos05.setValue(self.Coastdown_PosBank5)
        self.ui.CoastDown_RodPos04.setValue(self.Coastdown_PosBank4)
        self.ui.CoastDown_RodPos03.setValue(self.Coastdown_PosBank3)
        self.ui.CoastDown_RodPos_P.setValue(self.Coastdown_PosBankP)

        self.RodPosUnit = index
        self.replaceRodPosition()
        self.unitChangeFlag = False

    def makeStartButton(self, startFunc, saveFunc, frame, grid):
        self.ui.Coastdown_save_button.clicked['bool'].connect(saveFunc)
        self.ui.Coastdown_run_button.clicked['bool'].connect(startFunc)
        """
        self.CoastDown_StartButton = startB.startButton(df.CalcOpt_Coastdown, startFunc, frame)

        self.CoastDown_StartButton.setObjectName(u"CoastDown_StartButton")
        sizePolicy3 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy3.setHeightForWidth(self.CoastDown_StartButton.sizePolicy().hasHeightForWidth())
        self.CoastDown_StartButton.setSizePolicy(sizePolicy3)
        self.CoastDown_StartButton.setMinimumSize(QSize(200, 40))
        self.CoastDown_StartButton.setMaximumSize(QSize(16777215, 40))
        font2 = QFont()
        font2.setFamily(u"Segoe UI")
        font2.setPointSize(14)
        font2.setBold(False)
        font2.setWeight(50)
        self.CoastDown_StartButton.setFont(font2)
        self.CoastDown_StartButton.setStyleSheet(u"QPushButton {\n"
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

        self.CoastDown_SaveButton = startB.startButton(df.CalcOpt_Coastdown, saveFunc, frame)
        self.CoastDown_SaveButton.setObjectName(u"CoastDown_SaveButton")
        sizePolicy3 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy3.setHeightForWidth(self.CoastDown_SaveButton.sizePolicy().hasHeightForWidth())
        self.CoastDown_SaveButton.setSizePolicy(sizePolicy3)
        self.CoastDown_SaveButton.setMinimumSize(QSize(200, 40))
        self.CoastDown_SaveButton.setMaximumSize(QSize(16777215, 40))
        font2 = QFont()
        font2.setFamily(u"Segoe UI")
        font2.setPointSize(14)
        font2.setBold(False)
        font2.setWeight(50)
        self.CoastDown_SaveButton.setFont(font2)
        self.CoastDown_SaveButton.setStyleSheet(u"QPushButton {\n"
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

        grid.addWidget(self.CoastDown_SaveButton, 1, 0, 1, 1)
        grid.addWidget(self.CoastDown_StartButton, 1, 1, 1, 2)
        self.CoastDown_SaveButton.setText(QCoreApplication.translate("MainWindow", u"Save", None))
        self.CoastDown_StartButton.setText(QCoreApplication.translate("MainWindow", u"Run", None))
        """

    def get_input(self, calculation_object):
        return calculation_object.coastdown_input

    def get_default_input(self, user):
        now = datetime.datetime.now()
        coastdown_input = CoastDownInput.create(
            calculation_type=0,
            search_type=0,
            ss_snapshot_file="",
            ndr_burnup=0,
            ndr_target_asi=0,
            ndr_stopping_criterion=0,
            ndr_depletion_interval=0,
            ndr_bank_position_5=0,
            ndr_bank_position_4=0,
            ndr_bank_position_3=0,
            ndr_bank_position_P=0,
        )

        coastdown_calculation = Calculations.create(user=user,
                                                    calculation_type=cs.CALCULATION_COASTDOWN,
                                                    created_date=now,
                                                    modified_date=now,
                                                    coastdown_input=coastdown_input
                                                    )

        return coastdown_calculation, coastdown_input
    """
    def saveFunc(self):

        msgBox = PySaveMessageBox(self.current_calculation, self.ui.Lifetime_run_button)
        msgBox.setWindowTitle("Save Input?")
        #msgBox.setText("Load this input?")
        #msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        #msgBox.setDefaultButton(QMessageBox.Yes)
        result = msgBox.exec_()
        #if result == QMessageBox.Yes:
        #    self.load_input(self.table_objects[tableItem.row()])
        if msgBox.result == "Saved":
            pass
        elif msgBox.result == "SavedAs":
            query = LoginUser.get(LoginUser.username == cs.ADMIN_USER)
            user = query.login_user
            coastdown_calculation, coastdown_input = ut.get_last_input(user, CoastDownInput)

            calculation_object = self.current_calculation
            calculation_input = self.current_calculation.coastdown_input

            coastdown_calculation, coastdown_input  = self.getDefaultInput(user)
            for pointer_key in pointers:
                exec("coastdown_input.{} = calculation_input.{}".format(pointers[pointer_key], pointers[pointer_key]))

            coastdown_calculation.filename = msgBox.lineEdit1.text()
            coastdown_calculation.comments = msgBox.lineEdit2.text()
            coastdown_calculation.save()
            coastdown_input.save()

        self.load_recent_calculation(CoastDownInput, _POINTER_A_, _MINIMUM_ITEM_COUNT, _NUM_PRINTOUT_)
    """