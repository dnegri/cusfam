
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

pointers = {

    "ECP_Input01": "bs_ndr_date_time",
    "ECP_Input03": "bs_ndr_power",
    "ECP_Input04": "bs_ndr_burnup",
    "ECP_Input05": "bs_ndr_average_temperature",
    "ECP_Input06": "bs_ndr_target_eigen",
    "ECP_Input07": "bs_ndr_bank_position_P",
    "ECP_Input08": "bs_ndr_bank_position_5",
    "ECP_Input09": "bs_ndr_bank_position_4",

    "ECP_Input12": "as_ndr_date_time",
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

        # 01. Setting Initial ECP Control Input Data
        self.posBeforeBankP = 381.0
        self.posBeforeBank5 = 381.0
        self.posBeforeBank4 = 381.0
        self.posBeforeBank3 = 381.0

        self.posAfterBankP = 381.0
        self.posAfterBank5 = 381.0
        self.posAfterBank4 = 381.0
        self.posAfterBank3 = 381.0

        self.ECP_CalcOpt = df.select_none
        self.ECP_TargetOpt = df.select_none
        self.flag_RodPosAfterShutdown = False

        self.RodPosUnitBefore = df.RodPosUnit_cm
        self.unitChangeFlagBefore = False
        self.initSnapshotFlagBefore = False

        self.unitChangeFlag = False
        self.RodPosUnitAfter = df.RodPosUnit_cm
        self.initSnapshotFlag = False

        # 03. Setting Widget Interactions
        self.buttonGroupInput = QtWidgets.QButtonGroup()
        self.buttonGroupTarget = QtWidgets.QButtonGroup()
        self.settingAutoExclusive()
        self.settingLinkAction()

        # 04. Setting Initial UI widget
        self.ui.ECP_Main02.hide()
        self.ui.ECP_Main03.hide()

        # 05. Insert ChartView to chartWidget
        sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy5.setHorizontalStretch(6)
        sizePolicy5.setVerticalStretch(10)
        sizePolicy5.setHeightForWidth(self.ui.ECP_widgetChart.sizePolicy().hasHeightForWidth())
        self.ui.ECP_widgetChart.setSizePolicy(sizePolicy5)

        layA = QtWidgets.QVBoxLayout(self.ui.ECP_WidgetAxial)
        layA.setContentsMargins(0, 0, 0, 0)
        self.axialWidget = AxialWidget()
        layA.addWidget(self.axialWidget)

        # 03. Insert Table
        # self.tableItem = ["Time\n(hour)", "Power\n(%)", "Burnup\n(MWD/MTU)", "Keff",]
        # self.ECP_TableWidget = table01.ShutdownTableWidget(self.ui.frame_ECP_TableWidget_2, self.tableItem)
        # layoutTableButton = self.ECP_TableWidget.returnButtonLayout()
        # self.ui.gridlayout_ECP_TableWidget.addWidget(self.ECP_TableWidget, 0, 0, 1, 1)
        # #self.ui.gridlayout_ECP_TableWidget.addLayout(layoutTableButton, 1, 0, 1, 1)

        self.RodPosUnit = df.RodPosUnit_cm
        self.unitChartClass = unitSplineChart.UnitSplineChart(self.RodPosUnit)

        sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy5.setHorizontalStretch(10)
        sizePolicy5.setVerticalStretch(10)
        sizePolicy5.setHeightForWidth(self.ui.ECP_widgetChart.sizePolicy().hasHeightForWidth())
        self.ui.ECP_widgetChart.setSizePolicy(sizePolicy5)

        lay = QtWidgets.QVBoxLayout(self.ui.ECP_widgetChart)
        lay.setContentsMargins(0, 0, 0, 0)
        unitChartView = self.unitChartClass.returnChart()
        lay.addWidget(unitChartView)
        self.unitChartClass.axisX.setMax(10)
        self.current_index = 0
        # self.load()

    def load(self):
        self.load_input()

    def set_all_component_values(self, ecp_input):
        super().set_all_component_values(ecp_input)

        if ecp_input.search_type == 0:
            self.ui.ECP_CalcTarget01.setChecked(True)
        else:
            self.ui.ECP_CalcTarget02.setChecked(True)


    def settingAutoExclusive(self):
        self.buttonGroupTarget.addButton(self.ui.ECP_CalcTarget01)
        self.buttonGroupTarget.addButton(self.ui.ECP_CalcTarget02)

    def settingLinkAction(self):

        self.ui.ECP_CalcTarget01.toggled['bool'].connect(self.settingTargetOpt)
        self.ui.ECP_CalcTarget02.toggled['bool'].connect(self.settingTargetOpt)

        self.ui.ECP_Output1.currentIndexChanged.connect(self.ECP_output_index_changed)

        self.ui.ECP_save_button.clicked['bool'].connect(self.start_save)
        self.ui.ECP_run_button.clicked['bool'].connect(self.start_calc)

    def settingTargetOpt(self):
        search_type = 0
        if self.ui.ECP_CalcTarget01.isChecked():
            self.ECP_TargetOpt = df.select_Boron
            # show Reactor Condition After Shutdown Input Frame
            self.ui.ECP_Main02.show()
            self.ui.ECP_Main03.show()
            # show Rod Position Input Widget
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

        elif (self.ui.ECP_CalcTarget02.isChecked()):
            self.ECP_TargetOpt = df.select_RodPos
            # show Reactor Condition After Shutdown Input Frame
            self.ui.ECP_Main02.show()
            self.ui.ECP_Main03.show()
            # Hide Rod Position Input Widget
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

    def get_input(self, calculation_object):
        return calculation_object.ecp_input


    def get_default_input(self, user):
        now = datetime.datetime.now()
        ecp_input = ECP_Input.create(
            search_type=0,
            bs_ndr_date_time=now,
            bs_ndr_power=0,
            bs_ndr_burnup=0,
            bs_ndr_average_temperature=0,
            bs_ndr_target_eigen = 1.0,
            bs_ndr_bank_position_P=381.0,
            bs_ndr_bank_position_5=381.0,
            bs_ndr_bank_position_4=381.0,
            as_ndr_date_time=now,
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
        #self.setSuccecciveInput()
        # Set Input Variable
        self.startTime = self.ui.ECP_Input01.dateTime()
        self.endTime = self.ui.ECP_Input12.dateTime()
        self.pw = self.ui.ECP_Input03.value()
        self.bp = self.ui.ECP_Input04.value()
        #self.mtc = self.ui.ECP_Input05.value()
        #self.eigen = self.ui.ECP_Input06.value()
        #self.P_Pos = self.ui.ECP_Input07.value()
        #self.r5Pos = self.ui.ECP_Input08.value()
        #self.r4Pos = self.ui.ECP_Input09.value()
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

        error = self.check_calculation_input()

        if not error:
            self.cal_start = True
            self.current_index = 0
            self.start_calculation_message = SplashScreen()
            self.start_calculation_message.killed.connect(self.killManagerProcess)
            self.start_calculation_message.init_progress(10, 500)
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

        # self.show_output_values(0)
        self.ECP_output_index_changed()

    def show_output_values(self, show_index=0):
        startTime_Day  = self.startTime.date()
        startTime_Time = self.startTime.time()
        startTime_STR = "{:4d}-{:02d}-{:02d} {:02d}:{:02d}".format(startTime_Day.year(),startTime_Day.month(),startTime_Day.day(),startTime_Time.hour(),startTime_Time.minute())
        endTime_Day  = self.endTime.date()
        endTime_Time = self.endTime.time()
        endTime_STR = "{:4d}-{:02d}-{:02d} {:02d}:{:02d}".format(endTime_Day.year(),endTime_Day.month(),endTime_Day.day(),endTime_Time.hour(),endTime_Time.minute())
        self.ui.labelECP001.setText(startTime_STR)
        self.ui.labelECP002.setText("{:.2f}".format(self.pw))
        self.ui.labelECP003.setText("{:.2f}".format(self.bp))
        self.ui.labelECP004.setText("{:.2f}".format(self.mtc))
        self.ui.labelECP005.setText("{:.5f}".format(self.eigen))
        self.ui.labelECP006.setText("{:.2f}".format(self.calcManager.out_ppm_old[show_index]))
        # self.ui.labelECP007.setText("{:.2f}".format(self.P_Pos))
        # self.ui.labelECP008.setText("{:.2f}".format(self.r5Pos))
        # self.ui.labelECP009.setText("{:.2f}".format(self.r4Pos))
        self.ui.labelECP011.setText(endTime_STR)
        if self.ECP_TargetOpt==df.select_Boron:
            self.ui.labelECP012.setText("{:.2f}".format(self.calcManager.out_ppm[show_index]))
            self.ui.labelECP013.setText("{:.2f}".format(self.calcManager.rodPos[show_index][0]))
            self.ui.labelECP014.setText("{:.2f}".format(self.calcManager.rodPos[show_index][1]))
            self.ui.labelECP015.setText("{:.2f}".format(self.calcManager.rodPos[show_index][2]))

            [self.pw, self.bp, self.mtc, eigen, P_Pos, r5Pos, r4Pos, shutdown_P_Pos, shutdown_r5Pos,
             shutdown_r4Pos] = self.calcManager.inputArray

            self.unitChartClass.axisY_CBC.setMax(max(1000, self.calcManager.out_ppm[0]+100))
            output_array = [[self.calcManager.out_ppm[0], 0.0, shutdown_P_Pos, shutdown_r5Pos, shutdown_r4Pos, 381],]
            for i in range(0, len(self.calcManager.out_ppm)):
                output_array.append([self.calcManager.out_ppm[i], 0.0,
                                     self.calcManager.rodPos[i][0],
                                     self.calcManager.rodPos[i][1],
                                     self.calcManager.rodPos[i][2],
                                     381])
            self.appendPointData(output_array, 0)

        if self.ECP_TargetOpt==df.select_RodPos:
            self.ui.labelECP012.setText("{:.2f}".format(self.calcManager.out_ppm[show_index]))
            self.ui.labelECP013.setText("{:.2f}".format(self.calcManager.rodPos[show_index][0]))
            self.ui.labelECP014.setText("{:.2f}".format(self.calcManager.rodPos[show_index][1]))
            self.ui.labelECP015.setText("{:.2f}".format(self.calcManager.rodPos[show_index][2]))

            [ self.pw, self.bp, self.mtc, eigen, P_Pos, r5Pos, r4Pos, target_ppm ] = self.calcManager.inputArray

            self.unitChartClass.axisY_CBC.setMax(max(1000, target_ppm+100))
            #self.unitChartClass.axisX.setMax(target_ppm+100)
            output_array = [[self.calcManager.out_ppm[0], 0.0,
                             self.calcManager.rodPos[0][0],
                             self.calcManager.rodPos[0][1],
                             self.calcManager.rodPos[0][2], 381],]
            for i in range(0, len(self.calcManager.out_ppm)):
                output_array.append([self.calcManager.out_ppm[i], 0.0,
                                     self.calcManager.rodPos[i][0],
                                     self.calcManager.rodPos[i][1],
                                     self.calcManager.rodPos[i][2],
                                     381])
            self.appendPointData(output_array, 0)

    def appendPointData(self, outputArray, start):

        self.unitChartClass.clear()

        posP = []
        posR5 = []
        posR4 = []
        posR3 = []
        posCBC = []
        for i in range(len(outputArray)):
            posP.append(QPointF(start+i, outputArray[i][2]))
            posR5.append(QPointF(start+i, outputArray[i][3]))
            posR4.append(QPointF(start+i, outputArray[i][4]))
            posR3.append(QPointF(start+i, outputArray[i][5]))
            posCBC.append(QPointF(start+i, outputArray[i][0]))

        rodPos = [posP, posR5, posR4, posR3, ]

        self.unitChartClass.appendRodPosition(len(rodPos), rodPos, posCBC)


    def setSuccecciveInput(self):
        # Initialize
        self.inputArray = []

        initBU = self.ui.ECP_Input04.value()
        targetEigen = self.ui.ECP_Input06.value()

        timeInterval = 3.0
        core_power = self.ui.ECP_Input03.value()

        nStep = 10
        for i in range(nStep):
            time = timeInterval * i
            power = core_power
            unitArray = [time, power, initBU, targetEigen]
            self.inputArray.append(unitArray)

        self.nStep = nStep
        self.ECP_TableWidget.addInputArray(self.inputArray)
        self.last_table_created = datetime.datetime.now()

