from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication, QPointF
from PyQt5.QtGui import QFont
from PyQt5 import QtGui

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QSizePolicy

from model import *
import datetime

import Definitions as df

import constants as cs

from widgets.calculations.calculation_widget import CalculationWidget

import widgets.utils.PyUnitTableWidget as table00
import widgets.utils.PyShutdownTableWidget as table01
import widgets.utils.PyDecayTableWidget as table02
import widgets.utils.PyRodPositionSplineChart as unitSplineChart
from widgets.output.axial.axial_plot import AxialWidget
from widgets.output.radial.radial_graph import RadialWidget
import math
from widgets.output.trend.trends_graph import trendWidget

from widgets.utils.PySaveMessageBox import PySaveMessageBox

from PyQt5.QtWidgets import *

from widgets.utils.splash_screen import SplashScreen

_POINTER_A_ = "self.ui.RO_DB"
_NUM_PRINTOUT_ = 5
_MINIMUM_ITEM_COUNT = 20

pointers = {
    "RO_Input01": "ndr_cal_type",
    "RO_Input02": "ndr_burnup",
    "RO_Input03": "ndr_time",
    "RO_Input04": "ndr_bank_position_P",
    "RO_Input05": "ndr_bank_position_5",
    "RO_Input06": "ndr_power_ratio",
    "RO_Input07": "ndr_asi",
    "RO_Input08": "ndr_end_power",


    # "RO_Input00": "ndr_cal_type",
    #
    # "RO_Input01": "ndr_burnup",
    #
    #
    # "RO_Input08": "ndr_time",
    #
    # "RO_Input03": "ndr_bank_position_P",
    # "RO_Input05": "ndr_bank_position_5",
    # "RO_Input06": "ndr_bank_position_4",
    # "RO_Input07": "ndr_bank_position_3",
    #
    #
    # "RO_Input09": "ndr_power_ratio",
    # "RO_Input10": "ndr_asi",
    # "RO_Input11": "ndr_end_power",
}


class ReoperationWidget(CalculationWidget):

    def __init__(self, db, ui, calManager, queue, message_ui):

        pass
        super().__init__(db, ui, None, RO_Input, pointers, calManager, queue, message_ui)

        self.input_pointer = "self.current_calculation.asi_input"

        # 01. Setting Initial Shutdown Input Setting
        self.RO_CalcOpt = df.select_none
        self.inputArray = []
        # Input Dataset for RO_TableWidget Rod Position

        self.recalculationIndex = -1

        self.decayArray = []
        self.operationArray = []
        self.decayStep = 0
        self.operationStep = 0
        self.fixedTime_flag = True
        self.ASI_flag = True
        self.CBC_flag = True
        self.Power_flag = True


        # 03. Insert Table
        self.tableItem = ["Time\n(hour)","Power\n(%)","Burnup\n(MWD/MTU)","Keff","ASI","Boron\n(ppm)",
                           "Bank P\n(cm)", "Bank 5\n(cm)","Bank 4\n(cm)","Bank 3\n(cm)",]
        self.RO_TableWidget = table01.ShutdownTableWidget(self.ui.frame_RO_TableWidget, self.tableItem)
        layoutTableButton = self.RO_TableWidget.returnButtonLayout()
        self.ui.gridlayout_RO_TableWidget.addWidget(self.RO_TableWidget, 0, 0, 1, 1)
        self.ui.gridlayout_RO_TableWidget.addLayout(layoutTableButton,1,0,1,1)
        [ self.unitButton01, self.unitButton02, self.unitButton03 ] = self.RO_TableWidget.returnTableButton()

        # 04. trend, radial, axial graph add
        self.unitChart = None
        self.radialWidget = None
        self.axialWidget = None
        self.addOutput()

        # 05. Setting Widget Interactions
        self.settingLinkAction()

        self.all_rod_in = True

    def load(self):
        if len(self.inputArray) == 0:
            self.ui.RO_run_button.setText("Create Scenario")
            self.ui.RO_run_button.setStyleSheet(df.styleSheet_Create_Scenarios)
        else:
            self.ui.RO_run_button.setText("Run")
            self.ui.RO_run_button.setStyleSheet(df.styleSheet_Run)

        self.load_input()

    def set_all_component_values(self, ecp_input):
        super().set_all_component_values(ecp_input)

        # self.ui.LabelSub_RO01_2.hide()
        # self.ui.RO_Input02.hide()
        # self.ui.LabelSub_RO02_2.hide()
        # self.ui.RO_Input08.hide()

    def index_changed(self, key):
        super().index_changed(key)
        if self.current_calculation and self.last_table_created:
            if self.current_calculation.modified_date > self.last_table_created:
                self.ui.RO_run_button.setText("Create Scenario")
                self.ui.RO_run_button.setStyleSheet(df.styleSheet_Create_Scenarios)

    def value_changed(self, key):
        super().value_changed(key)
        if self.current_calculation and self.last_table_created:
            if self.current_calculation.modified_date > self.last_table_created:
                self.ui.RO_run_button.setText("Create Scenario")
                self.ui.RO_run_button.setStyleSheet(df.styleSheet_Create_Scenarios)

    def settingLinkAction(self):

        self.ui.RO_run_button.clicked['bool'].connect(self.start_calc)
        self.ui.RO_save_button.clicked['bool'].connect(self.start_save)

        self.unitButton01.clicked['bool'].connect(self.clickSaveAsExcel)
        self.unitButton02.clicked['bool'].connect(self.resetPositionData)
        self.RO_TableWidget.itemSelectionChanged.connect(self.cell_changed)
        self.unitButton03.clicked['bool'].connect(self.clearOuptut)

    def get_input(self, calculation_object):
        return calculation_object.ro_input

    def get_default_input(self, user):
        now = datetime.datetime.now()

        RO_input = RO_Input.create(
            ndr_cal_type="Restart",

            ndr_burnup=12000,
            ndr_power =0.0,

            ndr_bank_position_5 = 210,
            ndr_bank_position_4 = 381,
            ndr_bank_position_3 = 381,
            ndr_bank_position_P = 381,

            ndr_time = 30.0,

            ndr_target_keff = 1.0,
            ndr_power_ratio = 3.0,
            ndr_asi = 0.10,
            ndr_end_power = 100.0,
        )

        RO_calculation = Calculations.create(user=user,
                                              calculation_type=cs.CALCULATION_RO,
                                              created_date=now,
                                              modified_date=now,
                                              ro_input=RO_input
                                              )
        return RO_calculation, RO_input

    def setSuccecciveInput(self):
        #TODO SGH, MAKE DEFAULT SETTINGE FOR POWER ASCENSION
        self.RO_TableWidget.clearOutputArray()
        self.RO_TableWidget.clearOutputRodArray()

        # Initialize
        self.inputArray = []
        self.calc_rodPos = []
        self.calc_rodPosBox = []
        self.tableDatasetFlag = False

        initBU = self.ui.RO_Input02.value()
        #initPower = self.ui.RO_Input02.value()
        initPower = 0.0
        rodP = self.ui.RO_Input04.value()
        rod5 = self.ui.RO_Input05.value()
        rod4 = 381.0
        rod3 = 381.0

        overlap_time = self.ui.RO_Input03.value()

        targetEigen = 1.00000
        rdcPerHour = self.ui.RO_Input06.value()
        targetASI = self.ui.RO_Input07.value()
        EOF_Power = self.ui.RO_Input08.value()

        self.unitChart.adjustTime(overlap_time)
        # update Power Variation Per Hour and power increase flag
        # ( Power Ascention Mode == True, Power Reduction Mode == False)
        self.unitChart.updateRdcPerHour(rdcPerHour,True)


        #check input
        if initPower >= EOF_Power:
            msgBox = QMessageBox(self.get_ui_component())
            msgBox.setWindowTitle("Power Error")
            msgBox.setText("Initial Power {} is greater than Final Power {}".format(initPower, EOF_Power))
            msgBox.setStandardButtons(QMessageBox.Ok)
            #msgBox.setWindowFlag(QtCore.Qt.FramelessWindowHint)
            result = msgBox.exec_()
            return

        self.initBU = initBU
        self.initPower = initPower
        self.initRodPos = [rod5, rod4, rod3, rodP]

        self.overlap_time = overlap_time

        self.targetASI = targetASI
        self.targetEigen = targetEigen

        # TODO, make Except loop
        if(rdcPerHour==0.0):
            print("Error!")
            return

        powers = []
        currentPower = initPower
        while currentPower < EOF_Power:
            currentPower += rdcPerHour
            if currentPower > EOF_Power:
                currentPower = EOF_Power
            powers.append(currentPower)

        self.all_rod_in = True
        # if initPower == 0:
        #     msgBox = QMessageBox(self.ui.RO_Input02)
        #     msgBox.setWindowTitle("Model Setup")
        #     msgBox.setText("Initial Power is 0\nAll Rod In?")
        #     msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        #     #msgBox.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        #     result = msgBox.exec_()
        #     if result == QMessageBox.Yes:
        #         self.all_rod_in = True

        nStep = len(powers)
        if self.all_rod_in:
            initial_rod = [0, 0, 0, 0, 0, 0]
        else:
            initial_rod = [0, 0, rodP, rod5, rod4, rod3, ]
        self.inputArray.append([0, self.initPower, initBU, targetEigen])
        self.inputArray.append([self.overlap_time, self.initPower, initBU, targetEigen])
        self.inputArray.append([self.overlap_time, self.initPower, initBU, targetEigen])

        for i in range(nStep):
            time = self.overlap_time + 1.0 * (i+1)
            unitArray = [ time, powers[i], initBU, targetEigen ]
            self.inputArray.append(unitArray)

        self.RO_TableWidget.addInputArray(self.inputArray)
        self.RO_TableWidget.makeOutputTable([initial_rod,initial_rod,[0, 0, rodP, rod5, rod4, rod3,]])

    def clickSaveAsExcel(self):
        is_succ = self.RO_TableWidget.clickSaveAsExcel()
        if not is_succ:
            msgBox = QMessageBox(self.get_ui_component())
            msgBox.setWindowTitle("Output not found")
            msgBox.setText("Output not found to save to excel")
            msgBox.setStandardButtons(QMessageBox.Ok)
            result = msgBox.exec_()

    def resetPositionData(self):
        self.RO_TableWidget.resetPositionData()

    def empty_output(self):
            sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            sizePolicy5.setHorizontalStretch(0)
            sizePolicy5.setVerticalStretch(0)
            sizePolicy5.setHeightForWidth(self.ui.RO_widgetChart.sizePolicy().hasHeightForWidth())
            self.ui.RO_widgetChart.setSizePolicy(sizePolicy5)
            
            sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            sizePolicy5.setHorizontalStretch(0)
            sizePolicy5.setVerticalStretch(0)
            sizePolicy5.setHeightForWidth(self.ui.RO_WidgetRadial.sizePolicy().hasHeightForWidth())
            self.ui.RO_WidgetRadial.setSizePolicy(sizePolicy5)

            sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            sizePolicy5.setHorizontalStretch(0)
            sizePolicy5.setVerticalStretch(0)
            sizePolicy5.setHeightForWidth(self.ui.RO_WidgetAxial.sizePolicy().hasHeightForWidth())
            self.ui.RO_WidgetAxial.setSizePolicy(sizePolicy5)

    def addOutput(self):

        if not self.unitChart:
            # 02. Insert Spline Chart
            # self.unitChartClass = unitSplineChart.UnitSplineChart(df.RodPosUnit_cm)
            #
            # sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            # sizePolicy5.setHorizontalStretch(10)
            # sizePolicy5.setVerticalStretch(10)
            # sizePolicy5.setHeightForWidth(self.ui.RO_widgetChart.sizePolicy().hasHeightForWidth())
            # self.ui.RO_widgetChart.setSizePolicy(sizePolicy5)
            #
            # lay = QtWidgets.QVBoxLayout(self.ui.RO_widgetChart)
            # lay.setContentsMargins(0,0,0,0)
            # unitChartView = self.unitChartClass.returnChart()
            # lay.addWidget(unitChartView)



            lay = self.ui.grid_RO_frameChart
            self.unitChart = trendWidget(self.fixedTime_flag, self.ASI_flag, self.CBC_flag, self.Power_flag)
            sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            sizePolicy5.setHorizontalStretch(10)
            sizePolicy5.setVerticalStretch(10)
            self.unitChart.setSizePolicy(sizePolicy5)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(self.unitChart, 0, 0, 1, 1)




        if not self.radialWidget:
            # 04. Insert Radial Chart
            # sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            # sizePolicy5.setHorizontalStretch(6)
            # sizePolicy5.setVerticalStretch(10)
            # sizePolicy5.setHeightForWidth(self.ui.RO_WidgetRadial.sizePolicy().hasHeightForWidth())
            # self.ui.RO_WidgetRadial.setSizePolicy(sizePolicy5)
            # layR = QtWidgets.QVBoxLayout(self.ui.RO_WidgetRadial)
            # layR.setContentsMargins(0,0,0,0)
            # self.radialWidget = RadialWidget()
            # layR.addWidget(self.radialWidget)

            self.radialWidget = RadialWidget(self.ui.RO_LP_frame, self.ui.RO_InputLPgrid)

        if not self.axialWidget:
            # 05. Insert Axial Chart
            sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            sizePolicy5.setHorizontalStretch(6)
            sizePolicy5.setVerticalStretch(10)
            sizePolicy5.setHeightForWidth(self.ui.RO_WidgetAxial.sizePolicy().hasHeightForWidth())
            self.ui.RO_WidgetAxial.setSizePolicy(sizePolicy5)
            layA = QtWidgets.QVBoxLayout(self.ui.RO_WidgetAxial)
            layA.setContentsMargins(0,0,0,0)
            self.axialWidget = AxialWidget()
            layA.addWidget(self.axialWidget)

    def start_calc(self):

        super().start_calc()

        if self.unitChart:
            self.unitChart.clearData()

        if self.ui.RO_run_button.text() == cs.RUN_BUTTON_CREATE_SCENARIO:
            self.setSuccecciveInput()
            self.load()
        else:
            calcOpt, targetASI, pArray, error = self.get_calculation_input()

            #self.unitChartClass.axisX.setMax(len(pArray))
            if not error and calcOpt != df.CalcOpt_KILL:

                # Only consider restart
                x_20 = 0
                for p_index, values in enumerate(pArray):
                    if values[1] >= 20:
                        x_20 = p_index
                        break

                #self.unitChartClass.setXPoints(x_20, len(pArray))
                #self.unitChartClass.drawASIBand(x_20, len(pArray), 0.27, 0.60)

                if calcOpt == df.CalcOpt_RO:

                    if self.unitChart:
                        self.unitChart.clearData()
                    self.setSuccecciveInput()
                    self.RO_TableWidget.outputArray = []
                    self.calcManager.reoperation_output = []

                self.RO_TableWidget.last_update = 0
                self.start_calculation_message = SplashScreen()
                self.start_calculation_message.killed.connect(self.killManagerProcess)
                self.start_calculation_message.init_progress(len(pArray), 500)

                self.ui.RO_run_button.setText(cs.RUN_BUTTON_RUNNING)
                self.ui.RO_run_button.setDisabled(True)
                self.RO_TableWidget.clearOutputArray()
                self.queue.put((calcOpt, targetASI, pArray))

    def finished(self):

        if self.start_calculation_message:
            self.start_calculation_message.close()

        #self.definePointData(self.calcManager.reoperation_output)
        self.RO_TableWidget.selectRow(3)
        self.ui.RO_run_button.setText(cs.RUN_BUTTON_RUN)
        self.ui.RO_run_button.setDisabled(False)
        # self.cell_changed()

    def showOutput(self):
        self.start_calculation_message.progress()
        last_update = self.RO_TableWidget.last_update
        self.RO_TableWidget.appendOutputTable(self.calcManager.reoperation_output[last_update:], last_update)
        #self.appendPointData(self.calcManager.outputArray[last_update:], last_update)
        self.appendPointData(self.calcManager.reoperation_output, 0)

        # pd2d = self.calcManager.outputArray[-1][-1]
        # pd1d = self.calcManager.outputArray[-1][-2]
        # self.axialWidget.drawAxial(pd1d, self.calcManager.axial_position)
        # self.radialWidget.slot_astra_data(pd2d)

    def cell_changed(self):
        import numpy
        model_index = self.RO_TableWidget.selectedIndexes()
        if len(model_index) > 0:
            row = model_index[-1].row()
            #print(row)
            if row < len(self.calcManager.reoperation_output):

                pd2d = self.calcManager.reoperation_output[row][-2]
                pd1d = self.calcManager.reoperation_output[row][-3]

                p = self.calcManager.reoperation_output[row][2]
                r5 = self.calcManager.reoperation_output[row][3]
                r4 = self.calcManager.reoperation_output[row][4]
                r3 = self.calcManager.reoperation_output[row][5]

                data = {' P': p, 'R5': r5, 'R4': r4, 'R3': r3}
                self.axialWidget.drawBar(data)
                #print(data)

                power = self.RO_TableWidget.InputArray[row][1]

                if power == 0:
                    self.axialWidget.clearAxial()
                    self.radialWidget.clear_data()
                else:
                    self.axialWidget.drawAxial(pd1d[self.calcManager.kbc:self.calcManager.kec],
                                               self.calcManager.axial_position)
                    self.radialWidget.slot_astra_data(pd2d)

    def get_calculation_input(self):

        model_input_index = 3
        calcOpt = df.CalcOpt_RO
        pArray = []
        if self.RO_TableWidget.checkModified():
            msgBox = QMessageBox(self.get_ui_component())
            msgBox.setWindowTitle("Rod Modification Detected")
            msgBox.setText("Rod: Rod Position Search with (Target ASI)\n"
                           "CBC: Boron Search with (Target Rod Position)")
            msgBox.addButton(QPushButton('Rod Search'), QMessageBox.YesRole)
            msgBox.addButton(QPushButton('CBC Search'), QMessageBox.NoRole)
            msgBox.addButton(QPushButton('Cancel'), QMessageBox.NoRole)
            # msgBox.setWindowFlag(QtCore.Qt.FramelessWindowHint)
            result = msgBox.exec_()
            if result == 1:
                calcOpt = df.CalcOpt_RO_RESTART
            elif result == 2:
                calcOpt = df.CalcOpt_KILL

        rod_values = self.RO_TableWidget.getRodValues()
        for iStep in range(len(self.inputArray)):

            if calcOpt == df.CalcOpt_RO:
                if iStep < model_input_index:
                    input_element = self.RO_TableWidget.InputArray[iStep] + [0, 0] + rod_values[iStep]
                else:
                    input_element = self.RO_TableWidget.InputArray[iStep]
            elif calcOpt == df.CalcOpt_RO_RESTART:
                input_element = self.RO_TableWidget.InputArray[iStep] + [0, 0] + rod_values[iStep]
            else:
                input_element = []
            pArray.append(input_element)
        if calcOpt == df.CalcOpt_KILL:
            error = True
        else:
            error = self.check_calculation_input()
        return calcOpt, self.targetASI, pArray, error

    def check_calculation_input(self):

        if self.ui.RO_Input02.value() >= self.calcManager.cycle_burnup_values[-1]:
            msgBox = QMessageBox(self.get_ui_component())
            msgBox.setWindowTitle("Burnup Out of Range")
            msgBox.setText("{}MWD/MTU excedes EOC Cycle Burnup({} MWD/MTU)\n"
                           "Cycle Burnup must be less than {}MWD/MTU".format(self.ui.RO_Input02.value(),
                                                            self.calcManager.cycle_burnup_values[-1],
                                                            self.calcManager.cycle_burnup_values[-1]))
            msgBox.setStandardButtons(QMessageBox.Ok)
            #msgBox.setWindowFlag(QtCore.Qt.FramelessWindowHint)
            result = msgBox.exec_()
            # if result == QMessageBox.Cancel:
            return True
        return False

    def updateDecayTime(self):
        totalTime = self.ui.RO_Input03.value()
        self.RO_decayTableWidget.setTotalTime(totalTime)

    def addDecayTableRow(self):
        self.RO_decayTableWidget.addDecayTableRow()

    def deleteDecayTableRow(self):
        self.RO_decayTableWidget.deleteDecayTableRow()

    def definePointData(self, outputArray):
        posP = []
        posR5 = []
        posR4 = []
        posR3 = []
        posCBC = []

        added_hour = 0

        for iStep in range(len(self.inputArray)):
            # dt = (1.0 - self.debugData01) / 0.03
            if outputArray[iStep][-1] != 1:
                added_hour = outputArray[iStep][-1]-1
            posP.append(QPointF(self.inputArray[iStep][0]+added_hour, outputArray[iStep][2]))
            posR5.append(QPointF(self.inputArray[iStep][0]+added_hour, outputArray[iStep][3]))
            posR4.append(QPointF(self.inputArray[iStep][0]+added_hour, outputArray[iStep][4]))
            posR3.append(QPointF(self.inputArray[iStep][0]+added_hour, outputArray[iStep][5]))
            posCBC.append(QPointF(self.inputArray[iStep][0]+added_hour, outputArray[iStep][0]))

        # rodPos = [posP, posR5, posR4, posR3, ]
        #
        # self.unitChartClass.replaceRodPosition(len(rodPos), rodPos, posCBC)

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

        # self.unitChart.clearData()
        #
        # posP = []
        # posR5 = []
        # posR4 = []
        # posR3 = []
        # posCBC = []
        #
        # added_hour = 0
        #
        # for i in range(len(outputArray)):
        #     if outputArray[i][-1] != 1:
        #         added_hour += outputArray[i][-1]-1
        #         self.unitChartClass.drawASIBand(self.unitChartClass.x_20+outputArray[i][-1]-1, self.unitChartClass.x_100+(outputArray[i][-1]-1), 0.27, 0.60)
        #         self.unitChartClass.axisX.setMax(self.unitChartClass.x_100+outputArray[i][-1]-1)
        #     posP.append(QPointF(start+i+added_hour, outputArray[i][2]))
        #     posR5.append(QPointF(start+i+added_hour, outputArray[i][3]))
        #     posR4.append(QPointF(start+i+added_hour, outputArray[i][4]))
        #     posR3.append(QPointF(start+i+added_hour, outputArray[i][5]))
        #     posCBC.append(QPointF(start+i+added_hour, outputArray[i][0]))
        #
        # rodPos = [posP, posR5, posR4, posR3, ]

        #self.unitChartClass.appendRodPosition(len(rodPos), rodPos, posCBC)

    def clearOuptut(self):
        self.RO_TableWidget.clearOutputArray()
        self.RO_TableWidget.clearOutputRodArray()