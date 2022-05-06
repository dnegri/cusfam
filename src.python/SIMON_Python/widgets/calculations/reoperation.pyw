from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication, QPointF
from PyQt5.QtGui import QFont
from PyQt5 import QtGui

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtCore import QSize, pyqtSlot, pyqtSignal

from model import *
import datetime

import Definitions as df

import constants as cs

from widgets.calculations.calculation_widget import CalculationWidget
from widgets.calculations.IO_table import table_IO_widget

import widgets.utils.PyUnitTableWidget as table00
import widgets.utils.PyShutdownTableWidget as table01
import widgets.utils.PyDecayTableWidget as table02
import widgets.utils.PyRodPositionSplineChart as unitSplineChart
from widgets.output.axial.axial_plot import AxialWidget
from widgets.output.radial.radial_graph import RadialWidget
import math
from widgets.output.trend.trends_graph import trendWidget
from widgets.output.trend.trends02_graph import trend02Widget
from widgets.utils.Map_Quarter import Ui_unitWidget_OPR1000_quarter as opr
from widgets.utils.Map_Quarter import Ui_unitWidget_APR1400_quarter as apr
from widgets.utils.PySaveMessageBox import PySaveMessageBox, QMessageBoxWithStyle

from PyQt5.QtWidgets import *

from widgets.utils.splash_screen import SplashScreen

_POINTER_A_ = "self.ui.RO_DB"
_NUM_PRINTOUT_ = 5
_MINIMUM_ITEM_COUNT = 20

pointers = {
    #"RO_Input01": "ndr_cal_type",
    "RO_InputSelect": "calculation_type",
    "pushButton_InputModel": "snapshot_text",
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

        super().__init__(db, ui, None, RO_Input, pointers, calManager, queue, message_ui)

        self.input_pointer = "self.current_calculation.asi_input"

        # 01. Setting Initial Shutdown Input Setting
        self.RO_CalcOpt = df.select_none
        self.inputArray = []
        self.snapshotArray = []
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
        self.loadInputData = None


        # 03. Insert Table

        # self.tableItem = ["Time\n(hour)","Power\n(%)","Burnup\n(MWD/MTU)","Keff","ASI","Boron\n(ppm)",
        #                    "Bank P\n(cm)", "Bank 5\n(cm)","Bank 4\n(cm)","Bank 3\n(cm)",]
        # self.tableItemFormat = ["%.1f","%.2f","%.1f","%.5f","%.3f",
        #                         "%.1f","%.1f","%.1f","%.1f","%.1f"]
        self.tableItem = ["Time\n(hour)", "Power\n(%)"  ,
                          "ASI",          "Boron\n(ppm)", "Fr", "Fxy", "Fq",
                          "Bank P\n(cm)" ,"Bank 5\n(cm)", "Bank 4\n(cm)", "Bank 3\n(cm)", ]
        self.tableItemFormat = ["%.1f","%.2f",
                                "%.3f","%.1f","%.3f","%.3f","%.3f",
                                "%.1f","%.1f","%.1f","%.1f"]
        self.RO_TableWidget = table01.ShutdownTableWidget(self.ui.frame_RO_OutputSet, self.tableItem,self.tableItemFormat)
        self.IO_table = table_IO_widget()
        layoutTableButton = self.RO_TableWidget.returnButtonLayout()
        self.ui.gridlayout_RO_TableWidget.addWidget(self.RO_TableWidget, 0, 0, 1, 1)
        self.ui.gridlayout_RO_TableWidget.addLayout(layoutTableButton,1,0,1,1)
        self.RO_TableWidget.hide()
        #[ self.unitButton01, self.unitButton02, self.unitButton03 ] = self.RO_TableWidget.returnTableButton()

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

        # 05. Setting Widget Interactions
        self.settingLinkAction()

        self.all_rod_in = True

        # 06. Setting Input Type
        self.ui.RO_InputSelect.setCurrentIndex(df._INPUT_TYPE_USER_)
        self.inputType = df._INPUT_TYPE_USER_
        self.ui.LabelSub_Selete01.setVisible(False)
        self.ui.pushButton_InputModel.setVisible(False)
        # #self.ui.LabelSub_Selete01.hide()

        self.delete_current_calculation()

        self.ui.RO_run_button.setText("Run")
        self.ui.RO_run_button.setStyleSheet(df.styleSheet_Run)
        self.ui.RO_run_button.setDisabled(False)

        self.is_run_selected = False

    def linkInputModule(self,module):
        self.loadInputData = module

    def load(self, a_calculation=None):

        self.load_input(a_calculation)

        self.load_output(a_calculation)
        self.load_snapshot(a_calculation)

        self.ui.RO_Input02.setMaximum(self.calcManager.cycle_burnup_values[-1] + 1000)

        # if len(self.inputArray) == 0:
        #     self.ui.RO_run_button.setText("Create Scenario")
        #     self.ui.RO_run_button.setStyleSheet(df.styleSheet_Create_Scenarios)
        # else:
        #     self.ui.RO_run_button.setText("Run")
        #     self.ui.RO_run_button.setStyleSheet(df.styleSheet_Run)

    def set_all_component_values(self, ecp_input):
        super().set_all_component_values(ecp_input)

        # self.ui.LabelSub_RO01_2.hide()
        # self.ui.RO_Input02.hide()
        # self.ui.LabelSub_RO02_2.hide()
        # self.ui.RO_Input08.hide()

    def index_changed(self, key):
        super().index_changed(key)
        # if self.current_calculation and self.last_table_created:
        #     if self.current_calculation.modified_date > self.last_table_created:
        #         self.ui.RO_run_button.setText("Create Scenario")
        #         self.ui.RO_run_button.setStyleSheet(df.styleSheet_Create_Scenarios)

    def value_changed(self, key):
        super().value_changed(key)
        # if self.current_calculation and self.last_table_created:
        #     if self.current_calculation.modified_date > self.last_table_created:
        #         self.ui.RO_run_button.setText("Create Scenario")
        #         self.ui.RO_run_button.setStyleSheet(df.styleSheet_Create_Scenarios)

    def settingLinkAction(self):
        self.ui.pushButton_InputModel.clicked['bool'].connect(self.readModel)
        self.ui.RO_InputSelect.currentIndexChanged['int'].connect(self.changeInputType)

        self.ui.RO_run_button.clicked['bool'].connect(self.start_calc)
        self.ui.RO_IO_button.clicked['bool'].connect(self.open_IO_table)
        self.ui.RO_save_button.clicked['bool'].connect(self.start_save)

        # self.unitButton01.clicked['bool'].connect(self.clickSaveAsExcel)
        # self.unitButton02.clicked['bool'].connect(self.resetPositionData)
        self.RO_TableWidget.itemSelectionChanged.connect(self.cell_changed)

        self.unitChart.canvas.mpl_connect('pick_event', self.clickEvent01)
        self.unitChart02.canvas.mpl_connect('pick_event', self.clickEvent02)
        # self.unitButton03.clicked['bool'].connect(self.clearOuptut)
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
                self.current_calculation.ro_input.snapshot_text = burnup_text
                self.ui.RO_Input02.setValue(inputArray[-1][1])
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
            #     self.ui.SD_run_button.setDisabled(False)f
            # self.SD_TableWidget.addInputArray(self.inputArray)
            # self.last_table_created = datetime.datetime.now()

    def resizeRadialWidget(self,size):
        self.map_opr1000_frame.setMaximumSize(QSize(size, size))
        self.map_apr1400_frame.setMaximumSize(QSize(size, size))

    def get_input(self, calculation_object):
        return calculation_object.ro_input

    def get_output(self, calculation_object):
        return calculation_object.ro_output

    def get_manager_output(self):
        return self.calcManager.results.reoperation_output

    def set_manager_output(self, output):
        self.calcManager.results.reoperation_output = output

    def create_default_input_output(self, user):
        now = datetime.datetime.now()

        ro_input = RO_Input.create()
        ro_output = RO_Output.create()
        RO_calculation = Calculations.create(user=user,
                                              calculation_type=cs.CALCULATION_RO,
                                              created_date=now,
                                              modified_date=now,
                                              ro_input=ro_input,
                                              ro_output=ro_output
                                              )

        return RO_calculation, ro_input, ro_output

    def setSuccecciveInput(self):

        self.clearOutput()

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

        self.rdcPerHour = rdcPerHour

        # update Power Variation Per Hour and power increase flag
        # ( Power Ascention Mode == True, Power Reduction Mode == False)
        self.unitChart.updateRdcPerHour(rdcPerHour,True)
        self.unitChart.resizeMaxTimeAxes(overlap_time+100/rdcPerHour)
        self.unitChart02.resizeMaxTimeAxes(overlap_time+100/rdcPerHour)

        #check input
        if initPower >= EOF_Power:
            msgBox = QMessageBoxWithStyle(self.get_ui_component())
            msgBox.setWindowTitle("Power Error")
            msgBox.setText("Initial Power {} is greater than Final Power {}".format(initPower, EOF_Power))
            msgBox.setStandardButtons(QMessageBox.Ok)
            msgBox.setCustomStyle()
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
            #print("Error!")
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
        currentPower = initPower
        while currentPower < EOF_Power:
            for power in powerPerTimes:
                currentPower += power
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
            initial_rod = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        else:
            initial_rod = [0, 0, 0, 0, 0, rodP, rod5, rod4, rod3, ]
        self.inputArray.append([0, self.initPower, initBU, targetEigen])
        self.inputArray.append([self.overlap_time, self.initPower, initBU, targetEigen])
        self.inputArray.append([self.overlap_time, self.initPower, initBU, targetEigen])

        for i in range(nStep//len(powerPerTimes)):
            start_power = 0
            for j, power in enumerate(powerPerTimes):
                start_power += power
                time = self.overlap_time + start_power/rdcPerHour + i
                unitArray = [ time, powers[i*len(powerPerTimes)+j], initBU, targetEigen ]
                self.inputArray.append(unitArray)

        self.RO_TableWidget.addInputArray(self.inputArray)
        self.IO_table.IO_TableWidget.addInputArray(self.inputArray)
        self.RO_TableWidget.makeOutputTable([initial_rod,initial_rod,[0, 0, 0, 0, 0, rodP, rod5, rod4, rod3,]])
        self.IO_table.IO_TableWidget.makeOutputTable([initial_rod,initial_rod,[0, 0, 0, 0, 0, rodP, rod5, rod4, rod3,]])

    def clickSaveAsExcel(self):
        is_succ = self.RO_TableWidget.clickSaveAsExcel()
        if not is_succ:
            msgBox = QMessageBoxWithStyle(self.get_ui_component())
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
            # self.unitChart = trendWidget(self.fixedTime_flag, self.ASI_flag, self.CBC_flag, self.Power_flag)
            self.unitChart = trendWidget(self.fixedTime_flag, self.ASI_flag, False, self.Power_flag)
            sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            sizePolicy5.setHorizontalStretch(10)
            sizePolicy5.setVerticalStretch(10)
            self.unitChart.setSizePolicy(sizePolicy5)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(self.unitChart, 0, 0, 1, 1)

        if not self.unitChart02:

            lay02 = self.ui.grid_RO_frameChart02
            self.unitChart02 = trend02Widget(self.fixedTime_flag, False, True, False)#self.ui.SD_widgetChart)
            sizePolicy6 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            sizePolicy6.setHorizontalStretch(10)
            sizePolicy6.setVerticalStretch(10)
            self.unitChart02.setSizePolicy(sizePolicy6)
            lay02.setContentsMargins(0, 0, 0, 0)
            lay02.addWidget(self.unitChart02, 0, 0, 1, 1)


        if not self.radialWidget:
            self.map_opr1000 = opr(self.ui.RO_InputLP_Dframe,self.ui.gridLayout_RO_InputLP_Dframe)
            self.map_opr1000_frame , self.map_opr1000_grid = self.map_opr1000.return_opr_frame()
            self.ui.gridLayout_RO_InputLP_Dframe.addWidget(self.map_opr1000_frame , 0, 0, 1, 1)
            self.radialWidget = RadialWidget(self.map_opr1000_frame , self.map_opr1000_grid)
            #self.map_opr1000_frame.hide()



            self.map_apr1400 = apr(self.ui.RO_InputLP_Dframe,self.ui.gridLayout_RO_InputLP_Dframe)
            self.map_apr1400_frame, self.map_apr1400_grid = self.map_apr1400.return_apr_frame()
            self.ui.gridLayout_RO_InputLP_Dframe.addWidget(self.map_apr1400_frame, 0, 0, 1, 1)
            self.radialWidget02 = RadialWidget(self.map_apr1400_frame,self.map_apr1400_grid)
            self.map_apr1400_frame.hide()

        if not self.axialWidget:
            # 05. Insert Axial Chart
            sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
            sizePolicy5.setHorizontalStretch(6)
            sizePolicy5.setVerticalStretch(10)
            #sizePolicy5.setHeightForWidth(self.ui.RO_WidgetAxial.sizePolicy().hasHeightForWidth())
            self.ui.RO_WidgetAxial.setSizePolicy(sizePolicy5)
            layA = QtWidgets.QVBoxLayout(self.ui.RO_WidgetAxial)
            layA.setContentsMargins(0,0,0,0)
            self.axialWidget = AxialWidget()
            layA.addWidget(self.axialWidget)

    def start_calc(self):

        super().start_calc()

        if self.unitChart:
            self.unitChart.clearData()

        if self.unitChart02:
            self.unitChart02.clearData()

        if self.RO_TableWidget:
            self.RO_TableWidget.clearOutputArray()
            self.IO_table.IO_TableWidget.clearOutputArray()

        # if self.ui.RO_run_button.text() == cs.RUN_BUTTON_CREATE_SCENARIO:
        #
        #     self.current_calculation.ro_output.success = False
        #
        #     self.setSuccecciveInput()
        #     self.load()
        #
        # else:

        self.current_calculation.ro_output.success = False

        self.setSuccecciveInput()
        self.load()

        calcOpt, targetASI, pArray, snap_length, error = self.get_calculation_input()

        #self.unitChartClass.axisX.setMax(len(pArray))
        if not error and calcOpt != df.CalcOpt_KILL:

            # Only consider restart
            x_20 = 0
            for p_index, values in enumerate(pArray):
                if values[1] >= 20:
                    x_20 = p_index
                    break

            if calcOpt == df.CalcOpt_RO:

                if self.unitChart:
                    self.unitChart.clearData()
                self.setSuccecciveInput()
                self.RO_TableWidget.outputArray = []
                self.IO_table.IO_TableWidget.outputArray = []
                self.calcManager.results.reoperation_output = []

            self.RO_TableWidget.last_update = 0
            self.IO_table.IO_TableWidget.last_update = 0
            self.start_calculation_message = SplashScreen()
            self.start_calculation_message.killed.connect(self.killManagerProcess)
            self.start_calculation_message.init_progress(len(pArray), 500)

            # self.ui.RO_run_button.setText(cs.RUN_BUTTON_RUNNING)
            # self.ui.RO_run_button.setDisabled(True)
            self.RO_TableWidget.clearOutputArray()
            self.IO_table.IO_TableWidget.clearOutputArray()
            self.queue.put((calcOpt, targetASI, pArray, snap_length))

    def finished(self, is_success):

        self.is_run_selected = False

        if self.start_calculation_message:
            self.start_calculation_message.close()

        #self.definePointData(self.calcManager.reoperation_output)
        self.RO_TableWidget.selectRow(3)
        self.IO_table.IO_TableWidget.selectRow(3)
        # self.ui.RO_run_button.setText(cs.RUN_BUTTON_RUN)
        # self.ui.RO_run_button.setDisabled(False)

        if is_success == self.calcManager.SUCC:
            self.save_output(self.calcManager.results.reoperation_output)
            self.showOutput()
        else:

            ro_output = RO_Output.create(
                success=False,
                table="",
            )

            self.current_calculation.ro_output = ro_output
            self.current_calculation.ro_output.save()
            self.current_calculation.save()

        # self.cell_changed()


    def showOutput(self):
        if self.start_calculation_message:
            self.start_calculation_message.progress()

        #self.appendPointData(self.calcManager.outputArray[last_update:], last_update)

        if len(self.calcManager.results.reoperation_output) > 0:

            # last_update = self.RO_TableWidget.last_update
            # last_update = self.IO_table.IO_TableWidget.last_update

            last_update = 0

            self.RO_TableWidget.appendOutputTable(self.calcManager.results.reoperation_output[last_update:],
                                                  last_update)
            self.IO_table.IO_TableWidget.appendOutputTable(self.calcManager.results.reoperation_output[last_update:],
                                                           last_update)

            self.unitChart.adjustTime(self.overlap_time)
            self.unitChart02.adjustTime(self.overlap_time)

            self.appendPointData(self.calcManager.results.reoperation_output, 0)
            # outputs = self.calcManager.results.reoperation_output
            # pd2d = outputs[-1][df.asi_o_p2d]
            # pd1d = outputs[-1][df.asi_o_p1d]
            # self.axialWidget.drawAxial(pd1d[self.calcManager.results.kbc:self.calcManager.results.kec],
            #                            self.calcManager.results.axial_position)
            # self.radialWidget.slot_astra_data(pd2d)

    def cell_changed(self):
        import numpy
        model_index = self.RO_TableWidget.selectedIndexes()
        model_index = self.IO_table.IO_TableWidget.selectedIndexes()
        if len(model_index) > 0:
            row = model_index[-1].row()
            #print(row)
            if row < len(self.calcManager.results.reoperation_output):

                pd2d = self.calcManager.results.reoperation_output[row][df.asi_o_p2d]
                pd1d = self.calcManager.results.reoperation_output[row][df.asi_o_p1d]

                p = self.calcManager.results.reoperation_output[row][df.asi_o_bp]
                r5 = self.calcManager.results.reoperation_output[row][df.asi_o_b5]
                r4 = self.calcManager.results.reoperation_output[row][df.asi_o_b4]
                r3 = self.calcManager.results.reoperation_output[row][df.asi_o_b3]

                data = {' P': p, 'R5': r5, 'R4': r4, 'R3': r3}
                self.axialWidget.drawBar(data)
                #print(data)

                power = self.RO_TableWidget.InputArray[row][1]
                power = self.IO_table.IO_TableWidget.InputArray[row][1]

                if power == 0:
                    self.axialWidget.clearAxial()
                    self.radialWidget.clear_data()
                else:
                    self.axialWidget.drawAxial(pd1d[self.calcManager.results.kbc:self.calcManager.results.kec],
                                               self.calcManager.results.axial_position)
                    self.radialWidget.slot_astra_data(pd2d)

    def get_calculation_input(self):

        model_input_index = 3
        calcOpt = df.CalcOpt_RO
        pArray = []
        if self.RO_TableWidget.checkModified():
            msgBox = QMessageBoxWithStyle(self.get_ui_component())
            msgBox.setWindowTitle("Rod Modification Detected")
            msgBox.setText("Rod: Rod Position Search with (Target ASI)\n"
                           "CBC: Boron Search with (Target Rod Position)")
            msgBox.addButton(QPushButton('Rod Search'), QMessageBox.YesRole)
            msgBox.addButton(QPushButton('CBC Search'), QMessageBox.NoRole)
            msgBox.addButton(QPushButton('Cancel'), QMessageBox.NoRole)
            msgBox.setCustomStyle()
            # msgBox.setWindowFlag(QtCore.Qt.FramelessWindowHint)
            result = msgBox.exec_()
            if result == 1:
                calcOpt = df.CalcOpt_RO_RESTART
            elif result == 2:
                calcOpt = df.CalcOpt_KILL

        snap_length = len(self.snapshotArray)

        if self.ui.RO_InputSelect.currentIndex() == 1:
            for value in self.snapshotArray:
                pArray.append([value[0], value[2], value[1], value[3]]+[0,0,0]+value[4:])

        rod_values = self.RO_TableWidget.getRodValues()
        rod_values = self.IO_table.IO_TableWidget.getRodValues()
        for iStep in range(len(self.inputArray)):

            if calcOpt == df.CalcOpt_RO:
                if iStep < model_input_index:
                    input_element = self.RO_TableWidget.InputArray[iStep] + [0, 0, 0, 0, 0] + rod_values[iStep]
                    input_element = self.IO_table.IO_TableWidget.InputArray[iStep] + [0, 0, 0, 0, 0] + rod_values[iStep]
                else:
                    input_element = self.RO_TableWidget.InputArray[iStep]
                    input_element = self.IO_table.IO_TableWidget.InputArray[iStep]
            elif calcOpt == df.CalcOpt_RO_RESTART:
                input_element = self.RO_TableWidget.InputArray[iStep] + [0, 0, 0, 0, 0] + rod_values[iStep]
                input_element = self.IO_table.IO_TableWidget.InputArray[iStep] + [0, 0, 0, 0, 0] + rod_values[iStep]
            else:
                input_element = []
            pArray.append(input_element)
        if calcOpt == df.CalcOpt_KILL:
            error = True
        else:
            error = self.check_calculation_input()
        return calcOpt, self.targetASI, pArray, snap_length, error

    def check_calculation_input(self):

        if self.ui.RO_Input02.value() >= self.calcManager.cycle_burnup_values[-1]+1000:
            msgBox = QMessageBoxWithStyle(self.get_ui_component())
            msgBox.setWindowTitle("Burnup Out of Range")
            msgBox.setText("{}MWD/MTU excedes EOC Cycle Burnup({} MWD/MTU)\n"
                           "Cycle Burnup must be less than {}MWD/MTU".format(self.ui.RO_Input02.value(),
                                                            self.calcManager.cycle_burnup_values[-1],
                                                            self.calcManager.cycle_burnup_values[-1]+1000))
            msgBox.setStandardButtons(QMessageBox.Ok)
            msgBox.setCustomStyle()
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
            posP.append(QPointF(self.inputArray[iStep][0]+added_hour, outputArray[iStep][df.asi_o_bp]))
            posR5.append(QPointF(self.inputArray[iStep][0]+added_hour, outputArray[iStep][df.asi_o_b5]))
            posR4.append(QPointF(self.inputArray[iStep][0]+added_hour, outputArray[iStep][df.asi_o_b4]))
            posR3.append(QPointF(self.inputArray[iStep][0]+added_hour, outputArray[iStep][df.asi_o_b3]))
            posCBC.append(QPointF(self.inputArray[iStep][0]+added_hour, outputArray[iStep][df.asi_o_boron]))

        # rodPos = [posP, posR5, posR4, posR3, ]
        #
        # self.unitChartClass.replaceRodPosition(len(rodPos), rodPos, posCBC)

    def appendPointData(self, outputArray, start):

        # self.unitChartClass.clear()
        num = len(outputArray)
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
            pos_Fr.append(-1.0)
            pos_Fq.append(-1.0)
            current_time += outputArray[i][df.asi_o_time]
            time.append(current_time)
            power.append(outputArray[i][df.asi_o_power]*100)
            if power[-1] < 20.0:
                asi_band_time += outputArray[i][df.asi_o_time]
            # posP.append(QPointF(start+i, outputArray[i][2]))
            # posR5.append(QPointF(start+i, outputArray[i][3]))
            # posR4.append(QPointF(start+i, outputArray[i][4]))
            # posR3.append(QPointF(start+i, outputArray[i][5]))
            # posCBC.append(QPointF(start+i, outputArray[i][0]))

        self.unitChart.insertTime(time)
        self.unitChart02.insertTime(time)

        #rodPos = [posP, posR5, posR4, posR3, ]
        posOpt = [ posASI, posCBC, power ]
        posOpt02 = [ posASI, posCBC, power  ]
        # print("hello", len(outputArray), time, posASI)
        self.unitChart.updateRdcPerHour(self.rdcPerHour,True, asi_band_time)
        self.unitChart.insertDataSet(time, posP, posR5, posR4, posOpt )
        self.unitChart02.insertDataSet(time, pos_Fxy, pos_Fr, pos_Fq, posOpt02 )
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

    def clearOutput(self):
        self.RO_TableWidget.clearOutputArray()
        self.RO_TableWidget.clearOutputRodArray()
        self.RO_TableWidget.last_update = 0
        self.unitChart.clearData()
        self.unitChart02.clearData()