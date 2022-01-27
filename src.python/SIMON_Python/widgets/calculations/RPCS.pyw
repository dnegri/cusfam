from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication, QPointF
from PyQt5.QtGui import QFont

from PyQt5 import QtWidgets

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

from widgets.utils.PySaveMessageBox import PySaveMessageBox

_POINTER_A_ = "self.ui.RPCS_DB"
_NUM_PRINTOUT_ = 5
_MINIMUM_ITEM_COUNT = 40

# pointers = {
#
#     #"ASI_Snapshot": "ss_snapshot_file",
#     #"rdc01": "ndr_burnup",
#     #"rdc02": "ndr_power"#,
#     # "ASI_RodPos05": "ndr_bank_position_5",
#     # "ASI_RodPos05_Unit": "ndr_bank_type",
#     # "ASI_RodPos04": "ndr_bank_position_4",
#     # "ASI_RodPos03": "ndr_bank_position_3",
#     # "ASI_RodPos_P": "ndr_bank_position_P",
#
# }

pointers = {

    "RPCS_Snapshot": "ss_snapshot_file",
    "RPCS_Input01": "ndr_burnup",
    "RPCS_Input02": "ndr_target_keff",
    "RPCS_Input03": "ndr_target_keff",

}


class RPCSWidget(CalculationWidget):

    def __init__(self, db, ui):

        pass
        super().__init__(db, ui, ui.RPCS_DB, RPCS_Input, pointers)

        self.input_pointer = "self.current_calculation.asi_input"

        # TODO, SGH Remove input
        self.RodPosUnit = df.RodPosUnit_cm
        # 01. Setting Initial Shutdown Input Setting
        self.RPCS_CalcOpt = df.select_none
        self.InputArray = []
        # Input Dataset for RPCS_TableWidget Rod Position
        self.calc_rodPosBox = []
        self.calc_rodPos = []
        self.recalculationIndex = -1

        # 02. Insert Spline Chart
        self.unitChartClass = unitSplineChart.UnitSplineChart(self.RodPosUnit)
        lay = QtWidgets.QVBoxLayout(self.ui.RPCS_widgetChart)
        lay.setContentsMargins(0,0,0,0)
        unitChartView = self.unitChartClass.returnChart()
        lay.addWidget(unitChartView)

        # 03. Insert Table
        self.tableItem = ["Time\n(hour)","Power\n(%)","Burnup\n(MWD/MTU)","Keff","ASI","Boron\n(ppm)","Bank P\n(cm)","Bank 5\n(cm)","Bank 4\n(cm)","Bank 3\n(cm)"]
        self.RPCS_TableWidget = table01.ShutdownTableWidget(self.ui.frame_RPCS_TableWidget, self.tableItem)
        self.layoutTableButton = self.RPCS_TableWidget.returnButtonLayout()
        self.ui.gridlayout_RPCS_TableWidget.addWidget(self.RPCS_TableWidget, 0, 0, 1, 1)
        self.ui.gridlayout_RPCS_TableWidget.addLayout(self.layoutTableButton,1,0,1,1)
        [ self.unitButton01, self.unitButton02, self.unitButton03 ] = self.RPCS_TableWidget.returnTableButton()

        # 04. Insert Radial Chart
        layR = QtWidgets.QVBoxLayout(self.ui.RPCS_WidgetRadial)
        layR.setContentsMargins(0,0,0,0)
        self.RadialWidget = RadialWidget()
        layR.addWidget(self.RadialWidget)

        # 05. Insert Axial Chart
        layA = QtWidgets.QVBoxLayout(self.ui.RPCS_WidgetAxial)
        layA.setContentsMargins(0,0,0,0)
        self.AxialWidget = AxialWidget()
        layA.addWidget(self.AxialWidget)

        # 06. Link Signal and Slot for Start Calculation
        self.ui.RPCS_tabWidget.setCurrentIndex(0)
        self.ui.RPCS_tabWidget.setTabEnabled(1,False)
        self.ui.RPCS_tabWidget.setTabVisible(1,False)
        self.ui.RPCS_tabInput.hide()

        # 07. Setting Widget Interactions
        self.buttonGroupInput = QtWidgets.QButtonGroup()
        self.reductionGroupInput = QtWidgets.QButtonGroup()
        self.settingAutoExclusive()
        self.settingLinkAction()
        self.ui.RPCS_Main03.hide()

    def load(self):
        pass

        # self.load_input()
        # self.load_recent_calculation()

    def set_all_component_values(self, RPCS_input):

        if RPCS_input.calculation_type == 0:
            self.ui.RPCS_InpOpt2_NDR.setChecked(True)
        else:
            self.ui.RPCS_InpOpt1_Snapshot.setChecked(True)

        #Set values to components
        # for key in pointers.keys():
        #
        #     component = eval("self.ui.{}".format(key))
        #     value = eval("RPCS_input.{}".format(pointers[key]))
        #
        #     if pointers[key]:
        #         if isinstance(component, QComboBox):
        #             component.setCurrentText(value)
        #         else:
        #             component.setValue(float(value))

    def settingAutoExclusive(self):
        self.buttonGroupInput.addButton(self.ui.RPCS_InpOpt1_Snapshot)
        self.buttonGroupInput.addButton(self.ui.RPCS_InpOpt2_NDR)

        self.reductionGroupInput.addButton(self.ui.RPCS_RdcOpt01)
        self.reductionGroupInput.addButton(self.ui.RPCS_RdcOpt02)

    def settingLinkAction(self):
        pass
        self.ui.RPCS_InpOpt1_Snapshot.clicked['bool'].connect(self.settingInputOpt)
        self.ui.RPCS_InpOpt2_NDR.clicked['bool'].connect(self.settingInputOpt)

        self.ui.RPCS_RDC_apply.clicked['bool'].connect(self.setSuccecciveInput)
        self.unitButton01.clicked['bool'].connect(self.clickSaveAsExcel)
        self.unitButton02.clicked['bool'].connect(self.resetPositionData)
        # self.RPCS_tableWidget_button01.clicked['bool'].connect(self.clickSaveAsExcel)
        # self.RPCS_tableWidget_button02.clicked['bool'].connect(self.resetPositionData)
        # self.ui.RPCS_tableWidget.cellClicked.connect(self.cell_click)
        # self.ui.RPCS_tableWidget.cellChanged['int','int'].connect(self.cell_change)
        # self.ui.RPCS_RodPos05.valueChanged['double'].connect(self.rodPosChangedEvent)
        # self.ui.RPCS_RodPos04.valueChanged['double'].connect(self.rodPosChangedEvent)
        # self.ui.RPCS_RodPos03.valueChanged['double'].connect(self.rodPosChangedEvent)
        # self.ui.RPCS_RodPos_P.valueChanged['double'].connect(self.rodPosChangedEvent)

        # self.ui.RPCS_RodPos05_Unit.currentIndexChanged['int'].connect(self.changeRodPosUnit)

    def settingInputOpt(self):
        calculatin_type = 0
        if (self.ui.RPCS_InpOpt2_NDR.isChecked()):
            self.RPCS_CalcOpt = df.select_NDR
            self.ui.RPCS_Main03.show()
        elif (self.ui.RPCS_InpOpt1_Snapshot.isChecked()):
            self.RPCS_CalcOpt = df.select_snapshot
            self.ui.RPCS_Main03.hide()
            calculatin_type = 1

        if self.current_calculation:
            current_input = self.get_input(self.current_calculation)
            current_input.calculation_type = calculatin_type
            current_input.save()

    def get_input(self, calculation_object):
        return calculation_object.RPCS_input

    def getDefaultInput(self, user):
        now = datetime.datetime.now()

        RPCS_input = RPCS_Input.create(
            calculation_type=0,
            search_type=0,
            ss_snapshot_file="",
            ndr_burnup=0,
            ndr_target_keff=0,
            ndr_power=0,
        )

        RPCS_calculation = Calculations.create(user=user,
                                              calculation_type=cs.CALCULATION_SD,
                                              created_date=now,
                                              modified_date=now,
                                              RPCS_input=RPCS_input
                                              )
        return RPCS_calculation, RPCS_input

    def setSuccecciveInput(self):
        # Initialize
        self.InputArray = []

        initBU     = self.ui.RPCS_Input01.value()
        targetEigen = self.ui.RPCS_Input02.value()

        rdcPerHour = self.ui.RPCS_Input03.value()
        EOF_Power  = 0.0 #self.ui.rdc04.value()

        # TODO, make Except loop
        if(rdcPerHour==0.0):
            print("Error!")
            return

        nStep = math.ceil(( 100.0 - EOF_Power ) / rdcPerHour + 1.0)

        self.recalculationIndex = nStep

        for i in range(nStep-1):
            time = 1.0 * i
            power = 100.0 - i * rdcPerHour
            unitArray = [ time, power, initBU, targetEigen ]
            self.InputArray.append(unitArray)

        time = 100.0 / rdcPerHour
        power = 0.0
        unitArray = [ time, power, initBU, targetEigen]
        self.InputArray.append(unitArray)
        self.RPCS_TableWidget.addInputArray(self.InputArray)



    def clickSaveAsExcel(self):
        self.RPCS_TableWidget.clickSaveAsExcel()

    def resetPositionData(self):
        self.RPCS_TableWidget.resetPositionData()

