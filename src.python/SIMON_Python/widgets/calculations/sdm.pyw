
from PyQt5 import QtCore
from PyQt5.QtCore import (QCoreApplication, QSize, pyqtSlot)
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import *
import datetime
import Definitions as df
import constants as cs

from model import *
from widgets.calculations.calculation_widget import CalculationWidget

import widgets.utils.PyUnitButton as ts

#import ui_unitWidget_SDM_report1 as reportWidget1
#import ui_unitWidget_SDM_report2 as reportWidget2

import widgets.utils.PyShutdownTableWidget as table01

from widgets.utils.splash_screen import SplashScreen

from widgets.utils.PySaveMessageBox import PySaveMessageBox, QMessageBoxWithStyle

pointers = {
    "SDM_InputSelect": "calculation_type",
    "pushButton_InputModel": "snapshot_text",
    "SDM_Input01": "ndr_burnup",
    "SDM_Input02": "ndr_mode_selection",
    "SDM_Input03": "ndr_power",
}


class SDMWidget(CalculationWidget):

    def __init__(self, db, ui, calManager, queue, message_ui):

        super().__init__(db, ui, None, SDM_Input, pointers, calManager, queue, message_ui)

        # 02. Rod Input Settings
        self.funcType = eval("ts.outputButton")

        _BLANK_ASM_ = " "
        self.linkRCS = [[_BLANK_ASM_ for col in range(df._OPR1000_XPOS_)] for row in range(df._OPR1000_YPOS_)]
        self.bClassRCS = [[_BLANK_ASM_ for col in range(df._OPR1000_XPOS_)] for row in range(df._OPR1000_YPOS_)]
        self.controlRodMap = [[False for col in range(df._OPR1000_XPOS_)] for row in range(df._OPR1000_YPOS_)]
        self.globalButtonInfo = {}
        self.selectedStuckRods = []
        self.stuckRodPos = []
        self.settingLP(self.ui.SDM_InputLP_frame, self.ui.SDM_InputLPgrid)

        self.snapshotArray = []

        # 01 Set Components
        self.set_all_components()
        self.settingLinkAction()

        self.RodPosUnit = df.RodPosUnit_cm
        # 04. Load
        # self.load()

        # 08. Setting Input Type
        self.loadInputData = None
        self.ui.SDM_InputSelect.setCurrentIndex(df._INPUT_TYPE_USER_)
        self.inputType = df._INPUT_TYPE_USER_
        self.ui.LabelSub_Selete01.setVisible(False)
        self.ui.pushButton_InputModel.setVisible(False)

        self.delete_current_calculation()

    def linkInputModule(self,module):
        self.loadInputData = module

    ########################
    # Calculation Widget
    ########################
    def load(self, a_calculation=None):
        self.load_input(a_calculation)
        self.load_output(a_calculation)
        self.load_snapshot(a_calculation)

        self.ui.sdm1_plant.setText(self.calcManager.plant_name)
        self.ui.sdm1_cycle.setText(self.calcManager.cycle_name)

        self.ui.sdm2_plant.setText(self.calcManager.plant_name)
        self.ui.sdm2_cycle.setText(self.calcManager.cycle_name)

        self.ui.SDM_run_button.setText("Run")
        self.ui.SDM_run_button.setStyleSheet(df.styleSheet_Run)

        self.ui.SDM_Input01.setMaximum(self.calcManager.cycle_burnup_values[-1] + 1000)


    def load_input(self, a_calculation=None):
        sdm_input = super().load_input()

        # self.load_input(a_calculation)
        # self.load_output(a_calculation)

        if (sdm_input.ndr_stuckrod1_x > -1 and sdm_input.ndr_stuckrod1_y > -1) and \
                not self.controlRodMap[sdm_input.ndr_stuckrod1_x][sdm_input.ndr_stuckrod1_y]:
            # print(sdm_input.ndr_stuckrod1_x, sdm_input.ndr_stuckrod1_y)
            self.showAssemblyLoading(self.getRodName(sdm_input.ndr_stuckrod1_x, sdm_input.ndr_stuckrod1_y))

        if sdm_input.ndr_stuckrod2_x > -1 and sdm_input.ndr_stuckrod2_y > -1 and \
                not self.controlRodMap[sdm_input.ndr_stuckrod2_x][sdm_input.ndr_stuckrod2_y]:
            # print(sdm_input.ndr_stuckrod2_x, sdm_input.ndr_stuckrod2_y)
            self.showAssemblyLoading(self.getRodName(sdm_input.ndr_stuckrod2_x, sdm_input.ndr_stuckrod2_y))

    def get_input(self, calculation_object):
        return calculation_object.sdm_input

    def get_output(self, calculation_object):
        return calculation_object.sdm_output

    def create_default_input_output(self, user):
        now = datetime.datetime.now()

        sdm_input = SDM_Input.create()
        sdm_output = SDM_Output.create()

        sdm_calculation = Calculations.create(user=user,
                                              calculation_type=cs.CALCULATION_SDM,
                                              created_date=now,
                                              modified_date=now,
                                              sdm_input=sdm_input,
                                              sdm_output=sdm_output,
                                              )
        return sdm_calculation, sdm_input, sdm_output

    def get_output_pointers(self):
        return ["m1_success",
            "m1_core_burnup",
            "m1_temperature",
            "m1_cea_configuration",
            "m1_stuck_rod",
            "m1_n1_worth",
            "m1_defect_worth",
            "m1_required_worth",
            "m1_sdm_worth",

            "m2_success",
            "m2_core_burnup",
            "m2_temperature",
            "m2_cea_configuration",
            "m2_stuck_rod",
            "m2_required_worth",
            "m2_required_cbc"]

    def load_output(self, a_calculation=None):
        if not a_calculation:
            a_calculation = self.current_calculation

        if not a_calculation:
            raise("Error")

        sdm_output = a_calculation.sdm_output

        if sdm_output.m2_success:

            self.ui.sdm2_rods.setText(sdm_output.m2_stuck_rod)
            self.ui.sdm2_cea.setText(sdm_output.m2_cea_configuration)
            self.ui.sdm2_burnup.setText("{:.1f}".format(sdm_output.m2_core_burnup))
            self.ui.sdm2_temp.setText("{:.1f}".format(sdm_output.m2_temperature))
            self.ui.sdm2_rsdm.setText("{:.1f}".format(sdm_output.m2_required_worth))
            self.ui.sdm2_rb.setText("{:.1f}".format(sdm_output.m2_required_cbc))

        if sdm_output.m1_success:
            self.ui.sdm1_rods.setText(sdm_output.m1_stuck_rod)
            self.ui.sdm1_burnup.setText("{:.1f}".format(sdm_output.m1_core_burnup))
            self.ui.sdm1_cea.setText("N-1")
            self.ui.sdm1_margin.setText("{:.2f}".format(sdm_output.m1_sdm_worth))
            self.ui.sdm1_defect.setText("{:.2f}".format(sdm_output.m1_defect_worth))
            self.ui.sdm1_n1w.setText("{:.2f}".format(sdm_output.m1_n1_worth))
            self.ui.sdm1_rsdm.setText("{:.2f}".format(sdm_output.m1_required_worth))
            self.ui.sdm1_temp.setText("{:.1f}".format(sdm_output.m1_temperature))

    def settingLinkAction(self):
        self.ui.pushButton_InputModel.clicked['bool'].connect(self.readModel)
        self.ui.SDM_InputSelect.currentIndexChanged['int'].connect(self.changeInputType)

        self.ui.SDM_save_button.clicked['bool'].connect(self.start_save)
        self.ui.SDM_run_button.clicked['bool'].connect(self.start_calc)
        # self.ui.SDM_create_report_button.clicked['bool'].connect(self.printReport)

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
            self.loadInputData.readSnapshotData()
            [self.nSnapshot, inputArray] = self.loadInputData.returnSnapshot()
            self.snapshotArray = inputArray

            if len(inputArray) > 0:
                burnup_text = "B:{:d}".format(int(inputArray[-1][1]))
                self.ui.pushButton_InputModel.setText(burnup_text)
                self.ui.pushButton_InputModel.setStyleSheet(df.styleSheet_Run)
                self.current_calculation.sdm_input.snapshot_text = burnup_text
                self.ui.SDM_Input01.setValue(inputArray[-1][1])
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


    def mode_changed(self, ix):
        search_type = 0
        if ix == 0:
            self.ui.SDM_stackedWidget_reports.setCurrentWidget(self.ui.SDM_report1)
            self.ui.LabelSub_SDM03.show()
            self.ui.SDM_Input03.show()
        else:
            self.ui.SDM_stackedWidget_reports.setCurrentWidget(self.ui.SDM_report2)
            self.ui.LabelSub_SDM03.hide()
            self.ui.SDM_Input03.hide()
            search_type = 1

        if self.current_calculation:
            current_input = self.get_input(self.current_calculation)
            current_input.search_type = search_type
            current_input.save()

        self.load_input()

    def set_all_components(self):
        self.ui.SDM_Input02.currentIndexChanged['int'].connect(
            lambda: self.mode_changed(self.ui.SDM_Input02.currentIndex()))
        self.mode_changed(self.ui.SDM_Input02.currentIndex())

    def calculation_options(self, b):
        calculation_type = 0

        if b == self.ui.SDM_InpOpt01_NDR:
            self.ui.SDM_Main02.show()
            self.ui.SDM_Main03.hide()
            self.mode_changed(self.ui.SDM_Input02.currentIndex())
        if b == self.ui.SDM_InpOpt02_Snapshot:
            self.ui.SDM_Main02.hide()
            self.ui.SDM_Main03.show()
            calculation_type = 1

        if self.current_calculation:
            current_input = self.get_input(self.current_calculation)
            current_input.calculation_type = calculation_type
            current_input.save()

    ########################
    # Rod Position
    ########################
    def settingLP(self, frame, gridLayout):
        _translate = QCoreApplication.translate
        xID = df.OPR1000_xPos
        yID = df.OPR1000_yPos

        x_start_filled = []
        for i in range(15):
            x_start_filled.append(i)


        for yPos in range(len(yID)):
            y_start = ["Pop"]
            for xPos in range(len(xID)):
                if (df.OPR1000MAP[xPos][yPos] == True):
                    bName = "Core_%s%s" % (xID[xPos], yID[yPos])
                    self.globalButtonInfo[bName] = [df._POSITION_CORE_, 0, xPos, yPos]
                    # generate button geometry
                    buttonCore = self.funcType(bName, self.showAssemblyLoading, self.swapAssembly,
                                               frame)  # type: QPushButton
                    buttonCore.setEnabled(True)
                    sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
                    sizePolicy.setHorizontalStretch(0)
                    sizePolicy.setVerticalStretch(0)
                    sizePolicy.setHeightForWidth(buttonCore.sizePolicy().hasHeightForWidth())
                    buttonCore.setSizePolicy(sizePolicy)
                    buttonCore.setMinimumSize(QSize(15, 15))
                    buttonCore.setMaximumSize(QSize(16777215, 16777215))
                    buttonCore.setBaseSize(QSize(60, 60))
                    buttonCore.setText(df.OPR1000MAP_BP[xPos][yPos])
                    buttonCore.setIconSize(QSize(16, 16))
                    buttonCore.setFlat(False)
                    buttonCore.setObjectName(bName)
                    buttonCore.setStyle(QStyleFactory.create("Windows"))
                    if xPos == 14 and yPos == 14:
                        gridLayout.addWidget(buttonCore, yPos + 1, xPos + 1, 1, 1)
                    else:
                        gridLayout.addWidget(buttonCore, yPos + 1, xPos + 1, 1, 1)

                    self.bClassRCS[xPos][yPos] = buttonCore

                    style_sheet = "background-color: rgb({},{},{});" \
                                  "border-radius: 0px;" \
                                  "border-width: 0.5px;" \
                                  "border-style: solid;" \
                                  "color: white;". \
                        format(df.rgbSet[-3][0],
                               df.rgbSet[-3][1],
                               df.rgbSet[-3][2])
                    try:
                        popped = x_start_filled.pop(x_start_filled.index(xPos))
                    except ValueError:
                        popped = None
                    if len(y_start) == 1 and popped:
                        style_sheet += "border-color: rgb(150,150,150);"
                        y_start.pop()
                    elif popped:
                        style_sheet += "border-color: rgb(150,150,150) rgb(150,150,150) rgb(150,150,150) rgb({},{},{});".format(
                            df.rgbSet[-3][0], df.rgbSet[-3][1], df.rgbSet[-3][2])
                    elif len(y_start) == 1:
                        y_start.pop()
                        style_sheet += "border-color: rgb({},{},{}) rgb(150,150,150) rgb(150,150,150) rgb(150,150,150);".format(
                            df.rgbSet[-3][0], df.rgbSet[-3][1], df.rgbSet[-3][2])
                    else:
                        if xPos == 0 and yPos == 0:
                            style_sheet += "border-color: rgb(150,150,150);"
                        elif xPos == 0:
                            style_sheet += "border-color: rgb({},{},{}) rgb(150,150,150) rgb(150,150,150) rgb(150,150,150);".format(
                                df.rgbSet[-3][0], df.rgbSet[-3][1], df.rgbSet[-3][2])
                        elif yPos == 0:
                            style_sheet += "border-color: rgb(150,150,150) rgb(150,150,150) rgb(150,150,150) rgb({},{},{});".format(
                                df.rgbSet[-3][0], df.rgbSet[-3][1], df.rgbSet[-3][2])
                        else:
                            style_sheet += "border-color: rgb({},{},{}) rgb(150,150,150) rgb(150,150,150) rgb({},{},{});".format(
                                df.rgbSet[-3][0], df.rgbSet[-3][1], df.rgbSet[-3][2], df.rgbSet[-3][0], df.rgbSet[-3][1],
                                df.rgbSet[-3][1])

                    buttonCore.setStyleSheet(style_sheet)

                    if xPos == 14 and yPos == 14:
                        buttonCore.setStyleSheet(
                            "background-color: rgb({},{},{});border-radius = 0px;color: white;".format(df.rgbSet[-1][0],
                                                                                                       df.rgbSet[-1][1],
                                                                                                       df.rgbSet[-1][
                                                                                                           2]))
                        font = QFont()
                        font.setPointSize(10)
                        self.bClassRCS[xPos][yPos].setFont(font)

                    buttonCore.setDisabled(df.OPR1000MAP_INFO[xPos][yPos])
                    if df.OPR1000MAP_INFO[xPos][yPos]:
                        buttonCore.setDisabled(df.OPR1000MAP_INFO[xPos][yPos])

                    if len(df.OPR1000MAP_BP[xPos][yPos]) == 0:
                        buttonCore.setDisabled(True)


    def getAssemblyPosition(self, asmName):
        [pos, iBlock, xPos, yPos] = self.globalButtonInfo[asmName]
        xID = df.OPR1000_xPos[xPos]
        yID = df.OPR1000_yPos[yPos]
        if (pos == df._POSITION_CORE_):
            fullID = self.linkRCS[xPos][yPos]
            posName = "%s-%s" % (xID, yID)
        elif (pos == df._POSITION_SFP01_):
            fullID = self.linkSFP01[iBlock][xPos][yPos]
            posName = "SFP01_%s-%s" % (xID, yID)
        else:
            fullID = self.linkSFP02[iBlock][xPos][yPos]
            posName = "SFP02_%s-%s" % (xID, yID)

        cycleNum = 0
        startHist = []
        dischargeHist = []
        if (fullID == df._BLANK_ASM_):
            id = "Blank"
            type = " "
            stepNum = " "
            eocB = " "
        else:
            if (fullID[2].isalpha()):
                id = fullID[2:]
            else:
                id = fullID[3:]
            type = self.assemblies_dict[fullID].getAsmType()
            coreStep = self.assemblies_dict[fullID].getCoreStep()
            stepNum = str(len(coreStep))
            bp = self.assemblies_dict[fullID].getAsmBP()
            eocB = "%.1f MWD/MTU" % bp[-1][-1]

            cycleStep = self.assemblies_dict[fullID].getCoreStep()
            cycleNum = len(cycleStep)
            startHist = self.assemblies_dict[fullID].getStartHist()
            dischargeHist = self.assemblies_dict[fullID].getDischargeHistory()
        return id, type, posName, stepNum, eocB, cycleNum, startHist, dischargeHist

    # @pyqtSlot(str)
    def showAssemblyLoading(self, loc01):
        [pos, iBlock, xPos, yPos] = self.globalButtonInfo[loc01]
        [id, type, posName, stepNum, eocB, cycleNum, startHist, dischargeHist] = self.getAssemblyPosition(loc01)
        if (self.controlRodMap[xPos][yPos] == False):
            # self.bClassRCS[xPos][yPos].
            self.controlRodMap[xPos][yPos] = True

            colorR = int(df.rgbSet[11][0], 16)
            colorG = int(df.rgbSet[11][1], 16)
            colorB = int(df.rgbSet[11][2], 16)
            self.bClassRCS[xPos][yPos].setStyleSheet(
                "background-color: rgb({},{},{});border-radius = 0px;color: white;".format(df.rgbSet[-1][0],
                                                                                           df.rgbSet[-1][1],
                                                                                           df.rgbSet[-1][2]))

            self.selectedStuckRods.append((xPos, yPos))
            if len(self.selectedStuckRods) > 2:
                self.showAssemblyLoading(self.getRodName(self.selectedStuckRods[0][0], self.selectedStuckRods[0][1]))

        else:
            self.controlRodMap[xPos][yPos] = False

            style_sheet = "background-color: rgb({},{},{});" \
                          "border-radius: 0px;" \
                          "border-width: 0.5px;" \
                          "border-style: solid;" \
                          "color: white;". \
                format(df.rgbSet[-3][0],
                       df.rgbSet[-3][1],
                       df.rgbSet[-3][2])

            if xPos == 0 and yPos == 0:
                style_sheet += "border-color: rgb(150,150,150);"
            elif xPos == 0:
                style_sheet += "border-color: rgb({},{},{}) rgb(150,150,150) rgb(150,150,150) rgb(150,150,150);".format(
                    df.rgbSet[-3][0], df.rgbSet[-3][1], df.rgbSet[-3][2])
            elif yPos == 0:
                style_sheet += "border-color: rgb(150,150,150) rgb(150,150,150) rgb(150,150,150) rgb({},{},{});".format(
                    df.rgbSet[-3][0], df.rgbSet[-3][1], df.rgbSet[-3][2])
            else:
                style_sheet += "border-color: rgb({},{},{}) rgb(150,150,150) rgb(150,150,150) rgb({},{},{});".format(
                    df.rgbSet[-3][0], df.rgbSet[-3][1], df.rgbSet[-3][2], df.rgbSet[-3][0], df.rgbSet[-3][1],
                    df.rgbSet[-3][1])

            self.bClassRCS[xPos][yPos].setStyleSheet(style_sheet)

            if (xPos, yPos) in self.selectedStuckRods:
                self.selectedStuckRods.remove((xPos, yPos))
            # self.ui.LP_Info01.setText(id)

        self.current_calculation.sdm_input.ndr_stuckrod1_x = -1
        self.current_calculation.sdm_input.ndr_stuckrod1_y = -1
        self.current_calculation.sdm_input.ndr_stuckrod2_x = -1
        self.current_calculation.sdm_input.ndr_stuckrod2_y = -1

        for index_pos in range(2):
            if index_pos < len(self.selectedStuckRods):
                xPos = self.selectedStuckRods[index_pos][0]
                yPos = self.selectedStuckRods[index_pos][1]
                if index_pos == 0:
                    #print(xPos, yPos)
                    self.current_calculation.sdm_input.ndr_stuckrod1_x = xPos
                    self.current_calculation.sdm_input.ndr_stuckrod1_y = yPos

            if index_pos < len(self.selectedStuckRods):
                xPos = self.selectedStuckRods[index_pos][0]
                yPos = self.selectedStuckRods[index_pos][1]
                if index_pos == 1:
                    #print(xPos, yPos)
                    self.current_calculation.sdm_input.ndr_stuckrod2_x = xPos
                    self.current_calculation.sdm_input.ndr_stuckrod2_y = yPos

        self.get_input(self.current_calculation).save()
        self.current_calculation.modified_date = datetime.datetime.now()
        self.current_calculation.save()

    def getRodName(self, xPos, yPos):
        xID = df.OPR1000_xPos
        yID = df.OPR1000_yPos

        return "Core_%s%s" % (xID[xPos], yID[yPos])

    @pyqtSlot(str, str)
    def swapAssembly(self, loc01, loc02):
        pass

    def changeButtonSetting(self, id, pos, iBlock, xPos, yPos):
        font = QFont()
        # 01. Setting geometry

        if (self.controlRodMap[xPos][yPos] == False):
            colorR = int(df.rgbSet[0][0], 16)
            colorG = int(df.rgbSet[0][1], 16)
            colorB = int(df.rgbSet[0][2], 16)
        else:
            colorR = int(df.rgbDummy[0], 16)
            colorG = int(df.rgbDummy[1], 16)
            colorB = int(df.rgbDummy[2], 16)

        if (pos == df._POSITION_CORE_):
            if (id != df._BLANK_ASM_):
                if (id[2].isalpha()):
                    subID0 = id[2:]
                else:
                    subID0 = id[3:]
            else:
                subID0 = ""
            self.bClassRCS[xPos][yPos].setStyleSheet(
                "background-color: rgb({},{},{});border-radius = 0px".format(colorR, colorG, colorB))
            font.setPointSize(10)
            self.bClassRCS[xPos][yPos].setFont(font)
            self.bClassRCS[xPos][yPos].setText(subID0)
            # self.bClassRCS[xPos][yPos].rename(newName)

        elif (pos == df._POSITION_SFP01_):
            self.bClassSFP01[iBlock][xPos][yPos].setStyleSheet(
                "background-color: rgb({},{},{});border-radius = 0px".format(colorR, colorG, colorB))
            self.bClassSFP01[iBlock][xPos][yPos].setText("")
            # self.bClassSFP01[iBlock][xPos][yPos].rename(newName)

        else:
            self.bClassSFP02[iBlock][xPos][yPos].setStyleSheet(
                "background-color: rgb({},{},{});border-radius = 0px".format(colorR, colorG, colorB))
            self.bClassSFP02[iBlock][xPos][yPos].setText("")
            # self.bClassSFP02[iBlock][xPos][yPos].rename(newName)


    #########################
    # Calculation Start
    ########################
    def start_save(self):
        super().start_save()

    def start_calc(self):
        super().start_calc()
        calOption, bp, stuckRodPos, rsdm, error = self.get_calculation_input()

        if not error and not self.cal_start:

            pArray = []

            if self.ui.SDM_InputSelect.currentIndex() == 1:
                for value in self.snapshotArray:
                    pArray.append([value[0], value[2], value[1], value[3]] + [0, 0, 0] + value[4:])

            # eigen = 1 / (1 / (1.0007 + (1.00239 - 1.0007) * bp / self.calcManager.eoc_cycle_burnup_value)
            #      + df.sdm_astra_uncertainty / 100
            #      + 0.010)

            if len(pArray) == 0:
                pArray.append([0, 100, bp, 1.0, 0, 0, 0,])

            self.start_calculation_message = SplashScreen()
            self.start_calculation_message.killed.connect(self.killManagerProcess)
            self.start_calculation_message.init_progress(1, 500, is_nano=True)
            # self.ui.SDM_run_button.setText(cs.RUN_BUTTON_RUNNING)
            # self.ui.SDM_run_button.setDisabled(True)
            self.cal_start = True
            self.queue.put((calOption, bp, stuckRodPos, self.ui.SDM_Input02.currentIndex(), rsdm, pArray))

    def get_calculation_input(self):

        stuckRodPos = []
        for iRow in range(df._OPR1000_XPOS_):
            for iCol in range(df._OPR1000_YPOS_):
                if (self.controlRodMap[iRow][iCol] == True):
                    stuckRodPos.append(df.OPR1000_CR_ID[iCol][iRow])
        self.stuckRodPos = stuckRodPos
        bp = self.ui.SDM_Input01.value()
        rod_error = self.check_calculation_input()
        rsdm = df.sdm_required_sdm

        if rod_error:
            print("Rod")

        return df.CalcOpt_SDM, bp, stuckRodPos, rsdm, rod_error

    def check_calculation_input(self):

        if self.ui.SDM_Input01.value() >= self.calcManager.cycle_burnup_values[-1]+1000:
            msgBox = QMessageBoxWithStyle(self.get_ui_component())
            msgBox.setWindowTitle("Burnup Out of Range")
            msgBox.setText("{}MWD/MTU excedes EOC Cycle Burnup({} MWD/MTU)\n"
                           "Put Cycle Burnup less than {}MWD/MTU".format(self.ui.SDM_Input01.value(),
                                                            self.calcManager.cycle_burnup_values[-1],
                                                            self.calcManager.cycle_burnup_values[-1]+1000))
            msgBox.setStandardButtons(QMessageBox.Ok)
            msgBox.setCustomStyle()
            #msgBox.setWindowFlag(QtCore.Qt.FramelessWindowHint)
            result = msgBox.exec_()
            # if result == QMessageBox.Cancel:
            return True

        # check burnup range
        if self.ui.SDM_Input02.currentIndex() == 0:

            if len(self.stuckRodPos) != 1:
                msgBox = QMessageBoxWithStyle(self.get_ui_component())
                msgBox.setWindowTitle("Stuck Rod Error")
                msgBox.setText("N-{} Rod Selected.\nSelect only 1 Rod.".format(len(self.stuckRodPos)))
                msgBox.setStandardButtons(QMessageBox.Ok)
                #msgBox.setWindowFlag(QtCore.Qt.FramelessWindowHint)
                msgBox.setCustomStyle()
                result = msgBox.exec_()
                return True
        else:
            if len(self.stuckRodPos) > 2:
                msgBox = QMessageBoxWithStyle(self.get_ui_component())
                msgBox.setWindowTitle("Stuck Rod Error")
                msgBox.setText("Too much stuck rod N-{}".format(self.stuckRodPos.join(", ")))
                msgBox.setStandardButtons(QMessageBox.Ok)
                #msgBox.setWindowFlag(QtCore.Qt.FramelessWindowHint)
                msgBox.setCustomStyle()
                result = msgBox.exec_()
                return True

        return False

    #########################
    # End Calculation
    ########################
    @pyqtSlot(str)
    def finished(self, is_success):

        self.is_run_selected = False

        if self.start_calculation_message:
            self.start_calculation_message.close()

        if is_success == self.calcManager.SUCC:
            if self.calcManager.mode_index > 0:
                if len(self.calcManager.stuckRodPos) == 0:
                    rods = "ARO"
                    cea = "ARO"
                else:
                    if len(self.calcManager.stuckRodPos) == 1:
                        cea = "N-1"
                    else:
                        cea = "N-2"
                    rods = ",".join(self.calcManager.stuckRodPos)

                self.ui.sdm2_rods.setText(rods)
                self.ui.sdm2_cea.setText(cea)
                self.ui.sdm2_burnup.setText("{:.1f}".format(self.calcManager.bp))
                self.ui.sdm2_temp.setText("{:.1f}".format(self.calcManager.results.sdm2_temp))
                self.ui.sdm2_rsdm.setText("{:.1f}".format(self.calcManager.results.sdm2_rsdm))
                self.ui.sdm2_rb.setText("{:.1f}".format(self.calcManager.results.sdm2_ppm))

                self.current_calculation.sdm_output.m2_success = True
                self.current_calculation.sdm_output.m2_core_burnup = self.calcManager.bp
                self.current_calculation.sdm_output.m2_temperature = self.calcManager.results.sdm2_temp
                self.current_calculation.sdm_output.m2_cea_configuration = cea
                self.current_calculation.sdm_output.m2_stuck_rod = rods
                self.current_calculation.sdm_output.m2_required_worth = self.calcManager.results.sdm2_rsdm
                self.current_calculation.sdm_output.m2_required_cbc = self.calcManager.results.sdm2_ppm

            else:
                self.ui.sdm1_rods.setText(self.calcManager.stuckRodPos[-1])
                self.ui.sdm1_burnup.setText("{:.1f}".format(self.calcManager.bp))
                self.ui.sdm1_cea.setText("N-1")
                self.ui.sdm1_margin.setText("{:.2f}".format(self.calcManager.results.sdm1_sdm))
                self.ui.sdm1_defect.setText("{:.2f}".format(self.calcManager.results.sdm1_defect))
                self.ui.sdm1_n1w.setText("{:.2f}".format(self.calcManager.results.sdm1_n1w))
                self.ui.sdm1_rsdm.setText("{:.2f}".format(self.calcManager.results.sdm1_rsdm))
                self.ui.sdm1_temp.setText("{:.1f}".format(self.calcManager.results.sdm1_temp))

                self.current_calculation.sdm_output.m1_success = True
                self.current_calculation.sdm_output.m1_core_burnup = self.calcManager.bp
                self.current_calculation.sdm_output.m1_temperature = self.calcManager.results.sdm1_temp
                self.current_calculation.sdm_output.m1_cea_configuration = "N-1"
                self.current_calculation.sdm_output.m1_stuck_rod = self.calcManager.stuckRodPos[-1]
                self.current_calculation.sdm_output.m1_n1_worth = self.calcManager.results.sdm1_n1w
                self.current_calculation.sdm_output.m1_defect_worth = self.calcManager.results.sdm1_defect
                self.current_calculation.sdm_output.m1_required_worth = self.calcManager.results.sdm1_rsdm
                self.current_calculation.sdm_output.m1_sdm_worth = self.calcManager.results.sdm1_sdm

            self.current_calculation.save()
            self.current_calculation.sdm_output.save()

        self.cal_start = False
        # self.ui.SDM_run_button.setText(cs.RUN_BUTTON_RUN)
        # self.ui.SDM_run_button.setDisabled(False)


    # def printReport(self):
    #
    #     if self.ui.SDM_Input02.currentIndex() > 0:
    #
    #         uiReport = QWidget()
    #         uiReport.ui = reportWidget2.Ui_unitWidget_SDM()
    #         uiReport.ui.setupUi(uiReport)
    #
    #         uiReport.ui.sdm2_plant.setText(self.ui.sdm2_plant.text())
    #         uiReport.ui.sdm2_cycle.setText(self.ui.sdm2_cycle.text())
    #
    #         uiReport.ui.sdm2_rods.setText(self.ui.sdm2_rods.text())
    #         uiReport.ui.sdm2_cea.setText(self.ui.sdm2_cea.text())
    #         uiReport.ui.sdm2_burnup.setText(self.ui.sdm2_burnup.text())
    #         uiReport.ui.sdm2_temp.setText(self.ui.sdm2_temp.text())
    #         uiReport.ui.sdm2_rsdm.setText(self.ui.sdm2_rsdm.text())
    #         uiReport.ui.sdm2_rb.setText(self.ui.sdm2_rb.text())
    #
    #     else:
    #
    #         uiReport = QWidget()
    #         uiReport.ui = reportWidget1.Ui_unitWidget_SDM()
    #         uiReport.ui.setupUi(uiReport)
    #
    #         uiReport.ui.sdm1_plant.setText(self.ui.sdm1_plant.text())
    #         uiReport.ui.sdm1_cycle.setText(self.ui.sdm1_cycle.text())
    #
    #         uiReport.ui.sdm1_rods.setText(self.ui.sdm1_rods.text())
    #         uiReport.ui.sdm1_burnup.setText(self.ui.sdm1_burnup.text())
    #         uiReport.ui.sdm1_cea.setText(self.ui.sdm1_cea.text())
    #         uiReport.ui.sdm1_margin.setText(self.ui.sdm1_margin.text())
    #         uiReport.ui.sdm1_defect.setText(self.ui.sdm1_defect.text())
    #         uiReport.ui.sdm1_n1w.setText(self.ui.sdm1_n1w.text())
    #         uiReport.ui.sdm1_rsdm.setText(self.ui.sdm1_rsdm.text())
    #         uiReport.ui.sdm1_temp.setText(self.ui.sdm1_temp.text())
    #
    #     self.printPDF(uiReport)
    #
    # def clear_position(self):
    #     self.selectedStuckRods = []
    #     self.stuckRodPos = []
    #
    #     xID = df.OPR1000_xPos
    #     yID = df.OPR1000_yPos
    #     for yPos in range(len(yID)):
    #         for xPos in range(len(xID)):
    #             if df.OPR1000MAP[xPos][yPos] == True:
    #                 if self.controlRodMap[xPos][yPos]:
    #                     self.controlRodMap[xPos][yPos] = False
    #                     self.bClassRCS[xPos][yPos].setStyleSheet(
    #                         "background-color: rgb({},{},{});border-radius = 0px;color: white;".format(df.rgbSet[-3][0],
    #                                                                                                    df.rgbSet[-3][1],
    #                                                                                                df.rgbSet[-3][2]))
