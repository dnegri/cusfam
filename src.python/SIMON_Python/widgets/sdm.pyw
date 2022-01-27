import sys
import platform
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import (QCoreApplication, QPropertyAnimation, QDate, QDateTime, QMetaObject, QObject, QPoint, QRect, QSize, QTime, QUrl, Qt, QEvent)
from PyQt5.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont, QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter, QPixmap, QRadialGradient)
from PyQt5.QtWidgets import *
from model import *
import datetime
import os
import glob
from ui_main_rev18 import Ui_MainWindow
import constants as cs
import Definitions as df
import widgets.utils.PyUnitButton as ts
import utils as ut


class SDMWidget:

    def __init__(self, db, ui):
        self.ui = ui  # type: Ui_MainWindow
        self.db = db

        _BLANK_ASM_ = " "
        self.linkRCS     =  [[_BLANK_ASM_ for col in range(df._OPR1000_XPOS_)] for row in range(df._OPR1000_YPOS_)]
        self.bClassRCS   =  [[_BLANK_ASM_ for col in range(df._OPR1000_XPOS_)] for row in range(df._OPR1000_YPOS_)]
        self.globalButtonInfo = {}
        self.controlRodMap = [[ False for col in range(df._OPR1000_XPOS_)] for row in range(df._OPR1000_YPOS_)]

        self.funcType = eval("ts.outputButton")

        self.settingLP(self.ui.SDM_InputLP_frame,self.ui.SDM_InputLP_grid)
        self.set_all_components()

    def set_all_components(self):
        self.ui.SDM_radioButtonOpt01.toggled.connect(lambda: self.calculation_options(self.ui.SDM_radioButtonOpt01))
        self.ui.SDM_radioButtonOpt02.toggled.connect(lambda: self.calculation_options(self.ui.SDM_radioButtonOpt02))
        self.ui.SDM_radioButtonOpt03.toggled.connect(lambda: self.calculation_options(self.ui.SDM_radioButtonOpt03))
        self.ui.SDM_NDR01.currentIndexChanged.connect(self.mode_changed)
        self.ui.SDM_UserInput01.currentIndexChanged.connect(self.mode_changed)

        self.ui.SDM_UserInput02.clear()
        self.ui.SDM_UserInput03.clear()

        query = LoginUser.get(LoginUser.username == cs.ADMIN_USER)
        user = query.login_user
        plants, errors = ut.getPlantFiles(user)
        for plant in plants:
            self.ui.SDM_PlantName.addItem(plant)
        self.ui.SDM_PlantName.setCurrentText(user.plant_file)

        restarts, errors = ut.getRestartFiles(user)
        for restart in restarts:
            self.ui.SDM_PlantCycle.addItem(restart)
        self.ui.SDM_PlantCycle.setCurrentText(user.restart_file)

        self.ui.SDM_tabWidget.setCurrentIndex(0)
        # self.ui.SDM_radioButtonOpt01.setChecked(True)
        self.mode_changed(self.ui.SDM_NDR01.currentIndex())

        self.ui.frame_SDM_NDR.hide()
        self.ui.frame_SDM_Snapshot.hide()
        self.ui.frame_SDM_UserInput.hide()

        # self.ui.

    def calculation_options(self, b):
        if b == self.ui.SDM_radioButtonOpt01:
           # self.ui.SDM_stackedWidget.setCurrentWidget(self.ui.SDM_Stack01_Basic)
           self.ui.frame_SDM_NDR.show()
           self.ui.frame_SDM_Snapshot.hide()
           self.ui.frame_SDM_UserInput.hide()
        if b == self.ui.SDM_radioButtonOpt02:
           # self.ui.SDM_stackedWidget.setCurrentWidget(self.ui.SDM_Stack02_SnapShot)
           self.ui.frame_SDM_NDR.hide()
           self.ui.frame_SDM_Snapshot.show()
           self.ui.frame_SDM_UserInput.hide()
        if b == self.ui.SDM_radioButtonOpt03:
           # self.ui.SDM_stackedWidget.setCurrentWidget(self.ui.SDM_Stack03_UserInput)
           self.ui.frame_SDM_NDR.hide()
           self.ui.frame_SDM_Snapshot.hide()
           self.ui.frame_SDM_UserInput.show()

    def mode_changed(self, ix):
        if ix <= 1:
            self.ui.SDM_NDR03.hide()
            self.ui.SDM_LabelNDR03.hide()
            self.ui.SDM_NDR04.hide()
            self.ui.SDM_LabelNDR04.hide()
            self.ui.SDM_UserInput05.hide()
            self.ui.SDM_UserInputLabel05.hide()
            self.ui.SDM_UserInput06.hide()
            self.ui.SDM_UserInputLabel06.hide()
            self.ui.SDM_stackedWidget_reports.setCurrentWidget(self.ui.SDM_report1)
            self.ui.SDM_stackedWidget_inputs.setCurrentWidget(self.ui.SDM_input1)
        if ix > 1:
            self.ui.SDM_NDR03.show()
            self.ui.SDM_LabelNDR03.show()
            self.ui.SDM_NDR04.show()
            self.ui.SDM_LabelNDR04.show()
            self.ui.SDM_UserInput05.show()
            self.ui.SDM_UserInputLabel05.show()
            self.ui.SDM_UserInput06.show()
            self.ui.SDM_UserInputLabel06.show()
            self.ui.SDM_stackedWidget_reports.setCurrentWidget(self.ui.SDM_report2)
            self.ui.SDM_stackedWidget_inputs.setCurrentWidget(self.ui.SDM_input2)


    def settingLP(self,frame, gridLayout):
        _translate = QCoreApplication.translate
        xID = [ "A" ,"B" ,"C" ,"D" ,"E" ,"F" ,"G" ,"H" ,"J" ,"K" ,"L" ,"M" ,"N" ,"P" ,"R" ]
        yID = [ "01","02","03","04","05","06","07","08","09","10","11","12","13","14","15"]
        for yPos in range(len(yID)):
            for xPos in range(len(xID)):
                if(df.OPR1000MAP[xPos][yPos]==True):
                    bName ="Core_%s%s" % (xID[xPos], yID[yPos])
                    self.globalButtonInfo[bName] = [df._POSITION_CORE_, 0, xPos, yPos]
                    # generate button geometry
                    buttonCore = self.funcType( bName, self.showAssemblyLoading,self.swapAssembly,frame) # type: QPushButton
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
                    if xPos == 0 and yPos == 0:
                        gridLayout.addWidget(buttonCore, yPos + 1, xPos + 1, 2, 2)
                    else:
                        gridLayout.addWidget(buttonCore, yPos+1, xPos+1, 1, 1)

                    self.bClassRCS[xPos][yPos] = buttonCore
                    colorR = int(df.rgbDummy[0], 16)
                    colorG = int(df.rgbDummy[1], 16)
                    colorB = int(df.rgbDummy[2], 16)

                    buttonCore.setStyleSheet( "background-color: rgb({},{},{});border-radius = 0px;color: white;".format(df.rgbSet[-1][0],
                                                                                                   df.rgbSet[-1][1],
                                                                                                   df.rgbSet[-1][2]))
                    buttonCore.setDisabled(df.OPR1000MAP_INFO[xPos][yPos])
                    if df.OPR1000MAP_INFO[xPos][yPos]:
                        buttonCore.setDisabled(df.OPR1000MAP_INFO[xPos][yPos])

                    if len(df.OPR1000MAP_BP[xPos][yPos]) == 0:
                        buttonCore.setDisabled(True)

                    if xPos == 0 and yPos == 13:
                        buttonCore.setStyleSheet(
                            "background-color: blue;border-radius = 0px;color: white;")
                        font = QFont()
                        font.setPointSize(7)
                        self.bClassRCS[xPos][yPos].setFont(font)

                    if xPos == 0 and yPos == 14:
                        buttonCore.setStyleSheet(
                            "background-color: red;border-radius = 0px;color: white;")
                        font = QFont()
                        font.setPointSize(7)
                        self.bClassRCS[xPos][yPos].setFont(font)

                    #buttonCore.setStyleSheet(
                    #    "background-color: rgb({},{},{});border-radius = 0px;color: black;".format(colorR, colorG, colorB))

    def getAssemblyPosition(self,asmName):
        [pos, iBlock, xPos, yPos] = self.globalButtonInfo[asmName]
        xID = df.OPR1000_xPos[xPos]
        yID = df.OPR1000_yPos[yPos]
        if  ( pos == df._POSITION_CORE_ ):
            fullID = self.linkRCS[xPos][yPos]
            posName = "%s-%s" % (xID, yID)
        elif( pos == df._POSITION_SFP01_):
            fullID = self.linkSFP01[iBlock][xPos][yPos]
            posName = "SFP01_%s-%s" % (xID, yID)
        else:
            fullID = self.linkSFP02[iBlock][xPos][yPos]
            posName = "SFP02_%s-%s" % (xID, yID)

        cycleNum = 0
        startHist = []
        dischargeHist = []
        if( fullID == df._BLANK_ASM_):
            id = "Blank"
            type = " "
            stepNum = " "
            eocB = " "
        else:
            if (fullID[2].isalpha()):
                id = fullID[2:]
            else:
                id = fullID[3:]
            type     = self.assemblies_dict[fullID].getAsmType()
            coreStep = self.assemblies_dict[fullID].getCoreStep()
            stepNum = str(len(coreStep))
            bp      = self.assemblies_dict[fullID].getAsmBP()
            eocB    = "%.1f MWD/MTU" % bp[-1][-1]

            cycleStep = self.assemblies_dict[fullID].getCoreStep()
            cycleNum = len(cycleStep)
            startHist = self.assemblies_dict[fullID].getStartHist()
            dischargeHist = self.assemblies_dict[fullID].getDischargeHistory()
        return id, type, posName, stepNum, eocB, cycleNum, startHist, dischargeHist



    @Slot(str)
    def showAssemblyLoading(self,loc01):
        [pos, iBlock, xPos, yPos] = self.globalButtonInfo[loc01]
        [ id, type, posName, stepNum, eocB, cycleNum, startHist, dischargeHist ] = self.getAssemblyPosition(loc01)
        if(self.controlRodMap[xPos][yPos]==False):
            # self.bClassRCS[xPos][yPos].
            self.controlRodMap[xPos][yPos] = True

            colorR = int(df.rgbSet[11][0], 16)
            colorG = int(df.rgbSet[11][1], 16)
            colorB = int(df.rgbSet[11][2], 16)
            self.bClassRCS[xPos][yPos].setStyleSheet(
                "background-color: red;border-radius = 0px;color: white;".format(colorR, colorG, colorB))

            #self.bClassRCS[xPos][yPos].setStyleSheet("background-color: rgb({},{},{});border-radius = 0px;color: black;".format(colorR, colorG, colorB))
        else:
            self.controlRodMap[xPos][yPos] = False

            #colorR = int(df.rgbSet[-1][0], 16)
            #colorG = int(df.rgbSet[-1][1], 16)
            #colorB = int(df.rgbSet[-1][2], 16)
            #self.bClassRCS[xPos][yPos].setStyleSheet(
            #    "background-color: red;border-radius = 0px;color: black;")
            # self.bClassRCS[xPos][yPos].setStyleSheet("background-color: rgb({},{},{});border-radius = 0px;color: black;".format(df.rgbSet[-1][0], df.rgbSet[-1][1], df.rgbSet[-1][2]))

            self.bClassRCS[xPos][yPos].setStyleSheet(
                "background-color: rgb({},{},{});border-radius = 0px;color: white;".format(df.rgbSet[-1][0],
                                                                                           df.rgbSet[-1][1],
                                                                                           df.rgbSet[-1][2]))

            # self.ui.LP_Info01.setText(id)
        # self.ui.LP_Info02.setText(type)
        # self.ui.LP_Info03.setText(posName)
        # self.ui.LP_Info04.setText(stepNum)
        # self.ui.LP_Info05.setText(eocB)
        #
        # if(cycleNum==1):
        #     self.ui.LP_History01.setText(startHist[0].strftime("%Y-%m-%d %H:%M"))
        #     self.ui.LP_History02.setText(dischargeHist[0].strftime("%Y-%m-%d %H:%M"))
        #     self.ui.LP_History03.setText("-")
        #     self.ui.LP_History04.setText("-")
        #     self.ui.LP_History05.setText("-")
        #     self.ui.LP_History06.setText("-")
        # elif(cycleNum==2):
        #     self.ui.LP_History01.setText(startHist[0].strftime("%Y-%m-%d %H:%M"))
        #     self.ui.LP_History02.setText(dischargeHist[0].strftime("%Y-%m-%d %H:%M"))
        #     self.ui.LP_History03.setText(startHist[1].strftime("%Y-%m-%d %H:%M"))
        #     self.ui.LP_History04.setText(dischargeHist[1].strftime("%Y-%m-%d %H:%M"))
        #     self.ui.LP_History05.setText("-")
        #     self.ui.LP_History06.setText("-")
        # elif(cycleNum==3):
        #     self.ui.LP_History01.setText(startHist[0].strftime("%Y-%m-%d %H:%M"))
        #     self.ui.LP_History02.setText(dischargeHist[0].strftime("%Y-%m-%d %H:%M"))
        #     self.ui.LP_History03.setText(startHist[1].strftime("%Y-%m-%d %H:%M"))
        #     self.ui.LP_History04.setText(dischargeHist[1].strftime("%Y-%m-%d %H:%M"))
        #     self.ui.LP_History05.setText(startHist[2].strftime("%Y-%m-%d %H:%M"))
        #     self.ui.LP_History06.setText(dischargeHist[2].strftime("%Y-%m-%d %H:%M"))
        # else:
        #     self.ui.LP_History01.setText("-")
        #     self.ui.LP_History02.setText("-")
        #     self.ui.LP_History03.setText("-")
        #     self.ui.LP_History04.setText("-")
        #     self.ui.LP_History05.setText("-")
        #     self.ui.LP_History06.setText("-")
        pass

    @Slot(str, str)
    def swapAssembly(self,loc01,loc02):
        pass
        # [posFrom, iBlockFrom, xPosFrom, yPosFrom] = self.globalButtonInfo[loc01]
        # [posTo, iBlockTo, xPosTo, yPosTo] = self.globalButtonInfo[loc02]
        #
        # if (posFrom == df._POSITION_CORE_):
        #     id_From = self.linkRCS[xPosFrom][yPosFrom]
        # elif (posFrom == df._POSITION_SFP01_):
        #     id_From = self.linkSFP01[iBlockFrom][xPosFrom][yPosFrom]
        # else:
        #     id_From = self.linkSFP02[iBlockFrom][xPosFrom][yPosFrom]
        # NewControlRodFlag_From = self.controlRodMap[xPosTo][yPosTo]
        #
        #
        # if (posTo == df._POSITION_CORE_):
        #     id_To = self.linkRCS[xPosTo][yPosTo]
        # elif (posTo == df._POSITION_SFP01_):
        #     id_To = self.linkSFP01[iBlockTo][xPosTo][yPosTo]
        # else:
        #     id_To = self.linkSFP02[iBlockTo][xPosTo][yPosTo]
        # NewControlRodFlag_To = self.controlRodMap[xPosFrom][yPosFrom]
        #
        # NewID_To = id_From
        # NewID_From = id_To
        #
        # if (posFrom == df._POSITION_CORE_):
        #     self.linkRCS[xPosFrom][yPosFrom] = NewID_From
        # elif (posFrom == df._POSITION_SFP01_):
        #     self.linkSFP01[iBlockFrom][xPosFrom][yPosFrom] = NewID_From
        # else:
        #     self.linkSFP02[iBlockFrom][xPosFrom][yPosFrom] = NewID_From
        # self.controlRodMap[xPosFrom][yPosFrom] = NewControlRodFlag_From
        #
        # if (posTo == df._POSITION_CORE_):
        #     self.linkRCS[xPosTo][yPosTo] = NewID_To
        # elif (posTo == df._POSITION_SFP01_):
        #     self.linkSFP01[iBlockTo][xPosTo][yPosTo] = NewID_To
        # else:
        #     self.linkSFP02[iBlockTo][xPosTo][yPosTo] = NewID_To
        # self.controlRodMap[xPosTo][yPosTo] = NewControlRodFlag_To
        #
        # self.changeButtonSetting(NewID_From, posFrom, iBlockFrom, xPosFrom, yPosFrom)
        # self.changeButtonSetting(NewID_To, posTo, iBlockTo, xPosTo, yPosTo)

    def changeButtonSetting(self, id, pos, iBlock, xPos, yPos):
        font = QFont()
        # 01. Setting geometry
        subID0 = ""
        #if( id != df._BLANK_ASM_):
        #     self.assemblies_dict[id].setCurrentLoc(pos)
        #     assembly = self.assemblies_dict[id]
        #     subID = assembly.getSubID()
        #     if subID in self.subIDFilter:
        #         subNum = self.subIDFilter.index(subID)
        #     else:
        #         subNum = 0
        #     colorR = int(df.rgbSet[subNum][0], 16)
        #     colorG = int(df.rgbSet[subNum][1], 16)
        #     colorB = int(df.rgbSet[subNum][2], 16)
        # else:
        #     colorR = int(df.rgbDummy[0], 16)
        #     colorG = int(df.rgbDummy[1], 16)
        #     colorB = int(df.rgbDummy[2], 16)
        if(self.controlRodMap[xPos][yPos]==False):
            colorR = int(df.rgbSet[0][0], 16)
            colorG = int(df.rgbSet[0][1], 16)
            colorB = int(df.rgbSet[0][2], 16)
        else:
            colorR = int(df.rgbDummy[0], 16)
            colorG = int(df.rgbDummy[1], 16)
            colorB = int(df.rgbDummy[2], 16)

        if  (pos== df._POSITION_CORE_):
            if (id != df._BLANK_ASM_):
                if(id[2].isalpha()):
                    subID0 = id[2:]
                else:
                    subID0 = id[3:]
            else:
                subID0 = ""
            self.bClassRCS[xPos][yPos].setStyleSheet("background-color: rgb({},{},{});border-radius = 0px".format(colorR, colorG, colorB))
            font.setPointSize(10)
            self.bClassRCS[xPos][yPos].setFont(font)
            self.bClassRCS[xPos][yPos].setText(subID0)
            # self.bClassRCS[xPos][yPos].rename(newName)

        elif(pos== df._POSITION_SFP01_):
            self.bClassSFP01[iBlock][xPos][yPos].setStyleSheet(
                "background-color: rgb({},{},{});border-radius = 0px".format(colorR, colorG, colorB))
            self.bClassSFP01[iBlock][xPos][yPos].setText("")
            # self.bClassSFP01[iBlock][xPos][yPos].rename(newName)

        else:
            self.bClassSFP02[iBlock][xPos][yPos].setStyleSheet(
                "background-color: rgb({},{},{});border-radius = 0px".format(colorR, colorG, colorB))
            self.bClassSFP02[iBlock][xPos][yPos].setText("")
            # self.bClassSFP02[iBlock][xPos][yPos].rename(newName)