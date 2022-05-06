import widgets.utils.PyUnitTableWidget as unitTable
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QTableView, QSizePolicy, QFrame, QAbstractItemView, \
    QAbstractScrollArea, QHeaderView, QPushButton, QSpacerItem, QHBoxLayout, QWidget, QAbstractSpinBox, QFileDialog
from PyQt5.QtCore import QSize, QSizeF, QCoreApplication, Qt
from PyQt5.QtGui import QPalette, QBrush, QColor, QFont

import widgets.utils.PyTableDoubleSpinBox as unitDoubleBox
import widgets.utils.PyTableSpinBox as unitSpinBox
import math
import os
import pandas as pd
import Definitions as df


class DecayTableWidget(unitTable.unitTableWidget):
    def __init__(self, frame,  headerItem):
        super().__init__(frame,headerItem)

        # Initialize Dataset
        self.InputArray = []
        self.decayInputBox = []
        self.decayInputData = []
        self.rodPosChangedHistory = []
        self.totalTime = 0.0

        self.setRowCount(df.tableDecayColumnNum)
        self.setTableSpinBox()

        self.setLastSpinBox()

    def returnTimeData(self):
        return self.decayInputData

    def setTotalTime(self,totalTime):
        self.totalTime = totalTime

    def setTableSpinBox(self):
        nRow    = self.rowCount()
        #nColumn = self.columnCount()
        # Define
        for iRow in range(nRow):
            unitRodPos = []
            unitData = []

            unitBoxDouble = self.makeUnitDoubleBox(iRow,0)
            unitRodPos.append(unitBoxDouble)
            unitData.append(unitBoxDouble.value())

            unitBox = self.makeUnitSpinBox(iRow,1)
            unitRodPos.append(unitBox)
            unitData.append(unitBox.value())

            self.decayInputBox.append(unitRodPos)
            self.decayInputData.append(unitData)



    def makeUnitDoubleBox(self,iRow,iColumn):

        # Set Font for DoubleSpinBox
        font = QFont()
        font.setFamily(u"Segoe UI")
        font.setBold(True)
        font.setPointSize(9)


        doubleSpinBoxWidget = QWidget()
        doubleSpinBoxWidget.setStyleSheet("QWidget { background-color: rgb(38, 44, 53);}")

        doubleSpinBox = unitDoubleBox.tableDoubleSpinBox()
        doubleSpinBox.saveBoxPosition(iRow,iColumn)
        doubleSpinBox.setObjectName(u"RO_DecayTime%02d" %(iRow+1))

        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(doubleSpinBox.sizePolicy().hasHeightForWidth())
        doubleSpinBox.setSizePolicy((sizePolicy))
        doubleSpinBox.setMinimumSize(QSize(0, 0))
        doubleSpinBox.setMaximumSize(QSize(80, 16777215))
        doubleSpinBox.setAlignment(Qt.AlignCenter)
        doubleSpinBox.setFont(font)
        doubleSpinBox.setButtonSymbols(QAbstractSpinBox.NoButtons)
        doubleSpinBox.setProperty("showGroupSeparator", True)
        doubleSpinBox.setDecimals(2)
        doubleSpinBox.setMinimum(0.000000000000000)
        doubleSpinBox.setMaximum(999.999999999999)
        doubleSpinBox.setSingleStep(1.000000000000000)
        doubleSpinBox.setStepType(QAbstractSpinBox.DefaultStepType)
        doubleSpinBox.setValue(0.000000000000000)

        layout = QHBoxLayout(doubleSpinBoxWidget)
        layout.addWidget(doubleSpinBox)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(0,0,0,0)

        item = QTableWidgetItem()
        item.setTextAlignment(Qt.AlignCenter)

        # (a,b) = doubleSpinBox.returnBoxPosition()
        # print(a,b)
        self.setCellWidget(iRow,iColumn,doubleSpinBoxWidget)


        doubleSpinBox.editingFinished.connect(lambda: self.changeUnitDecayInterval(iRow,iColumn))

        # self.SD_tableWidget_button01.clicked['bool'].connect(self.clickSaveAsExcel)
        # self.SD_tableWidget_button02.clicked['bool'].connect(self.resetPositionData)

        return doubleSpinBox


    def makeUnitSpinBox(self,iRow,iColumn):

        # Set Font for DoubleSpinBox
        font = QFont()
        font.setFamily(u"Segoe UI")
        font.setBold(True)
        font.setPointSize(9)


        spinBoxWidget = QWidget()
        spinBoxWidget.setStyleSheet("QWidget { background-color: rgb(38, 44, 53);}")

        spinBox = unitSpinBox.tableSpinBox()
        spinBox.saveBoxPosition(iRow,iColumn)
        spinBox.setObjectName(u"RO_DecayNum%02d" %(iRow+1))

        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(spinBoxWidget.sizePolicy().hasHeightForWidth())
        spinBox.setSizePolicy((sizePolicy))
        spinBox.setMinimumSize(QSize(0, 0))
        spinBox.setMaximumSize(QSize(80, 16777215))
        spinBox.setAlignment(Qt.AlignCenter)
        spinBox.setFont(font)
        spinBox.setButtonSymbols(QAbstractSpinBox.NoButtons)
        spinBox.setProperty("showGroupSeparator", True)
        # spinBox.setDecimals(2)
        spinBox.setMinimum(0)
        spinBox.setMaximum(1000)
        spinBox.setSingleStep(1)
        spinBox.setStepType(QAbstractSpinBox.DefaultStepType)
        spinBox.setValue(0)

        layout = QHBoxLayout(spinBoxWidget)
        layout.addWidget(spinBox)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(0,0,0,0)

        item = QTableWidgetItem()
        item.setTextAlignment(Qt.AlignCenter)

        # (a,b) = doubleSpinBox.returnBoxPosition()
        # print(a,b)
        self.setCellWidget(iRow,iColumn,spinBoxWidget)


        #doubleSpinBox.editingFinished.connect(lambda: self.checkChangedCondition(iRow,iColumn))
        spinBox.editingFinished.connect(lambda: self.changeUnitTimeStepNum(iRow, iColumn))

        # self.SD_tableWidget_button01.clicked['bool'].connect(self.clickSaveAsExcel)
        # self.SD_tableWidget_button02.clicked['bool'].connect(self.resetPositionData)

        return spinBox

    def setLastSpinBox(self):

        lastRow = self.rowCount()
        self.decayInputBox[lastRow-1][1].setReadOnly(True)
        for idx in range(lastRow-1):
            self.decayInputBox[idx][1].setReadOnly(False)

    def changeUnitDecayInterval(self,iRow,iColumn):
        self.decayInputData[iRow][iColumn] = self.decayInputBox[iRow][iColumn].value()

        if(self.totalTime ==0.0):
            return
        if(self.decayInputData[iRow][0]==0.0):
            return

        if(self.rowCount()==iRow+1):

            totalTime = self.totalTime

            nRow = self.rowCount()
            for idx in range(nRow-1):
                if(idx == iRow):
                    continue
                minusTime = self.decayInputData[idx][0] * self.decayInputData[idx][1]
                totalTime = totalTime - minusTime

            self.decayInputData[iRow][0] = self.decayInputBox[iRow][0].value()
            nStep = math.ceil(totalTime /  self.decayInputData[iRow][0])

            self.decayInputBox[iRow][1].setValue(nStep)
            self.decayInputData[iRow][1] = nStep

    def changeUnitTimeStepNum(self,iRow,iColumn):
        self.decayInputData[iRow][iColumn] = self.decayInputBox[iRow][iColumn].value()

        if(self.decayInputData[-1][0]==0.0):
            return

        totalTime = self.totalTime


        nRow = self.rowCount()
        for idx in range(nRow-1):
            minusTime = self.decayInputData[idx][0] * self.decayInputData[idx][1]
            totalTime = totalTime - minusTime

        nStep = math.ceil(totalTime / self.decayInputData[-1][0])
        self.decayInputData[-1][1] = nStep
        self.decayInputBox[-1][1].setValue(nStep)



    def addDecayTableRow(self):
        nRow = self.rowCount()
        self.setRowCount(nRow+1)

        unitRodPos = []
        unitData = []

        unitBoxDouble = self.makeUnitDoubleBox(nRow,0)
        unitRodPos.append(unitBoxDouble)
        unitData.append(unitBoxDouble.value())

        unitBox = self.makeUnitSpinBox(nRow,1)
        unitRodPos.append(unitBox)
        unitData.append(unitBox.value())

        self.decayInputBox.append(unitRodPos)
        self.decayInputData.append(unitData)

        self.setLastSpinBox()
        if(self.rowCount() > 3):
            self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

    def deleteDecayTableRow(self):

        nRow = self.rowCount()
        if(nRow==1):
            return
        self.setRowCount(nRow-1)

        tmp = self.decayInputBox.pop(-1)
        tmp = self.decayInputData.pop(-1)

        self.setLastSpinBox()
        if(self.rowCount() <= 3):
            self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def setTableSizePolicy(self):
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)