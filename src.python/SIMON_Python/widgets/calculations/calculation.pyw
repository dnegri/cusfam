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
import widgets.utils.PyUnitButton   as ts

import constants as cs
import utils as ut
from widgets.utils.PySaveMessageBox import PySaveMessageBox, QMessageBoxWithStyle

_STR_DEFAULT_    = "Default"
_STR_MONITOR_    = "Monitor"
_STR_ECC_        = "ECC"
_STR_SDM_        = "SDM"
_STR_AO_CONTROL_ = "AO"
_STR_LIFETIME_   = "LIFETIME"
_STR_COASTDOWN_  = "COASTDOWN"

_POINTER_A_ = "self.ui.tableWidgetAll"

_CALC_DEFAULT_    = 0
_CALC_MONITOR_    = 1
_CALC_ECC_        = 2
_CALC_SDM_        = 3
_CALC_AO_CONTROL_ = 4
_CALC_LIFETIME_   = 5
_CALC_COASTDOWN_  = 6

STRING_CALC_OPT = [_STR_DEFAULT_,_STR_MONITOR_,_STR_ECC_,_STR_SDM_,_STR_AO_CONTROL_,_STR_LIFETIME_,_STR_COASTDOWN_]

_NUM_PRINTOUT_ = 4
_MINIMUM_ITEM_COUNT = 40



class RecentCalculationWidget:

    def __init__(self, db, ui, queue):
        super().__init__()
        self.ui = ui  # type: Ui_MainWindow
        self.db = db
        self.queue = queue

        self.checkbox01_Flag = False
        self.checkbox02_Flag = False
        self.checkbox03_Flag = False
        self.checkbox04_Flag = False

        self.ui.tableWidgetAll.horizontalHeader().setVisible(True)
        width = self.ui.Search004_A.width()
        self.ui.Search004_A.setMinimumWidth(width+40)
        self.ui.Search004_B.setMinimumWidth(width+40)
        now = datetime.datetime.now()
        self.ui.Search004_A.setDate(QtCore.QDate(now.year-1, now.month, now.day))
        self.ui.Search004_B.setDate(QtCore.QDate(now.year, now.month, now.day))

        self.ui.pushButton_Reset.clicked.connect(self.load)
        self.ui.pushButton_Search.clicked.connect(self.load)
        self.ui.Search001.textChanged.connect(self.load)
        self.ui.Search003.currentIndexChanged.connect(self.load)
        self.ui.Search004_A.dateChanged.connect(self.load)
        self.ui.Search004_B.dateChanged.connect(self.load)

        self.ui.tableWidgetAll.itemSelectionChanged.connect(self.cell_changed)
        self.input_array = None

    def setAllComponents(self):

        query = User.select().order_by(-User.last_login, )

        self.ui.username_dropdown.clear()
        for a_user in query:
            self.ui.username_dropdown.addItem(a_user.username)

        self.ui.username_dropdown.currentIndexChanged.connect(self.resetData)
        self.ui.username_button.clicked.connect(self.setUsername)
        self.ui.working_button.clicked.connect(self.setWorking)
        self.ui.plant_button.clicked.connect(self.setPlantFile)
        self.ui.restart_button.clicked.connect(self.setRestart)
        self.ui.snapshot_button.clicked.connect(self.setSnapshot)

    def load(self):

        eval(_POINTER_A_).setRowCount(0)
        pointerName = eval(_POINTER_A_)
        _translate = QtCore.QCoreApplication.translate

        self.checkbox01_Flag = False #self.ui.checkBoxSearch01.isChecked()
        self.checkbox02_Flag = False #self.ui.checkBoxSearch02.isChecked()
        self.checkbox03_Flag = False #self.ui.checkBoxSearch03.isChecked()
        self.checkbox04_Flag = False #self.ui.checkBoxSearch04.isChecked()

        #if (self.checkbox03_Flag == True):
        calcName = 'NULL'
        calcType = self.ui.Search003.currentIndex()
        if(calcType==1): calcName = cs.CALCULATION_SDM
        elif(calcType==2): calcName = cs.CALCULATION_ECP
        elif(calcType==3): calcName = cs.CALCULATION_RO
        elif(calcType==4): calcName = cs.CALCULATION_SD

        #set
        inputs_query = []
        if len(self.ui.Search001.text()) > 0:
            self.checkbox01_Flag = True
            inputs_query.append(Calculations.filename % (self.ui.Search001.text()))
            inputs_query.append(Calculations.comments % (self.ui.Search001.text()))

        if self.ui.Search003.currentIndex() != 0:
            self.checkbox03_Flag = True
            inputs_query.append(Calculations.calculation_type == calcName)

        start_datetime = datetime.datetime.combine(self.ui.Search004_A.date().toPyDate(), datetime.datetime.min.time())
        end_datetime = datetime.datetime.combine(self.ui.Search004_B.date().toPyDate(), datetime.datetime.max.time())
        inputs_query.append(Calculations.created_date.between(start_datetime, end_datetime))
        inputs = Calculations.select().where(Calculations.filename % (self.ui.Search001.text()) if len(self.ui.Search001.text()) > 0 else Calculations.created_date.between(start_datetime, end_datetime) |
                                             Calculations.comments % (self.ui.Search001.text()) if len(self.ui.Search001.text()) > 0 else Calculations.created_date.between(start_datetime, end_datetime),
                                             Calculations.calculation_type == calcName if self.ui.Search003.currentIndex() != 0 else Calculations.created_date.between(start_datetime, end_datetime),
                                             Calculations.modified_date.between(start_datetime, end_datetime),
                                             Calculations.filename != cs.RECENT_CALCULATION
                                             ).order_by(-Calculations.modified_date)
        self.input_array = inputs
        # input_array_temp = []
        # for input in inputs:
        #     if input.filename != cs.RECENT_CALCULATION:
        #         input_array_temp.append(input)
        # self.input_array = input_array_temp

        nRow = max(_MINIMUM_ITEM_COUNT, len(inputs))
        if (nRow == 0):
            return
        pointerName = eval(_POINTER_A_)
        _translate = QtCore.QCoreApplication.translate
        pointerName.setRowCount(nRow)
        pointerName.setColumnCount(_NUM_PRINTOUT_)

        for restartIdx in range(nRow):
            item = QtWidgets.QTableWidgetItem()
            pointerName.setVerticalHeaderItem(restartIdx, item)
            # print(item.size())

        # Insert Restart Dataset From TableWidget
        # for restartIdx in range(nRow-1,-1,-1):
        for restartIdx in range(nRow):
            item = QtWidgets.QTableWidgetItem()
            font = QtGui.QFont()
            font.setPointSize(11)
            font.setBold(False)
            font.setWeight(75)
            item.setFont(font)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            # calcOpt = inputs[restartIdx][0]
            if restartIdx < len(inputs):
                # Check if it is a default filename
                if inputs[restartIdx].filename != cs.RECENT_CALCULATION:
                    name = "%s" % (cs.CALCULATION_PRINT_OUT[inputs[restartIdx].calculation_type])
                    item.setText(_translate("Form", name))
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                    # inverseIdx = nRow - restartIdx - 1
                    # pointerName.setItem(inverseIdx, 1, item)
                    pointerName.setItem(restartIdx, 0, item)

                    for columnIdx in range(1, _NUM_PRINTOUT_):
                        item = QtWidgets.QTableWidgetItem()
                        font = QtGui.QFont()
                        font.setPointSize(11)
                        font.setBold(False)
                        font.setWeight(75)
                        item.setFont(font)
                        item.setTextAlignment(QtCore.Qt.AlignCenter)
                        if columnIdx == 1:
                            name = "%s" % (inputs[restartIdx].filename)
                        elif columnIdx == 2:
                            name = "%s" % (inputs[restartIdx].modified_date.strftime("%d/%m/%y %H:%M"))
                        elif columnIdx == 3:
                            name = "%s" % (inputs[restartIdx].comments)

                        item.setText(_translate("Form", name))
                        item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                        # inverseIdx = nRow - restartIdx - 1
                        pointerName.setItem(restartIdx, columnIdx, item)
            else:

                for columnIdx in range(0, _NUM_PRINTOUT_):
                    item = QtWidgets.QTableWidgetItem()
                    pointerName.setItem(restartIdx, columnIdx, item)

        # Resize TableWidget
        pointerName.resizeColumnsToContents()
        pointerName.horizontalHeader().setStretchLastSection(True)

        # pointerName.horizontalHeader().setMinimumHeight(200)
        pointerName.verticalHeader().setStretchLastSection(True)
        pointerName.setColumnWidth(5, 160)
        for i in range(_NUM_PRINTOUT_):
            if (i == 5):
                continue
            pointerName.horizontalHeader().setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)

    def resetData(self):
        eval(_POINTER_A_).setRowCount(0)

        self.ui.Search001 = ""
        #self.ui.Search002 = ""
        self.ui.Search003.setCurrentIndex(0)

        now = datetime.datetime.now()
        self.ui.Search004_A.setDate(QtCore.QDate(now.year-1, now.month, now.day))
        self.ui.Search004_B.setDate(QtCore.QDate(now.year, now.month, now.day))
        self.load()

    def cell_changed(self):
        model_index = self.ui.tableWidgetAll.selectedIndexes()
        if len(model_index) > 0:
            row = model_index[-1].row()
            if self.input_array:
                if row < len(self.input_array):
                    name = cs.CALCULATION_PRINT_OUT[self.input_array[row].calculation_type]
                    # result = QMessageBox.information(self.ui.tableWidgetAll,
                    #                         MESSAGE_TITLE.format(name),
                    #                         MESSAGE_CONTENT.format(name),
                    #                         QMessageBox.Ok,
                    #                         QMessageBox.Cancel,
                    #                         )
                    #
                    # if result == QMessageBox.Ok:
                    #     self.queue.put([df.CalcOpt_RECENT, row])

                    msgBox = QMessageBoxWithStyle(self.ui.tableWidgetAll)
                    msgBox.setWindowTitle(cs.MESSAGE_LOAD_INPUT_TITLE.format(name))
                    msgBox.setText(cs.MESSAGE_LOAD_INPUT_CONTENT.format(name))
                    msgBox.setStandardButtons(QMessageBox.Discard | QMessageBox.Open | QMessageBox.Cancel )
                    msgBox.setCustomStyle()
                    #msgBox.setWindowFlag(QtCore.Qt.FramelessWindowHint)
                    result = msgBox.exec_()

                    if result == QMessageBox.Open:
                        self.queue.put([df.CalcOpt_RECENT, row])
                    elif result == QMessageBox.Discard:
                        msgBox = QMessageBoxWithStyle(self.ui.tableWidgetAll)
                        msgBox.setWindowTitle(cs.MESSAGE_DISCARD_CONFIRM_TITLE.format(name))
                        msgBox.setText(cs.MESSAGE_DISCARD_CONFIRM_CONTENT.format(name))
                        msgBox.setStandardButtons(QMessageBox.Discard | QMessageBox.Cancel)
                        msgBox.setCustomStyle()
                        # msgBox.setWindowFlag(QtCore.Qt.FramelessWindowHint)
                        result = msgBox.exec_()
                        if result == QMessageBox.Discard:
                            if self.input_array[row].filename != cs.RECENT_CALCULATION:
                                ut.delete_calculations(self.input_array[row])

                                self.load()

                                msgBox1 = QMessageBoxWithStyle(self.ui.tableWidgetAll)
                                msgBox1.setWindowTitle(cs.MESSAGE_DISCARD_COMPLETE_TITLE.format(name))
                                msgBox1.setText(cs.MESSAGE_DISCARD_COMPLETE_CONTENT.format(name))
                                msgBox1.setStandardButtons(QMessageBox.Ok)
                                msgBox1.setCustomStyle()
                                msgBox1.exec_()
