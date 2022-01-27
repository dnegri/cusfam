import sys
import platform
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtPrintSupport import QPrinter
from PyQt5.QtCore import (QCoreApplication, QPropertyAnimation, QDate, QDateTime, QMetaObject, QObject, QPoint, QRect,
                            QSize, QTime, QUrl, Qt, QEvent, QFileInfo)
from PyQt5.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont, QFontDatabase, QIcon, QKeySequence,
                           QLinearGradient, QPalette, QPainter, QPixmap, QRadialGradient, QPainter)
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

import widgets.utils.PyRodPositionBarChart as unitChart
import widgets.utils.PyStartButton as startB

import constants as cs
import utils as ut

from widgets.utils.PySaveMessageBox import PySaveMessageBox
#import ui_unitWidget_SDM_report1 as unit_Report1


_NUM_PRINTOUT_ = 5
_MINIMUM_ITEM_COUNT = 40


class CalculationWidget:

    def __init__(self, db, ui, table_ui, input_type, ui_pointers, calcManager, queue=None, message_ui=None):

        self.ui = ui  # type: Ui_MainWindow
        self.db = db
        self.ui_pointers = []

        self.queue = queue

        self.current_calculation = None

        self.tableHeight = []
        self.tableWidth  = []
        self.table_objects = []
        if table_ui:
            self.table_ui = table_ui
        self.input_type = input_type
        self.ui_pointers = ui_pointers

        for key_i, key in enumerate(self.ui_pointers.keys()):

            component = eval("self.ui.{}".format(key))

            if self.ui_pointers[key]:
                if isinstance(component, QComboBox):
                    component.currentIndexChanged.connect(lambda state, x=key: self.index_changed(x))
                elif isinstance(component, QDoubleSpinBox):
                    component.textChanged.connect(lambda state, x=key: self.value_changed(x))
                elif isinstance(component, QDateTimeEdit):
                    component.dateTimeChanged.connect(lambda state, x=key: self.date_time_changed(x))

        # Setup
        #self.table_ui.itemClicked.connect(self.inputSelected)
        
        if table_ui:
            self.table_ui.selectionModel().selectionChanged.connect(self.inputSelected)

            self.table_ui.doubleClicked.connect(self.saveFunc)

        self.calcManager = calcManager

        self.last_table_created = datetime.datetime.now()

        self.cal_start = False
        self.message_ui = message_ui

        self.start_calculation_message = None

    def index_changed(self, key):
        if self.current_calculation:
            exec("self.get_input(self.current_calculation).{} = self.ui.{}.currentText()".format(self.ui_pointers[key], key))
            self.current_calculation.modified_date = datetime.datetime.now()


    def value_changed(self, key):
        if self.current_calculation:
            exec("self.get_input(self.current_calculation).{} = self.ui.{}.value()".format(self.ui_pointers[key], key))
            self.current_calculation.modified_date = datetime.datetime.now()


    def date_time_changed(self, key):
        if self.current_calculation:
            exec("self.get_input(self.current_calculation).{} = self.ui.{}.dateTime().toPyDateTime()".format(self.ui_pointers[key], key))
            self.current_calculation.modified_date = datetime.datetime.now()


    def save_input(self):
        if self.current_calculation:
            self.get_input(self.current_calculation).save()
            self.current_calculation.save()

    def load_recent_calculation(self):

        inputs = ut.get_all_inputs(self.input_type)

        self.table_objects = inputs

        nRow = max(_MINIMUM_ITEM_COUNT, len(inputs))
        if (nRow == 0):
            return
        pointerName = self.table_ui
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
                name = "%s" % (inputs[restartIdx].filename)
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
                        name = "%s" % (inputs[restartIdx].user.username)
                    elif columnIdx == 2:
                        name = "%s" % (inputs[restartIdx].created_date.strftime("%d/%m/%y %H:%M"))
                    elif columnIdx == 3:
                        name = "%s" % (inputs[restartIdx].modified_date.strftime("%d/%m/%y %H:%M"))
                    elif columnIdx == 4:
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
        #self.setTableWidth(pointerName)

        self.select_current_input()

    def setTableWidth(self, pointerName):
        width = pointerName.verticalHeader().width()
        width += pointerName.horizontalHeader().length()
        if (pointerName.verticalScrollBar().isVisible()):
            width += pointerName.verticalScrollBar().width()
        width += pointerName.frameWidth() * 2

        height = 0
        height += pointerName.verticalHeader().length() + 30
        # height+= pointerName.horizontalHeader().length()
        if (pointerName.horizontalScrollBar().isVisible()):
            height += pointerName.horizontalScrollBar().length()
        height += pointerName.frameWidth() * 2
        # pointerName.setFixedWidth(width)
        pointerName.setFixedHeight(height)

        # width  = pointerName.verticalHeader().width()
        # width += pointerName.frameWidth() * 2
        # width += pointerName.horizontalHeader().length()
        #
        # height = pointerName.frameWidth() * 2
        # height+= pointerName.verticalHeader().width()
        # height += pointerName.horizontalHeader().height()

        self.tableWidth.append(width)
        self.tableHeight.append(height)

    def get_default_input(self):
        pass

    def load(self):
        pass

    def load_input(self, a_calculation=None):

        # Input file specific load
        if a_calculation:
            calculation_object = a_calculation
            input_object = self.get_input(calculation_object)
        else:
            query = LoginUser.get(LoginUser.username == cs.ADMIN_USER)
            user = query.login_user
            calculation_object, input_object = ut.get_last_input(user, self.input_type)
        if not calculation_object:
            calculation_object, input_object = self.get_default_input(user)

        if calculation_object:
            self.current_calculation = calculation_object
            self.set_all_component_values(input_object)

        return input_object

    def set_all_component_values(self, input_object):
        for key in self.ui_pointers.keys():
            component = eval("self.ui.{}".format(key))
            value = eval("input_object.{}".format(self.ui_pointers[key]))

            if self.ui_pointers[key]:
                if isinstance(component, QComboBox):
                    component.setCurrentText(value)
                elif isinstance(component, QDateTimeEdit):
                    component.setDateTime(QDateTime(QDate.fromString(value.strftime("%Y-%m-%d"), "yyyy-MM-dd"),
                                                    QTime.fromString(value.strftime("%H:%M:%S"), "H:m:s")))
                else:
                    component.setValue(float(value))

    def settingAutoExclusive(self):
        pass

    def settingLinkAction(self):
        pass

    def settingInputOpt(self):
        pass

    def settingTargetOpt(self):
        pass

    def get_ui_component(self):
        return self.message_ui

    def start_save(self):


        msgBox = PySaveMessageBox(self.current_calculation, self.get_ui_component())
        msgBox.setWindowTitle("Save Input?")
        msgBox.exec_()

        if msgBox.result == msgBox.SAVE:
            if self.current_calculation.filename == msgBox.lineEdit1.text():
                self.current_calculation.comments = msgBox.lineEdit2.text()
                self.current_calculation.saved = True
                self.current_calculation.save()

            # self.load_recent_calculation()
        elif msgBox.result == msgBox.SAVE_AS:

            query = LoginUser.get(LoginUser.username == cs.ADMIN_USER)
            user = query.login_user

            calcultion_old = self.current_calculation
            input_old = self.get_input(calcultion_old)

            calcultion_new, input_new = self.get_default_input(user)

            calcultion_new.filename = msgBox.lineEdit1.text()
            calcultion_new.comments = msgBox.lineEdit2.text()

            calcultion_new.save()

            for pointer_key in self.ui_pointers:
                exec("input_new.{} = input_old.{}".format(self.ui_pointers[pointer_key], self.ui_pointers[pointer_key]))

            input_new.save()
            self.current_calculation = calcultion_new

            msgBox1 = QMessageBox(self.get_ui_component())
            msgBox1.setWindowTitle(cs.MESSAGE_SAVE_COMPLETE_TITLE.format(calcultion_new.filename))
            msgBox1.setText(cs.MESSAGE_SAVE_COMPLETE_CONTENT.format(calcultion_new.filename))
            msgBox1.setStandardButtons(QMessageBox.Ok)
            msgBox1.exec_()

            self.load()

            # self.load_recent_calculation()
        elif msgBox.result == msgBox.DELETE:


            if self.current_calculation.filename == cs.RECENT_CALCULATION:
                msgBox1 = QMessageBox(self.get_ui_component())
                msgBox1.setWindowTitle("Delete Error")
                msgBox1.setText("Temp file cannot be deleted.\nChange {} to different one".format(cs.RECENT_CALCULATION))
                msgBox1.setStandardButtons(QMessageBox.Ok)
                msgBox1.exec_()
                return

            msgBox = QMessageBox(self.get_ui_component())
            msgBox.setWindowTitle(cs.MESSAGE_DISCARD_CONFIRM_TITLE.format(self.current_calculation.filename))
            msgBox.setText(cs.MESSAGE_DISCARD_CONFIRM_CONTENT.format(self.current_calculation.filename))
            msgBox.setStandardButtons(QMessageBox.Discard | QMessageBox.Cancel)
            # msgBox.setWindowFlag(QtCore.Qt.FramelessWindowHint)
            result = msgBox.exec_()

            if result == msgBox.Discard:
                name = self.current_calculation.filename
                input_deleted = self.get_input(self.current_calculation)
                calculation_deleted = self.current_calculation
                input_deleted.delete_instance()
                calculation_deleted.delete_instance()

                msgBox1 = QMessageBox(self.get_ui_component())
                msgBox1.setWindowTitle(cs.MESSAGE_DISCARD_COMPLETE_TITLE.format(name))
                msgBox1.setText(cs.MESSAGE_DISCARD_COMPLETE_CONTENT.format(name))
                msgBox1.setStandardButtons(QMessageBox.Ok)
                msgBox1.exec_()

            # self.load_recent_calculation()
        pass

    def start_calc(self):
        self.save_input()

    def get_calculation_input(self):
        pass

    def check_calculation_input(self):
        pass

    def inputSelected(self, selected):
        if len(selected.indexes()) > 0:
            index = selected.indexes()[0]
            if index.row() < len(self.table_objects):
                self.load_input(self.table_objects[index.row()])
                """
                msgBox = QMessageBox(self.ui.Lifetime_run_button)
                msgBox.setWindowTitle("Load Input?")
                msgBox.setText("Load this input?")
                msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
                msgBox.setDefaultButton(QMessageBox.Yes)
                result = msgBox.exec_()
                if result == QMessageBox.Yes:
    
                else:
                """
            else:
                self.select_current_input()

    def get_input(self, calculation_object):
        return None

    def select_current_input(self):
        self.table_ui.clearSelection()
        for calculation_index, calculation_object in enumerate(self.table_objects):
            if calculation_object == self.current_calculation:
                self.table_ui.selectRow(calculation_index)
                break

    def printPDF(self, widget_report):
        # fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Export PDF', None, 'PDF files (.pdf);;All Files()')
        # copyFrame = type('copyFrame',self.ui.frameSummary.__bases__, dict(self.ui.frameSummary.__dict__))
        # copy.deepcopy(self.ui.frameSummary)

        # tmpFrame02 = copy.deepcopy(tmpFrame)

        # tmpFrame = self.ui.frameSummary
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(widget_report, 'Export PDF', ".", 'PDF files (*.pdf);; All Files()')
        if fn != '':
            if QFileInfo(fn).suffix() == "": fn += '.pdf'
            # 01. Initialize printer setting
            printer = QPrinter(QPrinter.HighResolution)
            printer.setOrientation(QPrinter.Portrait)
            printer.setPaperSize(QPrinter.A4)
            printer.setPageSize(QPrinter.A4)
            printer.setPageMargins(15,15,15,15,QPrinter.Millimeter)
            printer.setOutputFormat(QPrinter.PdfFormat)
            printer.setOutputFileName(fn)
            printer.newPage()

            # 03. Start QPainter
            painter = QtGui.QPainter()
            painter.begin(printer)

            painter.resetTransform()
            xscale = printer.pageRect().width()  / widget_report.width() #printer.pageRect()
            yscale = printer.pageRect().height() / widget_report.height()
            scale = min(xscale,yscale) * 0.9
            painter.scale(scale,scale)
            xMove = widget_report.width() * 0.05
            painter.translate(xMove, 0)
            widget_report.render(painter)
            # printer.newPage()
            del widget_report

            painter.end()

    def killManagerProcess(self):
        self.calcManager.restartProcess()