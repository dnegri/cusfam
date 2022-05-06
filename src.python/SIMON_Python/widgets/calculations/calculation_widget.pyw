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

from widgets.utils.PySaveMessageBox import PySaveMessageBox, QMessageBoxWithStyle

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

        self.snapshotArray = []
        # self.current_calculation.
        self.is_run_selected = False

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

    def save_snapshot(self):
        if len(self.snapshotArray) > 0:
            snap_length = len(self.snapshotArray)
            storage_text = "{:d},".format(snap_length)
            for values in self.snapshotArray:
                for value in values:
                    storage_text += "{:.3f},".format(value)
            snapshot_input = self.get_input(self.current_calculation)
            snapshot_input.snapshot_table = storage_text
            snapshot_input.save()

    def save_output(self, ouptuts, ):

        if len(ouptuts) == 0:
            return

        # storage_text = "{:d},".format(len(sd_outputs))

        p1ds = ouptuts[0][df.asi_o_p1d]
        p2ds = ouptuts[0][df.asi_o_p2d]
        p1d_length = len(p1ds)
        p2d_length = len(p2ds)
        storage_text = "{:d},{:d},{:d},".format(len(ouptuts), p1d_length, p2d_length)

        for sd_output in ouptuts:
            asi = sd_output[df.asi_o_asi]
            ppm = sd_output[df.asi_o_boron]
            fr = sd_output[df.asi_o_fr]
            fxy = sd_output[df.asi_o_fxy]
            fq = sd_output[df.asi_o_fq]
            rP = sd_output[df.asi_o_bp]
            r5 = sd_output[df.asi_o_b5]
            r4 = sd_output[df.asi_o_b4]
            r3 = sd_output[df.asi_o_b3]

            storage_text = storage_text+"{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},".format(asi,ppm, fr, fxy, fq, rP, r5, r4, r3)
            p1ds = sd_output[df.asi_o_p1d]
            p2ds = sd_output[df.asi_o_p2d]

            for p1d in p1ds:
                storage_text = storage_text + "{:.3f},".format(p1d)

            for p2d in p2ds:
                for p2d_v in p2d:
                    storage_text = storage_text + "{:.3f},".format(p2d_v)

            hour = sd_output[df.asi_o_time]
            power = sd_output[df.asi_o_power]
            react = sd_output[df.asi_o_reactivity]

            storage_text += "{:.3f},{:.3f},{:.3f},".format(hour, power, react)

        # print(storage_text)
        self.get_output(self.current_calculation).success = True
        self.get_output(self.current_calculation).table = storage_text
        self.get_output(self.current_calculation).save()
        self.current_calculation.save()

    def load_snapshot(self, a_calculation=None):

        if not a_calculation:
            a_calculation = self.current_calculation

        if not a_calculation:
            raise("Snapshot Load Error")

        # if len(self.snapshotArray) > 0:
        snapshot_input = self.get_input(a_calculation)
        # print(snapshot_input.snapshot_table)
        if len(snapshot_input.snapshot_table):
            read_table = snapshot_input.snapshot_table
            snapshotArrayStrings = read_table.split(",")
            length_array = int(snapshotArrayStrings[0])

            self.snapshotArray = []
            col_length = 10
            for row in range(length_array):
                self.snapshotArray.append([])
                for col in range(col_length):
                    self.snapshotArray[-1].append(float(snapshotArrayStrings[1+row*col_length+col]))

            for a_a in self.snapshotArray:
                print(a_a)

    def load_output(self, a_calculation=None):

        if not a_calculation:
            a_calculation = self.current_calculation

        if not a_calculation:
            raise("Error")

        if self.get_output(a_calculation).success:
            self.setSuccecciveInput()
            read_table = self.get_output(a_calculation).table
            # print(read_table)
            values = read_table.split(",")
            length_elements = 9
            length_array = int(values[0])
            length_p1d = int(values[1])
            length_p2d = int(values[2])
            # print(len(values))
            total_array_length = length_elements + length_p1d + length_p2d * length_p2d + 3
            start_index = 3

            ro_output = []
            element_array = []
            p1d_array = []
            p2d_array = []
            for _ in range(length_p2d):
                p2d_array.append([])
            # print()
            for element_i in range(len(values)-start_index):
                e_i = start_index + element_i
                # print(element_i, values[e_i], length_array, len(ro_output))
                if element_i%total_array_length < length_elements:
                    element_array.append(float(values[e_i]))

                if length_elements <= element_i%total_array_length < length_elements+length_p1d:
                    if length_elements == element_i%total_array_length:
                        element_array.append(p1d_array)
                    p1d_array.append(float(values[e_i]))

                if length_elements+length_p1d <= element_i%total_array_length < length_elements+length_p1d+length_p2d*length_p2d:
                    if length_elements+length_p1d == element_i%total_array_length:
                        element_array.append(p2d_array)
                    # print("row", element_i, total_array_length, length_p2d)
                    row = ((element_i-(length_elements+length_p1d)) % total_array_length) // length_p2d
                    # print("row", row)
                    p2d_array[row].append(float(values[e_i]))

                if element_i%total_array_length >= length_elements + length_p1d+length_p2d * length_p2d:
                    element_array.append(float(values[e_i]))

                # print(element_i%total_array_length, total_array_length-2)
                if element_i%total_array_length == total_array_length-1:
                    ro_output.append(element_array)
                    if length_array == len(ro_output):
                        break
                    element_array = []
                    p1d_array = []
                    p2d_array = []
                    for _ in range(length_p2d):
                        p2d_array.append([])

            self.set_manager_output(ro_output)
            self.clearOutput()
            self.showOutput()

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
                        name = "%s" % inputs[restartIdx].user.username
                    elif columnIdx == 2:
                        name = "%s" % inputs[restartIdx].created_date.strftime("%d/%m/%y %H:%M")
                    elif columnIdx == 3:
                        name = "%s" % inputs[restartIdx].modified_date.strftime("%d/%m/%y %H:%M")
                    elif columnIdx == 4:
                        name = "%s" % inputs[restartIdx].comments

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

    def create_default_input_output(self, user):
        return None, None, None

    def load(self, a_calculation=None):
        pass

    def clear_input_output(self, user):
        calculation_object, input_object = ut.get_default_input(user, self.input_type)

    def load_input(self, a_calculation=None):

        # Input file specific load
        if a_calculation:
            calculation_object = a_calculation
            input_object = self.get_input(calculation_object)
        else:
            if self.current_calculation:
                input_object = self.get_input(self.current_calculation)
                self.set_all_component_values(input_object)
                return input_object

            query = LoginUser.get(LoginUser.username == cs.ADMIN_USER)
            user = query.login_user
            calculation_object, input_object = ut.get_default_input(user, self.input_type)

        if not calculation_object:
            calculation_object, input_object, output_object = self.create_default_input_output(user)
            calculation_object.save()
            input_object.save()
            output_object.save()

        # print(input_object.calculation_type)
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
                elif isinstance(component, QPushButton):
                    component.setText(value)
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

    def get_output_pointers(self):
        return ["success", "table"]

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
            output_old = self.get_output(calcultion_old)

            calcultion_new, input_new, output_new = self.create_default_input_output(user)

            calcultion_new.filename = msgBox.lineEdit1.text()
            calcultion_new.comments = msgBox.lineEdit2.text()

            calcultion_new.save()

            for pointer_key in self.ui_pointers:
                exec("input_new.{} = input_old.{}".format(self.ui_pointers[pointer_key], self.ui_pointers[pointer_key]))

            input_new.save()

            for pointer_key in self.get_output_pointers():
                exec("output_new.{} = output_old.{}".format(pointer_key, pointer_key))

            output_new.save()

            msgBox1 = QMessageBoxWithStyle(self.get_ui_component())
            msgBox1.setWindowTitle(cs.MESSAGE_SAVE_COMPLETE_TITLE.format(calcultion_new.filename))
            msgBox1.setText(cs.MESSAGE_SAVE_COMPLETE_CONTENT.format(calcultion_new.filename))
            msgBox1.setStandardButtons(QMessageBox.Ok)
            msgBox1.setCustomStyle()
            msgBox1.exec_()

            self.load()

            # self.load_recent_calculation()
        elif msgBox.result == msgBox.DELETE:

            if self.current_calculation.filename == cs.RECENT_CALCULATION:
                msgBox1 = QMessageBoxWithStyle(self.get_ui_component())
                msgBox1.setWindowTitle("Delete Error")
                msgBox1.setText("Temp file cannot be deleted.\nChange {} to different one".format(cs.RECENT_CALCULATION))
                msgBox1.setStandardButtons(QMessageBox.Ok)
                msgBox1.setCustomStyle()
                msgBox1.exec_()
                return

            msgBox = QMessageBoxWithStyle(self.get_ui_component())
            msgBox.setWindowTitle(cs.MESSAGE_DISCARD_CONFIRM_TITLE.format(self.current_calculation.filename))
            msgBox.setText(cs.MESSAGE_DISCARD_CONFIRM_CONTENT.format(self.current_calculation.filename))
            msgBox.setStandardButtons(QMessageBox.Discard | QMessageBox.Cancel)
            # msgBox.setWindowFlag(QtCore.Qt.FramelessWindowHint)
            msgBox.setCustomStyle()
            result = msgBox.exec_()

            if result == msgBox.Discard:
                name = self.current_calculation.filename
                input_deleted = self.get_input(self.current_calculation)
                calculation_deleted = self.current_calculation
                input_deleted.delete_instance()
                calculation_deleted.delete_instance()

                msgBox1 = QMessageBoxWithStyle(self.get_ui_component())
                msgBox1.setWindowTitle(cs.MESSAGE_DISCARD_COMPLETE_TITLE.format(name))
                msgBox1.setText(cs.MESSAGE_DISCARD_COMPLETE_CONTENT.format(name))
                msgBox1.setStandardButtons(QMessageBox.Ok)
                msgBox1.setCustomStyle()
                msgBox1.exec_()

            # self.load_recent_calculation()
        pass

    def start_calc(self):
        self.save_input()

        if self.is_run_selected:
            return

        self.is_run_selected = True


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

    def get_output(self, calculation_object):
        return None

    def set_manager_output(self, output):
        return None

    def get_manager_output(self):
        return None

    def select_current_input(self):
        self.table_ui.clearSelection()
        for calculation_index, calculation_object in enumerate(self.table_objects):
            if calculation_object == self.current_calculation:
                self.table_ui.selectRow(calculation_index)
                break

    def killManagerProcess(self):
        self.calcManager.restartProcess()

    def delete_current_calculation(self):
        self.load_input()
        try:
            ut.delete_calculations(self.current_calculation)
        except:
            pass

        # if self.current_calculation.ecp_input:
        #     self.current_calculation.ecp_input.delete_instance(recursive=False)
        # if self.current_calculation.sd_input:
        #     self.current_calculation.sd_input.delete_instance(recursive=False)
        # if self.current_calculation.ro_input:
        #     self.current_calculation.ro_input.delete_instance(recursive=False)
        # if self.current_calculation.sdm_input:
        #     self.current_calculation.sdm_input.delete_instance(recursive=False)
        # if self.current_calculation.ecp_output:
        #     self.current_calculation.ecp_output.delete_instance(recursive=False)
        # if self.current_calculation.sd_output:
        #     self.current_calculation.sd_output.delete_instance(recursive=False)
        # if self.current_calculation.ro_output:
        #     self.current_calculation.ro_output.delete_instance(recursive=False)
        # if self.current_calculation.sdm_output:
        #     self.current_calculation.sdm_output.delete_instance(recursive=False)
        # self.current_calculation.delete_instance(recursive=False)

    def clearOutput(self):
        pass

    def showOutput(self):
        pass

    def clickEvent01(self,event):
        xpos = self.unitChart.clickEvent(event)
        self.unitChart02.annot1.set_visible(False)
        self.unitChart02.canvas.draw()
        self.clickEvent(xpos, self.get_manager_output())

    def clickEvent02(self, event):
        xpos = self.unitChart02.clickEvent(event)
        self.unitChart.annot1.set_visible(False)
        self.unitChart.canvas.draw()
        self.clickEvent(xpos, self.get_manager_output())

    def clickEvent(self, xpos, outputs):

        if (xpos != False):
            # Find matching index for link
            nStep = len(outputs)
            # tolerance = 0.4
            row = 0
            current_time = 0
            print(xpos)
            for idx in range(nStep):

                current_time += outputs[idx][df.asi_o_time]
                if nStep:
                    tolerance = min(outputs[idx][df.asi_o_time] / 2, 1.0)
                else:
                    tolerance = max(min(outputs[idx][df.asi_o_time] / 2, outputs[idx+1][df.asi_o_time] / 2), 0.3)

                tmp = abs(xpos - current_time)
                if (tmp < tolerance):
                    if (idx == nStep - 1):
                        row = idx
                        break
                    else:
                        tmp02 = abs(xpos - (current_time + outputs[idx+1][df.asi_o_time]))
                        if (tmp < tmp02):
                            row = idx
                        else:
                            row = idx + 1
                        break


            #     #print(self.inputArray[idx][0])
            # row = round(xpos)
            print(row)
            # model_index = self.SD_TableWidget.selectedIndexes()
            # if row > 0:
            # row = model_index[-1].row()
            # print(row)
            # print(len(self.calcManager.results.shutdown_output))
            if row < len(outputs):
                pd2d = outputs[row][df.asi_o_p2d]
                pd1d = outputs[row][df.asi_o_p1d]

                p = outputs[row][df.asi_o_bp]
                r5 = outputs[row][df.asi_o_b5]
                r4 = outputs[row][df.asi_o_b4]
                r3 = outputs[row][df.asi_o_b3]

                data = {' P': p, 'R5': r5, 'R4': r4, 'R3': r3}

                self.axialWidget.drawBar(data)

                power = outputs[row][df.asi_o_power]
                # if power == 0:
                #     self.axialWidget.clearAxial()
                #     self.radialWidget.clear_data()
                # else:
                self.axialWidget.drawAxial(pd1d[self.calcManager.results.kbc:self.calcManager.results.kec],
                                           self.calcManager.results.axial_position)
                self.radialWidget.slot_astra_data(pd2d)
        else:
            pass

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