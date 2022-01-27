import sys
import platform
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import (QCoreApplication, QPropertyAnimation, QDate, QDateTime, QMetaObject, QObject, QPoint, QRect, QSize, QTime, QUrl, Qt, QEvent)
from PyQt5.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont, QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter, QPixmap, QRadialGradient)
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QMdiArea
import sys
from PyQt5.QtChart import QtCharts
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from model import *
import datetime
import os
import glob
from ui_main_rev15 import Ui_MainWindow
import Definitions as df
import widgets.utils.PyUnitButton as ts

import constants as cs
import utils as ut

from widgets.calculations.calculation_widget import CalculationWidget

import widgets.utils.PyRodPositionBarChart as unitChart
import widgets.utils.PyStartButton as startB

from widgets.utils.PySaveMessageBox import PySaveMessageBox

from widgets.output.axial.axial_plot import AxialWidget
from widgets.output.radial.radial_graph import RadialWidget
from widgets.output.trend.trends_graph import TrendsWidget


class TileWidget(QWidget):
    count = 0

    def __init__(self, parent=None):
        QWidget.__init__(self,parent)
        self.mdi_widget = QMdiArea()
        self.mdi_widget.setBackground(QBrush(QColor(51, 58, 72)))

        self.previousCheckStates = [False, False, False, False]
        self.checkboxes = []

        radial_text = "Radial"
        axial_text = "Axial"
        trend_text = "Trend"
        report_text = "Report"

        self.radial = QCheckBox(radial_text)
        self.axial = QCheckBox(axial_text)
        self.trend = QCheckBox(trend_text)
        self.report = QCheckBox(report_text)

        self.checkboxes.append(self.radial)
        self.checkboxes.append(self.axial)
        self.checkboxes.append(self.trend)
        self.checkboxes.append(self.report)

        self.radial.stateChanged.connect(lambda x:self.windowAdd(0))
        self.axial.stateChanged.connect(lambda x:self.windowAdd(1))
        self.trend.stateChanged.connect(lambda x:self.windowAdd(2))
        self.report.stateChanged.connect(lambda x:self.windowAdd(3))

        self.sub_radial = QMdiSubWindow()
        self.sub_radial.setWidget(RadialWidget())
        self.sub_radial.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.sub_radial.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.mdi_widget.addSubWindow(self.sub_radial)

        self.sub_axial = QMdiSubWindow()
        self.sub_axial.setWidget(AxialWidget())
        self.sub_axial.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.sub_axial.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.mdi_widget.addSubWindow(self.sub_axial)

        self.sub_trend = QMdiSubWindow()
        self.sub_trend.setWidget(TrendsWidget())
        self.sub_trend.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.sub_trend.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.mdi_widget.addSubWindow(self.sub_trend)

        self.sub_report = QMdiSubWindow()
        self.sub_report.setWidget(QTextEdit("Report"))
        self.sub_report.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.sub_report.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.mdi_widget.addSubWindow(self.sub_report)

        """
        """
        self.init_ui()

        self.radial.setChecked(True)
        self.axial.setChecked(True)
        self.trend.setChecked(True)
        self.report.setChecked(True)

    def init_ui(self):

        main_layout_h = QHBoxLayout()
        main_layout = QVBoxLayout()

        main_layout_h.addStretch()
        main_layout_h.addWidget(self.radial)
        main_layout_h.addWidget(self.axial)
        main_layout_h.addWidget(self.trend)
        main_layout_h.addWidget(self.report)
        main_layout.addLayout(main_layout_h)
        main_layout.addWidget(self.mdi_widget)

        self.setLayout(main_layout)

    def windowAdd(self, check_index):

        #for check_index, checkbox in enumerate(self.checkboxes):
        if self.checkboxes[check_index].isChecked():
            if check_index == 0:
                self.sub_radial.show()
            elif check_index == 1:
                self.sub_axial.show()
            elif check_index == 2:
                self.sub_trend.show()
            elif check_index == 3:
                self.sub_report.show()
        else:
            if check_index == 0:
                if self.sub_radial:
                    self.sub_radial.hide()
            elif check_index == 1:
                if self.sub_axial:
                    self.sub_axial.hide()
            elif check_index == 2:
                if self.sub_trend:
                    self.sub_trend.hide()
            elif check_index == 3:
                if self.sub_report:
                    self.sub_report.hide()

        self.mdi_widget.tileSubWindows()
