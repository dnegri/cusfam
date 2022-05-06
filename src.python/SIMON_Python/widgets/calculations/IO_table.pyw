
from PyQt5 import QtCore
from PyQt5.QtCore import (QCoreApplication, QSize, pyqtSlot)
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import *
import datetime
import Definitions as df
import constants as cs
import ui_unitWidget_IO as unit_IO_table

from model import *
#from widgets.calculations.calculation_widget import CalculationWidget

import widgets.utils.PyUnitButton as ts

#import ui_unitWidget_SDM_report1 as reportWidget1
#import ui_unitWidget_SDM_report2 as reportWidget2

#import widgets.utils.PySnapshotTableWidget as table01
import widgets.utils.PyShutdownTableWidget as table01

from widgets.utils.splash_screen import SplashScreen
from PyQt5.QtCore import Qt

#class SnapshotWidget:
class table_IO_widget(QDialog):

    def __init__(self, ECP_Flag=False):
        super().__init__()
        self.setWindowModality(Qt.ApplicationModal)
        self.ui = unit_IO_table.Ui_unitWidget_Contents()
        self.ui.setupUi(self)
        self.localizeSnapshotWidget()
        self.inputArray = []
        self.snapshotData = None
        self.ECP_Flag = ECP_Flag

        self.insertTableWidget()
        #self.IO_TableWidget.setRowCount(0)

    def linkData(self, snapshotClass):
        self.snapshotClass = snapshotClass
        self.loadSnapshot()

    # # TODO SGH Make Snapshot Read and Write Model
    # def addInputArray(self,inputArray):

    def open_IO_table(self):
        self.show()

    def localizeSnapshotWidget(self):

        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint)


    def insertTableWidget(self):

        # 01. Initialize horizontal header setting
        if self.ECP_Flag:
            boron = "Boron/React\n(ppm)"
        else:
            boron = "Boron\n(ppm)"
        self.tableItem = ["Time\n(hour)", "Power\n(%)"  ,
                          "ASI",          boron, "Fr", "Fxy", "Fq",
                          "Bank P\n(cm)"     , "Bank 5\n(cm)", "Bank 4\n(cm)", "Bank 3\n(cm)", ]
        self.tableItemFormat = ["%.1f","%.2f",
                                "%.3f","%.1f","%.3f","%.3f","%.3f",
                                "%.1f","%.1f","%.1f","%.1f"]

        # 02. Define Snapshot Table Widget and gridlayout

        #self.IO_TableWidget = table01.SnapshotTableWidget(self.ui.frameTable_IO, self.tableItem,self.tableItemFormat,7,0,0)
        self.IO_TableWidget = table01.ShutdownTableWidget(self.ui.frameTable_IO, self.tableItem,self.tableItemFormat)

        self.layoutTableButton = self.IO_TableWidget.returnButtonLayout()
        self.ui.gridLayout_IO.addWidget(self.IO_TableWidget, 0, 0, 1, 1)
        self.ui.gridLayout_IO.addLayout(self.layoutTableButton, 1, 0, 1, 1)

        # 03. Define Button

        [self.unitButton01, self.unitButton02, self.unitButton03] = self.IO_TableWidget.returnTableButton()

        self.unitButton01.hide()
        #self.unitButton02.show()
        self.unitButton03.show()
        font = QFont()
        font.setFamilies([u"Segoe UI"])
        font.setPointSize(14)
        self.unitButton03.setFont(font)
        #self.IO_TableWidget.setRowCount(0)
        self.unitButton02.setText(QCoreApplication.translate("unitWidget_accept", u"Select", None))
        self.unitButton03.setText(QCoreApplication.translate("unitWidget_reject", u"Close", None))
        #self.unitButton02.setDisabled(True)
        self.unitButton02.setStyleSheet(df.styleSheet_Run)
        self.unitButton02.setMinimumSize(QtCore.QSize(150, 50))

        self.unitButton03.setMinimumSize(QtCore.QSize(100, 50))

        self.unitButton02.clicked['bool'].connect(self.accepted)
        self.unitButton03.clicked['bool'].connect(self.rejected)

        self.ui.gridLayout.setContentsMargins(4, 4, 4, 4)

        #self.unitButton03.show()
        #
        #self.IO_TableWidget.cellClicked.connect(self.updateClick)

    def loadSnapshot(self):
        self.inputArray = self.snapshotClass.getSnapshotData()

        self.IO_TableWidget.addInputArray(self.inputArray)

    def appendOutputTable(self,shutdown_output,last_update):
        self.IO_TableWidget.appendOutputTable(shutdown_output, last_update)


    def accepted(self):
        print("done")
        self.accept()

    def rejected(self):
        self.reject()