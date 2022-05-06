
from PyQt5 import QtCore
from PyQt5.QtCore import (QCoreApplication, QSize, pyqtSlot)
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import *
import datetime
import Definitions as df
import constants as cs

from model import *
#from widgets.calculations.calculation_widget import CalculationWidget

import widgets.utils.PyUnitButton as ts

#import ui_unitWidget_SDM_report1 as reportWidget1
#import ui_unitWidget_SDM_report2 as reportWidget2

import widgets.utils.PySnapshotTableWidget as table01

from widgets.utils.splash_screen import SplashScreen

pointers = {
    "SDM_Input01": "ndr_burnup",
    "SDM_Input02": "ndr_mode_selection",
}


class SnapshotWidget:

    def __init__(self, db, ui, queue):
        super().__init__()
        self.ui = ui  # type: Ui_MainWindow
        self.db = db
        self.queue = queue
        self.inputArray = []
        self.snapshotData = None

        self.localizeSnapshotWidget()

        self.insertTableWidget()


    def linkData(self, snapshotClass):
        self.snapshotClass = snapshotClass
        self.loadSnapshot()

    # def readModel(self):


    # # TODO SGH Make Snapshot Read and Write Model
    # def addInputArray(self,inputArray):
    #
    #
    #     nStep = len(self.inputArray)
    #     self.setRowCount(0)
    #     self.setRowCount(nStep)

    def localizeSnapshotWidget(self):

        # 01. Hide Lower grip frame
        self.ui.frame_grip.hide()

        # # 02. Resize name search Line Edit
        # width = self.ui.Search004_A.width()
        # self.ui.Search004_A.setMinimumWidth(width+40)
        # self.ui.Search004_B.setMinimumWidth(width+40)
        #
        # # 03. Setting Time and Date
        # now = datetime.datetime.now()
        # self.ui.Search004_A.setDate(QtCore.QDate(now.year-1, now.month, now.day))
        # self.ui.Search004_B.setDate(QtCore.QDate(now.year, now.month, now.day))

    def insertTableWidget(self):

        # 01. Initialize horizontal header setting
        self.tableItem = ["Calculation\nOption", "Burnup\n(MWD/MTU)", "Power\n(%)", "Bank P\n(cm)", "Bank 5\n(cm)",
                          "Bank 4\n(cm)", "Bank 3\n(cm)", "Calculation Time" ]
        self.tableItemFormat = ["%s"  ,"%.1f","%.2f","%.1f","%.1f",
                                "%.1f","%.1f","%s"]

        # 02. Define Snapshot Table Widget and gridlayout
        self.Snapshot_TableWidget = table01.SnapshotTableWidget(self.ui.frametableWidgetAll, self.tableItem,self.tableItemFormat,7,0,0)
        self.layoutTableButton = self.Snapshot_TableWidget.returnButtonLayout()
        self.ui.gridLayout_Snapshot.addWidget(self.Snapshot_TableWidget, 0, 0, 1, 1)
        self.ui.gridLayout_Snapshot.addLayout(self.layoutTableButton, 1, 0, 1, 1)

        # 03. Define Button
        [self.unitButton01, self.unitButton02, self.unitButton03] = self.Snapshot_TableWidget.returnTableButton()
        # self.unitButton01.hide()
        # self.unitButton02.hide()


    def loadSnapshot(self):
        self.inputArray = self.snapshotClass.getSnapshotData()
        #
        # # TODO SGH Make Snapshot Read and Write Model
        # self.inputArray = [["Default", 12000.0, 100.0, 350.0, 250.0, 300.0, 381.0, "2022-02-04 15:33:11"],
        #                    ["Default",  8000.0,  70.0, 260.0, 190.5, 220.0, 289.0, "2022-02-05 08:23:43"],
        #                    ["Default", 13050.0, 100.0, 291.0, 381.0, 381.0, 381.0, "2022-02-01 01:11:19"],
        #                    #["Default",  6500.0,   0.0,   0.0,   0.0,   0.0,   0.0, "2022-02-08 13:51:34"],
        #                    ["Default", 13030.0, 100.0, 200.0, 190.5, 293.0, 381.0, "2022-02-11 09:40:19"]]
        self.Snapshot_TableWidget.addInputArray(self.inputArray)