from PyQt5 import QtCore
from PyQt5.QtCore import (QCoreApplication, QSize, pyqtSlot)
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
import datetime

import Definitions as df
import widgets.utils.PySnapshotTableWidget as table01

# import datetime
# import Definitions as df
# import constants as cs
#
# from model import *
# #from widgets.calculations.calculation_widget import CalculationWidget
#
# import widgets.utils.PyUnitButton as ts
#
# #import ui_unitWidget_SDM_report1 as reportWidget1
# #import ui_unitWidget_SDM_report2 as reportWidget2
#
# import widgets.utils.PySnapshotTableWidget as table01

class SnapshotPopupWidget(QDialog):

    def __init__(self,ui):
        super().__init__()

        self.setWindowModality(Qt.ApplicationModal)

        self.ui = ui

        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(17)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(0)
        self.shadow.setColor(QColor(0, 0, 0, 150))
        self.ui.frame_content.setGraphicsEffect(self.shadow)

        self.ui.setupUi(self)
        self.localizeSnapshotWidget()

        self.insertTableWidget()

        self.currentIndex = -1

        #
        #
        # width = self.ui.Search004_A.width()
        # self.ui.Search004_A.setMinimumWidth(width+40)
        # self.ui.Search004_B.setMinimumWidth(width+40)
        # #self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        # self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint)
        # self.ui.Search004_A.setMinimumWidth(width+40)
        # self.ui.Search004_B.setMinimumWidth(width+40)
        # now = datetime.datetime.now()
        # self.ui.Search004_A.setDate(QtCore.QDate(now.year-1, now.month, now.day))
        # self.ui.Search004_B.setDate(QtCore.QDate(now.year, now.month, now.day))
        #
        # # # 03. Insert Table
        # self.tableItem = ["Calculation\nOption", "Burnup\n(MWD/MTU)", "Power\n(%)", "Bank P\n(cm)", "Bank 5\n(cm)",
        #                   "Bank 4\n(cm)", "Bank 3\n(cm)", "Calculation Time" ]
        # self.tableItemFormat = ["%s"  ,"%.1f","%.2f","%.1f","%.1f",
        #                         "%.1f","%.1f","%s"]
        # self.Snapshot_TableWidget = table01.SnapshotTableWidget(self.ui.frametableWidgetAll, self.tableItem,self.tableItemFormat,7,0,0)
        # self.layoutTableButton = self.Snapshot_TableWidget.returnButtonLayout()
        # self.ui.gridLayout_Snapshot.addWidget(self.Snapshot_TableWidget, 0, 0, 1, 1)
        # self.ui.gridLayout_Snapshot.addLayout(self.layoutTableButton, 1, 0, 1, 1)
        # [self.unitButton01, self.unitButton02, self.unitButton03] = self.Snapshot_TableWidget.returnTableButton()
        #
        # # TODO SGH Make Snapshot Read and Write Model
        # self.inputArray = [["Default", 12000.0, 100.0, 350.0, 250.0, 300.0, 381.0, "2022-02-04 15:33:11"],
        #                    ["Default",  8000.0,  70.0, 260.0, 190.5, 220.0, 289.0, "2022-02-05 08:23:43"],
        #                    ["Default", 13050.0, 100.0, 291.0, 381.0, 381.0, 381.0, "2022-02-01 01:11:19"],
        #                    ["Default",  6500.0,   0.0,   0.0,   0.0,   0.0,   0.0, "2022-02-08 13:51:34"],
        #                    ["Default", 13030.0, 100.0, 200.0, 190.5, 293.0, 381.0, "2022-02-11 09:40:19"]]
        # self.Snapshot_TableWidget.addInputArray(self.inputArray)
        #
        # self.unitButton01.clicked['bool'].connect(self.importDataSet)
        # self.unitButton02.clicked['bool'].connect(self.clickSaveAsExcel)
        self.unitButton02.clicked['bool'].connect(self.selectDataSet)
        self.unitButton03.clicked['bool'].connect(self.rejected)
        #
        self.Snapshot_TableWidget.cellClicked.connect(self.updateClick)
        self.Snapshot_TableWidget.doubleClicked.connect(self.table_double_clicked)


        #self.show()

    def localizeSnapshotWidget(self):

        # 01. Set Window Flag
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint)

        self.ui.gridLayout.setContentsMargins(4, 4, 4, 4)
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
        self.Snapshot_TableWidget.changeButtonText(u"Select",u"Cancel")

        self.unitButton01.hide()
        #self.unitButton02.hide()
        self.unitButton02.setDisabled(True)
        self.unitButton02.setStyleSheet(df.styleSheet_Run)
        self.unitButton02.setMinimumSize(QtCore.QSize(150, 50))
        font = QFont()
        font.setFamilies([u"Segoe UI"])
        font.setPointSize(14)
        self.unitButton02.setFont(font)
        self.unitButton03.setMinimumSize(QtCore.QSize(100, 50))
        self.unitButton03.setFont(font)
        #self.unitButton02.setStyleSheet(df.styleSheet_Create_Scenarios)

        #self.unitButton03.setDisabled(True)
        # print("delete")

        # # TODO SGH Make Snapshot Read and Write Model
        # self.inputArray = [["Default", 12000.0, 100.0, 350.0, 250.0, 300.0, 381.0, "2022-02-04 15:33:11"],
        #                    ["Default",  8000.0,  70.0, 260.0, 190.5, 220.0, 289.0, "2022-02-05 08:23:43"],
        #                    ["Default", 13050.0, 100.0, 291.0, 381.0, 381.0, 381.0, "2022-02-01 01:11:19"],
        #                    ["Default",  6500.0,   0.0,   0.0,   0.0,   0.0,   0.0, "2022-02-08 13:51:34"],
        #                    ["Default", 13030.0, 100.0, 200.0, 190.5, 293.0, 381.0, "2022-02-11 09:40:19"]]
        # self.Snapshot_TableWidget.addInputArray(self.inputArray)

    def updatePopup(self,inputArray):
        self.inputArray = inputArray
        self.Snapshot_TableWidget.addInputArray(self.inputArray)

    def importDataSet(self):
        pass

    def clickSaveAsExcel(self):
        pass

    def updateClick(self,row,col):
        self.unitButton02.setDisabled(False)
        #self.unitButton03.setDisabled(False)


        currentClickR = row
        currentClickC = col
        # print(currentClickR, currentClickC)
        self.currentIndex = row
        #self.checkBoxGroup[row].setChecked(True)


    def selectDataSet(self):
        #print(self.currentIndex)
        # print(self.currentIndex)
        # print(self.inputArray[0:self.currentIndex+1])
        self.accept()
        #self.currentIndex = -1
        #self.unitButton03.setDisabled(True)
        #tmp = QDialog.accept()
        #self.link = QtCore.QObject.connect()


        pass
    def rejected(self):
        self.reject()

    def getSnapshotDataset(self):
        return self.currentIndex #, self.inputArray[0:self.currentIndex+1]

    def table_double_clicked(self):
        self.selectDataSet()
