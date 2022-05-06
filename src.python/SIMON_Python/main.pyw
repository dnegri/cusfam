################################################################################
##
## BY: WANDERSON M.PIMENTA
## PROJECT MADE WITH: Qt Designer and PyQt5
## V: 1.0.0
##
## This project can be used freely for all uses, as long as they maintain the
## respective credits only in the Python scripts, any information in the visual
## interface (GUI) can be modified without any implication.
##
## There are limitations on Qt licenses if you want to use your products
## commercially, I recommend reading them on the official website:
## https://doc.qt.io/qtforpython/licenses.html
##
################################################################################

import ctypes
import sys
import os
import platform
import multiprocessing as mp
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import (QCoreApplication, QPropertyAnimation, QDate, QDateTime, QMetaObject, QObject, QPoint, QRect, QSize, QTime, QUrl, Qt, QEvent, pyqtSlot)
from PyQt5.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont, QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter, QPixmap, QRadialGradient)
from PyQt5.QtWidgets import *

# GUI FILE

from app_modules import *
import peewee
from model import *

from widgets.calculations.calculation import RecentCalculationWidget
from widgets.calculations.sdm import SDMWidget
# from widgets.calculations.ecp import ECPWidget
# from widgets.calculations.ecp4 import ECPWidget
from widgets.calculations.ecp import ECPWidget
#from widgets.calculations.lifetime import LifetimeWidget
from widgets.calculations.shutdown import Shutdown_Widget
from widgets.calculations.reoperation import ReoperationWidget
from widgets.calculations.calculationManagerProcess import CalculationManager
from widgets.calculations.snapshot import SnapshotWidget
from widgets.utils.PySnapshotWindow import SnapshotPopupWidget
from settings import Settings

# import ui_unitWidget_ECP_rev3 as unit_ECP
# import ui_unitWidget_ECP_rev4 as unit_ECP
import ui_unitWidget_ECP_revS01 as unit_ECP
import ui_unitWidget_snapshot as unit_Snapshot
#import ui_unitWidget_Lifetime_rev1 as unit_Lifetime
import ui_unitWidget_RO_revS01 as unit_RO
import ui_unitWidget_SD_revS01 as unit_SD
import ui_unitWidget_SDM_revS01 as unit_SDM
import ui_unitWidget_contents as unit_contents
import ui_unitWidget_setting as unit_setting

from PyQt5.QtCore import QPointF, QThread, QProcess

from queue import Queue
import Definitions as df
import constants as cs

from widgets.utils.splash_screen import SplashScreen
import multiprocessing

class MainWindow(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.db = db

        self.db.connect()
        self.db.create_tables([User,
                               LoginUser,
                               Calculations,
                               InputModel,
                               ECP_Input, SD_Input, RO_Input, SDM_Input,
                               ECP_Output, SD_Output, RO_Output, SDM_Output,
                               Cecor_Output,
                               ])

        ## PRINT ==> SYSTEM
        # print('System: ' + platform.system())
        # print('Version: ' +platform.release())

        ########################################################################
        ## START - WINDOW ATTRIBUTES
        ########################################################################

        ## REMOVE ==> STANDARD TITLE BAR
        UIFunctions.removeTitleBar(True)
        ## ==> END ##

        ## SET ==> WINDOW TITLE
        self.setWindowTitle(' ')
        UIFunctions.labelTitle(self, ' ')
        UIFunctions.labelDescription(self, 'Simulation and Monitoring')
        ## ==> END ##

        ## WINDOW SIZE ==> DEFA
        # ULT SIZE
        startSize = QSize(1920, 1080)
        self.resize(startSize)
        self.setMinimumSize(startSize)
        # UIFunctions.enableMaximumSize(self, 500, 720)
        ## ==> END ##
        self.unitWidgetFlag = False
        self.unitWidget_old = None
        self.unitWidgetLay = QtWidgets.QVBoxLayout(self.ui.frameUnitContent)

        ## ==> CREATE MENUS
        ########################################################################

        ## ==> TOGGLE MENU SIZE
        self.ui.btn_toggle_menu.clicked.connect(lambda: UIFunctions.toggleMenu(self, 220, True))
        ## ==> END ##

        ## ==> ADD CUSTOM MENUS
        #self.ui.stackedWidget.setMinimumWidth(20)
        self.recentCalculation = UIFunctions.addNewMenu(self, cs.CALCULATION_RC_TITLE, cs.CALCULATION_RC_BUTTON, "url(:/24x24/icons/24x24/cil-star.png)", True)
        # self.snapchotButton = UIFunctions.addNewMenu(self, cs.CALCULATION_SNAPSHOT_TITLE, cs.CALCULATION_SNAPSHOT_BUTTON, u":/24x24/icons/24x24/cil-view-stream.png", True)
        self.sdmButton = UIFunctions.addNewMenu(self, cs.CALCULATION_SDM_TITLE, cs.CALCULATION_SDM_BUTTON, "url(:/24x24/icons/24x24/cil-fullscreen-exit.png)", True)
        self.ecpButton = UIFunctions.addNewMenu(self, cs.CALCULATION_ECP_TITLE, cs.CALCULATION_ECP_BUTTON, "url(:/24x24/icons/24x24/cil-chevron-bottom.png)", True)
        self.roButton = UIFunctions.addNewMenu(self, cs.CALCULATION_PA_TITLE, cs.CALCULATION_PA_BUTTON, "url(:/24x24/icons/24x24/cil-arrow-circle-top.png)", True)
        self.sdButton = UIFunctions.addNewMenu(self, cs.CALCULATION_PR_TITLE, cs.CALCULATION_PR_BUTTON, "url(:/24x24/icons/24x24/cil-arrow-circle-bottom.png)", True)
        self.settingsButton = UIFunctions.addNewMenu(self, cs.CALCULATION_SETTINGS_TITLE, cs.CALCULATION_SETTINGS_BUTTON, "url(:/24x24/icons/24x24/cil-settings.png)", False)

        self.sdButton.clicked['bool'].connect(self.click_sd_button)
        self.roButton.clicked['bool'].connect(self.click_ro_button)
        self.ecpButton.clicked['bool'].connect(self.click_ecp_button)

        ## ==> END ##

        ## ==> MOVE WINDOW / MAXIMIZE / RESTORE
        ########################################################################
        def moveWindow(event):
            # IF MAXIMIZED CHANGE TO NORMAL
            if UIFunctions.returStatus() == 1:
                UIFunctions.maximize_restore(self)

            # MOVE WINDOW
            if event.buttons() == Qt.LeftButton:
                self.move(self.pos() + event.globalPos() - self.dragPos)
                self.dragPos = event.globalPos()
                event.accept()

        # WIDGET TO MOVE
        self.ui.frame_label_top_btns.mouseMoveEvent = moveWindow
        ## ==> END ##

        ## ==> LOAD DEFINITIONS
        ########################################################################
        UIFunctions.uiDefinitions(self)
        ## ==> END ##

        ########################################################################
        ## END - WINDOW ATTRIBUTES
        ############################## ---/--/--- ##############################

        ########################################################################
        #                                                                      #
        ## START -------------- WIDGETS FUNCTIONS/PARAMETERS ---------------- ##
        #                                                                      #
        ## ==> USER CODES BELLOW                                              ##
        ########################################################################

        ## ==> QTableWidget RARAMETERS
        ########################################################################
        #self.ui.tableWidget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        ## ==> END ##

        self.queue = Queue(10)
        self.calcManager = CalculationManager(queue=self.queue)

        self.unitWidget_calc = QWidget()
        self.ui_calc = unit_contents.Ui_unitWidget_Contents()
        self.ui_calc.setupUi(self.unitWidget_calc)
        self.calculation_widget = RecentCalculationWidget(self.db, self.ui_calc, self.queue)

        self.unitWidget_Snapshot = QWidget()
        self.ui_Snapshot = unit_Snapshot.Ui_unitWidget_Contents()
        self.ui_Snapshot.setupUi(self.unitWidget_Snapshot)
        #self.snapshot = SnapshotWidget(self.db, self.ui_Snapshot, self.calcManager, self.queue, self.ui_calc.tableWidgetAll)
        self.snapshot = SnapshotWidget(self.db, self.ui_Snapshot, self.queue)

        self.unitWidget_setting = QWidget()
        self.ui_setting = unit_setting.Ui_Form()
        self.ui_setting.setupUi(self.unitWidget_setting)
        self.settings_widget = Settings(self.db, self.ui_setting, self.ui_Snapshot, self.calcManager)
        # self.settings_widget.createTempUser()

        self.unitWidget_ECP = QWidget()
        self.ui_ECP       = unit_ECP.Ui_unitWidget_ECP()
        self.ui_ECP.setupUi(self.unitWidget_ECP)
        self.ecp_widget = ECPWidget(self.db, self.ui_ECP, self.calcManager,self.queue, self.ui_calc.tableWidgetAll)
        #self.unitWidget_ECP.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling,True)
        #self.unitWidget_ECP.setAttribute(Qt.AA_EnableHighDpiScaling,)

        self.unitWidget_SD = QWidget()
        self.ui_SD = unit_SD.Ui_unitWidget_SD()
        self.ui_SD.setupUi(self.unitWidget_SD)
        self.shutdown_widget = Shutdown_Widget(self.db, self.ui_SD, self.calcManager, self.queue, self.ui_calc.tableWidgetAll)

        self.unitWidget_SDM = QWidget()
        self.ui_SDM = unit_SDM.Ui_unitWidget_SDM()
        self.ui_SDM.setupUi(self.unitWidget_SDM)
        self.sdm_widget = SDMWidget(self.db, self.ui_SDM, self.calcManager, self.queue, self.ui_calc.tableWidgetAll)

        self.unitWidget_RO = QWidget()
        self.ui_RO = unit_RO.Ui_unitWidget_RO()
        self.ui_RO.setupUi(self.unitWidget_RO)
        self.reoperation = ReoperationWidget(self.db, self.ui_RO, self.calcManager, self.queue, self.ui_calc.tableWidgetAll)

        self.snapshot.linkData(self.settings_widget.loadInput)
        #self.snapshotPopup = SnapshotPopupWidget(self.ui_Snapshot)
        self.shutdown_widget.linkInputModule(self.settings_widget.loadInput)
        self.reoperation.linkInputModule(self.settings_widget.loadInput)
        self.ecp_widget.linkInputModule(self.settings_widget.loadInput)
        self.sdm_widget.linkInputModule(self.settings_widget.loadInput)


        ########################################################################
        ## SHOW ==> MAIN WINDOW
        ########################################################################

        self.calcManager.update_working.connect(self.update_worker)
        self.calcManager.finished_working.connect(self.finished_worker)
        self.calcManager.start()

        # self.start_calculation_message = SplashScreen()
        # self.start_calculation_message.killed.connect(self.killManagerProcess)
        # self.start_calculation_message.init_progress(1, 500, is_nano=True)

        ## ==> END ##W
        self.setModule(self.recentCalculation)
        # UIFunctions.toggleMenu(self, 220, True)
        self.show()


    @pyqtSlot(str)
    def update_worker(self, value):
        option = self.calcManager.results.calcOption
        if self.calcManager.calcOption == df.CalcOpt_KILL:

            self.calcManager.setKillOutput()
        elif self.calcManager.results.calcOption == df.CalcOpt_CECOR:
            option = self.calcManager.calcOption

        if option == df.CalcOpt_ASI:
            self.shutdown_widget.showOutput()
        elif option == df.CalcOpt_RO:
            self.reoperation.showOutput()
        elif option == df.CalcOpt_ASI_RESTART:
            self.shutdown_widget.showOutput()
        elif option == df.CalcOpt_RO_RESTART:
            self.reoperation.showOutput()
        elif option == df.CalcOpt_ECP:
            self.ecp_widget.showOutput()
        elif option == df.CalcOpt_INIT:
            if self.start_calculation_message:
                self.start_calculation_message.close()

    @pyqtSlot(str)
    def finished_worker(self, value):
        title = ""
        # print("options", self.calcManager.calcOption)

        if self.calcManager.calcOption == df.CalcOpt_KILL:
            self.calcManager.setKillOutput()
        else:
            option = self.calcManager.results.calcOption
            if option == df.CalcOpt_SDM:
                self.sdm_widget.finished(value)
                title = cs.CALCULATION_SDM_TITLE
            elif option == df.CalcOpt_ASI:
                self.shutdown_widget.finished(value)
                title = cs.CALCULATION_PR_TITLE
            elif option == df.CalcOpt_RO:
                self.reoperation.finished(value)
                title = cs.CALCULATION_PA_TITLE
            elif option == df.CalcOpt_ECP:
                self.ecp_widget.finished(value)
                title = cs.CALCULATION_ECP_TITLE
            elif option == df.CalcOpt_ASI_RESTART:
                self.shutdown_widget.finished(value)
                title = cs.CALCULATION_PR_TITLE
            elif option == df.CalcOpt_RO_RESTART:
                self.reoperation.finished(value)
                title = cs.CALCULATION_PA_TITLE
            elif option == df.CalcOpt_RECENT:
                self.show_input(self.calcManager.row)

        # self.calcManager.setKillOutput()

        # if self.calcManager.calcOption != df.CalcOpt_RECENT and self.calcManager.calcOption != df.CalcOpt_HEIGHT:
        #     msgBox = QMessageBox(self.ui_calc.tableWidgetAll)
        #     msgBox.setWindowTitle("Calculation Finished", )
        #     msgBox.setText("{} calculation finished {}".format(title, value))
        #     msgBox.setStandardButtons(QMessageBox.Ok)
        #     # msgBox.setStyleSheet("QPushButton {\n	border: 2px solid rgb(52, 59, 72);\n}")
        #     result = msgBox.exec_()

    ########################################################################
    ## MENUS ==> DYNAMIC MENUS FUNCTIONS
    ########################################################################

    def show_input(self, row):
        calculation = self.calculation_widget.input_array[row]
        load_module = None
        if cs.CALCULATION_SDM == calculation.calculation_type:
            self.setModule(self.sdmButton)
            load_module = self.sdm_widget
        elif cs.CALCULATION_ECP == calculation.calculation_type:
            self.setModule(self.ecpButton)
            load_module = self.ecp_widget
        elif cs.CALCULATION_RO == calculation.calculation_type:
            self.setModule(self.roButton)
            load_module = self.reoperation
        elif cs.CALCULATION_SD == calculation.calculation_type:
            self.setModule(self.sdButton)
            load_module = self.shutdown_widget

        load_module.load(calculation)

    def Button(self):
        # GET BT CLICKED
        btnWidget = self.sender()
        self.setModule(btnWidget)

    def setModule(self, btnWidget):

        if 220 == self.ui.frame_left_menu.width():
            UIFunctions.toggleMenu(self, 220, True)

        self.settings_widget.login_main()
        if not self.calcManager.check_files(self.settings_widget.current_user):
            btnWidget = self.settingsButton
            # print("settings")
            # msgBox = QMessageBox(btnWidget)
            # msgBox.setWindowTitle("Plant or Restart file not found",)
            # msgBox.setText("Redirecting to Settings",)
            # msgBox.setStandardButtons(QMessageBox.Ok)
            # msgBox.setWindowFlag(QtCore.Qt.FramelessWindowHint)
            # result = msgBox.exec_()
        else:
            if not self.calcManager.initialized:
                self.settings_widget.login()
                self.calcManager.load(self.settings_widget.current_user)

        # if btnWidget != self.settingsButton:
        #     if self.calcManager.is_started:
        #         msgBox = QMessageBox(self.ui_calc.tableWidgetAll)
        #         msgBox.setWindowTitle("Unable to change", )
        #         msgBox.setText("Calculation still in progress")
        #         msgBox.setStandardButtons(QMessageBox.Ok)
        #         result = msgBox.exec_()
        #         return

        if not self.unitWidgetFlag:
            self.unitWidgetLay.setContentsMargins(0,0,0,0)
            self.unitWidget = QWidget()

        UIFunctions.selectStandardMenu(self, btnWidget.objectName())

        # CALCULATIONS
        if btnWidget.objectName() == cs.CALCULATION_RC_BUTTON:
            UIFunctions.resetStyle(self, cs.CALCULATION_RC_BUTTON)
            UIFunctions.labelPage(self, cs.CALCULATION_RC_TITLE)
            UIFunctions.labelDescription(self, cs.CALCULATION_RC_TITLE+"-"+self.calcManager.plant_name+"호기"+self.calcManager.cycle_name+"주기")

            self.calculation_widget.load()
            if not self.unitWidgetFlag:
                self.unitWidgetLay.addWidget(self.unitWidget_calc)
            else:
                self.unitWidget.hide()
                self.unitWidgetLay.replaceWidget(self.unitWidget,self.unitWidget_calc)
            self.unitWidget = self.unitWidget_calc
            self.unitWidgetFlag = True
            self.unitWidget_calc.show()
            self.unitWidgetLay.update()

        # SDM
        if btnWidget.objectName() == cs.CALCULATION_SDM_BUTTON:
            UIFunctions.resetStyle(self, cs.CALCULATION_SDM_BUTTON)
            UIFunctions.labelPage(self, cs.CALCULATION_SDM)
            UIFunctions.labelDescription(self, cs.CALCULATION_SDM+"-"+self.calcManager.plant_name+"호기"+self.calcManager.cycle_name+"주기")

            if not self.unitWidgetFlag:
                self.unitWidgetLay.addWidget(self.unitWidget_SDM)
            else:
                self.unitWidget.hide()
                self.unitWidgetLay.replaceWidget(self.unitWidget,self.unitWidget_SDM)
            self.unitWidget = self.unitWidget_SDM
            self.unitWidgetFlag = True
            self.unitWidget_SDM.show()
            self.unitWidgetLay.update()
            self.sdm_widget.load()

        # ECP
        if btnWidget.objectName() == cs.CALCULATION_ECP_BUTTON:
            UIFunctions.resetStyle(self, cs.CALCULATION_ECP_BUTTON)
            UIFunctions.labelPage(self, cs.CALCULATION_ECP_TITLE)
            UIFunctions.labelDescription(self, cs.CALCULATION_ECP_TITLE+"-"+self.calcManager.plant_name+"호기"+self.calcManager.cycle_name+"주기")

            if not self.unitWidgetFlag:
                self.unitWidgetLay.addWidget(self.unitWidget_ECP)
            else:
                self.unitWidget.hide()
                self.unitWidgetLay.replaceWidget(self.unitWidget,self.unitWidget_ECP)
            self.unitWidget = self.unitWidget_ECP

            self.unitWidgetFlag = True
            self.unitWidget_ECP.show()
            self.unitWidgetLay.update()
            self.ecp_widget.load()


        # shutdown
        if btnWidget.objectName() == cs.CALCULATION_PR_BUTTON:
            UIFunctions.resetStyle(self, cs.CALCULATION_PR_BUTTON)
            UIFunctions.labelPage(self, cs.CALCULATION_PR_TITLE)
            UIFunctions.labelDescription(self, cs.CALCULATION_PR_TITLE+"-"+self.calcManager.plant_name+"호기"+self.calcManager.cycle_name+"주기")
            if not self.unitWidgetFlag:
                self.unitWidgetLay.addWidget(self.unitWidget_SD)
            else:
                self.unitWidget.hide()
                self.unitWidgetLay.replaceWidget(self.unitWidget,self.unitWidget_SD)
            self.unitWidget = self.unitWidget_SD
            self.unitWidgetFlag = True
            self.unitWidget_SD.show()
            self.unitWidgetLay.update()
            self.shutdown_widget.load()

        # reoperation
        if btnWidget.objectName() == cs.CALCULATION_PA_BUTTON:
            UIFunctions.resetStyle(self, cs.CALCULATION_PA_BUTTON)
            UIFunctions.labelPage(self, cs.CALCULATION_PA_TITLE)
            UIFunctions.labelDescription(self, cs.CALCULATION_PA_TITLE+"-"+self.calcManager.plant_name+"호기"+self.calcManager.cycle_name+"주기")
            if(self.unitWidgetFlag==False):
                self.unitWidgetLay.addWidget(self.unitWidget_RO)
            else:
                self.unitWidget.hide()
                self.unitWidgetLay.replaceWidget(self.unitWidget,self.unitWidget_RO)
            self.unitWidget = self.unitWidget_RO
            self.unitWidgetFlag = True
            self.unitWidget_RO.show()
            self.unitWidgetLay.update()
            self.reoperation.load()

        # snapshot
        if btnWidget.objectName() == cs.CALCULATION_SNAPSHOT_BUTTON:
            UIFunctions.resetStyle(self, cs.CALCULATION_SNAPSHOT_BUTTON)
            UIFunctions.labelPage(self, cs.CALCULATION_SNAPSHOT_TITLE)
            UIFunctions.labelDescription(self, cs.CALCULATION_SNAPSHOT_TITLE+"-"+self.calcManager.plant_name+"호기"+self.calcManager.cycle_name+"주기")
            if(self.unitWidgetFlag==False):
                self.unitWidgetLay.addWidget(self.unitWidget_Snapshot)
            else:
                self.unitWidget.hide()
                self.unitWidgetLay.replaceWidget(self.unitWidget,self.unitWidget_Snapshot)
            self.unitWidget = self.unitWidget_Snapshot
            self.unitWidgetFlag = True
            self.unitWidget_Snapshot.show()
            self.unitWidgetLay.update()
#            self.snapshot.load()

        # SETTINGS
        if btnWidget.objectName() == cs.CALCULATION_SETTINGS_BUTTON:
            UIFunctions.resetStyle(self, cs.CALCULATION_SETTINGS_BUTTON)
            UIFunctions.labelPage(self, cs.CALCULATION_SETTINGS_TITLE)
            UIFunctions.labelDescription(self, cs.CALCULATION_SETTINGS_TITLE)
            if(self.unitWidgetFlag==False):
                self.unitWidgetLay.addWidget(self.unitWidget_setting)
            else:
                self.unitWidget.hide()
                self.unitWidgetLay.replaceWidget(self.unitWidget,self.unitWidget_setting)
            self.unitWidget = self.unitWidget_setting
            self.unitWidgetFlag = True
            self.unitWidgetLay.update()
            self.unitWidget_setting.show()
            #self.settings_widget.load()
            self.settings_widget.login()

    def click_sd_button(self):
        tmp = min(self.ui_SD.SD_InputLP_Dframe.height(), self.ui_SD.SD_InputLP_Dframe.width()) * 0.93
        self.shutdown_widget.resizeRadialWidget(tmp)
        #tmp = min(self.ui_SD.SD_InputLP_Dframe.height(), self.ui_SD.SD_InputLP_Dframe.width()) * 0.93
        #self.ui_SD.SD_InputLP_frame.setMaximumSize(QSize(tmp, tmp))

    def click_ro_button(self):
        tmp = min(self.ui_RO.RO_InputLP_Dframe.height(), self.ui_RO.RO_InputLP_Dframe.width()) * 0.93
        self.reoperation.resizeRadialWidget(tmp)
        #self.ui_RO.RO_InputLP_frame.setMaximumSize(QSize(tmp, tmp))

    def click_ecp_button(self):
        tmp = min(self.ui_ECP.ECP_InputLP_Dframe.height(), self.ui_ECP.ECP_InputLP_Dframe.width()) * 0.93
        self.ecp_widget.resizeRadialWidget(tmp)
        # tmp = min(self.ui_ECP.ECP_InputLP_Dframe.height(), self.ui_ECP.ECP_InputLP_Dframe.width()) * 0.93
        # self.ui_ECP.ECP_InputLP_frame.setMaximumSize(QSize(tmp, tmp))

    ## ==> END ##

    ########################################################################
    ## START ==> APP EVENTS
    ########################################################################

    ## EVENT ==> MOUSE DOUBLE CLICK
    ########################################################################
    def eventFilter(self, watched, event):
        pass
        # if watched == self.le and event.type() == QtCore.QEvent.MouseButtonDblClick:
        #     print("pos: ", event.pos())
    ## ==> END ##

    def closeEvent(self, event):
        if self.calcManager.agents[0]:
            self.calcManager.agents[0].kill()
            event.accept()

    ## EVENT ==> MOUSE CLICK
    ########################################################################
    def mousePressEvent(self, event):
        self.dragPos = event.globalPos()
        if event.buttons() == Qt.LeftButton:
            pass
            #print('Mouse click: LEFT CLICK')
        if event.buttons() == Qt.RightButton:
            pass
            #print('Mouse click: RIGHT CLICK')
    ## ==> END ##

    ## EVENT ==> KEY PRESSED
    ########################################################################
    def keyPressEvent(self, event):
        pass
        #print('Key: ' + str(event.key()) + ' | Text Press: ' + str(event.text()))
    ## ==> END ##

    ## EVENT ==> RESIZE EVENT
    ########################################################################
    def resizeEvent(self, event):
        self.resizeFunction()
        return super(MainWindow, self).resizeEvent(event)

    def resizeFunction(self):

        tmp = min(self.ui_SD.SD_InputLP_Dframe.height(), self.ui_SD.SD_InputLP_Dframe.width()) * 0.93
        size = max(300.0, tmp)
        self.shutdown_widget.resizeRadialWidget(size)

        tmp = min(self.ui_RO.RO_InputLP_Dframe.height(), self.ui_RO.RO_InputLP_Dframe.width()) * 0.93
        size = max(300.0, tmp)
        self.reoperation.resizeRadialWidget(size)

        tmp = min(self.ui_ECP.ECP_InputLP_Dframe.height(), self.ui_ECP.ECP_InputLP_Dframe.width()) * 0.93
        size = max(300.0, tmp)
        self.ecp_widget.resizeRadialWidget(size)
        #print('Height: ' + str(self.height()) + ' | Width: ' + str(self.width()))
    ## ==> END ##

    ########################################################################
    ## END ==> APP EVENTS
    ############################## ---/--/--- ##############################

if __name__ == "__main__":
    multiprocessing.freeze_support()
    awareness = ctypes.c_int()
    errorCode = ctypes.windll.shcore.SetProcessDpiAwareness(1)#0,ctypes.byref(awareness))
    #print(errorCode)
    #print(awareness.value)
    #ctypes.windll.shcore.SetProcessDpiAwareness(1)

    #os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "2"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    os.environ["QT_SCALE_FACTOR"] = "1"
    app = QApplication(sys.argv)
    app.setApplicationName("SIMON 0.8")
    # app.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling,True)
    # app.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps,True)
    screen = app.screens()[0]
    dpi = screen.physicalDotsPerInch()

    QtGui.QFontDatabase.addApplicationFont('fonts/segoeui.ttf')
    QtGui.QFontDatabase.addApplicationFont('fonts/segoeuib.ttf')
    #QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling,True)

    window = MainWindow()
    #winH = ctypes.wintypes.HWND(window)
    sys.exit(app.exec_())
