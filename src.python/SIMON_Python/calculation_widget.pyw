import sys
import platform
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import (QCoreApplication, QPropertyAnimation, QDate, QDateTime, QMetaObject, QObject, QPoint, QRect, QSize, QTime, QUrl, Qt, QEvent)
from PyQt5.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont, QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter, QPixmap, QRadialGradient)
from PyQt5.QtWidgets import *
from model import *
import datetime
import os

_STR_DEFAULT_    = "Default"
_STR_MONITOR_    = "Monitor"
_STR_ECC_        = "ECC"
_STR_SDM_        = "SDM"
_STR_AO_CONTROL_ = "AO"
_STR_LIFETIME_   = "LIFETIME"
_STR_COASTDOWN_  = "COASTDOWN"

_POINTER_D_ = "self.ui.tableWidgetDefault"
_POINTER_A_ = "self.ui.tableWidgetAll"
# _POINTER_M_ = "self.ui.tableWidgetM"
# _POINTER_E_ = "self.ui.tableWidgetE"
# _POINTER_S_ = "self.ui.tableWidgetS"
# _POINTER_A_ = "self.ui.tableWidgetA"
# _POINTER_L_ = "self.ui.tableWidgetL"
# _POINTER_C_ = "self.ui.tableWidgetL"

_STR_SUCCESS_    = "SUCC"
_STR_FAIL_       = "FAIL"

_CALC_NONE_       = -1
_CALC_ALL_        = 1

_CALC_DEFAULT_    = 0
_CALC_MONITOR_    = 1
_CALC_ECC_        = 2
_CALC_SDM_        = 3
_CALC_AO_CONTROL_ = 4
_CALC_LIFETIME_   = 5
_CALC_COASTDOWN_  = 6

_NUM_PRINTOUT_ = 9

# STRING_CALC_OPT = [_STR_DEFAULT_,_STR_MONITOR_,_STR_ECC_,_STR_SDM_,_STR_AO_CONTROL_,_STR_LIFETIME_,_STR_COASTDOWN_]
STRING_CALC_OPT = [_STR_DEFAULT_,_STR_MONITOR_,_STR_ECC_,_STR_SDM_,_STR_AO_CONTROL_,_STR_LIFETIME_,_STR_COASTDOWN_]
# _POINTER_LIST_  = [_POINTER_D_,_POINTER_M_,_POINTER_E_,_POINTER_S_,_POINTER_A_,_POINTER_L_,_POINTER_C_]
_POINTER_LIST_  = [_POINTER_D_,_POINTER_A_]


class Settings:

    def __init__(self, db, ui):
        self.ui = ui
        self.db = db
        self.setAllComponents()
        self.createTempUser()
        self.login()

    def setAllComponents(self):

        query = User.select().order_by(-User.last_login, )

        self.ui.username_dropdown.clear()
        for a_user in query:
            self.ui.username_dropdown.addItem(a_user.username)

        self.ui.username_dropdown.currentIndexChanged.connect(self.userChanged)
        self.ui.username_button.clicked.connect(self.setUsername)
        self.ui.working_button.clicked.connect(self.setWorking)
        self.ui.plant_button.clicked.connect(self.setPlantFile)
        self.ui.restart_button.clicked.connect(self.setRestart)
        self.ui.snapshot_button.clicked.connect(self.setSnapshot)

    def userChanged(self, i):
        now = datetime.datetime.now()
        (User.insert(username=self.ui.username_dropdown.currentText(),
                     working_directory=DEFAULT_WORKING,
                     plant_directory=DEFAULT_PLANT,
                     restart_directory=DEFAULT_RESTART,
                     snapshot_directory=DEFAULT_SNAPSHOT,
                     last_login=now)
         .on_conflict(conflict_target=[User.username],
                      preserve=[User.last_login,
                                ], )
         .execute())
        self.login()

    def createTempUser(self):
        now = datetime.datetime.now()
        (User.insert(username=TEMP_USER,
                     working_directory=DEFAULT_WORKING,
                     plant_directory=DEFAULT_PLANT,
                     restart_directory=DEFAULT_RESTART,
                     snapshot_directory=DEFAULT_SNAPSHOT,
                     last_login=now)
         .on_conflict_ignore()
         .execute())

        self.create_folder(DEFAULT_WORKING)
        self.create_folder(DEFAULT_PLANT)
        self.create_folder(DEFAULT_RESTART)
        self.create_folder(DEFAULT_SNAPSHOT)

    def create_folder(self, directory):
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except OSError as e:
                QMessageBox.information(self.ui.working_button,
                                        UNABLE_TO_CREATE_FOLDER,
                                        UNABLE_TO_CREATE_FOLDER+": "+directory,
                                        QMessageBox.Ok,
                                                QtCore.Qt.FramelessWindowHint
                                        )
                return False
        return True
    """
    def textChanged(self):
        try:
            user = User.get(User.username == self.ui.username_edit.text())
            result = QMessageBox.information(self.ui.working_button,
                                    LOGIN_DETECTED,
                                    LOGIN_TO_USER+": "+self.ui.username_edit.text(),
                                    QMessageBox.Yes|QMessageBox.No,
                                    )
            if result == QMessageBox.Yes:
                user.last_login = datetime.datetime.now()
                user.save()
                self.ui.username_button.setText(LOGIN_BUTTON)
                self.login()
        except DoesNotExist:
            self.ui.username_button.setText(SIGNUP_BUTTON)
            pass
    """
    def setUsername(self):
        if self.ui.username_edit.text() != "":
            now = datetime.datetime.now()
            (User.insert(username=self.ui.username_edit.text(),
                         working_directory=DEFAULT_WORKING,
                         plant_directory=DEFAULT_PLANT,
                         restart_directory=DEFAULT_RESTART,
                         snapshot_directory=DEFAULT_SNAPSHOT,
                         last_login=now)
                .on_conflict(conflict_target=[User.username],
                             preserve=[User.last_login,
                                       ],)
                .execute())
            self.login()

    def setWorking(self):
        directory = QFileDialog.getExistingDirectory(self.ui.working_button,"Select Working Directory", self.ui.working_edit.text())
        if directory:
            if os.path.exists(directory):
                directory = directory+"/"
                query = User.select().order_by(-User.last_login, )
                user = query[0]
                user.working_directory = directory
                user.save()
                self.ui.working_edit.setText(directory)

    def setPlantFile(self):
        directory = QFileDialog.getExistingDirectory(self.ui.plant_button, "Select Working Directory", self.ui.plant_edit.text())
        if directory:
            if os.path.exists(directory):
                directory = directory+"/"
                query = User.select().order_by(-User.last_login, )
                user = query[0]
                user.working_directory = directory
                user.save()
                self.ui.plant_edit.setText(directory)

    def setRestart(self):
        directory = QFileDialog.getExistingDirectory(self.ui.restart_button, "Select Working Directory", self.ui.restart_edit.text())
        if directory:
            if os.path.exists(directory):
                directory = directory+"/"
                query = User.select().order_by(-User.last_login, )
                user = query[0]
                user.working_directory = directory
                user.save()
                self.ui.restart_edit.setText(directory)

    def setSnapshot(self):
        directory = QFileDialog.getExistingDirectory(self.ui.snapshot_button, "Select Working Directory", self.ui.snapshot_edit.text())
        if directory:
            if os.path.exists(directory):
                directory = directory+"/"
                query = User.select().order_by(-User.last_login, )
                user = query[0]
                user.working_directory = directory
                user.save()
                self.ui.snapshot_edit.setText(directory)

    def login(self):
        query = User.select().order_by(-User.last_login, )

        user = query[0]

        self.ui.username_edit.setText(user.username)

        self.ui.label_top_info_1.setText('Simulation and Monitoring - ' + user.username)
        self.ui.label_user_icon.setText(user.username[:2].upper())
        self.ui.username_label.setText("Login: " + user.username + " at " + str(user.last_login))

        self.ui.working_edit.setText(user.working_directory)
        self.ui.plant_edit.setText(user.plant_directory)
        self.ui.restart_edit.setText(user.restart_directory)
        self.ui.snapshot_edit.setText(user.snapshot_directory)