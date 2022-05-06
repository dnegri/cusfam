import sys
import platform
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import (QCoreApplication, QPropertyAnimation, QDate, QDateTime, QMetaObject, QObject, QPoint, QRect, QSize, QTime, QUrl, Qt, QEvent)
from PyQt5.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont, QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter, QPixmap, QRadialGradient)
from PyQt5.QtWidgets import *
from model import *
import datetime
import os
import glob

from ui_main_rev18 import Ui_MainWindow
import constants as cs
import utils as ut

from widgets.utils.LoadInputData import LoadInputData as loadInput

from widgets.utils.PySaveMessageBox import PySaveMessageBox, QMessageBoxWithStyle

TEMP_USER = "Temp_User"
ADMIN_USER = "ADMIN"
UNABLE_TO_LOGIN = "Temp User Login Warning"
UNABLE_TO_LOGIN = "Temp User Login Error"
UNABLE_TO_CREATE_FOLDER = "Unable to create folder"
UNABLE_TO_LOGIN = "Temp User Login Error"

UNABLE_TO_FIND_FILE = "File Not Found"
UNABLE_TO_FIND_FILE_MESSAGE = "Unable to find file {}\n" \
                              "with following extension: {}\n" \
                              "go to the working directory?"

UNABLE_TO_FILE_CORRUPTION = "Plant file name corruption"
UNABLE_TO_FILE_CORRUPTION_MESSAGE = "There are corruption in naming following file {}\n" \
                                    "EX) UCN5.PLT, UCN5.FF, UCN5.XS\n" \
                                    "Change the name of the all 3 files to one of followings\n" \
                                    "Available file names are: "

UNABLE_TO_FILE_RESTART_CORRUPTION = "Restart file name corruption"
UNABLE_TO_FILE_RESTART_CORRUPTION_MESSAGE = "There are corruption in naming following file {}\n" \
                                    "EX) UCN513.GMT, UCN5.RFA\n" \
                                    "Change the name of the all 2 files to one of followings\n" \
                                    "Available file names are: "


UNABLE_TO_PLANT_FILE = "Plant file not found"
UNABLE_TO_PLANT_FILE_MESSAGE = "No plant file found in following folder:\n{}"


UNABLE_TO_RESTART_FILE = "Restart file not found"
UNABLE_TO_RESTART_FILE_MESSAGE = "No restart file found in following folder:\n{}"


UNABLE_TO_FIND_FILE_MESSAGE = "Unable to find file {}\n" \
                              "with following extension: {}\n" \
                              "go to the working directory?"

SIGNUP_BUTTON = "Sign Up"
LOGIN_BUTTON = "Login"

SIGN_OUT = "Sign Out"
SIGN_OUT_MESSAGE = "Delete this user? "

LOGIN_DETECTED = "Login"
LOGIN_TO_USER = "Login to this User?"

DEFAULT_WORKING = "c:/simon/working_directory/"
DEFAULT_PLANT = "c:/simon/plant_files/"
DEFAULT_RESTART = "c:/simon/restart_files/"
DEFAULT_CECOR= "c:/simon/cecor_files/"


class Settings:

    def __init__(self, db, ui, ui_Snapshot, calManager):
        self.ui = ui  # type: Ui_MainWindow
        self.db = db
        self.createTempUser()
        self.setAllComponents()
        self.current_user = None
        #tableview not initialized for settings
        self.message_parent_ui = self.ui.plant_dropdown
        self.calManager = calManager
        self.current_plant_file = ""
        self.current_restart_file = ""
        self.loadInput = loadInput(ui_Snapshot)

    def setAllComponents(self):
        self.ui.refreshButton.clicked.connect(self.refreshFiles)
        self.ui.plant_dropdown.activated.connect(self.plantChanged)
        self.ui.restart_dropdown.activated.connect(self.restartChanged)
        self.ui.plant_button.clicked.connect(self.setPlantFile)
        self.ui.restart_button.clicked.connect(self.setRestart)
        self.ui.cecor_output_button.clicked.connect(self.setCecorOutput)
        self.ui.cecor_check_files_button.clicked.connect(self.checkCecorOutput)

        self.ui.plant_edit.setEnabled(False)
        self.ui.restart_edit.setEnabled(False)
        self.ui.cecor_output_edit.setEnabled(False)

    def userChanged(self, i):
        now = datetime.datetime.now()
        (User.insert(username=self.ui.username_dropdown.currentText(),
                     modified=False,
                     working_directory=DEFAULT_WORKING,
                     plant_directory=DEFAULT_PLANT,
                     restart_directory=DEFAULT_RESTART,
                     cecor_directory=DEFAULT_CECOR,
                     last_login=now)
         .on_conflict(conflict_target=[User.username],
                      preserve=[User.last_login,
                                ], )
         .execute())
        self.login()

    def plantChanged(self, i):
        if self.current_user:
            user = self.current_user
        else:
            query = LoginUser.get(LoginUser.username == ADMIN_USER)
            user = query.login_user
        # print("hello")

        #if self.ui.plant_dropdown.currentText() != user.plant_file:
        user.plant_file = self.ui.plant_dropdown.currentText()
        user.save()
        # self.calManager.load(user)

    def restartChanged(self, i):
        if self.current_user:
            user = self.current_user
        else:
            query = LoginUser.get(LoginUser.username == ADMIN_USER)
            user = query.login_user
        # print("hello")
        # if self.ui.restart_dropdown.currentText() != user.restart_file:
        user.restart_file = self.ui.restart_dropdown.currentText()
        user.save()
        # self.calManager.load(user)


    def createTempUser(self):

        temp_username = os.path.basename(os.path.normpath(DEFAULT_WORKING))

        now = datetime.datetime.now()
        (User.insert(username=temp_username,
                     modified=False,
                     working_directory=DEFAULT_WORKING,
                     plant_directory=DEFAULT_PLANT,
                     restart_directory=DEFAULT_RESTART,
                     cecor_directory=DEFAULT_CECOR,
                     last_login=now)
         .on_conflict_ignore()
         .execute())

        user = User.get(User.username == temp_username)
        (LoginUser.insert(username=ADMIN_USER,
                          login_user=user)
         .on_conflict_ignore()
         .execute())

        #self.create_folder(DEFAULT_WORKING)
        self.create_folder(DEFAULT_PLANT)
        self.create_folder(DEFAULT_RESTART)
        self.create_folder(DEFAULT_CECOR)
        #self.create_folder(DEFAULT_SNAPSHOT)

    def create_folder(self, directory):
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except OSError as e:
                msgBox = QMessageBoxWithStyle(self.message_parent_ui)
                msgBox.setWindowTitle(UNABLE_TO_CREATE_FOLDER)
                msgBox.setText(UNABLE_TO_CREATE_FOLDER+": "+directory, )
                msgBox.setStandardButtons(QMessageBox.Ok)
                # msgBox.setWindowFlag(QtCore.Qt.FramelessWindowHint)
                msgBox.setCustomStyle()
                result = msgBox.exec_()
                return False
        return True

    def textChanged(self):
        try:
            user = User.get(User.username == self.ui.username_edit.text())
            self.ui.username_button.setText(LOGIN_BUTTON)
        except DoesNotExist:
            self.ui.username_button.setText(SIGNUP_BUTTON)
            pass

    def setUsername(self):
        if self.ui.username_edit.text() != "":
            now = datetime.datetime.now()

            try:
                user = User.get(User.username == self.ui.username_edit.text())

                admin_user = LoginUser.get(LoginUser.username == ADMIN_USER)
                admin_user.login_user = user
                admin_user.save()

                self.login()

            except DoesNotExist:
                self.current_user = User.create(username=self.ui.username_edit.text(),
                         working_directory=DEFAULT_WORKING,
                         plant_directory=DEFAULT_PLANT,
                         restart_directory=DEFAULT_RESTART,
                         cecor_directory=DEFAULT_CECOR,
                         last_login=now)

                query = User.select().order_by(-User.last_login, )
                self.ui.username_dropdown.clear()
                for a_user in query:
                    self.ui.username_dropdown.addItem(a_user.username)


    def setWorking(self):
        directory = QFileDialog.getExistingDirectory(self.ui.restart_button,
                                                     "Select Working Directory",
                                                     self.ui.working_edit.text(),)
        if directory:
            if os.path.exists(directory):
                directory = directory+"/"

                temp_username = os.path.basename(os.path.normpath(directory))
                try:
                    User.get(User.username == temp_username)
                except DoesNotExist:
                    now = datetime.datetime.now()
                    (User.insert(username=temp_username,
                                modified=False,
                                 working_directory=DEFAULT_WORKING,
                                 plant_directory=DEFAULT_PLANT,
                                 restart_directory=DEFAULT_RESTART,
                                 cecor_directory=DEFAULT_CECOR,
                                 last_login=now)
                     .on_conflict_ignore()
                     .execute())

                user = User.get(User.username == temp_username)
                (LoginUser.insert(username=ADMIN_USER,
                                  login_user=user)
                 .on_conflict(conflict_target=[LoginUser.username],
                                     preserve=[LoginUser.login_user])
                 .execute())

                query = LoginUser.get(LoginUser.username == ADMIN_USER)
                user = query.login_user
                user.working_directory = directory
                user.save()
                self.ui.working_edit.setText(directory)
                self.login()

    def setPlantFile(self):
        directory = QFileDialog.getExistingDirectory(self.ui.plant_button,
                                                     "Select Plant Directory",
                                                     os.path.dirname(os.path.dirname(self.ui.plant_edit.text())),
                                                     )
        if directory:
            if os.path.exists(directory):
                directory = directory+"/"
                query = LoginUser.get(LoginUser.username == ADMIN_USER)
                user = query.login_user
                user.plant_directory = directory
                user.save()
                self.ui.plant_edit.setText(directory)
                self.plantDropdownChanged(user)
                user.save()
                self.plantChanged(0)

    def setRestart(self):
        directory = QFileDialog.getExistingDirectory(self.ui.restart_button,
                                                     "Select Restart Directory",
                                                     os.path.dirname(os.path.dirname(self.ui.restart_edit.text())))
        if directory:
            if os.path.exists(directory):
                directory = directory+"/"
                query = LoginUser.get(LoginUser.username == ADMIN_USER)
                user = query.login_user
                user.restart_directory = directory
                user.save()
                self.ui.restart_edit.setText(directory)
                self.restartDropdownChanged(user)
                user.save()
                self.restartChanged(0)

    def setCecorOutput(self):
        directory = QFileDialog.getExistingDirectory(self.ui.cecor_output_button,
                                                     "Select CECOR Output Directory",
                                                     self.ui.cecor_output_edit.text())
        if directory:
            if os.path.exists(directory):
                directory = directory+"/"
                query = LoginUser.get(LoginUser.username == ADMIN_USER)
                user = query.login_user
                user.cecor_directory = directory
                user.save()
                self.ui.cecor_output_edit.setText(directory)
                self.loadInput.read_cecor_output(user)

    def checkCecorOutput(self):
        for key_plant in cs.DEFINED_PLANTS.keys():
            if cs.DEFINED_PLANTS[key_plant] in self.calManager.plant_name:
                plant_name = key_plant+self.calManager.plant_name[len(cs.DEFINED_PLANTS[key_plant]):]

        cecor_message = self.loadInput.read_cecor_output(self.current_user, plant_name,
                                                             self.calManager.cycle_name)

        if len(cecor_message) > 0:
            QtWidgets.qApp.processEvents()
            msgBox = QMessageBoxWithStyle(self.message_parent_ui)
            msgBox.setWindowTitle(cs.UNABLE_CECOR_OUTPUT)
            msgBox.setText(cecor_message)
            msgBox.setStandardButtons(QMessageBox.Ok)
            msgBox.setCustomStyle()
            result = msgBox.exec_()
            return

        self.loadInput.readSnapshotData()
        [nSnapshot, inputArray] = self.loadInput.returnSnapshot()

    def plantDropdownChanged(self, user=None, with_error=True):
        self.ui.plant_dropdown.clear()
        if not user:
            user = self.current_user
        if user:
            plants, errors = ut.getPlantFiles(user)
            if with_error:
                for error in errors:
                    message = UNABLE_TO_FILE_CORRUPTION_MESSAGE.format(error)
                    for plant in cs.DEFINED_PLANTS:
                        message += " " + plant + ","

                    msgBox = QMessageBoxWithStyle(self.message_parent_ui)
                    msgBox.setWindowTitle(UNABLE_TO_FILE_CORRUPTION)
                    msgBox.setText(message,)
                    msgBox.setStandardButtons(QMessageBox.Ok)
                    # msgBox.setWindowFlag(QtCore.Qt.FramelessWindowHint)
                    msgBox.setCustomStyle()
                    result = msgBox.exec_()


            is_plant_present = False
            for plant in plants:
                self.ui.plant_dropdown.addItem(plant)
                if plant == user.plant_file:
                    is_plant_present = True

            if with_error:
                if len(plants) == 0:

                    msgBox = QMessageBoxWithStyle(self.message_parent_ui)
                    msgBox.setWindowTitle(UNABLE_TO_FILE_CORRUPTION)
                    msgBox.setText(UNABLE_TO_PLANT_FILE_MESSAGE.format(user.plant_directory),)
                    msgBox.setStandardButtons(QMessageBox.Ok)
                    # msgBox.setWindowFlag(QtCore.Qt.FramelessWindowHint)
                    msgBox.setCustomStyle()
                    result = msgBox.exec_()

            if is_plant_present:
                self.ui.plant_dropdown.setCurrentText(user.plant_file)

            user.plant_file = self.ui.plant_dropdown.currentText()
            user.save()

            self.login_main()

    def restartDropdownChanged(self, user=None, with_error=True):

        self.ui.restart_dropdown.clear()

        if not user:
            user = self.current_user

        if user:

            restarts, errors = ut.getRestartFiles(user)
            if with_error:
                for error in errors:
                    msgBox = QMessageBoxWithStyle(self.message_parent_ui)
                    msgBox.setWindowTitle(UNABLE_TO_FILE_RESTART_CORRUPTION)
                    msgBox.setText(error,)
                    msgBox.setStandardButtons(QMessageBox.Ok)
                    msgBox.setCustomStyle()
                    # msgBox.setWindowFlag(QtCore.Qt.FramelessWindowHint)
                    result = msgBox.exec_()

            is_restart_present = False
            for plant in restarts:
                self.ui.restart_dropdown.addItem(plant)
                if plant == user.restart_file:
                    is_restart_present = True

            if with_error:
                if len(restarts) == 0:

                    msgBox = QMessageBoxWithStyle(self.message_parent_ui)
                    msgBox.setWindowTitle(UNABLE_TO_RESTART_FILE)
                    msgBox.setText(UNABLE_TO_RESTART_FILE_MESSAGE.format(user.plant_directory), )
                    msgBox.setStandardButtons(QMessageBox.Ok)
                    msgBox.setCustomStyle()
                    # msgBox.setWindowFlag(QtCore.Qt.FramelessWindowHint)
                    result = msgBox.exec_()

            if is_restart_present:
                self.ui.restart_dropdown.setCurrentText(user.restart_file)

            user.restart_file = self.ui.restart_dropdown.currentText()
            user.save()
            self.login_main()

    def refreshFiles(self):
        self.plantDropdownChanged(with_error=False)
        self.restartDropdownChanged(with_error=False)

    def login(self):
        login_errors = False
        query = LoginUser.get(LoginUser.username == ADMIN_USER)
        user = query.login_user
        #print(user.username)
        #self.ui.username_edit.setText(user.username)
        #self.ui.username_label.setText("Login: " + user.username + " at " + str(user.last_login))

        # self.ui.label_top_info_1.setText('Simulation and Monitoring - ' + user.username)

        #self.ui.working_edit.setText(user.working_directory)
        self.ui.plant_edit.setText(user.plant_directory)
        self.ui.restart_edit.setText(user.restart_directory)
        self.ui.cecor_output_edit.setText(user.cecor_directory)
        #self.ui.snapshot_edit.setText(user.snapshot_directory)

        self.current_user = user
        self.plantDropdownChanged()
        self.restartDropdownChanged()

        restarts, errors = ut.getRestartFiles(user)
        if len(errors) > 0:
            login_errors = True
        restarts, errors = ut.getPlantFiles(user)
        if len(errors) > 0:
            login_errors = True

        return login_errors

    def login_main(self):
        login_errors = False
        query = LoginUser.get(LoginUser.username == ADMIN_USER)
        user = query.login_user
        self.current_user = user

        restarts, errors = ut.getRestartFiles(user)
        if len(errors) > 0:
            login_errors = True
        restarts, errors = ut.getPlantFiles(user)
        if len(errors) > 0:
            login_errors = True
        # self.loadInput.read_cecor_output(user)

        # if self.calManager.user:
        # self.calManager.setKillPocessCondition(user)

        return login_errors
        # self.ui.label_top_info_1.setText('Simulation and Monitoring - ' + user.username)
