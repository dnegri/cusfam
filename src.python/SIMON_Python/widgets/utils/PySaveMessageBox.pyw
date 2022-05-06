
from PyQt5.QtWidgets import *
from PyQt5.QtGui import ( QFont,)
from PyQt5.QtCore import ( QSize,)
import constants as cs
from PyQt5 import QtCore
import Definitions as df

from PyQt5 import QtCore, QtGui, QtWidgets


# class PySaveMessageBox(QDialog):
#
#     SAVE_AS = "Save as"
#     SAVE = "Save"
#     DELETE = "DELETE"
#
#     def __init__(self, calculation_object, parent=None):
#         QDialog.__init__(self, parent)
#
#         # self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
#         # self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint)
#         self.calculation_object = calculation_object
#         font2 = QFont()
#         font2.setFamily(u"Segoe UI")
#         font2.setPointSize(14)
#         font2.setBold(False)
#         font2.setWeight(50)
#         nameLabel = QLabel("Input name:")
#         font = QtGui.QFont()
#         font.setFamily("Segoe UI")
#         font.setPointSize(14)
#         nameLabel.setFont(font)
#         #nameLabel.setFont(font2)
#         commentLabel = QLabel("Comments:")
#         commentLabel.setFont(font)
#         #commentLabel.setFont(font2)
#         self.lineEdit1 = QLineEdit()
#
#         if self.calculation_object.filename == cs.RECENT_CALCULATION:
#             name = ""
#         else:
#             name = calculation_object.filename
#         self.lineEdit1.setFont(font)
#         self.lineEdit1.setText(name)
#         self.lineEdit1.textChanged.connect(self.filename_changed)
#         self.lineEdit2 = QLineEdit()
#         self.lineEdit2.setFont(font)
#         if self.calculation_object.filename == cs.RECENT_CALCULATION:
#             name = ""
#         else:
#             name = calculation_object.comments
#         self.lineEdit2.setText(name)
#         nameLabel.setBuddy(self.lineEdit1)
#         commentLabel.setBuddy(self.lineEdit2)
#
#         # button_format = u"QPushButton {\n"\
#         #                 "	border: 2px solid rgb(85, 170, 255);\n"\
#         #                                          "	border-radius: 5px;	\n"\
#         #                                          "	color: white;	\n"\
#         #                                          "	background-color: rgb(85, 170, 255);\n"\
#         #                                          "}\n"\
#         #                                          "QPushButton:hover {\n"\
#         #                                          "	background-color: rgb(72, 144, 216);\n"\
#         #                                          "	border: 2px solid rgb(72, 144, 216);\n"\
#         #                                          "}\n"\
#         #                                          "QPushButton:pressed {	\n"\
#         #                                          "	background-color: rgb(52, 59, 72);\n"\
#         #                                          "	border: 2px solid rgb(52, 59, 72);\n"\
#         #                                          "}"\
#         #                                          "QPushButton:disabled {	\n"\
#         #                                          "	background-color: rgb(52, 59, 72);\n"\
#         #                                          "	border: 2px solid rgb(52, 59, 72);\n"\
#         #                                          "}"
#         # delete_button_format = u"QPushButton {\n" \
#         #                        "	border-radius: 5px;	\n" \
#         #                        "	color: white;	\n" \
#         #                        "	background-color: red;\n" \
#         #                        "	border: 2px solid red;\n" \
#         #                        "}\n" \
#         #                        "QPushButton:hover {\n" \
#         #                        "	background-color: rgb(72, 144, 216);\n" \
#         #                        "	border: 2px solid rgb(72, 144, 216);\n" \
#         #                        "}\n" \
#         #                        "QPushButton:pressed {	\n" \
#         #                        "	background-color: rgb(85, 170, 255);\n" \
#         #                        "	border: 2px solid rgb(85, 170, 255);\n" \
#         #                        "}" \
#         #                        "QPushButton:disabled {	\n" \
#         #                        "	background-color: rgb(52, 59, 72);\n" \
#         #                        "	border: 2px solid rgb(52, 59, 72);\n" \
#         #                        "}"
#
#         # button_format = u"QPushButton {\n"\
#         #                                          "	background-color: rgb(85, 170, 255);\n"\
#         #                                          "}\n"
#         delete_button_format = u"QPushButton {\n" \
#                                "	background-color: rgb(170, 0, 0);\n" \
#                                "}\n" \
#
#
#         self.saveButton = QPushButton(self.SAVE)
#         sizePolicy3 = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
#         sizePolicy3.setHeightForWidth(self.saveButton.sizePolicy().hasHeightForWidth())
#         self.saveButton.setSizePolicy(sizePolicy3)
#
#         # self.saveButton.setMinimumSize(QSize(120, 40))
#         # self.saveButton.setMaximumSize(QSize(16777215, 40))
#
#         #self.saveButton.setFont(font2)
#         #self.saveButton.setStyleSheet(button_format)
#         self.saveButton.setGeometry(QtCore.QRect(100, 189, 81, 41))
#         self.saveButton.setMinimumSize(QtCore.QSize(60, 30))
#         self.saveButton.setSizePolicy(sizePolicy3)
#         self.saveButton.setFont(font)
#
#         self.saveButton.setDefault(True)
#         self.saveButton.setStyleSheet(df.styleSheet_Table_Normal)
#         self.saveButton.setFixedWidth(80)
#         # self.saveButton.setSiz
#         self.saveAsButton = QPushButton(self.SAVE_AS)
#         sizePolicy3 = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
#         sizePolicy3.setHeightForWidth(self.saveButton.sizePolicy().hasHeightForWidth())
#
#         # self.saveAsButton.setMinimumSize(QSize(120, 40))
#         # self.saveAsButton.setMaximumSize(QSize(16777215, 40))
#         #self.saveAsButton.setFont(font2)
#
#         #self.saveAsButton.setStyleSheet(button_format)
#
#         self.deleteButton = QPushButton(self.DELETE)
#         sizePolicy3 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
#         sizePolicy3.setHeightForWidth(self.saveButton.sizePolicy().hasHeightForWidth())
#         self.deleteButton.setSizePolicy(sizePolicy3)
#
#         self.deleteButton.setGeometry(QtCore.QRect(100, 189, 81, 41))
#         self.deleteButton.setMinimumSize(QtCore.QSize(60, 30))
#         # self.deleteButton.setSizePolicy(sizePolicy3)
#         font = QtGui.QFont()
#         font.setFamily("Segoe UI")
#         font.setPointSize(14)
#         self.deleteButton.setFont(font)
#         self.deleteButton.setFixedWidth(80)
#
#         # self.deleteButton.setMinimumSize(QSize(120, 40))
#         # self.deleteButton.setMaximumSize(QSize(16777215, 40))
#
#         #self.deleteButton.setFont(font2)
#         self.deleteButton.setStyleSheet(df.styleSheet_Delete_Scenarios)
#
#         cancelButton = QPushButton("Cancel")
#         sizePolicy3 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
#         sizePolicy3.setHeightForWidth(cancelButton.sizePolicy().hasHeightForWidth())
#         cancelButton.setSizePolicy(sizePolicy3)
#         cancelButton.setStyleSheet(df.styleSheet_Table_Normal)
#
#         cancelButton.setGeometry(QtCore.QRect(100, 189, 81, 41))
#         cancelButton.setMinimumSize(QtCore.QSize(60, 30))
#         # self.deleteButton.setSizePolicy(sizePolicy3)
#         cancelButton.setFont(font)
#         cancelButton.setFixedWidth(80)
#         #cancelButton.setMinimumSize(QSize(120, 40))
#
#         #cancelButton.setMaximumSize(QSize(16777215, 40))
#         #cancelButton.setStyleSheet(button_format)
#
#         #cancelButton.setFont(font2)
#         #cancelButton.setStyleSheet(button_format)
#
#         gridLayout = QGridLayout()
#         gridLayout.addWidget(nameLabel, 0, 0)
#         gridLayout.addWidget(self.lineEdit1, 0, 1)
#         gridLayout.addWidget(commentLabel, 1, 0)
#         gridLayout.addWidget(self.lineEdit2, 1, 1)
#
#         buttonLayout = QHBoxLayout()
#         buttonLayout.addStretch(2)
#         buttonLayout.addWidget(self.saveButton)
#         #buttonLayout.addWidget(self.saveAsButton)
#         buttonLayout.addWidget(self.deleteButton)
#         buttonLayout.addWidget(cancelButton)
#
#         mainLayout = QVBoxLayout()
#         mainLayout.addLayout(gridLayout)
#         mainLayout.addLayout(buttonLayout)
#
#         self.setLayout(mainLayout)
#
#         self.saveButton.clicked.connect(self.saveAction)
#         self.saveAsButton.clicked.connect(self.saveAsAction)
#         self.deleteButton.clicked.connect(self.deleteAction)
#         cancelButton.clicked.connect(self.close)
#         self.result = "Cancel"
#         self.filename_changed()
#
#     def filename_changed(self):
#         if self.lineEdit1.text() != self.calculation_object.filename:
#             self.saveButton.setText(self.SAVE_AS)
#         else:
#             self.saveButton.setText(self.SAVE)
#
#     def saveAction(self):
#         name = self.lineEdit1.text().strip()
#         self.lineEdit1.setText(name)
#         if len(name) == 0:
#             name = "Empty"
#             msgBox1 = QMessageBox(self)
#             msgBox1.setWindowTitle(cs.MESSAGE_SAVE_UNSUCCESSFUL_TITLE.format(name))
#             msgBox1.setText(cs.MESSAGE_SAVE_UNSUCCESSFUL_CONTENT.format(name))
#             msgBox1.setStandardButtons(QMessageBox.Ok)
#             msgBox1.exec_()
#             return
#
#         self.result = self.saveButton.text()
#         self.close()
#
#     def deleteAction(self):
#         self.result = self.DELETE
#         self.close()
#
#     def saveAsAction(self):
#         self.result = self.SAVE_AS
#         self.close()
#
#

class PySaveMessageBox(QDialog):

    SAVE_AS = "Save as"
    SAVE = "Save"
    DELETE = "DELETE"

    def __init__(self, calculation_object, parent=None):
        QDialog.__init__(self, parent)

        # self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        # self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint)
        self.calculation_object = calculation_object
        font2 = QFont()
        font2.setFamily(u"Segoe UI")
        font2.setPointSize(14)
        font2.setBold(False)
        font2.setWeight(50)
        nameLabel = QLabel("Input name:")
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(14)
        nameLabel.setFont(font)
        #nameLabel.setFont(font2)
        commentLabel = QLabel("Comments:")
        commentLabel.setFont(font)
        #commentLabel.setFont(font2)
        self.lineEdit1 = QLineEdit()

        if self.calculation_object.filename == cs.RECENT_CALCULATION:
            name = ""
        else:
            name = calculation_object.filename
        self.lineEdit1.setFont(font)
        self.lineEdit1.setText(name)
        self.lineEdit1.textChanged.connect(self.filename_changed)
        self.lineEdit2 = QLineEdit()
        self.lineEdit2.setFont(font)
        if self.calculation_object.filename == cs.RECENT_CALCULATION:
            name = ""
        else:
            name = calculation_object.comments
        self.lineEdit2.setText(name)
        nameLabel.setBuddy(self.lineEdit1)
        commentLabel.setBuddy(self.lineEdit2)

        # button_format = u"QPushButton {\n"\
        #                 "	border: 2px solid rgb(85, 170, 255);\n"\
        #                                          "	border-radius: 5px;	\n"\
        #                                          "	color: white;	\n"\
        #                                          "	background-color: rgb(85, 170, 255);\n"\
        #                                          "}\n"\
        #                                          "QPushButton:hover {\n"\
        #                                          "	background-color: rgb(72, 144, 216);\n"\
        #                                          "	border: 2px solid rgb(72, 144, 216);\n"\
        #                                          "}\n"\
        #                                          "QPushButton:pressed {	\n"\
        #                                          "	background-color: rgb(52, 59, 72);\n"\
        #                                          "	border: 2px solid rgb(52, 59, 72);\n"\
        #                                          "}"\
        #                                          "QPushButton:disabled {	\n"\
        #                                          "	background-color: rgb(52, 59, 72);\n"\
        #                                          "	border: 2px solid rgb(52, 59, 72);\n"\
        #                                          "}"
        # delete_button_format = u"QPushButton {\n" \
        #                        "	border-radius: 5px;	\n" \
        #                        "	color: white;	\n" \
        #                        "	background-color: red;\n" \
        #                        "	border: 2px solid red;\n" \
        #                        "}\n" \
        #                        "QPushButton:hover {\n" \
        #                        "	background-color: rgb(72, 144, 216);\n" \
        #                        "	border: 2px solid rgb(72, 144, 216);\n" \
        #                        "}\n" \
        #                        "QPushButton:pressed {	\n" \
        #                        "	background-color: rgb(85, 170, 255);\n" \
        #                        "	border: 2px solid rgb(85, 170, 255);\n" \
        #                        "}" \
        #                        "QPushButton:disabled {	\n" \
        #                        "	background-color: rgb(52, 59, 72);\n" \
        #                        "	border: 2px solid rgb(52, 59, 72);\n" \
        #                        "}"

        # button_format = u"QPushButton {\n"\
        #                                          "	background-color: rgb(85, 170, 255);\n"\
        #                                          "}\n"
        delete_button_format = u"QPushButton {\n" \
                               "	background-color: rgb(170, 0, 0);\n" \
                               "}\n" \


        self.saveButton = QPushButton(self.SAVE)
        sizePolicy3 = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        sizePolicy3.setHeightForWidth(self.saveButton.sizePolicy().hasHeightForWidth())
        self.saveButton.setSizePolicy(sizePolicy3)

        # self.saveButton.setMinimumSize(QSize(120, 40))
        # self.saveButton.setMaximumSize(QSize(16777215, 40))

        #self.saveButton.setFont(font2)
        #self.saveButton.setStyleSheet(button_format)
        self.saveButton.setGeometry(QtCore.QRect(100, 189, 81, 41))
        self.saveButton.setMinimumSize(QtCore.QSize(60, 30))
        self.saveButton.setSizePolicy(sizePolicy3)
        self.saveButton.setFont(font)

        self.saveButton.setDefault(True)
        self.saveButton.setStyleSheet(df.styleSheet_Table_Normal)
        self.saveButton.setFixedWidth(90)
        # self.saveButton.setSiz
        self.saveAsButton = QPushButton(self.SAVE_AS)
        sizePolicy3 = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        sizePolicy3.setHeightForWidth(self.saveButton.sizePolicy().hasHeightForWidth())

        # self.saveAsButton.setMinimumSize(QSize(120, 40))
        # self.saveAsButton.setMaximumSize(QSize(16777215, 40))
        #self.saveAsButton.setFont(font2)

        #self.saveAsButton.setStyleSheet(button_format)

        self.deleteButton = QPushButton(self.DELETE)
        sizePolicy3 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        sizePolicy3.setHeightForWidth(self.saveButton.sizePolicy().hasHeightForWidth())
        self.deleteButton.setSizePolicy(sizePolicy3)

        self.deleteButton.setGeometry(QtCore.QRect(100, 189, 81, 41))
        self.deleteButton.setMinimumSize(QtCore.QSize(60, 30))
        # self.deleteButton.setSizePolicy(sizePolicy3)
        font = QtGui.QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(14)
        self.deleteButton.setFont(font)
        self.deleteButton.setFixedWidth(90)

        # self.deleteButton.setMinimumSize(QSize(120, 40))
        # self.deleteButton.setMaximumSize(QSize(16777215, 40))

        #self.deleteButton.setFont(font2)
        self.deleteButton.setStyleSheet(df.styleSheet_Delete_Scenarios)

        cancelButton = QPushButton("Cancel")
        sizePolicy3 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        sizePolicy3.setHeightForWidth(cancelButton.sizePolicy().hasHeightForWidth())
        cancelButton.setSizePolicy(sizePolicy3)
        cancelButton.setStyleSheet(df.styleSheet_Table_Normal)

        cancelButton.setGeometry(QtCore.QRect(100, 189, 81, 41))
        cancelButton.setMinimumSize(QtCore.QSize(60, 30))
        # self.deleteButton.setSizePolicy(sizePolicy3)
        cancelButton.setFont(font)
        cancelButton.setFixedWidth(80)
        #cancelButton.setMinimumSize(QSize(120, 40))

        #cancelButton.setMaximumSize(QSize(16777215, 40))
        #cancelButton.setStyleSheet(button_format)

        #cancelButton.setFont(font2)
        #cancelButton.setStyleSheet(button_format)

        gridLayout = QGridLayout()
        gridLayout.addWidget(nameLabel, 0, 0)
        gridLayout.addWidget(self.lineEdit1, 0, 1)
        gridLayout.addWidget(commentLabel, 1, 0)
        gridLayout.addWidget(self.lineEdit2, 1, 1)

        buttonLayout = QHBoxLayout()
        buttonLayout.addStretch(2)
        buttonLayout.addWidget(self.saveButton)
        #buttonLayout.addWidget(self.saveAsButton)
        buttonLayout.addWidget(self.deleteButton)
        buttonLayout.addWidget(cancelButton)

        mainLayout = QVBoxLayout()
        mainLayout.addLayout(gridLayout)
        mainLayout.addLayout(buttonLayout)

        self.setLayout(mainLayout)

        self.saveButton.clicked.connect(self.saveAction)
        self.saveAsButton.clicked.connect(self.saveAsAction)
        self.deleteButton.clicked.connect(self.deleteAction)
        cancelButton.clicked.connect(self.close)
        self.result = "Cancel"
        self.filename_changed()

    def filename_changed(self):
        if self.lineEdit1.text() != self.calculation_object.filename:
            self.saveButton.setText(self.SAVE_AS)
        else:
            self.saveButton.setText(self.SAVE)

    def saveAction(self):
        name = self.lineEdit1.text().strip()
        self.lineEdit1.setText(name)
        if len(name) == 0:
            name = "Empty"
            msgBox1 = QMessageBoxWithStyle(self)
            msgBox1.setWindowTitle(cs.MESSAGE_SAVE_UNSUCCESSFUL_TITLE.format(name))
            msgBox1.setText(cs.MESSAGE_SAVE_UNSUCCESSFUL_CONTENT.format(name))
            msgBox1.setStandardButtons(QMessageBox.Ok)
            msgBox1.setCustomStyle()
            msgBox1.exec_()
            return

        self.result = self.saveButton.text()
        self.close()

    def deleteAction(self):
        self.result = self.DELETE
        self.close()

    def saveAsAction(self):
        self.result = self.SAVE_AS
        self.close()

class QMessageBoxWithStyle(QMessageBox):

    def __init__(self, parent=None):
        QMessageBox.__init__(self, parent)

    def setCustomStyle(self):

        self.setStyleSheet(df.styleSheet_Message_Label)

        for button in self.buttons():
            if button.text() == "Discard" or button.text() == "DELETE" :
                button.setStyleSheet(df.styleSheet_Delete_Scenarios)
            else:
                button.setStyleSheet(df.styleSheet_Message_Button)

            button.setFixedWidth(80)
            font = QtGui.QFont()
            font.setFamily("Segoe UI")
            font.setPointSize(14)
            button.setFont(font)