# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'unitWidget_LifeTime.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class Ui_unitWidget_Lifetime(object):
    def setupUi(self, unitWidget_Lifetime):
        if not unitWidget_Lifetime.objectName():
            unitWidget_Lifetime.setObjectName(u"unitWidget_Lifetime")
        unitWidget_Lifetime.resize(1821, 1373)
        self.gridLayout = QGridLayout(unitWidget_Lifetime)
        self.gridLayout.setObjectName(u"gridLayout")
        self.Lifetime_frame_MainWindow = QFrame(unitWidget_Lifetime)
        self.Lifetime_frame_MainWindow.setObjectName(u"Lifetime_frame_MainWindow")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Lifetime_frame_MainWindow.sizePolicy().hasHeightForWidth())
        self.Lifetime_frame_MainWindow.setSizePolicy(sizePolicy)
        self.Lifetime_frame_MainWindow.setStyleSheet(u"background-color: rgb(44, 49, 60);\n"
"/* LINE EDIT */\n"
"QLineEdit {\n"
"	background-color: rgb(27, 29, 35);\n"
"	border-radius: 5px;\n"
"	border: 2px solid rgb(27, 29, 35);\n"
"	padding-left: 10px;\n"
"}\n"
"QLineEdit:hover {\n"
"	border: 2px solid rgb(64, 71, 88);\n"
"}\n"
"QLineEdit:focus {\n"
"	border: 2px solid rgb(91, 101, 124);\n"
"}\n"
"\n"
"/* SCROLL BARS */\n"
"QScrollBar:horizontal {\n"
"    border: none;\n"
"    background: rgb(52, 59, 72);\n"
"    height: 14px;\n"
"    margin: 0px 21px 0 21px;\n"
"	border-radius: 0px;\n"
"}\n"
"QScrollBar::handle:horizontal {\n"
"    background: rgb(85, 170, 255);\n"
"    min-width: 25px;\n"
"	border-radius: 7px\n"
"}\n"
"QScrollBar::add-line:horizontal {\n"
"    border: none;\n"
"    background: rgb(55, 63, 77);\n"
"    width: 20px;\n"
"	border-top-right-radius: 7px;\n"
"    border-bottom-right-radius: 7px;\n"
"    subcontrol-position: right;\n"
"    subcontrol-origin: margin;\n"
"}\n"
"QScrollBar::sub-line:horizontal {\n"
"    border: none;\n"
"    background:"
                        " rgb(55, 63, 77);\n"
"    width: 20px;\n"
"	border-top-left-radius: 7px;\n"
"    border-bottom-left-radius: 7px;\n"
"    subcontrol-position: left;\n"
"    subcontrol-origin: margin;\n"
"}\n"
"QScrollBar::up-arrow:horizontal, QScrollBar::down-arrow:horizontal\n"
"{\n"
"     background: none;\n"
"}\n"
"QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal\n"
"{\n"
"     background: none;\n"
"}\n"
" QScrollBar:vertical {\n"
"	border: none;\n"
"    background: rgb(52, 59, 72);\n"
"    width: 14px;\n"
"    margin: 21px 0 21px 0;\n"
"	border-radius: 0px;\n"
" }\n"
" QScrollBar::handle:vertical {	\n"
"	background: rgb(85, 170, 255);\n"
"    min-height: 25px;\n"
"	border-radius: 7px\n"
" }\n"
" QScrollBar::add-line:vertical {\n"
"     border: none;\n"
"    background: rgb(55, 63, 77);\n"
"     height: 20px;\n"
"	border-bottom-left-radius: 7px;\n"
"    border-bottom-right-radius: 7px;\n"
"     subcontrol-position: bottom;\n"
"     subcontrol-origin: margin;\n"
" }\n"
" QScrollBar::sub-line:vertical {\n"
"	b"
                        "order: none;\n"
"    background: rgb(55, 63, 77);\n"
"     height: 20px;\n"
"	border-top-left-radius: 7px;\n"
"    border-top-right-radius: 7px;\n"
"     subcontrol-position: top;\n"
"     subcontrol-origin: margin;\n"
" }\n"
" QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical {\n"
"     background: none;\n"
" }\n"
"\n"
" QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {\n"
"     background: none;\n"
" }\n"
"\n"
"/* CHECKBOX */\n"
"QCheckBox::indicator {\n"
"    border: 3px solid rgb(52, 59, 72);\n"
"	width: 15px;\n"
"	height: 15px;\n"
"	border-radius: 10px;\n"
"    background: rgb(44, 49, 60);\n"
"}\n"
"QCheckBox::indicator:hover {\n"
"    border: 3px solid rgb(58, 66, 81);\n"
"}\n"
"QCheckBox::indicator:checked {\n"
"    background: 3px solid rgb(52, 59, 72);\n"
"	border: 3px solid rgb(52, 59, 72);	\n"
"	background-image: url(:/16x16/icons/16x16/cil-check-alt.png);\n"
"}\n"
"\n"
"/* RADIO BUTTON */\n"
"QRadioButton::indicator {\n"
"    border: 3px solid rgb(52, 59, 72);\n"
"	width: "
                        "15px;\n"
"	height: 15px;\n"
"	border-radius: 10px;\n"
"    background: rgb(44, 49, 60);\n"
"}\n"
"QRadioButton::indicator:hover {\n"
"    border: 3px solid rgb(58, 66, 81);\n"
"}\n"
"QRadioButton::indicator:checked {\n"
"    background: 3px solid rgb(94, 106, 130);\n"
"	border: 3px solid rgb(52, 59, 72);	\n"
"}\n"
"\n"
"/* COMBOBOX */\n"
"QComboBox{\n"
"	background-color: rgb(27, 29, 35);\n"
"	border-radius: 5px;\n"
"	border: 2px solid rgb(27, 29, 35);\n"
"	padding: 5px;\n"
"	padding-left: 10px;\n"
"}\n"
"QComboBox:hover{\n"
"	border: 2px solid rgb(64, 71, 88);\n"
"}\n"
"QComboBox::drop-down {\n"
"	subcontrol-origin: padding;\n"
"	subcontrol-position: top right;\n"
"	width: 25px; \n"
"	border-left-width: 3px;\n"
"	border-left-color: rgba(39, 44, 54, 150);\n"
"	border-left-style: solid;\n"
"	border-top-right-radius: 3px;\n"
"	border-bottom-right-radius: 3px;	\n"
"	background-image: url(:/16x16/icons/16x16/cil-arrow-bottom.png);\n"
"	background-position: center;\n"
"	background-repeat: no-reperat;\n"
" }\n"
"QCo"
                        "mboBox QAbstractItemView {\n"
"	color: rgb(85, 170, 255);	\n"
"	background-color: rgb(27, 29, 35);\n"
"	padding: 10px;\n"
"	selection-background-color: rgb(39, 44, 54);\n"
"}\n"
"\n"
"/* SLIDERS */\n"
"QSlider::groove:horizontal {\n"
"    border-radius: 9px;\n"
"    height: 18px;\n"
"	margin: 0px;\n"
"	background-color: rgb(52, 59, 72);\n"
"}\n"
"QSlider::groove:horizontal:hover {\n"
"	background-color: rgb(55, 62, 76);\n"
"}\n"
"QSlider::handle:horizontal {\n"
"    background-color: rgb(85, 170, 255);\n"
"    border: none;\n"
"    height: 18px;\n"
"    width: 18px;\n"
"    margin: 0px;\n"
"	border-radius: 9px;\n"
"}\n"
"QSlider::handle:horizontal:hover {\n"
"    background-color: rgb(105, 180, 255);\n"
"}\n"
"QSlider::handle:horizontal:pressed {\n"
"    background-color: rgb(65, 130, 195);\n"
"}\n"
"\n"
"QSlider::groove:vertical {\n"
"    border-radius: 9px;\n"
"    width: 18px;\n"
"    margin: 0px;\n"
"	background-color: rgb(52, 59, 72);\n"
"}\n"
"QSlider::groove:vertical:hover {\n"
"	background-color: rgb(5"
                        "5, 62, 76);\n"
"}\n"
"QSlider::handle:vertical {\n"
"    background-color: rgb(85, 170, 255);\n"
"	border: none;\n"
"    height: 18px;\n"
"    width: 18px;\n"
"    margin: 0px;\n"
"	border-radius: 9px;\n"
"}\n"
"QSlider::handle:vertical:hover {\n"
"    background-color: rgb(105, 180, 255);\n"
"}\n"
"QSlider::handle:vertical:pressed {\n"
"    background-color: rgb(65, 130, 195);\n"
"}\n"
"/* QTableWidget*/\n"
"QTableWidget {	\n"
"	background-color: rgb(38, 44, 53);\n"
"	padding: 10px;\n"
"	border-radius: 15px;\n"
"	gridline-color: rgb(44, 49, 60);\n"
"	border-bottom: 1px solid rgb(44, 49, 60);\n"
"}\n"
"QTableWidget::item{\n"
"	border-color: rgb(44, 49, 60);\n"
"	padding-left: 5px;\n"
"	padding-right: 5px;\n"
"	gridline-color: rgb(44, 49, 60);\n"
"}\n"
"QTableWidget::item:selected{\n"
"	background-color: rgb(85, 170, 255);\n"
"}\n"
"QScrollBar:horizontal {\n"
"    border: none;\n"
"    background: rgb(52, 59, 72);\n"
"    height: 14px;\n"
"    margin: 0px 21px 0 21px;\n"
"	border-radius: 0px;\n"
"}\n"
"\n"
"QSc"
                        "rollBar:handle:horizontal {\n"
"    background: rgb(91, 101, 124);\n"
"\n"
"}\n"
"\n"
" QScrollBar:vertical {\n"
"	border: none;\n"
"    background: rgb(52, 59, 72);\n"
"    width: 14px;\n"
"    margin: 21px 0 21px 0;\n"
"	border-radius: 0px;\n"
" }\n"
"QHeaderView::section{\n"
"	Background-color: rgb(39, 44, 54);\n"
"	max-width: 30px;\n"
"	border: 1px solid rgb(44, 49, 60);\n"
"	border-style: none;\n"
"    border-bottom: 1px solid rgb(44, 49, 60);\n"
"    border-right: 1px solid rgb(44, 49, 60);\n"
"}\n"
"QTableWidget::horizontalHeader {	\n"
"	background-color: rgb(190, 190, 190);\n"
"}\n"
"\n"
"QHeaderView::section:horizontal\n"
"{\n"
"	background-color: rgb(51, 59, 70);\n"
"	padding: 3px;\n"
"	border-top-left-radius: 7px;\n"
"    border-top-right-radius: 7px;\n"
"	color: rgb(167,	174, 183);\n"
"}\n"
"QHeaderView::section:vertical\n"
"{\n"
"    border: 1px solid rgb(44, 49, 60);\n"
"}\n"
"\n"
"QScrollBar:handle:vertical {\n"
"    background: rgb(91, 101, 124);\n"
"\n"
"}\n"
"\n"
"QTableCornerButton::section{"
                        " \n"
"\n"
"	background-color: rgb(38, 44, 53);\n"
"}\n"
"\n"
"")
        self.Lifetime_frame_MainWindow.setFrameShape(QFrame.NoFrame)
        self.Lifetime_frame_MainWindow.setFrameShadow(QFrame.Raised)
        self.gridLayout_Lifetime_frame_MainWindow = QGridLayout(self.Lifetime_frame_MainWindow)
        self.gridLayout_Lifetime_frame_MainWindow.setSpacing(3)
        self.gridLayout_Lifetime_frame_MainWindow.setObjectName(u"gridLayout_Lifetime_frame_MainWindow")
        self.LF_Main05 = QFrame(self.Lifetime_frame_MainWindow)
        self.LF_Main05.setObjectName(u"LF_Main05")
        self.LF_Main05.setMinimumSize(QSize(0, 0))
        self.LF_Main05.setFrameShape(QFrame.StyledPanel)
        self.LF_Main05.setFrameShadow(QFrame.Raised)
        self.Lifetime_CalcButton_Grid = QGridLayout(self.LF_Main05)
        self.Lifetime_CalcButton_Grid.setObjectName(u"Lifetime_CalcButton_Grid")
        self.Lifetime_CalcButton_Grid.setContentsMargins(0, 0, 0, 0)
        self.Lifetime_save_button = QPushButton(self.LF_Main05)
        self.Lifetime_save_button.setObjectName(u"Lifetime_save_button")
        self.Lifetime_save_button.setMinimumSize(QSize(0, 50))
        font = QFont()
        font.setFamily(u"Segoe UI")
        font.setPointSize(14)
        self.Lifetime_save_button.setFont(font)
        self.Lifetime_save_button.setStyleSheet(u"QPushButton {\n"
"	border: 2px solid rgb(52, 59, 72);\n"
"	border-radius: 5px;\n"
"	background-color: rgb(52, 59, 72);\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(57, 65, 79);\n"
"	border: 2px solid rgb(61, 70, 86);\n"
"}\n"
"QPushButton:pressed {	\n"
"	background-color: rgb(67, 77, 93);\n"
"	border: 2px solid rgb(43, 50, 61);\n"
"}")

        self.Lifetime_CalcButton_Grid.addWidget(self.Lifetime_save_button, 0, 0, 1, 1)

        self.Lifetime_run_button = QPushButton(self.LF_Main05)
        self.Lifetime_run_button.setObjectName(u"Lifetime_run_button")
        self.Lifetime_run_button.setMinimumSize(QSize(220, 50))
        self.Lifetime_run_button.setFont(font)
        self.Lifetime_run_button.setStyleSheet(u"QPushButton {\n"
"	border: 2px solid rgb(52, 59, 72);\n"
"	border-radius: 5px;\n"
"	background-color: rgb(85, 170, 255);\n"
"}\n"
"QPushButton:hover {\n"
"	background-color: rgb(72, 144, 216);\n"
"	border: 2px solid rgb(61, 70, 86);\n"
"}\n"
"QPushButton:pressed {	\n"
"	background-color: rgb(52, 59, 72);\n"
"	border: 2px solid rgb(43, 50, 61);\n"
"}")

        self.Lifetime_CalcButton_Grid.addWidget(self.Lifetime_run_button, 0, 1, 1, 1)


        self.gridLayout_Lifetime_frame_MainWindow.addWidget(self.LF_Main05, 4, 0, 1, 1)

        self.Lifetime_InputSet_Total = QFrame(self.Lifetime_frame_MainWindow)
        self.Lifetime_InputSet_Total.setObjectName(u"Lifetime_InputSet_Total")
        self.Lifetime_InputSet_Total.setFrameShape(QFrame.StyledPanel)
        self.Lifetime_InputSet_Total.setFrameShadow(QFrame.Raised)
        self.gridLayout_Lifetime_InputSet_Total = QGridLayout(self.Lifetime_InputSet_Total)
        self.gridLayout_Lifetime_InputSet_Total.setSpacing(0)
        self.gridLayout_Lifetime_InputSet_Total.setObjectName(u"gridLayout_Lifetime_InputSet_Total")
        self.gridLayout_Lifetime_InputSet_Total.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_Lifetime_Total = QGridLayout()
        self.gridLayout_Lifetime_Total.setObjectName(u"gridLayout_Lifetime_Total")
        self.LF_Main02 = QFrame(self.Lifetime_InputSet_Total)
        self.LF_Main02.setObjectName(u"LF_Main02")
        self.LF_Main02.setFrameShape(QFrame.StyledPanel)
        self.LF_Main02.setFrameShadow(QFrame.Raised)
        self.gridLayout_Lifetime_Main04_Input = QGridLayout(self.LF_Main02)
        self.gridLayout_Lifetime_Main04_Input.setSpacing(6)
        self.gridLayout_Lifetime_Main04_Input.setObjectName(u"gridLayout_Lifetime_Main04_Input")
        self.gridLayout_Lifetime_Main04_Input.setContentsMargins(9, 9, 9, 9)
        self.LF_Main02Grid = QGridLayout()
        self.LF_Main02Grid.setObjectName(u"LF_Main02Grid")
        self.LabelSub_LF07 = QLabel(self.LF_Main02)
        self.LabelSub_LF07.setObjectName(u"LabelSub_LF07")
        self.LabelSub_LF07.setMinimumSize(QSize(105, 0))
        font1 = QFont()
        font1.setFamily(u"Segoe UI")
        font1.setPointSize(11)
        font1.setBold(False)
        font1.setWeight(50)
        self.LabelSub_LF07.setFont(font1)

        self.LF_Main02Grid.addWidget(self.LabelSub_LF07, 7, 1, 1, 1)

        self.LabelSub_LF04 = QLabel(self.LF_Main02)
        self.LabelSub_LF04.setObjectName(u"LabelSub_LF04")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.LabelSub_LF04.sizePolicy().hasHeightForWidth())
        self.LabelSub_LF04.setSizePolicy(sizePolicy1)
        self.LabelSub_LF04.setMinimumSize(QSize(105, 0))
        font2 = QFont()
        font2.setFamily(u"Segoe UI")
        font2.setPointSize(10)
        self.LabelSub_LF04.setFont(font2)

        self.LF_Main02Grid.addWidget(self.LabelSub_LF04, 4, 1, 1, 1)

        self.LabelSub_LF01 = QLabel(self.LF_Main02)
        self.LabelSub_LF01.setObjectName(u"LabelSub_LF01")
        sizePolicy1.setHeightForWidth(self.LabelSub_LF01.sizePolicy().hasHeightForWidth())
        self.LabelSub_LF01.setSizePolicy(sizePolicy1)
        self.LabelSub_LF01.setMinimumSize(QSize(105, 0))
        font3 = QFont()
        font3.setFamily(u"Segoe UI")
        font3.setPointSize(11)
        self.LabelSub_LF01.setFont(font3)

        self.LF_Main02Grid.addWidget(self.LabelSub_LF01, 1, 1, 1, 1)

        self.LabelSub_LF06 = QLabel(self.LF_Main02)
        self.LabelSub_LF06.setObjectName(u"LabelSub_LF06")
        self.LabelSub_LF06.setMinimumSize(QSize(105, 0))
        self.LabelSub_LF06.setFont(font1)

        self.LF_Main02Grid.addWidget(self.LabelSub_LF06, 6, 1, 1, 1)

        self.LabelSub_LF05 = QLabel(self.LF_Main02)
        self.LabelSub_LF05.setObjectName(u"LabelSub_LF05")
        self.LabelSub_LF05.setMinimumSize(QSize(105, 0))
        self.LabelSub_LF05.setFont(font1)

        self.LF_Main02Grid.addWidget(self.LabelSub_LF05, 5, 1, 1, 1)

        self.LabelSub_LF02 = QLabel(self.LF_Main02)
        self.LabelSub_LF02.setObjectName(u"LabelSub_LF02")
        sizePolicy1.setHeightForWidth(self.LabelSub_LF02.sizePolicy().hasHeightForWidth())
        self.LabelSub_LF02.setSizePolicy(sizePolicy1)
        self.LabelSub_LF02.setMinimumSize(QSize(105, 0))
        self.LabelSub_LF02.setFont(font3)

        self.LF_Main02Grid.addWidget(self.LabelSub_LF02, 2, 1, 1, 1)

        self.LabelSub_LF08 = QLabel(self.LF_Main02)
        self.LabelSub_LF08.setObjectName(u"LabelSub_LF08")
        self.LabelSub_LF08.setMinimumSize(QSize(105, 0))
        self.LabelSub_LF08.setFont(font1)

        self.LF_Main02Grid.addWidget(self.LabelSub_LF08, 8, 1, 1, 1)

        self.LabelSub_LF03 = QLabel(self.LF_Main02)
        self.LabelSub_LF03.setObjectName(u"LabelSub_LF03")
        sizePolicy1.setHeightForWidth(self.LabelSub_LF03.sizePolicy().hasHeightForWidth())
        self.LabelSub_LF03.setSizePolicy(sizePolicy1)
        self.LabelSub_LF03.setMinimumSize(QSize(105, 0))
        self.LabelSub_LF03.setFont(font2)

        self.LF_Main02Grid.addWidget(self.LabelSub_LF03, 3, 1, 1, 1)

        self.gridLayout_Lifetime_Input03_A_4 = QGridLayout()
        self.gridLayout_Lifetime_Input03_A_4.setSpacing(0)
        self.gridLayout_Lifetime_Input03_A_4.setObjectName(u"gridLayout_Lifetime_Input03_A_4")
        self.verticalSpacer_Lifetime_Input03_A_7 = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.gridLayout_Lifetime_Input03_A_4.addItem(self.verticalSpacer_Lifetime_Input03_A_7, 2, 0, 1, 1)

        self.verticalSpacer_Lifetime_Input03_A_6 = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.gridLayout_Lifetime_Input03_A_4.addItem(self.verticalSpacer_Lifetime_Input03_A_6, 0, 0, 1, 1)

        self.LF_Input01 = QDoubleSpinBox(self.LF_Main02)
        self.LF_Input01.setObjectName(u"LF_Input01")
        sizePolicy1.setHeightForWidth(self.LF_Input01.sizePolicy().hasHeightForWidth())
        self.LF_Input01.setSizePolicy(sizePolicy1)
        self.LF_Input01.setMinimumSize(QSize(0, 0))
        self.LF_Input01.setMaximumSize(QSize(16777215, 16777215))
        self.LF_Input01.setFont(font2)
        self.LF_Input01.setStyleSheet(u"padding: 3px;")
        self.LF_Input01.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.LF_Input01.setProperty("showGroupSeparator", True)
        self.LF_Input01.setDecimals(2)
        self.LF_Input01.setMinimum(0.000000000000000)
        self.LF_Input01.setMaximum(100000.000000000000000)
        self.LF_Input01.setSingleStep(100.000000000000000)
        self.LF_Input01.setStepType(QAbstractSpinBox.DefaultStepType)
        self.LF_Input01.setValue(0.000000000000000)

        self.gridLayout_Lifetime_Input03_A_4.addWidget(self.LF_Input01, 1, 0, 1, 1)


        self.LF_Main02Grid.addLayout(self.gridLayout_Lifetime_Input03_A_4, 1, 3, 1, 1)

        self.LabelTitle_LF02 = QLabel(self.LF_Main02)
        self.LabelTitle_LF02.setObjectName(u"LabelTitle_LF02")
        self.LabelTitle_LF02.setMinimumSize(QSize(0, 0))
        font4 = QFont()
        font4.setFamily(u"Segoe UI")
        font4.setPointSize(14)
        font4.setBold(True)
        font4.setWeight(75)
        self.LabelTitle_LF02.setFont(font4)

        self.LF_Main02Grid.addWidget(self.LabelTitle_LF02, 0, 0, 1, 4)

        self.gridLayout_Lifetime_Input03_A_5 = QGridLayout()
        self.gridLayout_Lifetime_Input03_A_5.setSpacing(0)
        self.gridLayout_Lifetime_Input03_A_5.setObjectName(u"gridLayout_Lifetime_Input03_A_5")
        self.verticalSpacer_Lifetime_Input03_A_9 = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.gridLayout_Lifetime_Input03_A_5.addItem(self.verticalSpacer_Lifetime_Input03_A_9, 2, 0, 1, 1)

        self.verticalSpacer_Lifetime_Input03_A_8 = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.gridLayout_Lifetime_Input03_A_5.addItem(self.verticalSpacer_Lifetime_Input03_A_8, 0, 0, 1, 1)

        self.LF_Input02 = QDoubleSpinBox(self.LF_Main02)
        self.LF_Input02.setObjectName(u"LF_Input02")
        sizePolicy1.setHeightForWidth(self.LF_Input02.sizePolicy().hasHeightForWidth())
        self.LF_Input02.setSizePolicy(sizePolicy1)
        self.LF_Input02.setMinimumSize(QSize(0, 0))
        self.LF_Input02.setMaximumSize(QSize(16777215, 16777215))
        self.LF_Input02.setFont(font2)
        self.LF_Input02.setStyleSheet(u"padding: 3px;")
        self.LF_Input02.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.LF_Input02.setProperty("showGroupSeparator", True)
        self.LF_Input02.setDecimals(2)
        self.LF_Input02.setMinimum(0.000000000000000)
        self.LF_Input02.setMaximum(100000.000000000000000)
        self.LF_Input02.setSingleStep(100.000000000000000)
        self.LF_Input02.setStepType(QAbstractSpinBox.DefaultStepType)
        self.LF_Input02.setValue(0.000000000000000)

        self.gridLayout_Lifetime_Input03_A_5.addWidget(self.LF_Input02, 1, 0, 1, 1)


        self.LF_Main02Grid.addLayout(self.gridLayout_Lifetime_Input03_A_5, 2, 3, 1, 1)

        self.gridLayout_Lifetime_Input03_A = QGridLayout()
        self.gridLayout_Lifetime_Input03_A.setSpacing(0)
        self.gridLayout_Lifetime_Input03_A.setObjectName(u"gridLayout_Lifetime_Input03_A")
        self.verticalSpacer_Lifetime_Input03_A_01 = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.gridLayout_Lifetime_Input03_A.addItem(self.verticalSpacer_Lifetime_Input03_A_01, 0, 0, 1, 1)

        self.verticalSpacer_Lifetime_Input03_A_02 = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.gridLayout_Lifetime_Input03_A.addItem(self.verticalSpacer_Lifetime_Input03_A_02, 2, 0, 1, 1)

        self.LF_Input03 = QDoubleSpinBox(self.LF_Main02)
        self.LF_Input03.setObjectName(u"LF_Input03")
        sizePolicy1.setHeightForWidth(self.LF_Input03.sizePolicy().hasHeightForWidth())
        self.LF_Input03.setSizePolicy(sizePolicy1)
        self.LF_Input03.setMinimumSize(QSize(0, 0))
        self.LF_Input03.setMaximumSize(QSize(16777215, 16777215))
        self.LF_Input03.setFont(font2)
        self.LF_Input03.setStyleSheet(u"padding: 3px;")
        self.LF_Input03.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.LF_Input03.setProperty("showGroupSeparator", True)
        self.LF_Input03.setDecimals(5)
        self.LF_Input03.setMaximum(10.000000000000000)
        self.LF_Input03.setSingleStep(0.010000000000000)
        self.LF_Input03.setStepType(QAbstractSpinBox.DefaultStepType)
        self.LF_Input03.setValue(1.000000000000000)

        self.gridLayout_Lifetime_Input03_A.addWidget(self.LF_Input03, 1, 0, 1, 1)


        self.LF_Main02Grid.addLayout(self.gridLayout_Lifetime_Input03_A, 3, 3, 1, 1)

        self.gridLayout_Lifetime_Input03_A_3 = QGridLayout()
        self.gridLayout_Lifetime_Input03_A_3.setSpacing(0)
        self.gridLayout_Lifetime_Input03_A_3.setObjectName(u"gridLayout_Lifetime_Input03_A_3")
        self.verticalSpacer_Lifetime_Input03_A_4 = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.gridLayout_Lifetime_Input03_A_3.addItem(self.verticalSpacer_Lifetime_Input03_A_4, 0, 0, 1, 1)

        self.verticalSpacer_Lifetime_Input03_A_5 = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.gridLayout_Lifetime_Input03_A_3.addItem(self.verticalSpacer_Lifetime_Input03_A_5, 2, 0, 1, 1)

        self.LF_Input04 = QDoubleSpinBox(self.LF_Main02)
        self.LF_Input04.setObjectName(u"LF_Input04")
        sizePolicy1.setHeightForWidth(self.LF_Input04.sizePolicy().hasHeightForWidth())
        self.LF_Input04.setSizePolicy(sizePolicy1)
        self.LF_Input04.setMinimumSize(QSize(0, 0))
        self.LF_Input04.setMaximumSize(QSize(16777215, 16777215))
        self.LF_Input04.setFont(font2)
        self.LF_Input04.setStyleSheet(u"padding: 3px;")
        self.LF_Input04.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.LF_Input04.setProperty("showGroupSeparator", True)
        self.LF_Input04.setMaximum(100000.000000000000000)
        self.LF_Input04.setStepType(QAbstractSpinBox.DefaultStepType)
        self.LF_Input04.setValue(0.000000000000000)

        self.gridLayout_Lifetime_Input03_A_3.addWidget(self.LF_Input04, 1, 0, 1, 1)


        self.LF_Main02Grid.addLayout(self.gridLayout_Lifetime_Input03_A_3, 4, 3, 1, 1)

        self.horizontalSpacer_Lifetime_Main04_01 = QSpacerItem(5, 5, QSizePolicy.Fixed, QSizePolicy.Minimum)

        self.LF_Main02Grid.addItem(self.horizontalSpacer_Lifetime_Main04_01, 1, 0, 8, 1)

        self.gridLayout_Lifetime_Input03_A_6 = QGridLayout()
        self.gridLayout_Lifetime_Input03_A_6.setSpacing(0)
        self.gridLayout_Lifetime_Input03_A_6.setObjectName(u"gridLayout_Lifetime_Input03_A_6")
        self.verticalSpacer_Lifetime_Input03_A_10 = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.gridLayout_Lifetime_Input03_A_6.addItem(self.verticalSpacer_Lifetime_Input03_A_10, 0, 0, 1, 1)

        self.verticalSpacer_Lifetime_Input03_A_11 = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.gridLayout_Lifetime_Input03_A_6.addItem(self.verticalSpacer_Lifetime_Input03_A_11, 2, 0, 1, 1)

        self.LF_Input05 = QDoubleSpinBox(self.LF_Main02)
        self.LF_Input05.setObjectName(u"LF_Input05")
        sizePolicy1.setHeightForWidth(self.LF_Input05.sizePolicy().hasHeightForWidth())
        self.LF_Input05.setSizePolicy(sizePolicy1)
        self.LF_Input05.setMinimumSize(QSize(0, 0))
        self.LF_Input05.setMaximumSize(QSize(16777215, 16777215))
        self.LF_Input05.setFont(font2)
        self.LF_Input05.setStyleSheet(u"padding: 3px;")
        self.LF_Input05.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.LF_Input05.setProperty("showGroupSeparator", True)
        self.LF_Input05.setMaximum(100000.000000000000000)
        self.LF_Input05.setStepType(QAbstractSpinBox.DefaultStepType)
        self.LF_Input05.setValue(50.000000000000000)

        self.gridLayout_Lifetime_Input03_A_6.addWidget(self.LF_Input05, 1, 0, 1, 1)


        self.LF_Main02Grid.addLayout(self.gridLayout_Lifetime_Input03_A_6, 5, 3, 1, 1)

        self.gridLayout_Lifetime_Input03_A_7 = QGridLayout()
        self.gridLayout_Lifetime_Input03_A_7.setSpacing(0)
        self.gridLayout_Lifetime_Input03_A_7.setObjectName(u"gridLayout_Lifetime_Input03_A_7")
        self.verticalSpacer_Lifetime_Input03_A_13 = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.gridLayout_Lifetime_Input03_A_7.addItem(self.verticalSpacer_Lifetime_Input03_A_13, 2, 0, 1, 1)

        self.verticalSpacer_Lifetime_Input03_A_12 = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.gridLayout_Lifetime_Input03_A_7.addItem(self.verticalSpacer_Lifetime_Input03_A_12, 0, 0, 1, 1)

        self.LF_Input06 = QDoubleSpinBox(self.LF_Main02)
        self.LF_Input06.setObjectName(u"LF_Input06")
        sizePolicy1.setHeightForWidth(self.LF_Input06.sizePolicy().hasHeightForWidth())
        self.LF_Input06.setSizePolicy(sizePolicy1)
        self.LF_Input06.setMinimumSize(QSize(0, 0))
        self.LF_Input06.setMaximumSize(QSize(16777215, 16777215))
        self.LF_Input06.setFont(font2)
        self.LF_Input06.setStyleSheet(u"padding: 3px;")
        self.LF_Input06.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.LF_Input06.setProperty("showGroupSeparator", True)
        self.LF_Input06.setMaximum(100000.000000000000000)
        self.LF_Input06.setStepType(QAbstractSpinBox.DefaultStepType)
        self.LF_Input06.setValue(50.000000000000000)

        self.gridLayout_Lifetime_Input03_A_7.addWidget(self.LF_Input06, 1, 0, 1, 1)


        self.LF_Main02Grid.addLayout(self.gridLayout_Lifetime_Input03_A_7, 6, 3, 1, 1)

        self.gridLayout_Lifetime_Input03_A_9 = QGridLayout()
        self.gridLayout_Lifetime_Input03_A_9.setSpacing(0)
        self.gridLayout_Lifetime_Input03_A_9.setObjectName(u"gridLayout_Lifetime_Input03_A_9")
        self.verticalSpacer_Lifetime_Input03_A_16 = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.gridLayout_Lifetime_Input03_A_9.addItem(self.verticalSpacer_Lifetime_Input03_A_16, 0, 0, 1, 1)

        self.verticalSpacer_Lifetime_Input03_A_17 = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.gridLayout_Lifetime_Input03_A_9.addItem(self.verticalSpacer_Lifetime_Input03_A_17, 2, 0, 1, 1)

        self.LF_Input07 = QDoubleSpinBox(self.LF_Main02)
        self.LF_Input07.setObjectName(u"LF_Input07")
        sizePolicy1.setHeightForWidth(self.LF_Input07.sizePolicy().hasHeightForWidth())
        self.LF_Input07.setSizePolicy(sizePolicy1)
        self.LF_Input07.setMinimumSize(QSize(0, 0))
        self.LF_Input07.setMaximumSize(QSize(16777215, 16777215))
        self.LF_Input07.setFont(font2)
        self.LF_Input07.setStyleSheet(u"padding: 3px;")
        self.LF_Input07.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.LF_Input07.setProperty("showGroupSeparator", True)
        self.LF_Input07.setMaximum(100000.000000000000000)
        self.LF_Input07.setStepType(QAbstractSpinBox.DefaultStepType)
        self.LF_Input07.setValue(50.000000000000000)

        self.gridLayout_Lifetime_Input03_A_9.addWidget(self.LF_Input07, 1, 0, 1, 1)


        self.LF_Main02Grid.addLayout(self.gridLayout_Lifetime_Input03_A_9, 7, 3, 1, 1)

        self.gridLayout_Lifetime_Input03_A_8 = QGridLayout()
        self.gridLayout_Lifetime_Input03_A_8.setSpacing(0)
        self.gridLayout_Lifetime_Input03_A_8.setObjectName(u"gridLayout_Lifetime_Input03_A_8")
        self.verticalSpacer_Lifetime_Input03_A_15 = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.gridLayout_Lifetime_Input03_A_8.addItem(self.verticalSpacer_Lifetime_Input03_A_15, 2, 0, 1, 1)

        self.verticalSpacer_Lifetime_Input03_A_14 = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.gridLayout_Lifetime_Input03_A_8.addItem(self.verticalSpacer_Lifetime_Input03_A_14, 0, 0, 1, 1)

        self.LF_Input08 = QDoubleSpinBox(self.LF_Main02)
        self.LF_Input08.setObjectName(u"LF_Input08")
        sizePolicy1.setHeightForWidth(self.LF_Input08.sizePolicy().hasHeightForWidth())
        self.LF_Input08.setSizePolicy(sizePolicy1)
        self.LF_Input08.setMinimumSize(QSize(0, 0))
        self.LF_Input08.setMaximumSize(QSize(16777215, 16777215))
        self.LF_Input08.setFont(font2)
        self.LF_Input08.setStyleSheet(u"padding: 3px;")
        self.LF_Input08.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.LF_Input08.setProperty("showGroupSeparator", True)
        self.LF_Input08.setMaximum(100000.000000000000000)
        self.LF_Input08.setStepType(QAbstractSpinBox.DefaultStepType)
        self.LF_Input08.setValue(50.000000000000000)

        self.gridLayout_Lifetime_Input03_A_8.addWidget(self.LF_Input08, 1, 0, 1, 1)


        self.LF_Main02Grid.addLayout(self.gridLayout_Lifetime_Input03_A_8, 8, 3, 1, 1)

        self.horizontalSpacer_Lifetime_Main04_02 = QSpacerItem(1, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.LF_Main02Grid.addItem(self.horizontalSpacer_Lifetime_Main04_02, 1, 2, 8, 1)


        self.gridLayout_Lifetime_Main04_Input.addLayout(self.LF_Main02Grid, 0, 0, 1, 1)


        self.gridLayout_Lifetime_Total.addWidget(self.LF_Main02, 1, 0, 1, 1)

        self.LF_Main01 = QFrame(self.Lifetime_InputSet_Total)
        self.LF_Main01.setObjectName(u"LF_Main01")
        self.LF_Main01.setFrameShape(QFrame.StyledPanel)
        self.LF_Main01.setFrameShadow(QFrame.Raised)
        self.gridLayout_Lifetime_Main01_InputSetup = QGridLayout(self.LF_Main01)
        self.gridLayout_Lifetime_Main01_InputSetup.setObjectName(u"gridLayout_Lifetime_Main01_InputSetup")
        self.LF_Main01Grid = QGridLayout()
        self.LF_Main01Grid.setSpacing(6)
        self.LF_Main01Grid.setObjectName(u"LF_Main01Grid")
        self.LF_Main01Grid.setContentsMargins(0, 0, 0, 0)
        self.horizontalSpacer_Lifetime_Main01_01 = QSpacerItem(1, 20, QSizePolicy.Fixed, QSizePolicy.Minimum)

        self.LF_Main01Grid.addItem(self.horizontalSpacer_Lifetime_Main01_01, 1, 0, 2, 1)

        self.Lifetime_InpOpt2_NDR = QRadioButton(self.LF_Main01)
        self.Lifetime_InpOpt2_NDR.setObjectName(u"Lifetime_InpOpt2_NDR")
        sizePolicy1.setHeightForWidth(self.Lifetime_InpOpt2_NDR.sizePolicy().hasHeightForWidth())
        self.Lifetime_InpOpt2_NDR.setSizePolicy(sizePolicy1)
        self.Lifetime_InpOpt2_NDR.setMinimumSize(QSize(0, 0))
        font5 = QFont()
        font5.setFamily(u"Segoe UI")
        font5.setPointSize(12)
        self.Lifetime_InpOpt2_NDR.setFont(font5)
        self.Lifetime_InpOpt2_NDR.setAutoExclusive(False)

        self.LF_Main01Grid.addWidget(self.Lifetime_InpOpt2_NDR, 2, 1, 1, 1)

        self.LabelTitle_LF01 = QLabel(self.LF_Main01)
        self.LabelTitle_LF01.setObjectName(u"LabelTitle_LF01")
        self.LabelTitle_LF01.setMinimumSize(QSize(0, 0))
        self.LabelTitle_LF01.setFont(font4)

        self.LF_Main01Grid.addWidget(self.LabelTitle_LF01, 0, 0, 1, 4)

        self.Lifetime_InpOpt1_Snapshot = QRadioButton(self.LF_Main01)
        self.Lifetime_InpOpt1_Snapshot.setObjectName(u"Lifetime_InpOpt1_Snapshot")
        sizePolicy1.setHeightForWidth(self.Lifetime_InpOpt1_Snapshot.sizePolicy().hasHeightForWidth())
        self.Lifetime_InpOpt1_Snapshot.setSizePolicy(sizePolicy1)
        self.Lifetime_InpOpt1_Snapshot.setMinimumSize(QSize(0, 0))
        self.Lifetime_InpOpt1_Snapshot.setFont(font5)
#if QT_CONFIG(whatsthis)
        self.Lifetime_InpOpt1_Snapshot.setWhatsThis(u"")
#endif // QT_CONFIG(whatsthis)
        self.Lifetime_InpOpt1_Snapshot.setAutoExclusive(False)

        self.LF_Main01Grid.addWidget(self.Lifetime_InpOpt1_Snapshot, 1, 1, 1, 1)

        self.horizontalSpacer_Lifetime_Main01_02 = QSpacerItem(1, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.LF_Main01Grid.addItem(self.horizontalSpacer_Lifetime_Main01_02, 2, 2, 1, 1)

        self.Lifetime_Snapshot = QComboBox(self.LF_Main01)
        self.Lifetime_Snapshot.setObjectName(u"Lifetime_Snapshot")
        sizePolicy1.setHeightForWidth(self.Lifetime_Snapshot.sizePolicy().hasHeightForWidth())
        self.Lifetime_Snapshot.setSizePolicy(sizePolicy1)
        self.Lifetime_Snapshot.setMinimumSize(QSize(100, 0))
        self.Lifetime_Snapshot.setMaximumSize(QSize(16777215, 30))
        self.Lifetime_Snapshot.setFont(font2)

        self.LF_Main01Grid.addWidget(self.Lifetime_Snapshot, 1, 2, 1, 2)


        self.gridLayout_Lifetime_Main01_InputSetup.addLayout(self.LF_Main01Grid, 0, 0, 1, 1)


        self.gridLayout_Lifetime_Total.addWidget(self.LF_Main01, 0, 0, 1, 1)


        self.gridLayout_Lifetime_InputSet_Total.addLayout(self.gridLayout_Lifetime_Total, 0, 0, 1, 1)

        self.verticalSpacer_20 = QSpacerItem(20, 4000, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_Lifetime_InputSet_Total.addItem(self.verticalSpacer_20, 1, 0, 1, 1)


        self.gridLayout_Lifetime_frame_MainWindow.addWidget(self.Lifetime_InputSet_Total, 0, 0, 3, 1)

        self.Lifetime_tabWidget = QTabWidget(self.Lifetime_frame_MainWindow)
        self.Lifetime_tabWidget.setObjectName(u"Lifetime_tabWidget")
        sizePolicy2 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy2.setHorizontalStretch(200)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.Lifetime_tabWidget.sizePolicy().hasHeightForWidth())
        self.Lifetime_tabWidget.setSizePolicy(sizePolicy2)
        self.Lifetime_tabWidget.setFont(font)
        self.Lifetime_tabWidget.setStyleSheet(u"\n"
"\n"
"QTabWidget{\n"
"	background-color: rgb(255, 255, 255);\n"
"	border-radius: 10px;\n"
"	padding-left: 0px;\n"
"	margin-left: 0px;\n"
"}\n"
"\n"
"QTabWidget::pane{\n"
"	background-color: blue;\n"
"	border-radius: 10px;\n"
"	border: 0px solid  rgb(27, 29, 35);\n"
"	padding-left: 0px;\n"
"	margin-left: 0px;\n"
"	border-bottom: 5px solid  rgb(85, 170, 255);\n"
"}\n"
"\n"
"QTabBar::tab{\n"
"	background-color: rgb(27, 29, 35);\n"
"	border-top-left-radius: 5px;\n"
"	border-top-right-radius: 5px;\n"
"	border-top: 5px solid  rgb(27, 29, 35);\n"
"	padding-left: 20px;\n"
"	padding-right: 20px;\n"
"	padding-top: 0px;\n"
"	padding-bottom: 5px;\n"
"}\n"
"\n"
"QTabBar::tab:selected{\n"
"	background-color: rgb(51, 58, 72);\n"
"	border-top-left-radius: 5px;\n"
"	border-top-right-radius: 5px;\n"
"	border-top: 5px solid  rgb(51, 58, 72);\n"
"	padding-top: 0px;\n"
"}")
        self.Lifetime_Database = QWidget()
        self.Lifetime_Database.setObjectName(u"Lifetime_Database")
        self.gridLayout_18 = QGridLayout(self.Lifetime_Database)
        self.gridLayout_18.setObjectName(u"gridLayout_18")
        self.Lifetime_DB = QTableWidget(self.Lifetime_Database)
        if (self.Lifetime_DB.columnCount() < 7):
            self.Lifetime_DB.setColumnCount(7)
        __qtablewidgetitem = QTableWidgetItem()
        self.Lifetime_DB.setHorizontalHeaderItem(0, __qtablewidgetitem)
        __qtablewidgetitem1 = QTableWidgetItem()
        self.Lifetime_DB.setHorizontalHeaderItem(1, __qtablewidgetitem1)
        __qtablewidgetitem2 = QTableWidgetItem()
        self.Lifetime_DB.setHorizontalHeaderItem(2, __qtablewidgetitem2)
        __qtablewidgetitem3 = QTableWidgetItem()
        self.Lifetime_DB.setHorizontalHeaderItem(3, __qtablewidgetitem3)
        __qtablewidgetitem4 = QTableWidgetItem()
        self.Lifetime_DB.setHorizontalHeaderItem(4, __qtablewidgetitem4)
        __qtablewidgetitem5 = QTableWidgetItem()
        self.Lifetime_DB.setHorizontalHeaderItem(5, __qtablewidgetitem5)
        __qtablewidgetitem6 = QTableWidgetItem()
        self.Lifetime_DB.setHorizontalHeaderItem(6, __qtablewidgetitem6)
        if (self.Lifetime_DB.rowCount() < 4):
            self.Lifetime_DB.setRowCount(4)
        __qtablewidgetitem7 = QTableWidgetItem()
        self.Lifetime_DB.setVerticalHeaderItem(0, __qtablewidgetitem7)
        __qtablewidgetitem8 = QTableWidgetItem()
        self.Lifetime_DB.setVerticalHeaderItem(1, __qtablewidgetitem8)
        __qtablewidgetitem9 = QTableWidgetItem()
        self.Lifetime_DB.setVerticalHeaderItem(2, __qtablewidgetitem9)
        __qtablewidgetitem10 = QTableWidgetItem()
        self.Lifetime_DB.setVerticalHeaderItem(3, __qtablewidgetitem10)
        self.Lifetime_DB.setObjectName(u"Lifetime_DB")
        sizePolicy3 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.Lifetime_DB.sizePolicy().hasHeightForWidth())
        self.Lifetime_DB.setSizePolicy(sizePolicy3)
        palette = QPalette()
        brush = QBrush(QColor(210, 210, 210, 255))
        brush.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.WindowText, brush)
        brush1 = QBrush(QColor(38, 44, 53, 255))
        brush1.setStyle(Qt.SolidPattern)
        palette.setBrush(QPalette.Active, QPalette.Button, brush1)
        palette.setBrush(QPalette.Active, QPalette.Text, brush)
        palette.setBrush(QPalette.Active, QPalette.ButtonText, brush)
        palette.setBrush(QPalette.Active, QPalette.Base, brush1)
        palette.setBrush(QPalette.Active, QPalette.Window, brush1)
        brush2 = QBrush(QColor(210, 210, 210, 128))
        brush2.setStyle(Qt.NoBrush)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette.setBrush(QPalette.Active, QPalette.PlaceholderText, brush2)
#endif
        palette.setBrush(QPalette.Inactive, QPalette.WindowText, brush)
        palette.setBrush(QPalette.Inactive, QPalette.Button, brush1)
        palette.setBrush(QPalette.Inactive, QPalette.Text, brush)
        palette.setBrush(QPalette.Inactive, QPalette.ButtonText, brush)
        palette.setBrush(QPalette.Inactive, QPalette.Base, brush1)
        palette.setBrush(QPalette.Inactive, QPalette.Window, brush1)
        brush3 = QBrush(QColor(210, 210, 210, 128))
        brush3.setStyle(Qt.NoBrush)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette.setBrush(QPalette.Inactive, QPalette.PlaceholderText, brush3)
#endif
        palette.setBrush(QPalette.Disabled, QPalette.WindowText, brush)
        palette.setBrush(QPalette.Disabled, QPalette.Button, brush1)
        palette.setBrush(QPalette.Disabled, QPalette.Text, brush)
        palette.setBrush(QPalette.Disabled, QPalette.ButtonText, brush)
        palette.setBrush(QPalette.Disabled, QPalette.Base, brush1)
        palette.setBrush(QPalette.Disabled, QPalette.Window, brush1)
        brush4 = QBrush(QColor(210, 210, 210, 128))
        brush4.setStyle(Qt.NoBrush)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette.setBrush(QPalette.Disabled, QPalette.PlaceholderText, brush4)
#endif
        self.Lifetime_DB.setPalette(palette)
        self.Lifetime_DB.setStyleSheet(u"QTableWidget {	\n"
"	background-color: rgb(38, 44, 53);\n"
"	padding: 10px;\n"
"	border-radius: 15px;\n"
"	gridline-color: rgb(44, 49, 60);\n"
"	border-bottom: 1px solid rgb(44, 49, 60);\n"
"}\n"
"QTableWidget::item{\n"
"	border-color: rgb(44, 49, 60);\n"
"	padding-left: 5px;\n"
"	padding-right: 5px;\n"
"	gridline-color: rgb(44, 49, 60);\n"
"}\n"
"QTableWidget::item:selected{\n"
"	background-color: rgb(85, 170, 255);\n"
"}\n"
"QScrollBar:horizontal {\n"
"    border: none;\n"
"    background: rgb(52, 59, 72);\n"
"    height: 14px;\n"
"    margin: 0px 21px 0 21px;\n"
"	border-radius: 0px;\n"
"}\n"
"QScrollBar:handle:horizontal {\n"
"    background: rgb(79, 110, 162);\n"
"}\n"
"QScrollBar:vertical {\n"
"	border: none;\n"
"    background: rgb(52, 59, 72);\n"
"    width: 14px;\n"
"    margin: 21px 0 21px 0;\n"
"	border-radius: 0px;\n"
"}\n"
"QScrollBar:handle:vertical {\n"
"    background: rgb(79, 110, 162);\n"
"\n"
"}\n"
"QHeaderView::section{\n"
"	Background-color: rgb(39, 44, 54);\n"
"	max-width: 30px;\n"
"	bord"
                        "er: 1px solid rgb(44, 49, 60);\n"
"	border-style: none;\n"
"    border-bottom: 1px solid rgb(44, 49, 60);\n"
"    border-right: 1px solid rgb(44, 49, 60);\n"
"}\n"
"\n"
"\n"
"QTableWidget::horizontalHeader {	\n"
"	background-color: rgb(190, 190, 190);\n"
"}\n"
"\n"
"QHeaderView::section:horizontal\n"
"{\n"
"	background-color: rgb(51, 59, 70);\n"
"	padding: 3px;\n"
"	border-top-left-radius: 7px;\n"
"    border-top-right-radius: 7px;\n"
"	color: rgb(167,	174, 183);\n"
"}\n"
"QHeaderView::section:vertical\n"
"{\n"
"    border: 1px solid rgb(44, 49, 60);\n"
"}\n"
"\n"
"QTableWidget::verticalHeader {	\n"
"	background-color: rgb(81, 255, 0);\n"
"}\n"
"QTableCornerButton::section{\n"
"	Background-color: rgb(39, 44, 54);\n"
"	border: 1px solid rgb(44, 49, 60);\n"
"	border-style: none;\n"
"    border-bottom: 1px solid rgb(44, 49, 60);\n"
"    border-right: 1px solid rgb(44, 49, 60);\n"
"}")
        self.Lifetime_DB.setFrameShape(QFrame.NoFrame)
        self.Lifetime_DB.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.Lifetime_DB.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.Lifetime_DB.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.Lifetime_DB.setAlternatingRowColors(False)
        self.Lifetime_DB.setSelectionMode(QAbstractItemView.SingleSelection)
        self.Lifetime_DB.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.Lifetime_DB.setShowGrid(True)
        self.Lifetime_DB.setGridStyle(Qt.SolidLine)
        self.Lifetime_DB.setSortingEnabled(False)
        self.Lifetime_DB.horizontalHeader().setVisible(False)
        self.Lifetime_DB.horizontalHeader().setCascadingSectionResizes(True)
        self.Lifetime_DB.horizontalHeader().setDefaultSectionSize(155)
        self.Lifetime_DB.horizontalHeader().setProperty("showSortIndicator", False)
        self.Lifetime_DB.horizontalHeader().setStretchLastSection(True)
        self.Lifetime_DB.verticalHeader().setVisible(False)
        self.Lifetime_DB.verticalHeader().setCascadingSectionResizes(True)
        self.Lifetime_DB.verticalHeader().setDefaultSectionSize(30)
        self.Lifetime_DB.verticalHeader().setHighlightSections(False)
        self.Lifetime_DB.verticalHeader().setStretchLastSection(False)

        self.gridLayout_18.addWidget(self.Lifetime_DB, 0, 0, 1, 1)

        self.Lifetime_tabWidget.addTab(self.Lifetime_Database, "")
        self.Lifetime_tabInput = QWidget()
        self.Lifetime_tabInput.setObjectName(u"Lifetime_tabInput")
        self.Lifetime_tabInput.setStyleSheet(u"background-color: rgb(51, 58, 72);")
        self.gridLayout_55 = QGridLayout(self.Lifetime_tabInput)
        self.gridLayout_55.setObjectName(u"gridLayout_55")
        self.frame_Lifetime_tabInput = QFrame(self.Lifetime_tabInput)
        self.frame_Lifetime_tabInput.setObjectName(u"frame_Lifetime_tabInput")
        self.frame_Lifetime_tabInput.setFrameShape(QFrame.StyledPanel)
        self.frame_Lifetime_tabInput.setFrameShadow(QFrame.Raised)
        self.gridLayout_56 = QGridLayout(self.frame_Lifetime_tabInput)
        self.gridLayout_56.setObjectName(u"gridLayout_56")
        self.label_189 = QLabel(self.frame_Lifetime_tabInput)
        self.label_189.setObjectName(u"label_189")
        font6 = QFont()
        font6.setFamily(u"Segoe UI")
        font6.setPointSize(20)
        font6.setUnderline(False)
        self.label_189.setFont(font6)
        self.label_189.setAlignment(Qt.AlignCenter)
        self.label_189.setMargin(20)

        self.gridLayout_56.addWidget(self.label_189, 1, 0, 1, 2)

        self.label_190 = QLabel(self.frame_Lifetime_tabInput)
        self.label_190.setObjectName(u"label_190")
        font7 = QFont()
        font7.setFamily(u"Segoe UI")
        font7.setPointSize(12)
        font7.setUnderline(False)
        self.label_190.setFont(font7)
        self.label_190.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_190.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_190.setMargin(8)

        self.gridLayout_56.addWidget(self.label_190, 7, 1, 1, 1)

        self.label_191 = QLabel(self.frame_Lifetime_tabInput)
        self.label_191.setObjectName(u"label_191")
        self.label_191.setFont(font7)
        self.label_191.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_191.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_191.setMargin(8)

        self.gridLayout_56.addWidget(self.label_191, 6, 0, 1, 1)

        self.label_192 = QLabel(self.frame_Lifetime_tabInput)
        self.label_192.setObjectName(u"label_192")
        self.label_192.setFont(font7)
        self.label_192.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_192.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_192.setMargin(8)

        self.gridLayout_56.addWidget(self.label_192, 3, 0, 1, 1)

        self.label_193 = QLabel(self.frame_Lifetime_tabInput)
        self.label_193.setObjectName(u"label_193")
        self.label_193.setFont(font7)
        self.label_193.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_193.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_193.setMargin(8)

        self.gridLayout_56.addWidget(self.label_193, 10, 0, 1, 1)

        self.label_194 = QLabel(self.frame_Lifetime_tabInput)
        self.label_194.setObjectName(u"label_194")
        self.label_194.setFont(font7)
        self.label_194.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_194.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_194.setMargin(8)

        self.gridLayout_56.addWidget(self.label_194, 4, 0, 1, 1)

        self.label_195 = QLabel(self.frame_Lifetime_tabInput)
        self.label_195.setObjectName(u"label_195")
        self.label_195.setFont(font7)
        self.label_195.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_195.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_195.setMargin(8)

        self.gridLayout_56.addWidget(self.label_195, 5, 1, 1, 1)

        self.label_196 = QLabel(self.frame_Lifetime_tabInput)
        self.label_196.setObjectName(u"label_196")
        self.label_196.setFont(font7)
        self.label_196.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_196.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_196.setMargin(8)

        self.gridLayout_56.addWidget(self.label_196, 11, 0, 1, 1)

        self.label_197 = QLabel(self.frame_Lifetime_tabInput)
        self.label_197.setObjectName(u"label_197")
        self.label_197.setFont(font7)
        self.label_197.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_197.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_197.setMargin(8)

        self.gridLayout_56.addWidget(self.label_197, 4, 1, 1, 1)

        self.verticalSpacer_26 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Fixed)

        self.gridLayout_56.addItem(self.verticalSpacer_26, 0, 0, 1, 1)

        self.label_198 = QLabel(self.frame_Lifetime_tabInput)
        self.label_198.setObjectName(u"label_198")
        self.label_198.setFont(font7)
        self.label_198.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_198.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_198.setMargin(8)

        self.gridLayout_56.addWidget(self.label_198, 9, 0, 1, 1)

        self.verticalSpacer_27 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_56.addItem(self.verticalSpacer_27, 14, 0, 1, 1)

        self.label_199 = QLabel(self.frame_Lifetime_tabInput)
        self.label_199.setObjectName(u"label_199")
        self.label_199.setFont(font7)
        self.label_199.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_199.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_199.setMargin(8)

        self.gridLayout_56.addWidget(self.label_199, 13, 1, 1, 1)

        self.label_200 = QLabel(self.frame_Lifetime_tabInput)
        self.label_200.setObjectName(u"label_200")
        self.label_200.setFont(font7)
        self.label_200.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_200.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_200.setMargin(8)

        self.gridLayout_56.addWidget(self.label_200, 8, 0, 1, 1)

        self.label_201 = QLabel(self.frame_Lifetime_tabInput)
        self.label_201.setObjectName(u"label_201")
        self.label_201.setFont(font7)
        self.label_201.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_201.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_201.setMargin(8)

        self.gridLayout_56.addWidget(self.label_201, 13, 0, 1, 1)

        self.label_202 = QLabel(self.frame_Lifetime_tabInput)
        self.label_202.setObjectName(u"label_202")
        self.label_202.setFont(font7)
        self.label_202.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_202.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_202.setMargin(8)

        self.gridLayout_56.addWidget(self.label_202, 6, 1, 1, 1)

        self.label_203 = QLabel(self.frame_Lifetime_tabInput)
        self.label_203.setObjectName(u"label_203")
        self.label_203.setFont(font7)
        self.label_203.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_203.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_203.setMargin(8)

        self.gridLayout_56.addWidget(self.label_203, 5, 0, 1, 1)

        self.label_204 = QLabel(self.frame_Lifetime_tabInput)
        self.label_204.setObjectName(u"label_204")
        self.label_204.setMaximumSize(QSize(100, 16777215))
        self.label_204.setFont(font7)
        self.label_204.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_204.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_204.setMargin(8)

        self.gridLayout_56.addWidget(self.label_204, 3, 1, 1, 1)

        self.label_205 = QLabel(self.frame_Lifetime_tabInput)
        self.label_205.setObjectName(u"label_205")
        self.label_205.setFont(font7)
        self.label_205.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_205.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_205.setMargin(8)

        self.gridLayout_56.addWidget(self.label_205, 8, 1, 1, 1)

        self.label_206 = QLabel(self.frame_Lifetime_tabInput)
        self.label_206.setObjectName(u"label_206")
        self.label_206.setFont(font7)
        self.label_206.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_206.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_206.setMargin(8)

        self.gridLayout_56.addWidget(self.label_206, 12, 0, 1, 1)

        self.label_207 = QLabel(self.frame_Lifetime_tabInput)
        self.label_207.setObjectName(u"label_207")
        self.label_207.setFont(font7)
        self.label_207.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_207.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_207.setMargin(8)

        self.gridLayout_56.addWidget(self.label_207, 11, 1, 1, 1)

        self.label_208 = QLabel(self.frame_Lifetime_tabInput)
        self.label_208.setObjectName(u"label_208")
        self.label_208.setFont(font7)
        self.label_208.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_208.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_208.setMargin(8)

        self.gridLayout_56.addWidget(self.label_208, 9, 1, 1, 1)

        self.label_209 = QLabel(self.frame_Lifetime_tabInput)
        self.label_209.setObjectName(u"label_209")
        self.label_209.setFont(font7)
        self.label_209.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_209.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_209.setMargin(8)

        self.gridLayout_56.addWidget(self.label_209, 7, 0, 1, 1)

        self.label_210 = QLabel(self.frame_Lifetime_tabInput)
        self.label_210.setObjectName(u"label_210")
        self.label_210.setFont(font7)
        self.label_210.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_210.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_210.setMargin(8)

        self.gridLayout_56.addWidget(self.label_210, 12, 1, 1, 1)

        self.line_7 = QFrame(self.frame_Lifetime_tabInput)
        self.line_7.setObjectName(u"line_7")
        self.line_7.setFrameShape(QFrame.HLine)
        self.line_7.setFrameShadow(QFrame.Sunken)

        self.gridLayout_56.addWidget(self.line_7, 2, 0, 1, 2)

        self.label_211 = QLabel(self.frame_Lifetime_tabInput)
        self.label_211.setObjectName(u"label_211")
        self.label_211.setFont(font7)
        self.label_211.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_211.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_211.setMargin(8)

        self.gridLayout_56.addWidget(self.label_211, 10, 1, 1, 1)


        self.gridLayout_55.addWidget(self.frame_Lifetime_tabInput, 0, 0, 1, 1)

        self.Lifetime_tabWidget.addTab(self.Lifetime_tabInput, "")
        self.Lifetime_tabRodPos = QWidget()
        self.Lifetime_tabRodPos.setObjectName(u"Lifetime_tabRodPos")
        self.gridLayout_57 = QGridLayout(self.Lifetime_tabRodPos)
        self.gridLayout_57.setObjectName(u"gridLayout_57")
        self.gridLayout_57.setContentsMargins(0, 0, 0, 0)
        self.frame_Lifetime_tabRodPos = QFrame(self.Lifetime_tabRodPos)
        self.frame_Lifetime_tabRodPos.setObjectName(u"frame_Lifetime_tabRodPos")
        self.frame_Lifetime_tabRodPos.setAutoFillBackground(False)
        self.frame_Lifetime_tabRodPos.setStyleSheet(u"background-color: rgb(51, 58, 72);")
        self.gridLayout_58 = QGridLayout(self.frame_Lifetime_tabRodPos)
        self.gridLayout_58.setObjectName(u"gridLayout_58")
        self.verticalSpacer_Lifetime_tabRodPos02 = QSpacerItem(20, 807, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_58.addItem(self.verticalSpacer_Lifetime_tabRodPos02, 3, 0, 1, 2)

        self.horizontalSpacer_Lifetime_tabRodPos01 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_58.addItem(self.horizontalSpacer_Lifetime_tabRodPos01, 2, 0, 1, 1)

        self.Lifetime_widgetChart = QWidget(self.frame_Lifetime_tabRodPos)
        self.Lifetime_widgetChart.setObjectName(u"Lifetime_widgetChart")
        self.Lifetime_widgetChart.setMinimumSize(QSize(660, 660))

        self.gridLayout_58.addWidget(self.Lifetime_widgetChart, 1, 1, 2, 1)

        self.horizontalSpacer_Lifetime_tabRodPos02 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_58.addItem(self.horizontalSpacer_Lifetime_tabRodPos02, 2, 2, 1, 1)

        self.Lifetime_widgetChart001 = QFrame(self.frame_Lifetime_tabRodPos)
        self.Lifetime_widgetChart001.setObjectName(u"Lifetime_widgetChart001")
        self.Lifetime_widgetChart001.setFrameShape(QFrame.StyledPanel)
        self.Lifetime_widgetChart001.setFrameShadow(QFrame.Raised)
        self.gridLayout_59 = QGridLayout(self.Lifetime_widgetChart001)
        self.gridLayout_59.setSpacing(6)
        self.gridLayout_59.setObjectName(u"gridLayout_59")
        self.gridLayout_59.setContentsMargins(-1, 9, 9, -1)

        self.gridLayout_58.addWidget(self.Lifetime_widgetChart001, 0, 0, 1, 1)


        self.gridLayout_57.addWidget(self.frame_Lifetime_tabRodPos, 0, 0, 1, 1)

        self.Lifetime_tabWidget.addTab(self.Lifetime_tabRodPos, "")
        self.Lifetime_tabReport = QWidget()
        self.Lifetime_tabReport.setObjectName(u"Lifetime_tabReport")
        self.Lifetime_tabReport.setStyleSheet(u"QWidget{\n"
"\n"
"}")
        self.gridLayout_69 = QGridLayout(self.Lifetime_tabReport)
        self.gridLayout_69.setObjectName(u"gridLayout_69")
        self.gridLayout_69.setHorizontalSpacing(5)
        self.gridLayout_69.setContentsMargins(0, 0, 0, 0)
        self.frame_Lifetime_tabReport = QWidget(self.Lifetime_tabReport)
        self.frame_Lifetime_tabReport.setObjectName(u"frame_Lifetime_tabReport")
        self.frame_Lifetime_tabReport.setStyleSheet(u"background-color: rgb(51, 58, 72);")
        self.gridLayout_70 = QGridLayout(self.frame_Lifetime_tabReport)
        self.gridLayout_70.setObjectName(u"gridLayout_70")
        self.verticalSpacer_Lifetime_tabReport01 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_70.addItem(self.verticalSpacer_Lifetime_tabReport01, 1, 1, 1, 1)

        self.horizontalSpacer_Lifetime_tabReport02 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_70.addItem(self.horizontalSpacer_Lifetime_tabReport02, 0, 2, 1, 1)

        self.horizontalSpacer_Lifetime_tabReport01 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_70.addItem(self.horizontalSpacer_Lifetime_tabReport01, 0, 0, 1, 1)


        self.gridLayout_69.addWidget(self.frame_Lifetime_tabReport, 0, 0, 1, 1)

        self.Lifetime_tabWidget.addTab(self.Lifetime_tabReport, "")

        self.gridLayout_Lifetime_frame_MainWindow.addWidget(self.Lifetime_tabWidget, 0, 1, 5, 1)

        self.verticalSpacer_35 = QSpacerItem(20, 4000, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_Lifetime_frame_MainWindow.addItem(self.verticalSpacer_35, 3, 0, 1, 1)


        self.gridLayout.addWidget(self.Lifetime_frame_MainWindow, 0, 0, 1, 1)


        self.retranslateUi(unitWidget_Lifetime)

        self.Lifetime_tabWidget.setCurrentIndex(1)


        QMetaObject.connectSlotsByName(unitWidget_Lifetime)
    # setupUi

    def retranslateUi(self, unitWidget_Lifetime):
        unitWidget_Lifetime.setWindowTitle(QCoreApplication.translate("unitWidget_Lifetime", u"Form", None))
        self.Lifetime_save_button.setText(QCoreApplication.translate("unitWidget_Lifetime", u"Save", None))
        self.Lifetime_run_button.setText(QCoreApplication.translate("unitWidget_Lifetime", u"Run", None))
        self.LabelSub_LF07.setText(QCoreApplication.translate("unitWidget_Lifetime", u"<html><head/><body><p>Bank 4 Position<br/>(cm withdrawn)</p></body></html>", None))
        self.LabelSub_LF04.setText(QCoreApplication.translate("unitWidget_Lifetime", u"Depletion Interval\n"
"(MWD/MTU)", None))
        self.LabelSub_LF01.setText(QCoreApplication.translate("unitWidget_Lifetime", u"Core Power (%)", None))
        self.LabelSub_LF06.setText(QCoreApplication.translate("unitWidget_Lifetime", u"<html><head/><body><p>Bank 5 Position<br/>(cm withdrawn)</p></body></html>", None))
        self.LabelSub_LF05.setText(QCoreApplication.translate("unitWidget_Lifetime", u"<html><head/><body><p>Bank P Position<br/>(cm withdrawn)</p></body></html>", None))
        self.LabelSub_LF02.setText(QCoreApplication.translate("unitWidget_Lifetime", u"<html><head/><body><p>Cycle Burnup<br>(MWD/MTU)</p></body></html>", None))
        self.LabelSub_LF08.setText(QCoreApplication.translate("unitWidget_Lifetime", u"<html><head/><body><p>Bank 3 Position<br/>(cm withdrawn)</p></body></html>", None))
        self.LabelSub_LF03.setText(QCoreApplication.translate("unitWidget_Lifetime", u"Stopping Criterion\n"
"(keff)", None))
        self.LabelTitle_LF02.setText(QCoreApplication.translate("unitWidget_Lifetime", u"Calculation Option", None))
        self.Lifetime_InpOpt2_NDR.setText(QCoreApplication.translate("unitWidget_Lifetime", u"NDR", None))
        self.LabelTitle_LF01.setText(QCoreApplication.translate("unitWidget_Lifetime", u"Input Setup", None))
#if QT_CONFIG(statustip)
        self.Lifetime_InpOpt1_Snapshot.setStatusTip("")
#endif // QT_CONFIG(statustip)
        self.Lifetime_InpOpt1_Snapshot.setText(QCoreApplication.translate("unitWidget_Lifetime", u"Snapshot", None))
        ___qtablewidgetitem = self.Lifetime_DB.horizontalHeaderItem(0)
        ___qtablewidgetitem.setText(QCoreApplication.translate("unitWidget_Lifetime", u"Database Name", None));
        ___qtablewidgetitem1 = self.Lifetime_DB.horizontalHeaderItem(1)
        ___qtablewidgetitem1.setText(QCoreApplication.translate("unitWidget_Lifetime", u"Avg. Temperature", None));
        ___qtablewidgetitem2 = self.Lifetime_DB.horizontalHeaderItem(2)
        ___qtablewidgetitem2.setText(QCoreApplication.translate("unitWidget_Lifetime", u"Boron Concentration", None));
        ___qtablewidgetitem3 = self.Lifetime_DB.horizontalHeaderItem(3)
        ___qtablewidgetitem3.setText(QCoreApplication.translate("unitWidget_Lifetime", u"Bank5 Position", None));
        ___qtablewidgetitem4 = self.Lifetime_DB.horizontalHeaderItem(4)
        ___qtablewidgetitem4.setText(QCoreApplication.translate("unitWidget_Lifetime", u"Bank4 Position", None));
        ___qtablewidgetitem5 = self.Lifetime_DB.horizontalHeaderItem(5)
        ___qtablewidgetitem5.setText(QCoreApplication.translate("unitWidget_Lifetime", u"Bank3 Position", None));
        ___qtablewidgetitem6 = self.Lifetime_DB.horizontalHeaderItem(6)
        ___qtablewidgetitem6.setText(QCoreApplication.translate("unitWidget_Lifetime", u"BankP Position", None));
        ___qtablewidgetitem7 = self.Lifetime_DB.verticalHeaderItem(0)
        ___qtablewidgetitem7.setText(QCoreApplication.translate("unitWidget_Lifetime", u"1", None));
        ___qtablewidgetitem8 = self.Lifetime_DB.verticalHeaderItem(1)
        ___qtablewidgetitem8.setText(QCoreApplication.translate("unitWidget_Lifetime", u"2", None));
        ___qtablewidgetitem9 = self.Lifetime_DB.verticalHeaderItem(2)
        ___qtablewidgetitem9.setText(QCoreApplication.translate("unitWidget_Lifetime", u"3", None));
        ___qtablewidgetitem10 = self.Lifetime_DB.verticalHeaderItem(3)
        ___qtablewidgetitem10.setText(QCoreApplication.translate("unitWidget_Lifetime", u"4", None));
        self.Lifetime_tabWidget.setTabText(self.Lifetime_tabWidget.indexOf(self.Lifetime_Database), QCoreApplication.translate("unitWidget_Lifetime", u"Database", None))
        self.label_189.setText(QCoreApplication.translate("unitWidget_Lifetime", u"    Lifetime Calculation Result      ", None))
        self.label_190.setText(QCoreApplication.translate("unitWidget_Lifetime", u"-", None))
        self.label_191.setText(QCoreApplication.translate("unitWidget_Lifetime", u"Inlet Temperature (\u2109)", None))
        self.label_192.setText(QCoreApplication.translate("unitWidget_Lifetime", u"Plant", None))
        self.label_193.setText(QCoreApplication.translate("unitWidget_Lifetime", u"N-1 Rod Worth (%\u0394\u03c1)", None))
        self.label_194.setText(QCoreApplication.translate("unitWidget_Lifetime", u"Cycle", None))
        self.label_195.setText(QCoreApplication.translate("unitWidget_Lifetime", u"0.00", None))
        self.label_196.setText(QCoreApplication.translate("unitWidget_Lifetime", u"Total Defect (%\u0394\u03c1)", None))
        self.label_197.setText(QCoreApplication.translate("unitWidget_Lifetime", u"13", None))
        self.label_198.setText(QCoreApplication.translate("unitWidget_Lifetime", u"Stuck Rod Worth (%\u0394\u03c1)", None))
        self.label_199.setText(QCoreApplication.translate("unitWidget_Lifetime", u"-", None))
        self.label_200.setText(QCoreApplication.translate("unitWidget_Lifetime", u"Stuck Rod (N-1 Condition)", None))
        self.label_201.setText(QCoreApplication.translate("unitWidget_Lifetime", u"Required Value (%\u0394\u03c1)", None))
        self.label_202.setText(QCoreApplication.translate("unitWidget_Lifetime", u"-", None))
        self.label_203.setText(QCoreApplication.translate("unitWidget_Lifetime", u"Burnup (MWD/MTU)", None))
        self.label_204.setText(QCoreApplication.translate("unitWidget_Lifetime", u"SKN01", None))
        self.label_205.setText(QCoreApplication.translate("unitWidget_Lifetime", u"-", None))
        self.label_206.setText(QCoreApplication.translate("unitWidget_Lifetime", u"Shutdown Margin (%\u0394\u03c1)", None))
        self.label_207.setText(QCoreApplication.translate("unitWidget_Lifetime", u"-", None))
        self.label_208.setText(QCoreApplication.translate("unitWidget_Lifetime", u"-", None))
        self.label_209.setText(QCoreApplication.translate("unitWidget_Lifetime", u"CEA Configuration", None))
        self.label_210.setText(QCoreApplication.translate("unitWidget_Lifetime", u"-", None))
        self.label_211.setText(QCoreApplication.translate("unitWidget_Lifetime", u"-", None))
        self.Lifetime_tabWidget.setTabText(self.Lifetime_tabWidget.indexOf(self.Lifetime_tabInput), QCoreApplication.translate("unitWidget_Lifetime", u"Input", None))
        self.Lifetime_tabWidget.setTabText(self.Lifetime_tabWidget.indexOf(self.Lifetime_tabRodPos), QCoreApplication.translate("unitWidget_Lifetime", u"Rod Input", None))
        self.Lifetime_tabWidget.setTabText(self.Lifetime_tabWidget.indexOf(self.Lifetime_tabReport), QCoreApplication.translate("unitWidget_Lifetime", u"Report", None))
    # retranslateUi

