# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'unitWidget_PA.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class Ui_unitWidget_PA(object):
    def setupUi(self, unitWidget_PA):
        if not unitWidget_PA.objectName():
            unitWidget_PA.setObjectName(u"unitWidget_PA")
        unitWidget_PA.resize(1945, 1354)
        self.gridLayout = QGridLayout(unitWidget_PA)
        self.gridLayout.setObjectName(u"gridLayout")
        self.PA_frame_MainWindow = QFrame(unitWidget_PA)
        self.PA_frame_MainWindow.setObjectName(u"PA_frame_MainWindow")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.PA_frame_MainWindow.sizePolicy().hasHeightForWidth())
        self.PA_frame_MainWindow.setSizePolicy(sizePolicy)
        self.PA_frame_MainWindow.setStyleSheet(u"background-color: rgb(44, 49, 60);\n"
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
        self.PA_frame_MainWindow.setFrameShape(QFrame.NoFrame)
        self.PA_frame_MainWindow.setFrameShadow(QFrame.Raised)
        self.gridLayout_PA_frame_MainWindow = QGridLayout(self.PA_frame_MainWindow)
        self.gridLayout_PA_frame_MainWindow.setSpacing(3)
        self.gridLayout_PA_frame_MainWindow.setObjectName(u"gridLayout_PA_frame_MainWindow")
        self.PA_Main07_CalcButton = QFrame(self.PA_frame_MainWindow)
        self.PA_Main07_CalcButton.setObjectName(u"PA_Main07_CalcButton")
        self.PA_Main07_CalcButton.setMinimumSize(QSize(0, 0))
        self.PA_Main07_CalcButton.setFrameShape(QFrame.StyledPanel)
        self.PA_Main07_CalcButton.setFrameShadow(QFrame.Raised)
        self.PA_Control_CalcButton_Grid = QGridLayout(self.PA_Main07_CalcButton)
        self.PA_Control_CalcButton_Grid.setObjectName(u"PA_Control_CalcButton_Grid")
        self.PA_Control_CalcButton_Grid.setContentsMargins(0, 0, 0, 0)
        self.PA_save_button = QPushButton(self.PA_Main07_CalcButton)
        self.PA_save_button.setObjectName(u"PA_save_button")
        self.PA_save_button.setMinimumSize(QSize(0, 50))
        font = QFont()
        font.setFamily(u"Segoe UI")
        font.setPointSize(14)
        self.PA_save_button.setFont(font)
        self.PA_save_button.setStyleSheet(u"QPushButton {\n"
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

        self.PA_Control_CalcButton_Grid.addWidget(self.PA_save_button, 1, 0, 1, 1)

        self.verticalSpacer_16 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.PA_Control_CalcButton_Grid.addItem(self.verticalSpacer_16, 0, 1, 1, 1)

        self.PA_run_button = QPushButton(self.PA_Main07_CalcButton)
        self.PA_run_button.setObjectName(u"PA_run_button")
        self.PA_run_button.setMinimumSize(QSize(220, 50))
        self.PA_run_button.setFont(font)
        self.PA_run_button.setStyleSheet(u"QPushButton {\n"
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

        self.PA_Control_CalcButton_Grid.addWidget(self.PA_run_button, 1, 1, 1, 1)


        self.gridLayout_PA_frame_MainWindow.addWidget(self.PA_Main07_CalcButton, 4, 0, 1, 1)

        self.PA_InputSet_Total = QFrame(self.PA_frame_MainWindow)
        self.PA_InputSet_Total.setObjectName(u"PA_InputSet_Total")
        self.PA_InputSet_Total.setFrameShape(QFrame.StyledPanel)
        self.PA_InputSet_Total.setFrameShadow(QFrame.Raised)
        self.gridLayout_PA_InputSet_Total = QGridLayout(self.PA_InputSet_Total)
        self.gridLayout_PA_InputSet_Total.setSpacing(0)
        self.gridLayout_PA_InputSet_Total.setObjectName(u"gridLayout_PA_InputSet_Total")
        self.gridLayout_PA_InputSet_Total.setContentsMargins(0, 0, 0, 0)
        self.verticalSpacer_99 = QSpacerItem(20, 4000, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_PA_InputSet_Total.addItem(self.verticalSpacer_99, 1, 0, 1, 1)

        self.gridLayout_PA_Total = QGridLayout()
        self.gridLayout_PA_Total.setObjectName(u"gridLayout_PA_Total")
        self.PA_Main01 = QFrame(self.PA_InputSet_Total)
        self.PA_Main01.setObjectName(u"PA_Main01")
        self.PA_Main01.setFrameShape(QFrame.StyledPanel)
        self.PA_Main01.setFrameShadow(QFrame.Raised)
        self.gridLayout_Lifetime_Main01_InputSetup_7 = QGridLayout(self.PA_Main01)
        self.gridLayout_Lifetime_Main01_InputSetup_7.setObjectName(u"gridLayout_Lifetime_Main01_InputSetup_7")
        self.PA_Main01Grid = QGridLayout()
        self.PA_Main01Grid.setSpacing(6)
        self.PA_Main01Grid.setObjectName(u"PA_Main01Grid")
        self.PA_Main01Grid.setContentsMargins(0, 0, 0, 0)
        self.PA_InpOpt2_NDR = QRadioButton(self.PA_Main01)
        self.PA_InpOpt2_NDR.setObjectName(u"PA_InpOpt2_NDR")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.PA_InpOpt2_NDR.sizePolicy().hasHeightForWidth())
        self.PA_InpOpt2_NDR.setSizePolicy(sizePolicy1)
        self.PA_InpOpt2_NDR.setMinimumSize(QSize(0, 0))
        font1 = QFont()
        font1.setFamily(u"Segoe UI")
        font1.setPointSize(12)
        self.PA_InpOpt2_NDR.setFont(font1)
        self.PA_InpOpt2_NDR.setAutoExclusive(False)

        self.PA_Main01Grid.addWidget(self.PA_InpOpt2_NDR, 2, 1, 1, 2)

        self.horizontalSpacer_PA_Main02_02 = QSpacerItem(1, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.PA_Main01Grid.addItem(self.horizontalSpacer_PA_Main02_02, 2, 3, 1, 1)

        self.horizontalSpacer_PA_Main02_01 = QSpacerItem(1, 20, QSizePolicy.Fixed, QSizePolicy.Minimum)

        self.PA_Main01Grid.addItem(self.horizontalSpacer_PA_Main02_01, 1, 0, 2, 1)

        self.PA_Snapshot = QComboBox(self.PA_Main01)
        self.PA_Snapshot.setObjectName(u"PA_Snapshot")
        sizePolicy1.setHeightForWidth(self.PA_Snapshot.sizePolicy().hasHeightForWidth())
        self.PA_Snapshot.setSizePolicy(sizePolicy1)
        self.PA_Snapshot.setMinimumSize(QSize(60, 0))
        self.PA_Snapshot.setMaximumSize(QSize(16777215, 30))
        font2 = QFont()
        font2.setFamily(u"Segoe UI")
        font2.setPointSize(10)
        self.PA_Snapshot.setFont(font2)

        self.PA_Main01Grid.addWidget(self.PA_Snapshot, 1, 3, 1, 2)

        self.PA_InpOpt1_Snapshot = QRadioButton(self.PA_Main01)
        self.PA_InpOpt1_Snapshot.setObjectName(u"PA_InpOpt1_Snapshot")
        sizePolicy1.setHeightForWidth(self.PA_InpOpt1_Snapshot.sizePolicy().hasHeightForWidth())
        self.PA_InpOpt1_Snapshot.setSizePolicy(sizePolicy1)
        self.PA_InpOpt1_Snapshot.setMinimumSize(QSize(0, 0))
        self.PA_InpOpt1_Snapshot.setFont(font1)
#if QT_CONFIG(whatsthis)
        self.PA_InpOpt1_Snapshot.setWhatsThis(u"")
#endif // QT_CONFIG(whatsthis)
        self.PA_InpOpt1_Snapshot.setAutoExclusive(False)

        self.PA_Main01Grid.addWidget(self.PA_InpOpt1_Snapshot, 1, 1, 1, 2)

        self.LabelTitle_PA01 = QLabel(self.PA_Main01)
        self.LabelTitle_PA01.setObjectName(u"LabelTitle_PA01")
        self.LabelTitle_PA01.setMinimumSize(QSize(0, 0))
        font3 = QFont()
        font3.setFamily(u"Segoe UI")
        font3.setPointSize(14)
        font3.setBold(True)
        font3.setWeight(75)
        self.LabelTitle_PA01.setFont(font3)

        self.PA_Main01Grid.addWidget(self.LabelTitle_PA01, 0, 0, 1, 5)


        self.gridLayout_Lifetime_Main01_InputSetup_7.addLayout(self.PA_Main01Grid, 0, 0, 1, 1)


        self.gridLayout_PA_Total.addWidget(self.PA_Main01, 0, 0, 1, 1)

        self.PA_Main05 = QFrame(self.PA_InputSet_Total)
        self.PA_Main05.setObjectName(u"PA_Main05")
        self.PA_Main05.setMaximumSize(QSize(16777215, 16777215))
        self.PA_Main05.setFrameShape(QFrame.StyledPanel)
        self.PA_Main05.setFrameShadow(QFrame.Raised)
        self.gridLayout_94 = QGridLayout(self.PA_Main05)
        self.gridLayout_94.setObjectName(u"gridLayout_94")
        self.PA_Main05Grid = QGridLayout()
        self.PA_Main05Grid.setObjectName(u"PA_Main05Grid")
        self.gridLayout_PA_Main06_Button_3 = QGridLayout()
        self.gridLayout_PA_Main06_Button_3.setObjectName(u"gridLayout_PA_Main06_Button_3")
        self.horizontalSpacer_PA_Main06Button_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_PA_Main06_Button_3.addItem(self.horizontalSpacer_PA_Main06Button_3, 0, 0, 1, 1)

        self.PA_Insert02 = QPushButton(self.PA_Main05)
        self.PA_Insert02.setObjectName(u"PA_Insert02")
        self.PA_Insert02.setMinimumSize(QSize(60, 0))
        self.PA_Insert02.setFont(font1)
        self.PA_Insert02.setStyleSheet(u"QPushButton {\n"
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

        self.gridLayout_PA_Main06_Button_3.addWidget(self.PA_Insert02, 0, 1, 1, 1)


        self.PA_Main05Grid.addLayout(self.gridLayout_PA_Main06_Button_3, 3, 1, 1, 3)

        self.LabelSub_PA06 = QLabel(self.PA_Main05)
        self.LabelSub_PA06.setObjectName(u"LabelSub_PA06")
        sizePolicy1.setHeightForWidth(self.LabelSub_PA06.sizePolicy().hasHeightForWidth())
        self.LabelSub_PA06.setSizePolicy(sizePolicy1)
        self.LabelSub_PA06.setMinimumSize(QSize(105, 0))
        font4 = QFont()
        font4.setFamily(u"Segoe UI")
        font4.setPointSize(11)
        self.LabelSub_PA06.setFont(font4)

        self.PA_Main05Grid.addWidget(self.LabelSub_PA06, 1, 1, 1, 1)

        self.gridLayout_PA_Main07_Input03 = QGridLayout()
        self.gridLayout_PA_Main07_Input03.setObjectName(u"gridLayout_PA_Main07_Input03")
        self.verticalSpacer_100 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.gridLayout_PA_Main07_Input03.addItem(self.verticalSpacer_100, 0, 0, 1, 1)

        self.verticalSpacer_101 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.gridLayout_PA_Main07_Input03.addItem(self.verticalSpacer_101, 2, 0, 1, 1)

        self.PA_Input07 = QDoubleSpinBox(self.PA_Main05)
        self.PA_Input07.setObjectName(u"PA_Input07")
        sizePolicy1.setHeightForWidth(self.PA_Input07.sizePolicy().hasHeightForWidth())
        self.PA_Input07.setSizePolicy(sizePolicy1)
        self.PA_Input07.setMinimumSize(QSize(0, 0))
        self.PA_Input07.setMaximumSize(QSize(16777215, 16777215))
        self.PA_Input07.setFont(font2)
        self.PA_Input07.setStyleSheet(u"padding: 3px;")
        self.PA_Input07.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.PA_Input07.setProperty("showGroupSeparator", True)
        self.PA_Input07.setDecimals(2)
        self.PA_Input07.setMinimum(0.000000000000000)
        self.PA_Input07.setMaximum(100.000000000000000)
        self.PA_Input07.setSingleStep(1.000000000000000)
        self.PA_Input07.setStepType(QAbstractSpinBox.DefaultStepType)
        self.PA_Input07.setValue(0.000000000000000)

        self.gridLayout_PA_Main07_Input03.addWidget(self.PA_Input07, 1, 0, 1, 1)


        self.PA_Main05Grid.addLayout(self.gridLayout_PA_Main07_Input03, 2, 3, 1, 1)

        self.gridLayout_PA_Main07_Input02 = QGridLayout()
        self.gridLayout_PA_Main07_Input02.setObjectName(u"gridLayout_PA_Main07_Input02")
        self.PA_Input06 = QDoubleSpinBox(self.PA_Main05)
        self.PA_Input06.setObjectName(u"PA_Input06")
        sizePolicy1.setHeightForWidth(self.PA_Input06.sizePolicy().hasHeightForWidth())
        self.PA_Input06.setSizePolicy(sizePolicy1)
        self.PA_Input06.setMinimumSize(QSize(0, 0))
        self.PA_Input06.setMaximumSize(QSize(16777215, 16777215))
        self.PA_Input06.setFont(font2)
        self.PA_Input06.setStyleSheet(u"padding: 3px;")
        self.PA_Input06.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.PA_Input06.setProperty("showGroupSeparator", True)
        self.PA_Input06.setDecimals(3)
        self.PA_Input06.setMinimum(0.000000000000000)
        self.PA_Input06.setMaximum(100.000000000000000)
        self.PA_Input06.setSingleStep(0.100000000000000)
        self.PA_Input06.setStepType(QAbstractSpinBox.DefaultStepType)
        self.PA_Input06.setValue(3.000000000000000)

        self.gridLayout_PA_Main07_Input02.addWidget(self.PA_Input06, 1, 0, 1, 1)

        self.verticalSpacer_102 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.gridLayout_PA_Main07_Input02.addItem(self.verticalSpacer_102, 0, 0, 1, 1)

        self.verticalSpacer_103 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.gridLayout_PA_Main07_Input02.addItem(self.verticalSpacer_103, 2, 0, 1, 1)


        self.PA_Main05Grid.addLayout(self.gridLayout_PA_Main07_Input02, 1, 3, 1, 1)

        self.LabelTitle_PA05 = QLabel(self.PA_Main05)
        self.LabelTitle_PA05.setObjectName(u"LabelTitle_PA05")
        self.LabelTitle_PA05.setMinimumSize(QSize(0, 0))
        self.LabelTitle_PA05.setFont(font3)

        self.PA_Main05Grid.addWidget(self.LabelTitle_PA05, 0, 0, 1, 4)

        self.horizontalSpacer_PA_Main05_7 = QSpacerItem(5, 0, QSizePolicy.Fixed, QSizePolicy.Minimum)

        self.PA_Main05Grid.addItem(self.horizontalSpacer_PA_Main05_7, 1, 0, 3, 1)

        self.LabelSub_PA07 = QLabel(self.PA_Main05)
        self.LabelSub_PA07.setObjectName(u"LabelSub_PA07")
        sizePolicy1.setHeightForWidth(self.LabelSub_PA07.sizePolicy().hasHeightForWidth())
        self.LabelSub_PA07.setSizePolicy(sizePolicy1)
        self.LabelSub_PA07.setMinimumSize(QSize(105, 0))
        self.LabelSub_PA07.setFont(font4)

        self.PA_Main05Grid.addWidget(self.LabelSub_PA07, 2, 1, 1, 1)

        self.horizontalSpacer_PA_Main05_6 = QSpacerItem(1, 0, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.PA_Main05Grid.addItem(self.horizontalSpacer_PA_Main05_6, 1, 2, 2, 1)


        self.gridLayout_94.addLayout(self.PA_Main05Grid, 0, 0, 1, 1)


        self.gridLayout_PA_Total.addWidget(self.PA_Main05, 4, 0, 1, 1)

        self.PA_Main04 = QFrame(self.PA_InputSet_Total)
        self.PA_Main04.setObjectName(u"PA_Main04")
        self.PA_Main04.setFrameShape(QFrame.StyledPanel)
        self.PA_Main04.setFrameShadow(QFrame.Raised)
        self.gridLayout_93 = QGridLayout(self.PA_Main04)
        self.gridLayout_93.setObjectName(u"gridLayout_93")
        self.PA_Main04Grid = QGridLayout()
        self.PA_Main04Grid.setObjectName(u"PA_Main04Grid")
        self.LabelSub_PA03 = QLabel(self.PA_Main04)
        self.LabelSub_PA03.setObjectName(u"LabelSub_PA03")
        sizePolicy1.setHeightForWidth(self.LabelSub_PA03.sizePolicy().hasHeightForWidth())
        self.LabelSub_PA03.setSizePolicy(sizePolicy1)
        self.LabelSub_PA03.setMinimumSize(QSize(105, 0))
        self.LabelSub_PA03.setFont(font2)

        self.PA_Main04Grid.addWidget(self.LabelSub_PA03, 1, 1, 1, 1)

        self.gridLayout_PA_Main05_Input02_4 = QGridLayout()
        self.gridLayout_PA_Main05_Input02_4.setObjectName(u"gridLayout_PA_Main05_Input02_4")
        self.verticalSpacer_117 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_PA_Main05_Input02_4.addItem(self.verticalSpacer_117, 2, 0, 1, 1)

        self.verticalSpacer_118 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_PA_Main05_Input02_4.addItem(self.verticalSpacer_118, 0, 0, 1, 1)

        self.PA_Input04 = QLineEdit(self.PA_Main04)
        self.PA_Input04.setObjectName(u"PA_Input04")
        sizePolicy2 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.PA_Input04.sizePolicy().hasHeightForWidth())
        self.PA_Input04.setSizePolicy(sizePolicy2)
        self.PA_Input04.setMinimumSize(QSize(0, 0))
        self.PA_Input04.setMaximumSize(QSize(100, 16777215))

        self.gridLayout_PA_Main05_Input02_4.addWidget(self.PA_Input04, 1, 0, 1, 1)


        self.PA_Main04Grid.addLayout(self.gridLayout_PA_Main05_Input02_4, 2, 3, 1, 1)

        self.horizontalSpacer_PA_Main05_5 = QSpacerItem(5, 0, QSizePolicy.Fixed, QSizePolicy.Minimum)

        self.PA_Main04Grid.addItem(self.horizontalSpacer_PA_Main05_5, 1, 0, 4, 1)

        self.gridLayout_PA_Main05_Input02_3 = QGridLayout()
        self.gridLayout_PA_Main05_Input02_3.setObjectName(u"gridLayout_PA_Main05_Input02_3")
        self.verticalSpacer_119 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_PA_Main05_Input02_3.addItem(self.verticalSpacer_119, 0, 0, 1, 1)

        self.verticalSpacer_120 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_PA_Main05_Input02_3.addItem(self.verticalSpacer_120, 2, 0, 1, 1)

        self.PA_Input03 = QDoubleSpinBox(self.PA_Main04)
        self.PA_Input03.setObjectName(u"PA_Input03")
        sizePolicy1.setHeightForWidth(self.PA_Input03.sizePolicy().hasHeightForWidth())
        self.PA_Input03.setSizePolicy(sizePolicy1)
        self.PA_Input03.setMinimumSize(QSize(0, 0))
        self.PA_Input03.setMaximumSize(QSize(16777215, 16777215))
        self.PA_Input03.setFont(font2)
        self.PA_Input03.setStyleSheet(u"padding: 3px;")
        self.PA_Input03.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.PA_Input03.setProperty("showGroupSeparator", True)
        self.PA_Input03.setDecimals(2)
        self.PA_Input03.setMinimum(0.000000000000000)
        self.PA_Input03.setMaximum(1000.000000000000000)
        self.PA_Input03.setSingleStep(1.000000000000000)
        self.PA_Input03.setStepType(QAbstractSpinBox.DefaultStepType)
        self.PA_Input03.setValue(0.000000000000000)

        self.gridLayout_PA_Main05_Input02_3.addWidget(self.PA_Input03, 1, 0, 1, 1)


        self.PA_Main04Grid.addLayout(self.gridLayout_PA_Main05_Input02_3, 1, 3, 1, 1)

        self.LabelSub_PA04 = QLabel(self.PA_Main04)
        self.LabelSub_PA04.setObjectName(u"LabelSub_PA04")
        sizePolicy1.setHeightForWidth(self.LabelSub_PA04.sizePolicy().hasHeightForWidth())
        self.LabelSub_PA04.setSizePolicy(sizePolicy1)
        self.LabelSub_PA04.setMinimumSize(QSize(105, 0))
        self.LabelSub_PA04.setFont(font2)

        self.PA_Main04Grid.addWidget(self.LabelSub_PA04, 2, 1, 1, 1)

        self.gridLayout_PA_Main06_Button_2 = QGridLayout()
        self.gridLayout_PA_Main06_Button_2.setObjectName(u"gridLayout_PA_Main06_Button_2")
        self.PA_Insert01 = QPushButton(self.PA_Main04)
        self.PA_Insert01.setObjectName(u"PA_Insert01")
        self.PA_Insert01.setMinimumSize(QSize(60, 0))
        self.PA_Insert01.setFont(font1)
        self.PA_Insert01.setStyleSheet(u"QPushButton {\n"
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

        self.gridLayout_PA_Main06_Button_2.addWidget(self.PA_Insert01, 0, 1, 1, 1)

        self.horizontalSpacer_PA_Main06Button_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_PA_Main06_Button_2.addItem(self.horizontalSpacer_PA_Main06Button_2, 0, 0, 1, 1)


        self.PA_Main04Grid.addLayout(self.gridLayout_PA_Main06_Button_2, 4, 1, 1, 3)

        self.LabelTitle_PA04 = QLabel(self.PA_Main04)
        self.LabelTitle_PA04.setObjectName(u"LabelTitle_PA04")
        self.LabelTitle_PA04.setMinimumSize(QSize(0, 0))
        self.LabelTitle_PA04.setFont(font3)

        self.PA_Main04Grid.addWidget(self.LabelTitle_PA04, 0, 0, 1, 4)

        self.LabelSub_PA05 = QLabel(self.PA_Main04)
        self.LabelSub_PA05.setObjectName(u"LabelSub_PA05")
        sizePolicy1.setHeightForWidth(self.LabelSub_PA05.sizePolicy().hasHeightForWidth())
        self.LabelSub_PA05.setSizePolicy(sizePolicy1)
        self.LabelSub_PA05.setMinimumSize(QSize(105, 0))
        self.LabelSub_PA05.setFont(font2)

        self.PA_Main04Grid.addWidget(self.LabelSub_PA05, 3, 1, 1, 1)

        self.horizontalSpacer_PA_Main05_4 = QSpacerItem(1, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.PA_Main04Grid.addItem(self.horizontalSpacer_PA_Main05_4, 1, 2, 3, 1)

        self.gridLayout_PA_Main05_Input02_5 = QGridLayout()
        self.gridLayout_PA_Main05_Input02_5.setObjectName(u"gridLayout_PA_Main05_Input02_5")
        self.verticalSpacer_121 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_PA_Main05_Input02_5.addItem(self.verticalSpacer_121, 2, 0, 1, 1)

        self.verticalSpacer_122 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_PA_Main05_Input02_5.addItem(self.verticalSpacer_122, 0, 0, 1, 1)

        self.PA_Input05 = QLineEdit(self.PA_Main04)
        self.PA_Input05.setObjectName(u"PA_Input05")
        sizePolicy2.setHeightForWidth(self.PA_Input05.sizePolicy().hasHeightForWidth())
        self.PA_Input05.setSizePolicy(sizePolicy2)
        self.PA_Input05.setMinimumSize(QSize(0, 0))
        self.PA_Input05.setMaximumSize(QSize(100, 16777215))

        self.gridLayout_PA_Main05_Input02_5.addWidget(self.PA_Input05, 1, 0, 1, 1)


        self.PA_Main04Grid.addLayout(self.gridLayout_PA_Main05_Input02_5, 3, 3, 1, 1)


        self.gridLayout_93.addLayout(self.PA_Main04Grid, 0, 0, 1, 1)


        self.gridLayout_PA_Total.addWidget(self.PA_Main04, 3, 0, 1, 1)

        self.PA_Main02 = QFrame(self.PA_InputSet_Total)
        self.PA_Main02.setObjectName(u"PA_Main02")
        self.PA_Main02.setFrameShape(QFrame.StyledPanel)
        self.PA_Main02.setFrameShadow(QFrame.Raised)
        self.gridLayout_Lifetime_Main04_Input_16 = QGridLayout(self.PA_Main02)
        self.gridLayout_Lifetime_Main04_Input_16.setSpacing(6)
        self.gridLayout_Lifetime_Main04_Input_16.setObjectName(u"gridLayout_Lifetime_Main04_Input_16")
        self.gridLayout_Lifetime_Main04_Input_16.setContentsMargins(9, 9, 9, 9)
        self.PA_Main02Grid = QGridLayout()
        self.PA_Main02Grid.setObjectName(u"PA_Main02Grid")
        self.PA_Opt01 = QRadioButton(self.PA_Main02)
        self.PA_Opt01.setObjectName(u"PA_Opt01")
        sizePolicy1.setHeightForWidth(self.PA_Opt01.sizePolicy().hasHeightForWidth())
        self.PA_Opt01.setSizePolicy(sizePolicy1)
        self.PA_Opt01.setMinimumSize(QSize(0, 0))
        self.PA_Opt01.setFont(font1)
#if QT_CONFIG(whatsthis)
        self.PA_Opt01.setWhatsThis(u"")
#endif // QT_CONFIG(whatsthis)
        self.PA_Opt01.setAutoExclusive(False)

        self.PA_Main02Grid.addWidget(self.PA_Opt01, 1, 1, 1, 1)

        self.PA_Opt02 = QRadioButton(self.PA_Main02)
        self.PA_Opt02.setObjectName(u"PA_Opt02")
        sizePolicy1.setHeightForWidth(self.PA_Opt02.sizePolicy().hasHeightForWidth())
        self.PA_Opt02.setSizePolicy(sizePolicy1)
        self.PA_Opt02.setMinimumSize(QSize(0, 0))
        self.PA_Opt02.setFont(font1)
        self.PA_Opt02.setAutoExclusive(False)

        self.PA_Main02Grid.addWidget(self.PA_Opt02, 2, 1, 1, 1)

        self.horizontalSpacer_PA_Opt01 = QSpacerItem(5, 5, QSizePolicy.Fixed, QSizePolicy.Minimum)

        self.PA_Main02Grid.addItem(self.horizontalSpacer_PA_Opt01, 1, 0, 3, 1)

        self.LabelTitle_PA02 = QLabel(self.PA_Main02)
        self.LabelTitle_PA02.setObjectName(u"LabelTitle_PA02")
        self.LabelTitle_PA02.setMinimumSize(QSize(0, 0))
        self.LabelTitle_PA02.setFont(font3)

        self.PA_Main02Grid.addWidget(self.LabelTitle_PA02, 0, 0, 1, 3)

        self.horizontalSpacer_PA_Opt02 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.PA_Main02Grid.addItem(self.horizontalSpacer_PA_Opt02, 1, 2, 3, 1)


        self.gridLayout_Lifetime_Main04_Input_16.addLayout(self.PA_Main02Grid, 0, 1, 1, 1)


        self.gridLayout_PA_Total.addWidget(self.PA_Main02, 1, 0, 1, 1)

        self.PA_Main03 = QFrame(self.PA_InputSet_Total)
        self.PA_Main03.setObjectName(u"PA_Main03")
        self.PA_Main03.setFrameShape(QFrame.StyledPanel)
        self.PA_Main03.setFrameShadow(QFrame.Raised)
        self.gridLayout_Lifetime_Main04_Input_17 = QGridLayout(self.PA_Main03)
        self.gridLayout_Lifetime_Main04_Input_17.setSpacing(6)
        self.gridLayout_Lifetime_Main04_Input_17.setObjectName(u"gridLayout_Lifetime_Main04_Input_17")
        self.gridLayout_Lifetime_Main04_Input_17.setContentsMargins(9, 9, 9, 9)
        self.PA_Main03Grid = QGridLayout()
        self.PA_Main03Grid.setObjectName(u"PA_Main03Grid")
        self.LabelSub_PA01 = QLabel(self.PA_Main03)
        self.LabelSub_PA01.setObjectName(u"LabelSub_PA01")
        sizePolicy1.setHeightForWidth(self.LabelSub_PA01.sizePolicy().hasHeightForWidth())
        self.LabelSub_PA01.setSizePolicy(sizePolicy1)
        self.LabelSub_PA01.setMinimumSize(QSize(105, 0))
        self.LabelSub_PA01.setFont(font2)

        self.PA_Main03Grid.addWidget(self.LabelSub_PA01, 1, 1, 1, 1)

        self.LabelTitle_PA03 = QLabel(self.PA_Main03)
        self.LabelTitle_PA03.setObjectName(u"LabelTitle_PA03")
        self.LabelTitle_PA03.setMinimumSize(QSize(0, 0))
        self.LabelTitle_PA03.setFont(font3)

        self.PA_Main03Grid.addWidget(self.LabelTitle_PA03, 0, 0, 1, 4)

        self.LabelSub_PA02 = QLabel(self.PA_Main03)
        self.LabelSub_PA02.setObjectName(u"LabelSub_PA02")
        sizePolicy1.setHeightForWidth(self.LabelSub_PA02.sizePolicy().hasHeightForWidth())
        self.LabelSub_PA02.setSizePolicy(sizePolicy1)
        self.LabelSub_PA02.setMinimumSize(QSize(105, 0))
        self.LabelSub_PA02.setFont(font2)

        self.PA_Main03Grid.addWidget(self.LabelSub_PA02, 2, 1, 1, 1)

        self.gridLayout_PA_Main05_Input02_2 = QGridLayout()
        self.gridLayout_PA_Main05_Input02_2.setObjectName(u"gridLayout_PA_Main05_Input02_2")
        self.verticalSpacer_123 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_PA_Main05_Input02_2.addItem(self.verticalSpacer_123, 0, 0, 1, 1)

        self.verticalSpacer_124 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_PA_Main05_Input02_2.addItem(self.verticalSpacer_124, 2, 0, 1, 1)

        self.PA_Input02 = QDoubleSpinBox(self.PA_Main03)
        self.PA_Input02.setObjectName(u"PA_Input02")
        sizePolicy1.setHeightForWidth(self.PA_Input02.sizePolicy().hasHeightForWidth())
        self.PA_Input02.setSizePolicy(sizePolicy1)
        self.PA_Input02.setMinimumSize(QSize(0, 0))
        self.PA_Input02.setMaximumSize(QSize(16777215, 16777215))
        self.PA_Input02.setFont(font2)
        self.PA_Input02.setStyleSheet(u"padding: 3px;")
        self.PA_Input02.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.PA_Input02.setProperty("showGroupSeparator", True)
        self.PA_Input02.setDecimals(2)
        self.PA_Input02.setMinimum(0.000000000000000)
        self.PA_Input02.setMaximum(100000.000000000000000)
        self.PA_Input02.setSingleStep(100.000000000000000)
        self.PA_Input02.setStepType(QAbstractSpinBox.DefaultStepType)
        self.PA_Input02.setValue(0.000000000000000)

        self.gridLayout_PA_Main05_Input02_2.addWidget(self.PA_Input02, 1, 0, 1, 1)


        self.PA_Main03Grid.addLayout(self.gridLayout_PA_Main05_Input02_2, 2, 3, 1, 1)

        self.horizontalSpacer_PA_Main05_2 = QSpacerItem(5, 5, QSizePolicy.Fixed, QSizePolicy.Minimum)

        self.PA_Main03Grid.addItem(self.horizontalSpacer_PA_Main05_2, 1, 0, 3, 1)

        self.gridLayout_PA_Main05_Input02 = QGridLayout()
        self.gridLayout_PA_Main05_Input02.setObjectName(u"gridLayout_PA_Main05_Input02")
        self.verticalSpacer_125 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_PA_Main05_Input02.addItem(self.verticalSpacer_125, 2, 0, 1, 1)

        self.verticalSpacer_126 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_PA_Main05_Input02.addItem(self.verticalSpacer_126, 0, 0, 1, 1)

        self.PA_Input01 = QDoubleSpinBox(self.PA_Main03)
        self.PA_Input01.setObjectName(u"PA_Input01")
        sizePolicy1.setHeightForWidth(self.PA_Input01.sizePolicy().hasHeightForWidth())
        self.PA_Input01.setSizePolicy(sizePolicy1)
        self.PA_Input01.setMinimumSize(QSize(0, 0))
        self.PA_Input01.setMaximumSize(QSize(16777215, 16777215))
        self.PA_Input01.setFont(font2)
        self.PA_Input01.setStyleSheet(u"padding: 3px;")
        self.PA_Input01.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.PA_Input01.setProperty("showGroupSeparator", True)
        self.PA_Input01.setDecimals(5)
        self.PA_Input01.setMinimum(0.000000000000000)
        self.PA_Input01.setMaximum(10.000000000000000)
        self.PA_Input01.setSingleStep(0.010000000000000)
        self.PA_Input01.setStepType(QAbstractSpinBox.DefaultStepType)
        self.PA_Input01.setValue(1.000000000000000)

        self.gridLayout_PA_Main05_Input02.addWidget(self.PA_Input01, 1, 0, 1, 1)


        self.PA_Main03Grid.addLayout(self.gridLayout_PA_Main05_Input02, 1, 3, 1, 1)

        self.horizontalSpacer_4 = QSpacerItem(1, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.PA_Main03Grid.addItem(self.horizontalSpacer_4, 1, 2, 2, 1)


        self.gridLayout_Lifetime_Main04_Input_17.addLayout(self.PA_Main03Grid, 0, 1, 1, 1)


        self.gridLayout_PA_Total.addWidget(self.PA_Main03, 2, 0, 1, 1)


        self.gridLayout_PA_InputSet_Total.addLayout(self.gridLayout_PA_Total, 0, 0, 1, 1)


        self.gridLayout_PA_frame_MainWindow.addWidget(self.PA_InputSet_Total, 0, 0, 3, 1)

        self.PA_tabWidget = QTabWidget(self.PA_frame_MainWindow)
        self.PA_tabWidget.setObjectName(u"PA_tabWidget")
        sizePolicy3 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy3.setHorizontalStretch(200)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.PA_tabWidget.sizePolicy().hasHeightForWidth())
        self.PA_tabWidget.setSizePolicy(sizePolicy3)
        self.PA_tabWidget.setFont(font)
        self.PA_tabWidget.setStyleSheet(u"\n"
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
        self.PA_Database = QWidget()
        self.PA_Database.setObjectName(u"PA_Database")
        self.gridLayout_170 = QGridLayout(self.PA_Database)
        self.gridLayout_170.setObjectName(u"gridLayout_170")
        self.PA_DB = QTableWidget(self.PA_Database)
        if (self.PA_DB.columnCount() < 7):
            self.PA_DB.setColumnCount(7)
        __qtablewidgetitem = QTableWidgetItem()
        self.PA_DB.setHorizontalHeaderItem(0, __qtablewidgetitem)
        __qtablewidgetitem1 = QTableWidgetItem()
        self.PA_DB.setHorizontalHeaderItem(1, __qtablewidgetitem1)
        __qtablewidgetitem2 = QTableWidgetItem()
        self.PA_DB.setHorizontalHeaderItem(2, __qtablewidgetitem2)
        __qtablewidgetitem3 = QTableWidgetItem()
        self.PA_DB.setHorizontalHeaderItem(3, __qtablewidgetitem3)
        __qtablewidgetitem4 = QTableWidgetItem()
        self.PA_DB.setHorizontalHeaderItem(4, __qtablewidgetitem4)
        __qtablewidgetitem5 = QTableWidgetItem()
        self.PA_DB.setHorizontalHeaderItem(5, __qtablewidgetitem5)
        __qtablewidgetitem6 = QTableWidgetItem()
        self.PA_DB.setHorizontalHeaderItem(6, __qtablewidgetitem6)
        if (self.PA_DB.rowCount() < 4):
            self.PA_DB.setRowCount(4)
        __qtablewidgetitem7 = QTableWidgetItem()
        self.PA_DB.setVerticalHeaderItem(0, __qtablewidgetitem7)
        __qtablewidgetitem8 = QTableWidgetItem()
        self.PA_DB.setVerticalHeaderItem(1, __qtablewidgetitem8)
        __qtablewidgetitem9 = QTableWidgetItem()
        self.PA_DB.setVerticalHeaderItem(2, __qtablewidgetitem9)
        __qtablewidgetitem10 = QTableWidgetItem()
        self.PA_DB.setVerticalHeaderItem(3, __qtablewidgetitem10)
        self.PA_DB.setObjectName(u"PA_DB")
        sizePolicy4 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy4.setHorizontalStretch(0)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.PA_DB.sizePolicy().hasHeightForWidth())
        self.PA_DB.setSizePolicy(sizePolicy4)
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
        self.PA_DB.setPalette(palette)
        self.PA_DB.setStyleSheet(u"QTableWidget {	\n"
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
        self.PA_DB.setFrameShape(QFrame.NoFrame)
        self.PA_DB.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.PA_DB.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.PA_DB.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.PA_DB.setAlternatingRowColors(False)
        self.PA_DB.setSelectionMode(QAbstractItemView.SingleSelection)
        self.PA_DB.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.PA_DB.setShowGrid(True)
        self.PA_DB.setGridStyle(Qt.SolidLine)
        self.PA_DB.setSortingEnabled(False)
        self.PA_DB.horizontalHeader().setVisible(False)
        self.PA_DB.horizontalHeader().setCascadingSectionResizes(True)
        self.PA_DB.horizontalHeader().setDefaultSectionSize(155)
        self.PA_DB.horizontalHeader().setProperty("showSortIndicator", False)
        self.PA_DB.horizontalHeader().setStretchLastSection(True)
        self.PA_DB.verticalHeader().setVisible(False)
        self.PA_DB.verticalHeader().setCascadingSectionResizes(True)
        self.PA_DB.verticalHeader().setDefaultSectionSize(30)
        self.PA_DB.verticalHeader().setHighlightSections(False)
        self.PA_DB.verticalHeader().setStretchLastSection(False)

        self.gridLayout_170.addWidget(self.PA_DB, 0, 0, 1, 1)

        self.PA_tabWidget.addTab(self.PA_Database, "")
        self.PA_tabInput = QWidget()
        self.PA_tabInput.setObjectName(u"PA_tabInput")
        self.PA_tabInput.setStyleSheet(u"background-color: rgb(51, 58, 72);")
        self.gridLayout_171 = QGridLayout(self.PA_tabInput)
        self.gridLayout_171.setObjectName(u"gridLayout_171")
        self.frame_PA_tabInput = QFrame(self.PA_tabInput)
        self.frame_PA_tabInput.setObjectName(u"frame_PA_tabInput")
        self.frame_PA_tabInput.setFrameShape(QFrame.StyledPanel)
        self.frame_PA_tabInput.setFrameShadow(QFrame.Raised)
        self.gridLayout_172 = QGridLayout(self.frame_PA_tabInput)
        self.gridLayout_172.setObjectName(u"gridLayout_172")
        self.label_469 = QLabel(self.frame_PA_tabInput)
        self.label_469.setObjectName(u"label_469")
        font5 = QFont()
        font5.setFamily(u"Segoe UI")
        font5.setPointSize(20)
        font5.setUnderline(False)
        self.label_469.setFont(font5)
        self.label_469.setAlignment(Qt.AlignCenter)
        self.label_469.setMargin(20)

        self.gridLayout_172.addWidget(self.label_469, 1, 0, 1, 2)

        self.label_470 = QLabel(self.frame_PA_tabInput)
        self.label_470.setObjectName(u"label_470")
        font6 = QFont()
        font6.setFamily(u"Segoe UI")
        font6.setPointSize(12)
        font6.setUnderline(False)
        self.label_470.setFont(font6)
        self.label_470.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_470.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_470.setMargin(8)

        self.gridLayout_172.addWidget(self.label_470, 7, 1, 1, 1)

        self.label_471 = QLabel(self.frame_PA_tabInput)
        self.label_471.setObjectName(u"label_471")
        self.label_471.setFont(font6)
        self.label_471.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_471.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_471.setMargin(8)

        self.gridLayout_172.addWidget(self.label_471, 6, 0, 1, 1)

        self.label_472 = QLabel(self.frame_PA_tabInput)
        self.label_472.setObjectName(u"label_472")
        self.label_472.setFont(font6)
        self.label_472.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_472.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_472.setMargin(8)

        self.gridLayout_172.addWidget(self.label_472, 3, 0, 1, 1)

        self.label_473 = QLabel(self.frame_PA_tabInput)
        self.label_473.setObjectName(u"label_473")
        self.label_473.setFont(font6)
        self.label_473.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_473.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_473.setMargin(8)

        self.gridLayout_172.addWidget(self.label_473, 10, 0, 1, 1)

        self.label_474 = QLabel(self.frame_PA_tabInput)
        self.label_474.setObjectName(u"label_474")
        self.label_474.setFont(font6)
        self.label_474.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_474.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_474.setMargin(8)

        self.gridLayout_172.addWidget(self.label_474, 4, 0, 1, 1)

        self.label_475 = QLabel(self.frame_PA_tabInput)
        self.label_475.setObjectName(u"label_475")
        self.label_475.setFont(font6)
        self.label_475.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_475.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_475.setMargin(8)

        self.gridLayout_172.addWidget(self.label_475, 5, 1, 1, 1)

        self.label_476 = QLabel(self.frame_PA_tabInput)
        self.label_476.setObjectName(u"label_476")
        self.label_476.setFont(font6)
        self.label_476.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_476.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_476.setMargin(8)

        self.gridLayout_172.addWidget(self.label_476, 11, 0, 1, 1)

        self.label_477 = QLabel(self.frame_PA_tabInput)
        self.label_477.setObjectName(u"label_477")
        self.label_477.setFont(font6)
        self.label_477.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_477.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_477.setMargin(8)

        self.gridLayout_172.addWidget(self.label_477, 4, 1, 1, 1)

        self.verticalSpacer_104 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Fixed)

        self.gridLayout_172.addItem(self.verticalSpacer_104, 0, 0, 1, 1)

        self.label_478 = QLabel(self.frame_PA_tabInput)
        self.label_478.setObjectName(u"label_478")
        self.label_478.setFont(font6)
        self.label_478.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_478.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_478.setMargin(8)

        self.gridLayout_172.addWidget(self.label_478, 9, 0, 1, 1)

        self.verticalSpacer_105 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_172.addItem(self.verticalSpacer_105, 14, 0, 1, 1)

        self.label_479 = QLabel(self.frame_PA_tabInput)
        self.label_479.setObjectName(u"label_479")
        self.label_479.setFont(font6)
        self.label_479.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_479.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_479.setMargin(8)

        self.gridLayout_172.addWidget(self.label_479, 13, 1, 1, 1)

        self.label_480 = QLabel(self.frame_PA_tabInput)
        self.label_480.setObjectName(u"label_480")
        self.label_480.setFont(font6)
        self.label_480.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_480.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_480.setMargin(8)

        self.gridLayout_172.addWidget(self.label_480, 8, 0, 1, 1)

        self.label_481 = QLabel(self.frame_PA_tabInput)
        self.label_481.setObjectName(u"label_481")
        self.label_481.setFont(font6)
        self.label_481.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_481.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_481.setMargin(8)

        self.gridLayout_172.addWidget(self.label_481, 13, 0, 1, 1)

        self.label_482 = QLabel(self.frame_PA_tabInput)
        self.label_482.setObjectName(u"label_482")
        self.label_482.setFont(font6)
        self.label_482.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_482.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_482.setMargin(8)

        self.gridLayout_172.addWidget(self.label_482, 6, 1, 1, 1)

        self.label_483 = QLabel(self.frame_PA_tabInput)
        self.label_483.setObjectName(u"label_483")
        self.label_483.setFont(font6)
        self.label_483.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_483.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_483.setMargin(8)

        self.gridLayout_172.addWidget(self.label_483, 5, 0, 1, 1)

        self.label_484 = QLabel(self.frame_PA_tabInput)
        self.label_484.setObjectName(u"label_484")
        self.label_484.setMaximumSize(QSize(100, 16777215))
        self.label_484.setFont(font6)
        self.label_484.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_484.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_484.setMargin(8)

        self.gridLayout_172.addWidget(self.label_484, 3, 1, 1, 1)

        self.label_485 = QLabel(self.frame_PA_tabInput)
        self.label_485.setObjectName(u"label_485")
        self.label_485.setFont(font6)
        self.label_485.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_485.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_485.setMargin(8)

        self.gridLayout_172.addWidget(self.label_485, 8, 1, 1, 1)

        self.label_486 = QLabel(self.frame_PA_tabInput)
        self.label_486.setObjectName(u"label_486")
        self.label_486.setFont(font6)
        self.label_486.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_486.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_486.setMargin(8)

        self.gridLayout_172.addWidget(self.label_486, 12, 0, 1, 1)

        self.label_487 = QLabel(self.frame_PA_tabInput)
        self.label_487.setObjectName(u"label_487")
        self.label_487.setFont(font6)
        self.label_487.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_487.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_487.setMargin(8)

        self.gridLayout_172.addWidget(self.label_487, 11, 1, 1, 1)

        self.label_488 = QLabel(self.frame_PA_tabInput)
        self.label_488.setObjectName(u"label_488")
        self.label_488.setFont(font6)
        self.label_488.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_488.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_488.setMargin(8)

        self.gridLayout_172.addWidget(self.label_488, 9, 1, 1, 1)

        self.label_489 = QLabel(self.frame_PA_tabInput)
        self.label_489.setObjectName(u"label_489")
        self.label_489.setFont(font6)
        self.label_489.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_489.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_489.setMargin(8)

        self.gridLayout_172.addWidget(self.label_489, 7, 0, 1, 1)

        self.label_490 = QLabel(self.frame_PA_tabInput)
        self.label_490.setObjectName(u"label_490")
        self.label_490.setFont(font6)
        self.label_490.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_490.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_490.setMargin(8)

        self.gridLayout_172.addWidget(self.label_490, 12, 1, 1, 1)

        self.line_19 = QFrame(self.frame_PA_tabInput)
        self.line_19.setObjectName(u"line_19")
        self.line_19.setFrameShape(QFrame.HLine)
        self.line_19.setFrameShadow(QFrame.Sunken)

        self.gridLayout_172.addWidget(self.line_19, 2, 0, 1, 2)

        self.label_491 = QLabel(self.frame_PA_tabInput)
        self.label_491.setObjectName(u"label_491")
        self.label_491.setFont(font6)
        self.label_491.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_491.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_491.setMargin(8)

        self.gridLayout_172.addWidget(self.label_491, 10, 1, 1, 1)


        self.gridLayout_171.addWidget(self.frame_PA_tabInput, 0, 0, 1, 1)

        self.PA_tabWidget.addTab(self.PA_tabInput, "")
        self.PA_tabRodPos = QWidget()
        self.PA_tabRodPos.setObjectName(u"PA_tabRodPos")
        self.gridLayout_173 = QGridLayout(self.PA_tabRodPos)
        self.gridLayout_173.setObjectName(u"gridLayout_173")
        self.gridLayout_173.setContentsMargins(0, 0, 0, 0)
        self.frame_PA_tabRodPos = QFrame(self.PA_tabRodPos)
        self.frame_PA_tabRodPos.setObjectName(u"frame_PA_tabRodPos")
        self.frame_PA_tabRodPos.setAutoFillBackground(False)
        self.frame_PA_tabRodPos.setStyleSheet(u"background-color: rgb(51, 58, 72);")
        self.gridLayout_174 = QGridLayout(self.frame_PA_tabRodPos)
        self.gridLayout_174.setObjectName(u"gridLayout_174")
        self.PA_widgetChart = QWidget(self.frame_PA_tabRodPos)
        self.PA_widgetChart.setObjectName(u"PA_widgetChart")
        self.PA_widgetChart.setMinimumSize(QSize(960, 660))

        self.gridLayout_174.addWidget(self.PA_widgetChart, 1, 1, 1, 1)

        self.horizontalSpacer_68 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_174.addItem(self.horizontalSpacer_68, 1, 2, 1, 1)

        self.PA_tableWidget = QTableWidget(self.frame_PA_tabRodPos)
        if (self.PA_tableWidget.columnCount() < 9):
            self.PA_tableWidget.setColumnCount(9)
        __qtablewidgetitem11 = QTableWidgetItem()
        self.PA_tableWidget.setHorizontalHeaderItem(0, __qtablewidgetitem11)
        __qtablewidgetitem12 = QTableWidgetItem()
        self.PA_tableWidget.setHorizontalHeaderItem(1, __qtablewidgetitem12)
        __qtablewidgetitem13 = QTableWidgetItem()
        self.PA_tableWidget.setHorizontalHeaderItem(2, __qtablewidgetitem13)
        __qtablewidgetitem14 = QTableWidgetItem()
        self.PA_tableWidget.setHorizontalHeaderItem(3, __qtablewidgetitem14)
        __qtablewidgetitem15 = QTableWidgetItem()
        self.PA_tableWidget.setHorizontalHeaderItem(4, __qtablewidgetitem15)
        __qtablewidgetitem16 = QTableWidgetItem()
        self.PA_tableWidget.setHorizontalHeaderItem(5, __qtablewidgetitem16)
        __qtablewidgetitem17 = QTableWidgetItem()
        self.PA_tableWidget.setHorizontalHeaderItem(6, __qtablewidgetitem17)
        __qtablewidgetitem18 = QTableWidgetItem()
        self.PA_tableWidget.setHorizontalHeaderItem(7, __qtablewidgetitem18)
        __qtablewidgetitem19 = QTableWidgetItem()
        self.PA_tableWidget.setHorizontalHeaderItem(8, __qtablewidgetitem19)
        if (self.PA_tableWidget.rowCount() < 4):
            self.PA_tableWidget.setRowCount(4)
        __qtablewidgetitem20 = QTableWidgetItem()
        self.PA_tableWidget.setVerticalHeaderItem(0, __qtablewidgetitem20)
        __qtablewidgetitem21 = QTableWidgetItem()
        self.PA_tableWidget.setVerticalHeaderItem(1, __qtablewidgetitem21)
        __qtablewidgetitem22 = QTableWidgetItem()
        self.PA_tableWidget.setVerticalHeaderItem(2, __qtablewidgetitem22)
        __qtablewidgetitem23 = QTableWidgetItem()
        self.PA_tableWidget.setVerticalHeaderItem(3, __qtablewidgetitem23)
        self.PA_tableWidget.setObjectName(u"PA_tableWidget")
        sizePolicy4.setHeightForWidth(self.PA_tableWidget.sizePolicy().hasHeightForWidth())
        self.PA_tableWidget.setSizePolicy(sizePolicy4)
        palette1 = QPalette()
        palette1.setBrush(QPalette.Active, QPalette.WindowText, brush)
        brush5 = QBrush(QColor(51, 58, 72, 255))
        brush5.setStyle(Qt.SolidPattern)
        palette1.setBrush(QPalette.Active, QPalette.Button, brush5)
        palette1.setBrush(QPalette.Active, QPalette.Text, brush)
        palette1.setBrush(QPalette.Active, QPalette.ButtonText, brush)
        palette1.setBrush(QPalette.Active, QPalette.Base, brush5)
        palette1.setBrush(QPalette.Active, QPalette.Window, brush5)
        brush6 = QBrush(QColor(210, 210, 210, 128))
        brush6.setStyle(Qt.NoBrush)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette1.setBrush(QPalette.Active, QPalette.PlaceholderText, brush6)
#endif
        palette1.setBrush(QPalette.Inactive, QPalette.WindowText, brush)
        palette1.setBrush(QPalette.Inactive, QPalette.Button, brush5)
        palette1.setBrush(QPalette.Inactive, QPalette.Text, brush)
        palette1.setBrush(QPalette.Inactive, QPalette.ButtonText, brush)
        palette1.setBrush(QPalette.Inactive, QPalette.Base, brush5)
        palette1.setBrush(QPalette.Inactive, QPalette.Window, brush5)
        brush7 = QBrush(QColor(210, 210, 210, 128))
        brush7.setStyle(Qt.NoBrush)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette1.setBrush(QPalette.Inactive, QPalette.PlaceholderText, brush7)
#endif
        palette1.setBrush(QPalette.Disabled, QPalette.WindowText, brush)
        palette1.setBrush(QPalette.Disabled, QPalette.Button, brush5)
        palette1.setBrush(QPalette.Disabled, QPalette.Text, brush)
        palette1.setBrush(QPalette.Disabled, QPalette.ButtonText, brush)
        palette1.setBrush(QPalette.Disabled, QPalette.Base, brush5)
        palette1.setBrush(QPalette.Disabled, QPalette.Window, brush5)
        brush8 = QBrush(QColor(210, 210, 210, 128))
        brush8.setStyle(Qt.NoBrush)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette1.setBrush(QPalette.Disabled, QPalette.PlaceholderText, brush8)
#endif
        self.PA_tableWidget.setPalette(palette1)
        self.PA_tableWidget.setStyleSheet(u"QTableWidget {	\n"
"	background-color: rgb(51, 58, 72);\n"
"	padding: 10px;\n"
"	border-radius: 5px;\n"
"	gridline-color: rgb(44, 49, 60);\n"
"	border-bottom: 1px solid rgb(44, 49, 60);\n"
"}\n"
"QTableWidget::item{\n"
"	border-color: rgb(51, 58, 72);\n"
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
" QScrollBar:vertical {\n"
"	border: none;\n"
"    background: rgb(52, 59, 72);\n"
"    width: 14px;\n"
"    margin: 21px 0 21px 0;\n"
"	border-radius: 0px;\n"
" }\n"
"QScrollBar:handle:vertical {\n"
"    background: rgb(79, 110, 162);\n"
"\n"
"}\n"
"QHeaderView::section{\n"
"	Background-color: rgb(61, 123, 184);\n"
"	max-width: 30px;\n"
"	b"
                        "order: 1px solid rgb(44, 49, 60);\n"
"	border-style: none;\n"
"    border-bottom: 1px solid rgb(44, 49, 60);\n"
"    border-right: 1px solid rgb(44, 49, 60);\n"
"}\n"
"\n"
"\n"
"QTableWidget::horizontalHeader {	\n"
"	background-color: rgb(81, 255, 0);\n"
"}\n"
"QHeaderView::section:horizontal\n"
"{\n"
"    border: 1px solid rgb(32, 34, 42);\n"
"	background-color: rgb(27, 29, 35);\n"
"	padding: 3px;\n"
"	border-top-left-radius: 7px;\n"
"    border-top-right-radius: 7px;\n"
"}\n"
"QHeaderView::section:vertical\n"
"{\n"
"    border: 1px solid rgb(32, 34, 42);\n"
"	background-color: rgb(27, 29, 35);\n"
"\n"
"}\n"
"QTableWidget::verticalHeader {	\n"
"	background-color: rgb(81, 255, 0);\n"
"}\n"
"QTableCornerButton::section{\n"
"    border: 1px solid rgb(32, 34, 42);\n"
"	background-color: rgb(27, 29, 35);\n"
"	padding: 3px;\n"
"	border-top-left-radius: 7px;\n"
"    border-top-right-radius: 7px;\n"
"}")
        self.PA_tableWidget.setFrameShape(QFrame.NoFrame)
        self.PA_tableWidget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.PA_tableWidget.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.PA_tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.PA_tableWidget.setAlternatingRowColors(False)
        self.PA_tableWidget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.PA_tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.PA_tableWidget.setShowGrid(True)
        self.PA_tableWidget.setGridStyle(Qt.SolidLine)
        self.PA_tableWidget.setSortingEnabled(False)
        self.PA_tableWidget.horizontalHeader().setVisible(False)
        self.PA_tableWidget.horizontalHeader().setCascadingSectionResizes(True)
        self.PA_tableWidget.horizontalHeader().setDefaultSectionSize(120)
        self.PA_tableWidget.horizontalHeader().setProperty("showSortIndicator", False)
        self.PA_tableWidget.horizontalHeader().setStretchLastSection(True)
        self.PA_tableWidget.verticalHeader().setVisible(False)
        self.PA_tableWidget.verticalHeader().setCascadingSectionResizes(True)
        self.PA_tableWidget.verticalHeader().setDefaultSectionSize(30)
        self.PA_tableWidget.verticalHeader().setHighlightSections(False)
        self.PA_tableWidget.verticalHeader().setStretchLastSection(False)

        self.gridLayout_174.addWidget(self.PA_tableWidget, 0, 0, 1, 3)


        self.gridLayout_173.addWidget(self.frame_PA_tabRodPos, 0, 0, 1, 1)

        self.PA_tabWidget.addTab(self.PA_tabRodPos, "")
        self.PA_tabReport = QWidget()
        self.PA_tabReport.setObjectName(u"PA_tabReport")
        self.PA_tabReport.setStyleSheet(u"QWidget{\n"
"\n"
"}")
        self.gridLayout_175 = QGridLayout(self.PA_tabReport)
        self.gridLayout_175.setObjectName(u"gridLayout_175")
        self.gridLayout_175.setHorizontalSpacing(5)
        self.gridLayout_175.setContentsMargins(0, 0, 0, 0)
        self.frame_Lifetime_tabReport_7 = QWidget(self.PA_tabReport)
        self.frame_Lifetime_tabReport_7.setObjectName(u"frame_Lifetime_tabReport_7")
        self.frame_Lifetime_tabReport_7.setStyleSheet(u"background-color: rgb(51, 58, 72);")
        self.gridLayout_176 = QGridLayout(self.frame_Lifetime_tabReport_7)
        self.gridLayout_176.setObjectName(u"gridLayout_176")
        self.verticalSpacer_ASI_tabReport_5 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_176.addItem(self.verticalSpacer_ASI_tabReport_5, 1, 1, 1, 1)

        self.horizontalSpacer_ASI_tabReport_8 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_176.addItem(self.horizontalSpacer_ASI_tabReport_8, 0, 2, 1, 1)

        self.horizontalSpacer_ASI_tabReport_9 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_176.addItem(self.horizontalSpacer_ASI_tabReport_9, 0, 0, 1, 1)


        self.gridLayout_175.addWidget(self.frame_Lifetime_tabReport_7, 0, 0, 1, 1)

        self.PA_tabWidget.addTab(self.PA_tabReport, "")

        self.gridLayout_PA_frame_MainWindow.addWidget(self.PA_tabWidget, 0, 1, 5, 1)

        self.verticalSpacer_106 = QSpacerItem(20, 4000, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_PA_frame_MainWindow.addItem(self.verticalSpacer_106, 3, 0, 1, 1)


        self.gridLayout.addWidget(self.PA_frame_MainWindow, 0, 0, 1, 1)


        self.retranslateUi(unitWidget_PA)

        self.PA_tabWidget.setCurrentIndex(1)


        QMetaObject.connectSlotsByName(unitWidget_PA)
    # setupUi

    def retranslateUi(self, unitWidget_PA):
        unitWidget_PA.setWindowTitle(QCoreApplication.translate("unitWidget_PA", u"Form", None))
        self.PA_save_button.setText(QCoreApplication.translate("unitWidget_PA", u"Save", None))
        self.PA_run_button.setText(QCoreApplication.translate("unitWidget_PA", u"Run", None))
        self.PA_InpOpt2_NDR.setText(QCoreApplication.translate("unitWidget_PA", u"NDR", None))
#if QT_CONFIG(statustip)
        self.PA_InpOpt1_Snapshot.setStatusTip("")
#endif // QT_CONFIG(statustip)
        self.PA_InpOpt1_Snapshot.setText(QCoreApplication.translate("unitWidget_PA", u"Snapshot", None))
        self.LabelTitle_PA01.setText(QCoreApplication.translate("unitWidget_PA", u"Input Setup", None))
        self.PA_Insert02.setText(QCoreApplication.translate("unitWidget_PA", u"Insert Input", None))
        self.LabelSub_PA06.setText(QCoreApplication.translate("unitWidget_PA", u"<html><head/><body><p><span style=\" font-size:10pt;\">Power Increase<br/>Ratio (%/hour)</span></p></body></html>", None))
        self.LabelTitle_PA05.setText(QCoreApplication.translate("unitWidget_PA", u"PA Input", None))
        self.LabelSub_PA07.setText(QCoreApplication.translate("unitWidget_PA", u"<html><head/><body><p><span style=\" font-size:10pt;\">Additional Calc. Time</span><br><span style=\" font-size:10pt;\">After PA</span></p></body></html>", None))
        self.LabelSub_PA03.setText(QCoreApplication.translate("unitWidget_PA", u"<html><head/><body><p>Zero Power<br/>Time (hour)</p></body></html>", None))
        self.PA_Input04.setPlaceholderText(QCoreApplication.translate("unitWidget_PA", u"ex)1 10 20", None))
        self.LabelSub_PA04.setText(QCoreApplication.translate("unitWidget_PA", u"<html><head/><body><p>Time Step Interval<br>(hour)</p></body></html>", None))
        self.PA_Insert01.setText(QCoreApplication.translate("unitWidget_PA", u"Insert Input", None))
        self.LabelTitle_PA04.setText(QCoreApplication.translate("unitWidget_PA", u"Decay Time Input", None))
        self.LabelSub_PA05.setText(QCoreApplication.translate("unitWidget_PA", u"<html><head/><body><p>Number of<br>Time Step</p></body></html>", None))
        self.PA_Input05.setPlaceholderText(QCoreApplication.translate("unitWidget_PA", u"ex) 30 7 20", None))
#if QT_CONFIG(statustip)
        self.PA_Opt01.setStatusTip("")
#endif // QT_CONFIG(statustip)
        self.PA_Opt01.setText(QCoreApplication.translate("unitWidget_PA", u"Succeccive Input", None))
        self.PA_Opt02.setText(QCoreApplication.translate("unitWidget_PA", u"Excel-Base Input", None))
        self.LabelTitle_PA02.setText(QCoreApplication.translate("unitWidget_PA", u"PA Input", None))
        self.LabelSub_PA01.setText(QCoreApplication.translate("unitWidget_PA", u"<html><head/><body><p>Cycle Burnup<br/>(MWD/MTU)</p></body></html>", None))
        self.LabelTitle_PA03.setText(QCoreApplication.translate("unitWidget_PA", u"Initial Condition Input", None))
        self.LabelSub_PA02.setText(QCoreApplication.translate("unitWidget_PA", u"<html><head/><body><p>Target<br>Eigenvalue</p></body></html>", None))
        ___qtablewidgetitem = self.PA_DB.horizontalHeaderItem(0)
        ___qtablewidgetitem.setText(QCoreApplication.translate("unitWidget_PA", u"Database Name", None));
        ___qtablewidgetitem1 = self.PA_DB.horizontalHeaderItem(1)
        ___qtablewidgetitem1.setText(QCoreApplication.translate("unitWidget_PA", u"Avg. Temperature", None));
        ___qtablewidgetitem2 = self.PA_DB.horizontalHeaderItem(2)
        ___qtablewidgetitem2.setText(QCoreApplication.translate("unitWidget_PA", u"Boron Concentration", None));
        ___qtablewidgetitem3 = self.PA_DB.horizontalHeaderItem(3)
        ___qtablewidgetitem3.setText(QCoreApplication.translate("unitWidget_PA", u"Bank5 Position", None));
        ___qtablewidgetitem4 = self.PA_DB.horizontalHeaderItem(4)
        ___qtablewidgetitem4.setText(QCoreApplication.translate("unitWidget_PA", u"Bank4 Position", None));
        ___qtablewidgetitem5 = self.PA_DB.horizontalHeaderItem(5)
        ___qtablewidgetitem5.setText(QCoreApplication.translate("unitWidget_PA", u"Bank3 Position", None));
        ___qtablewidgetitem6 = self.PA_DB.horizontalHeaderItem(6)
        ___qtablewidgetitem6.setText(QCoreApplication.translate("unitWidget_PA", u"BankP Position", None));
        ___qtablewidgetitem7 = self.PA_DB.verticalHeaderItem(0)
        ___qtablewidgetitem7.setText(QCoreApplication.translate("unitWidget_PA", u"1", None));
        ___qtablewidgetitem8 = self.PA_DB.verticalHeaderItem(1)
        ___qtablewidgetitem8.setText(QCoreApplication.translate("unitWidget_PA", u"2", None));
        ___qtablewidgetitem9 = self.PA_DB.verticalHeaderItem(2)
        ___qtablewidgetitem9.setText(QCoreApplication.translate("unitWidget_PA", u"3", None));
        ___qtablewidgetitem10 = self.PA_DB.verticalHeaderItem(3)
        ___qtablewidgetitem10.setText(QCoreApplication.translate("unitWidget_PA", u"4", None));
        self.PA_tabWidget.setTabText(self.PA_tabWidget.indexOf(self.PA_Database), QCoreApplication.translate("unitWidget_PA", u"Database", None))
        self.label_469.setText(QCoreApplication.translate("unitWidget_PA", u"     Power Ascention Calculation Result      ", None))
        self.label_470.setText(QCoreApplication.translate("unitWidget_PA", u"-", None))
        self.label_471.setText(QCoreApplication.translate("unitWidget_PA", u"Inlet Temperature (\u2109)", None))
        self.label_472.setText(QCoreApplication.translate("unitWidget_PA", u"Plant", None))
        self.label_473.setText(QCoreApplication.translate("unitWidget_PA", u"N-1 Rod Worth (%\u0394\u03c1)", None))
        self.label_474.setText(QCoreApplication.translate("unitWidget_PA", u"Cycle", None))
        self.label_475.setText(QCoreApplication.translate("unitWidget_PA", u"0.00", None))
        self.label_476.setText(QCoreApplication.translate("unitWidget_PA", u"Total Defect (%\u0394\u03c1)", None))
        self.label_477.setText(QCoreApplication.translate("unitWidget_PA", u"13", None))
        self.label_478.setText(QCoreApplication.translate("unitWidget_PA", u"Stuck Rod Worth (%\u0394\u03c1)", None))
        self.label_479.setText(QCoreApplication.translate("unitWidget_PA", u"-", None))
        self.label_480.setText(QCoreApplication.translate("unitWidget_PA", u"Stuck Rod (N-1 Condition)", None))
        self.label_481.setText(QCoreApplication.translate("unitWidget_PA", u"Required Value (%\u0394\u03c1)", None))
        self.label_482.setText(QCoreApplication.translate("unitWidget_PA", u"-", None))
        self.label_483.setText(QCoreApplication.translate("unitWidget_PA", u"Burnup (MWD/MTU)", None))
        self.label_484.setText(QCoreApplication.translate("unitWidget_PA", u"SKN01", None))
        self.label_485.setText(QCoreApplication.translate("unitWidget_PA", u"-", None))
        self.label_486.setText(QCoreApplication.translate("unitWidget_PA", u"Shutdown Margin (%\u0394\u03c1)", None))
        self.label_487.setText(QCoreApplication.translate("unitWidget_PA", u"-", None))
        self.label_488.setText(QCoreApplication.translate("unitWidget_PA", u"-", None))
        self.label_489.setText(QCoreApplication.translate("unitWidget_PA", u"CEA Configuration", None))
        self.label_490.setText(QCoreApplication.translate("unitWidget_PA", u"-", None))
        self.label_491.setText(QCoreApplication.translate("unitWidget_PA", u"-", None))
        self.PA_tabWidget.setTabText(self.PA_tabWidget.indexOf(self.PA_tabInput), QCoreApplication.translate("unitWidget_PA", u"Input", None))
        ___qtablewidgetitem11 = self.PA_tableWidget.horizontalHeaderItem(0)
        ___qtablewidgetitem11.setText(QCoreApplication.translate("unitWidget_PA", u"Delta Time\n"
"(hour)", None));
        ___qtablewidgetitem12 = self.PA_tableWidget.horizontalHeaderItem(1)
        ___qtablewidgetitem12.setText(QCoreApplication.translate("unitWidget_PA", u"Relative Power\n"
"(%)", None));
        ___qtablewidgetitem13 = self.PA_tableWidget.horizontalHeaderItem(2)
        ___qtablewidgetitem13.setText(QCoreApplication.translate("unitWidget_PA", u"Cycle Burnup\n"
"(MWD/MTU)", None));
        ___qtablewidgetitem14 = self.PA_tableWidget.horizontalHeaderItem(3)
        ___qtablewidgetitem14.setText(QCoreApplication.translate("unitWidget_PA", u"ASI", None));
        ___qtablewidgetitem15 = self.PA_tableWidget.horizontalHeaderItem(4)
        ___qtablewidgetitem15.setText(QCoreApplication.translate("unitWidget_PA", u"BankP Position", None));
        ___qtablewidgetitem16 = self.PA_tableWidget.horizontalHeaderItem(5)
        ___qtablewidgetitem16.setText(QCoreApplication.translate("unitWidget_PA", u"Bank5 Position", None));
        ___qtablewidgetitem17 = self.PA_tableWidget.horizontalHeaderItem(6)
        ___qtablewidgetitem17.setText(QCoreApplication.translate("unitWidget_PA", u"Bank4 Position", None));
        ___qtablewidgetitem18 = self.PA_tableWidget.horizontalHeaderItem(7)
        ___qtablewidgetitem18.setText(QCoreApplication.translate("unitWidget_PA", u"Bank3 Position", None));
        ___qtablewidgetitem19 = self.PA_tableWidget.horizontalHeaderItem(8)
        ___qtablewidgetitem19.setText(QCoreApplication.translate("unitWidget_PA", u"EigenValue", None));
        ___qtablewidgetitem20 = self.PA_tableWidget.verticalHeaderItem(0)
        ___qtablewidgetitem20.setText(QCoreApplication.translate("unitWidget_PA", u"1", None));
        ___qtablewidgetitem21 = self.PA_tableWidget.verticalHeaderItem(1)
        ___qtablewidgetitem21.setText(QCoreApplication.translate("unitWidget_PA", u"2", None));
        ___qtablewidgetitem22 = self.PA_tableWidget.verticalHeaderItem(2)
        ___qtablewidgetitem22.setText(QCoreApplication.translate("unitWidget_PA", u"3", None));
        ___qtablewidgetitem23 = self.PA_tableWidget.verticalHeaderItem(3)
        ___qtablewidgetitem23.setText(QCoreApplication.translate("unitWidget_PA", u"4", None));
        self.PA_tabWidget.setTabText(self.PA_tabWidget.indexOf(self.PA_tabRodPos), QCoreApplication.translate("unitWidget_PA", u"Rod Input", None))
        self.PA_tabWidget.setTabText(self.PA_tabWidget.indexOf(self.PA_tabReport), QCoreApplication.translate("unitWidget_PA", u"Report", None))
    # retranslateUi

