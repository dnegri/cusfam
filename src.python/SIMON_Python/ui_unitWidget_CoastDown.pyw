# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'unitWidget_CoastDown.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class Ui_unitWidget_Coastdown(object):
    def setupUi(self, unitWidget_Coastdown):
        if not unitWidget_Coastdown.objectName():
            unitWidget_Coastdown.setObjectName(u"unitWidget_Coastdown")
        unitWidget_Coastdown.resize(1812, 1615)
        self.gridLayout = QGridLayout(unitWidget_Coastdown)
        self.gridLayout.setObjectName(u"gridLayout")
        self.CoastDown_frame_MainWindow = QFrame(unitWidget_Coastdown)
        self.CoastDown_frame_MainWindow.setObjectName(u"CoastDown_frame_MainWindow")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.CoastDown_frame_MainWindow.sizePolicy().hasHeightForWidth())
        self.CoastDown_frame_MainWindow.setSizePolicy(sizePolicy)
        self.CoastDown_frame_MainWindow.setStyleSheet(u"background-color: rgb(44, 49, 60);\n"
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
        self.CoastDown_frame_MainWindow.setFrameShape(QFrame.NoFrame)
        self.CoastDown_frame_MainWindow.setFrameShadow(QFrame.Raised)
        self.gridLayout_43 = QGridLayout(self.CoastDown_frame_MainWindow)
        self.gridLayout_43.setSpacing(3)
        self.gridLayout_43.setObjectName(u"gridLayout_43")
        self.CoastDown_tabWidget = QTabWidget(self.CoastDown_frame_MainWindow)
        self.CoastDown_tabWidget.setObjectName(u"CoastDown_tabWidget")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(200)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.CoastDown_tabWidget.sizePolicy().hasHeightForWidth())
        self.CoastDown_tabWidget.setSizePolicy(sizePolicy1)
        font = QFont()
        font.setFamily(u"Segoe UI")
        font.setPointSize(14)
        self.CoastDown_tabWidget.setFont(font)
        self.CoastDown_tabWidget.setStyleSheet(u"\n"
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
        self.Coastdown_tabWidget_DB = QWidget()
        self.Coastdown_tabWidget_DB.setObjectName(u"Coastdown_tabWidget_DB")
        self.gridLayout_16 = QGridLayout(self.Coastdown_tabWidget_DB)
        self.gridLayout_16.setObjectName(u"gridLayout_16")
        self.Coastdown_DB = QTableWidget(self.Coastdown_tabWidget_DB)
        if (self.Coastdown_DB.columnCount() < 7):
            self.Coastdown_DB.setColumnCount(7)
        __qtablewidgetitem = QTableWidgetItem()
        self.Coastdown_DB.setHorizontalHeaderItem(0, __qtablewidgetitem)
        __qtablewidgetitem1 = QTableWidgetItem()
        self.Coastdown_DB.setHorizontalHeaderItem(1, __qtablewidgetitem1)
        __qtablewidgetitem2 = QTableWidgetItem()
        self.Coastdown_DB.setHorizontalHeaderItem(2, __qtablewidgetitem2)
        __qtablewidgetitem3 = QTableWidgetItem()
        self.Coastdown_DB.setHorizontalHeaderItem(3, __qtablewidgetitem3)
        __qtablewidgetitem4 = QTableWidgetItem()
        self.Coastdown_DB.setHorizontalHeaderItem(4, __qtablewidgetitem4)
        __qtablewidgetitem5 = QTableWidgetItem()
        self.Coastdown_DB.setHorizontalHeaderItem(5, __qtablewidgetitem5)
        __qtablewidgetitem6 = QTableWidgetItem()
        self.Coastdown_DB.setHorizontalHeaderItem(6, __qtablewidgetitem6)
        if (self.Coastdown_DB.rowCount() < 4):
            self.Coastdown_DB.setRowCount(4)
        __qtablewidgetitem7 = QTableWidgetItem()
        self.Coastdown_DB.setVerticalHeaderItem(0, __qtablewidgetitem7)
        __qtablewidgetitem8 = QTableWidgetItem()
        self.Coastdown_DB.setVerticalHeaderItem(1, __qtablewidgetitem8)
        __qtablewidgetitem9 = QTableWidgetItem()
        self.Coastdown_DB.setVerticalHeaderItem(2, __qtablewidgetitem9)
        __qtablewidgetitem10 = QTableWidgetItem()
        self.Coastdown_DB.setVerticalHeaderItem(3, __qtablewidgetitem10)
        self.Coastdown_DB.setObjectName(u"Coastdown_DB")
        sizePolicy2 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.Coastdown_DB.sizePolicy().hasHeightForWidth())
        self.Coastdown_DB.setSizePolicy(sizePolicy2)
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
        self.Coastdown_DB.setPalette(palette)
        self.Coastdown_DB.setStyleSheet(u"QTableWidget {	\n"
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
        self.Coastdown_DB.setFrameShape(QFrame.NoFrame)
        self.Coastdown_DB.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.Coastdown_DB.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.Coastdown_DB.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.Coastdown_DB.setAlternatingRowColors(False)
        self.Coastdown_DB.setSelectionMode(QAbstractItemView.SingleSelection)
        self.Coastdown_DB.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.Coastdown_DB.setShowGrid(True)
        self.Coastdown_DB.setGridStyle(Qt.SolidLine)
        self.Coastdown_DB.setSortingEnabled(False)
        self.Coastdown_DB.horizontalHeader().setVisible(False)
        self.Coastdown_DB.horizontalHeader().setCascadingSectionResizes(True)
        self.Coastdown_DB.horizontalHeader().setDefaultSectionSize(155)
        self.Coastdown_DB.horizontalHeader().setProperty("showSortIndicator", False)
        self.Coastdown_DB.horizontalHeader().setStretchLastSection(True)
        self.Coastdown_DB.verticalHeader().setVisible(False)
        self.Coastdown_DB.verticalHeader().setCascadingSectionResizes(True)
        self.Coastdown_DB.verticalHeader().setDefaultSectionSize(30)
        self.Coastdown_DB.verticalHeader().setHighlightSections(False)
        self.Coastdown_DB.verticalHeader().setStretchLastSection(False)

        self.gridLayout_16.addWidget(self.Coastdown_DB, 0, 0, 1, 1)

        self.CoastDown_tabWidget.addTab(self.Coastdown_tabWidget_DB, "")
        self.Coastdown_tabInput = QWidget()
        self.Coastdown_tabInput.setObjectName(u"Coastdown_tabInput")
        self.Coastdown_tabInput.setStyleSheet(u"background-color: rgb(51, 58, 72);")
        self.gridLayout_21 = QGridLayout(self.Coastdown_tabInput)
        self.gridLayout_21.setObjectName(u"gridLayout_21")
        self.frame_20 = QFrame(self.Coastdown_tabInput)
        self.frame_20.setObjectName(u"frame_20")
        self.frame_20.setFrameShape(QFrame.StyledPanel)
        self.frame_20.setFrameShadow(QFrame.Raised)
        self.gridLayout_39 = QGridLayout(self.frame_20)
        self.gridLayout_39.setObjectName(u"gridLayout_39")
        self.label_166 = QLabel(self.frame_20)
        self.label_166.setObjectName(u"label_166")
        font1 = QFont()
        font1.setFamily(u"Segoe UI")
        font1.setPointSize(20)
        font1.setUnderline(False)
        self.label_166.setFont(font1)
        self.label_166.setAlignment(Qt.AlignCenter)
        self.label_166.setMargin(20)

        self.gridLayout_39.addWidget(self.label_166, 1, 0, 1, 2)

        self.label_167 = QLabel(self.frame_20)
        self.label_167.setObjectName(u"label_167")
        font2 = QFont()
        font2.setFamily(u"Segoe UI")
        font2.setPointSize(12)
        font2.setUnderline(False)
        self.label_167.setFont(font2)
        self.label_167.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_167.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_167.setMargin(8)

        self.gridLayout_39.addWidget(self.label_167, 7, 1, 1, 1)

        self.label_168 = QLabel(self.frame_20)
        self.label_168.setObjectName(u"label_168")
        self.label_168.setFont(font2)
        self.label_168.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_168.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_168.setMargin(8)

        self.gridLayout_39.addWidget(self.label_168, 6, 0, 1, 1)

        self.label_169 = QLabel(self.frame_20)
        self.label_169.setObjectName(u"label_169")
        self.label_169.setFont(font2)
        self.label_169.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_169.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_169.setMargin(8)

        self.gridLayout_39.addWidget(self.label_169, 3, 0, 1, 1)

        self.label_170 = QLabel(self.frame_20)
        self.label_170.setObjectName(u"label_170")
        self.label_170.setFont(font2)
        self.label_170.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_170.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_170.setMargin(8)

        self.gridLayout_39.addWidget(self.label_170, 10, 0, 1, 1)

        self.label_171 = QLabel(self.frame_20)
        self.label_171.setObjectName(u"label_171")
        self.label_171.setFont(font2)
        self.label_171.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_171.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_171.setMargin(8)

        self.gridLayout_39.addWidget(self.label_171, 4, 0, 1, 1)

        self.label_172 = QLabel(self.frame_20)
        self.label_172.setObjectName(u"label_172")
        self.label_172.setFont(font2)
        self.label_172.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_172.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_172.setMargin(8)

        self.gridLayout_39.addWidget(self.label_172, 5, 1, 1, 1)

        self.label_173 = QLabel(self.frame_20)
        self.label_173.setObjectName(u"label_173")
        self.label_173.setFont(font2)
        self.label_173.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_173.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_173.setMargin(8)

        self.gridLayout_39.addWidget(self.label_173, 11, 0, 1, 1)

        self.label_174 = QLabel(self.frame_20)
        self.label_174.setObjectName(u"label_174")
        self.label_174.setFont(font2)
        self.label_174.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_174.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_174.setMargin(8)

        self.gridLayout_39.addWidget(self.label_174, 4, 1, 1, 1)

        self.verticalSpacer_21 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Fixed)

        self.gridLayout_39.addItem(self.verticalSpacer_21, 0, 0, 1, 1)

        self.label_175 = QLabel(self.frame_20)
        self.label_175.setObjectName(u"label_175")
        self.label_175.setFont(font2)
        self.label_175.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_175.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_175.setMargin(8)

        self.gridLayout_39.addWidget(self.label_175, 9, 0, 1, 1)

        self.verticalSpacer_22 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_39.addItem(self.verticalSpacer_22, 14, 0, 1, 1)

        self.label_176 = QLabel(self.frame_20)
        self.label_176.setObjectName(u"label_176")
        self.label_176.setFont(font2)
        self.label_176.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_176.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_176.setMargin(8)

        self.gridLayout_39.addWidget(self.label_176, 13, 1, 1, 1)

        self.label_177 = QLabel(self.frame_20)
        self.label_177.setObjectName(u"label_177")
        self.label_177.setFont(font2)
        self.label_177.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_177.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_177.setMargin(8)

        self.gridLayout_39.addWidget(self.label_177, 8, 0, 1, 1)

        self.label_178 = QLabel(self.frame_20)
        self.label_178.setObjectName(u"label_178")
        self.label_178.setFont(font2)
        self.label_178.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_178.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_178.setMargin(8)

        self.gridLayout_39.addWidget(self.label_178, 13, 0, 1, 1)

        self.label_179 = QLabel(self.frame_20)
        self.label_179.setObjectName(u"label_179")
        self.label_179.setFont(font2)
        self.label_179.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_179.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_179.setMargin(8)

        self.gridLayout_39.addWidget(self.label_179, 6, 1, 1, 1)

        self.label_180 = QLabel(self.frame_20)
        self.label_180.setObjectName(u"label_180")
        self.label_180.setFont(font2)
        self.label_180.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_180.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_180.setMargin(8)

        self.gridLayout_39.addWidget(self.label_180, 5, 0, 1, 1)

        self.label_181 = QLabel(self.frame_20)
        self.label_181.setObjectName(u"label_181")
        self.label_181.setMaximumSize(QSize(100, 16777215))
        self.label_181.setFont(font2)
        self.label_181.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_181.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_181.setMargin(8)

        self.gridLayout_39.addWidget(self.label_181, 3, 1, 1, 1)

        self.label_182 = QLabel(self.frame_20)
        self.label_182.setObjectName(u"label_182")
        self.label_182.setFont(font2)
        self.label_182.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_182.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_182.setMargin(8)

        self.gridLayout_39.addWidget(self.label_182, 8, 1, 1, 1)

        self.label_183 = QLabel(self.frame_20)
        self.label_183.setObjectName(u"label_183")
        self.label_183.setFont(font2)
        self.label_183.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_183.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_183.setMargin(8)

        self.gridLayout_39.addWidget(self.label_183, 12, 0, 1, 1)

        self.label_184 = QLabel(self.frame_20)
        self.label_184.setObjectName(u"label_184")
        self.label_184.setFont(font2)
        self.label_184.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_184.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_184.setMargin(8)

        self.gridLayout_39.addWidget(self.label_184, 11, 1, 1, 1)

        self.label_185 = QLabel(self.frame_20)
        self.label_185.setObjectName(u"label_185")
        self.label_185.setFont(font2)
        self.label_185.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_185.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_185.setMargin(8)

        self.gridLayout_39.addWidget(self.label_185, 9, 1, 1, 1)

        self.label_186 = QLabel(self.frame_20)
        self.label_186.setObjectName(u"label_186")
        self.label_186.setFont(font2)
        self.label_186.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_186.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_186.setMargin(8)

        self.gridLayout_39.addWidget(self.label_186, 7, 0, 1, 1)

        self.label_187 = QLabel(self.frame_20)
        self.label_187.setObjectName(u"label_187")
        self.label_187.setFont(font2)
        self.label_187.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_187.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_187.setMargin(8)

        self.gridLayout_39.addWidget(self.label_187, 12, 1, 1, 1)

        self.line_6 = QFrame(self.frame_20)
        self.line_6.setObjectName(u"line_6")
        self.line_6.setFrameShape(QFrame.HLine)
        self.line_6.setFrameShadow(QFrame.Sunken)

        self.gridLayout_39.addWidget(self.line_6, 2, 0, 1, 2)

        self.label_188 = QLabel(self.frame_20)
        self.label_188.setObjectName(u"label_188")
        self.label_188.setFont(font2)
        self.label_188.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_188.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_188.setMargin(8)

        self.gridLayout_39.addWidget(self.label_188, 10, 1, 1, 1)


        self.gridLayout_21.addWidget(self.frame_20, 0, 0, 1, 1)

        self.CoastDown_tabWidget.addTab(self.Coastdown_tabInput, "")
        self.Coastdown_tabRodPos = QWidget()
        self.Coastdown_tabRodPos.setObjectName(u"Coastdown_tabRodPos")
        self.gridLayout_53 = QGridLayout(self.Coastdown_tabRodPos)
        self.gridLayout_53.setObjectName(u"gridLayout_53")
        self.gridLayout_53.setContentsMargins(0, 0, 0, 0)
        self.frame_8 = QFrame(self.Coastdown_tabRodPos)
        self.frame_8.setObjectName(u"frame_8")
        self.frame_8.setAutoFillBackground(False)
        self.frame_8.setStyleSheet(u"background-color: rgb(51, 58, 72);")
        self.gridLayout_44 = QGridLayout(self.frame_8)
        self.gridLayout_44.setObjectName(u"gridLayout_44")
        self.frame_21 = QFrame(self.frame_8)
        self.frame_21.setObjectName(u"frame_21")
        self.frame_21.setFrameShape(QFrame.StyledPanel)
        self.frame_21.setFrameShadow(QFrame.Raised)
        self.gridLayout_54 = QGridLayout(self.frame_21)
        self.gridLayout_54.setSpacing(6)
        self.gridLayout_54.setObjectName(u"gridLayout_54")
        self.gridLayout_54.setContentsMargins(-1, 9, 9, -1)

        self.gridLayout_44.addWidget(self.frame_21, 0, 0, 2, 2)

        self.CoastDown_widgetChart = QWidget(self.frame_8)
        self.CoastDown_widgetChart.setObjectName(u"CoastDown_widgetChart")
        self.CoastDown_widgetChart.setMinimumSize(QSize(660, 660))

        self.gridLayout_44.addWidget(self.CoastDown_widgetChart, 1, 1, 2, 1)

        self.horizontalSpacer_45 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_44.addItem(self.horizontalSpacer_45, 2, 2, 1, 1)

        self.verticalSpacer_23 = QSpacerItem(20, 807, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_44.addItem(self.verticalSpacer_23, 3, 0, 1, 2)

        self.horizontalSpacer_46 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_44.addItem(self.horizontalSpacer_46, 2, 0, 1, 1)


        self.gridLayout_53.addWidget(self.frame_8, 0, 0, 1, 1)

        self.CoastDown_tabWidget.addTab(self.Coastdown_tabRodPos, "")
        self.Coastdown_tabReport = QWidget()
        self.Coastdown_tabReport.setObjectName(u"Coastdown_tabReport")
        self.Coastdown_tabReport.setStyleSheet(u"QWidget{\n"
"\n"
"}")
        self.gridLayout_67 = QGridLayout(self.Coastdown_tabReport)
        self.gridLayout_67.setObjectName(u"gridLayout_67")
        self.gridLayout_67.setHorizontalSpacing(5)
        self.gridLayout_67.setContentsMargins(0, 0, 0, 0)
        self.frame_CoastDown_tabReport = QWidget(self.Coastdown_tabReport)
        self.frame_CoastDown_tabReport.setObjectName(u"frame_CoastDown_tabReport")
        self.frame_CoastDown_tabReport.setStyleSheet(u"background-color: rgb(51, 58, 72);")
        self.gridLayout_68 = QGridLayout(self.frame_CoastDown_tabReport)
        self.gridLayout_68.setObjectName(u"gridLayout_68")
        self.verticalSpacer_25 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_68.addItem(self.verticalSpacer_25, 1, 1, 1, 1)

        self.horizontalSpacer_47 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_68.addItem(self.horizontalSpacer_47, 0, 2, 1, 1)

        self.horizontalSpacer_48 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_68.addItem(self.horizontalSpacer_48, 0, 0, 1, 1)


        self.gridLayout_67.addWidget(self.frame_CoastDown_tabReport, 0, 0, 1, 1)

        self.CoastDown_tabWidget.addTab(self.Coastdown_tabReport, "")

        self.gridLayout_43.addWidget(self.CoastDown_tabWidget, 0, 1, 4, 1)

        self.CoastDown_Main07 = QFrame(self.CoastDown_frame_MainWindow)
        self.CoastDown_Main07.setObjectName(u"CoastDown_Main07")
        self.CoastDown_Main07.setMinimumSize(QSize(0, 0))
        self.CoastDown_Main07.setFrameShape(QFrame.StyledPanel)
        self.CoastDown_Main07.setFrameShadow(QFrame.Raised)
        self.Coastdown_CalcButton_Grid = QGridLayout(self.CoastDown_Main07)
        self.Coastdown_CalcButton_Grid.setObjectName(u"Coastdown_CalcButton_Grid")
        self.Coastdown_CalcButton_Grid.setContentsMargins(0, 0, 0, 0)
        self.Coastdown_save_button = QPushButton(self.CoastDown_Main07)
        self.Coastdown_save_button.setObjectName(u"Coastdown_save_button")
        self.Coastdown_save_button.setMinimumSize(QSize(0, 50))
        self.Coastdown_save_button.setFont(font)
        self.Coastdown_save_button.setStyleSheet(u"QPushButton {\n"
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

        self.Coastdown_CalcButton_Grid.addWidget(self.Coastdown_save_button, 1, 0, 1, 1)

        self.Coastdown_run_button = QPushButton(self.CoastDown_Main07)
        self.Coastdown_run_button.setObjectName(u"Coastdown_run_button")
        self.Coastdown_run_button.setMinimumSize(QSize(220, 50))
        self.Coastdown_run_button.setFont(font)
        self.Coastdown_run_button.setStyleSheet(u"QPushButton {\n"
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

        self.Coastdown_CalcButton_Grid.addWidget(self.Coastdown_run_button, 1, 1, 1, 1)

        self.verticalSpacer_7 = QSpacerItem(20, 400, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.Coastdown_CalcButton_Grid.addItem(self.verticalSpacer_7, 0, 0, 1, 2)


        self.gridLayout_43.addWidget(self.CoastDown_Main07, 3, 0, 1, 1)

        self.Coastdown_InputSet_Total = QFrame(self.CoastDown_frame_MainWindow)
        self.Coastdown_InputSet_Total.setObjectName(u"Coastdown_InputSet_Total")
        self.Coastdown_InputSet_Total.setFrameShape(QFrame.StyledPanel)
        self.Coastdown_InputSet_Total.setFrameShadow(QFrame.Raised)
        self.gridLayout_Coastdown_InputSet_Total = QGridLayout(self.Coastdown_InputSet_Total)
        self.gridLayout_Coastdown_InputSet_Total.setSpacing(0)
        self.gridLayout_Coastdown_InputSet_Total.setObjectName(u"gridLayout_Coastdown_InputSet_Total")
        self.gridLayout_Coastdown_InputSet_Total.setContentsMargins(0, 0, 0, 0)
        self.verticalSpacer_34 = QSpacerItem(20, 4000, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_Coastdown_InputSet_Total.addItem(self.verticalSpacer_34, 1, 0, 1, 1)

        self.gridLayout_Coastdown_Total = QGridLayout()
        self.gridLayout_Coastdown_Total.setObjectName(u"gridLayout_Coastdown_Total")
        self.CD_Main02 = QFrame(self.Coastdown_InputSet_Total)
        self.CD_Main02.setObjectName(u"CD_Main02")
        self.CD_Main02.setFrameShape(QFrame.StyledPanel)
        self.CD_Main02.setFrameShadow(QFrame.Raised)
        self.layoutCoastdown0002 = QGridLayout(self.CD_Main02)
        self.layoutCoastdown0002.setObjectName(u"layoutCoastdown0002")
        self.CD_Main02Grid = QGridLayout()
        self.CD_Main02Grid.setSpacing(6)
        self.CD_Main02Grid.setObjectName(u"CD_Main02Grid")
        self.CD_Main02Grid.setContentsMargins(0, 0, 0, 0)
        self.CoastDown_CalcTarget02 = QRadioButton(self.CD_Main02)
        self.CoastDown_CalcTarget02.setObjectName(u"CoastDown_CalcTarget02")
        sizePolicy3 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.CoastDown_CalcTarget02.sizePolicy().hasHeightForWidth())
        self.CoastDown_CalcTarget02.setSizePolicy(sizePolicy3)
        self.CoastDown_CalcTarget02.setMinimumSize(QSize(0, 0))
        font3 = QFont()
        font3.setFamily(u"Segoe UI")
        font3.setPointSize(12)
        font3.setBold(False)
        font3.setWeight(50)
        self.CoastDown_CalcTarget02.setFont(font3)
        self.CoastDown_CalcTarget02.setAutoExclusive(False)

        self.CD_Main02Grid.addWidget(self.CoastDown_CalcTarget02, 2, 1, 1, 1)

        self.horizontalSpacer_CoastDown_Main02_01 = QSpacerItem(1, 20, QSizePolicy.Fixed, QSizePolicy.Minimum)

        self.CD_Main02Grid.addItem(self.horizontalSpacer_CoastDown_Main02_01, 1, 0, 2, 1)

        self.CoastDown_CalcTarget01 = QRadioButton(self.CD_Main02)
        self.CoastDown_CalcTarget01.setObjectName(u"CoastDown_CalcTarget01")
        sizePolicy3.setHeightForWidth(self.CoastDown_CalcTarget01.sizePolicy().hasHeightForWidth())
        self.CoastDown_CalcTarget01.setSizePolicy(sizePolicy3)
        self.CoastDown_CalcTarget01.setMinimumSize(QSize(0, 0))
        self.CoastDown_CalcTarget01.setFont(font3)
#if QT_CONFIG(whatsthis)
        self.CoastDown_CalcTarget01.setWhatsThis(u"")
#endif // QT_CONFIG(whatsthis)
        self.CoastDown_CalcTarget01.setAutoExclusive(False)

        self.CD_Main02Grid.addWidget(self.CoastDown_CalcTarget01, 1, 1, 1, 1)

        self.horizontalSpacer_CoastDown_Main02_02 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.CD_Main02Grid.addItem(self.horizontalSpacer_CoastDown_Main02_02, 1, 2, 2, 1)

        self.LabelTitle_CD02 = QLabel(self.CD_Main02)
        self.LabelTitle_CD02.setObjectName(u"LabelTitle_CD02")
        self.LabelTitle_CD02.setMinimumSize(QSize(0, 0))
        font4 = QFont()
        font4.setFamily(u"Segoe UI")
        font4.setPointSize(14)
        font4.setBold(True)
        font4.setWeight(75)
        self.LabelTitle_CD02.setFont(font4)

        self.CD_Main02Grid.addWidget(self.LabelTitle_CD02, 0, 0, 1, 3)


        self.layoutCoastdown0002.addLayout(self.CD_Main02Grid, 0, 0, 1, 1)


        self.gridLayout_Coastdown_Total.addWidget(self.CD_Main02, 1, 0, 1, 1)

        self.CD_Main01 = QFrame(self.Coastdown_InputSet_Total)
        self.CD_Main01.setObjectName(u"CD_Main01")
        self.CD_Main01.setFrameShape(QFrame.StyledPanel)
        self.CD_Main01.setFrameShadow(QFrame.Raised)
        self.layoutCoastdown0001 = QGridLayout(self.CD_Main01)
        self.layoutCoastdown0001.setObjectName(u"layoutCoastdown0001")
        self.CD_Main01Grid = QGridLayout()
        self.CD_Main01Grid.setSpacing(6)
        self.CD_Main01Grid.setObjectName(u"CD_Main01Grid")
        self.CD_Main01Grid.setContentsMargins(0, 0, 0, 0)
        self.LabelTitle_CD01 = QLabel(self.CD_Main01)
        self.LabelTitle_CD01.setObjectName(u"LabelTitle_CD01")
        self.LabelTitle_CD01.setMinimumSize(QSize(0, 0))
        self.LabelTitle_CD01.setFont(font4)

        self.CD_Main01Grid.addWidget(self.LabelTitle_CD01, 0, 0, 1, 4)

        self.CoastDown_InpOpt2_NDR = QRadioButton(self.CD_Main01)
        self.CoastDown_InpOpt2_NDR.setObjectName(u"CoastDown_InpOpt2_NDR")
        sizePolicy3.setHeightForWidth(self.CoastDown_InpOpt2_NDR.sizePolicy().hasHeightForWidth())
        self.CoastDown_InpOpt2_NDR.setSizePolicy(sizePolicy3)
        self.CoastDown_InpOpt2_NDR.setMinimumSize(QSize(0, 0))
        font5 = QFont()
        font5.setFamily(u"Segoe UI")
        font5.setPointSize(12)
        self.CoastDown_InpOpt2_NDR.setFont(font5)
        self.CoastDown_InpOpt2_NDR.setAutoExclusive(False)

        self.CD_Main01Grid.addWidget(self.CoastDown_InpOpt2_NDR, 2, 1, 1, 1)

        self.CoastDown_InpOpt1_Snapshot = QRadioButton(self.CD_Main01)
        self.CoastDown_InpOpt1_Snapshot.setObjectName(u"CoastDown_InpOpt1_Snapshot")
        sizePolicy3.setHeightForWidth(self.CoastDown_InpOpt1_Snapshot.sizePolicy().hasHeightForWidth())
        self.CoastDown_InpOpt1_Snapshot.setSizePolicy(sizePolicy3)
        self.CoastDown_InpOpt1_Snapshot.setMinimumSize(QSize(0, 0))
        self.CoastDown_InpOpt1_Snapshot.setFont(font5)
#if QT_CONFIG(whatsthis)
        self.CoastDown_InpOpt1_Snapshot.setWhatsThis(u"")
#endif // QT_CONFIG(whatsthis)
        self.CoastDown_InpOpt1_Snapshot.setAutoExclusive(False)

        self.CD_Main01Grid.addWidget(self.CoastDown_InpOpt1_Snapshot, 1, 1, 1, 1)

        self.horizontalSpacer_CoastDown_Main01_01 = QSpacerItem(1, 20, QSizePolicy.Fixed, QSizePolicy.Minimum)

        self.CD_Main01Grid.addItem(self.horizontalSpacer_CoastDown_Main01_01, 1, 0, 2, 1)

        self.CoastDown_Snapshot = QComboBox(self.CD_Main01)
        self.CoastDown_Snapshot.setObjectName(u"CoastDown_Snapshot")
        sizePolicy3.setHeightForWidth(self.CoastDown_Snapshot.sizePolicy().hasHeightForWidth())
        self.CoastDown_Snapshot.setSizePolicy(sizePolicy3)
        self.CoastDown_Snapshot.setMinimumSize(QSize(0, 0))
        self.CoastDown_Snapshot.setMaximumSize(QSize(16777215, 30))
        font6 = QFont()
        font6.setFamily(u"Segoe UI")
        font6.setPointSize(10)
        self.CoastDown_Snapshot.setFont(font6)

        self.CD_Main01Grid.addWidget(self.CoastDown_Snapshot, 1, 2, 1, 1)

        self.horizontalSpacer_CoastDown_Main01_02 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.CD_Main01Grid.addItem(self.horizontalSpacer_CoastDown_Main01_02, 2, 2, 1, 1)


        self.layoutCoastdown0001.addLayout(self.CD_Main01Grid, 0, 0, 1, 1)


        self.gridLayout_Coastdown_Total.addWidget(self.CD_Main01, 0, 0, 1, 1)

        self.CD_Main03 = QFrame(self.Coastdown_InputSet_Total)
        self.CD_Main03.setObjectName(u"CD_Main03")
        self.CD_Main03.setMinimumSize(QSize(0, 0))
        self.CD_Main03.setFrameShape(QFrame.StyledPanel)
        self.CD_Main03.setFrameShadow(QFrame.Raised)
        self.layoutCoastdown0005 = QGridLayout(self.CD_Main03)
        self.layoutCoastdown0005.setSpacing(6)
        self.layoutCoastdown0005.setObjectName(u"layoutCoastdown0005")
        self.layoutCoastdown0005.setContentsMargins(9, 9, 9, 9)
        self.CD_Main03Grid = QGridLayout()
        self.CD_Main03Grid.setObjectName(u"CD_Main03Grid")
        self.LabelTitle_CD03 = QLabel(self.CD_Main03)
        self.LabelTitle_CD03.setObjectName(u"LabelTitle_CD03")
        self.LabelTitle_CD03.setMinimumSize(QSize(0, 0))
        self.LabelTitle_CD03.setFont(font4)

        self.CD_Main03Grid.addWidget(self.LabelTitle_CD03, 0, 0, 1, 4)

        self.CD_Input02 = QDoubleSpinBox(self.CD_Main03)
        self.CD_Input02.setObjectName(u"CD_Input02")
        sizePolicy3.setHeightForWidth(self.CD_Input02.sizePolicy().hasHeightForWidth())
        self.CD_Input02.setSizePolicy(sizePolicy3)
        self.CD_Input02.setMinimumSize(QSize(0, 0))
        self.CD_Input02.setMaximumSize(QSize(16777215, 16777215))
        self.CD_Input02.setFont(font6)
        self.CD_Input02.setStyleSheet(u"padding: 3px;")
        self.CD_Input02.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.CD_Input02.setProperty("showGroupSeparator", True)
        self.CD_Input02.setDecimals(4)
        self.CD_Input02.setMinimum(-1.000000000000000)
        self.CD_Input02.setMaximum(1.000000000000000)
        self.CD_Input02.setSingleStep(0.001000000000000)
        self.CD_Input02.setStepType(QAbstractSpinBox.DefaultStepType)
        self.CD_Input02.setValue(0.000000000000000)

        self.CD_Main03Grid.addWidget(self.CD_Input02, 2, 3, 1, 1)

        self.gridLayout_CoastDown_Input04_A = QGridLayout()
        self.gridLayout_CoastDown_Input04_A.setSpacing(0)
        self.gridLayout_CoastDown_Input04_A.setObjectName(u"gridLayout_CoastDown_Input04_A")
        self.CD_Input04 = QDoubleSpinBox(self.CD_Main03)
        self.CD_Input04.setObjectName(u"CD_Input04")
        sizePolicy3.setHeightForWidth(self.CD_Input04.sizePolicy().hasHeightForWidth())
        self.CD_Input04.setSizePolicy(sizePolicy3)
        self.CD_Input04.setMinimumSize(QSize(0, 0))
        self.CD_Input04.setMaximumSize(QSize(16777215, 16777215))
        self.CD_Input04.setFont(font6)
        self.CD_Input04.setStyleSheet(u"padding: 3px;")
        self.CD_Input04.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.CD_Input04.setProperty("showGroupSeparator", True)
        self.CD_Input04.setMaximum(100000.000000000000000)
        self.CD_Input04.setStepType(QAbstractSpinBox.DefaultStepType)
        self.CD_Input04.setValue(0.000000000000000)

        self.gridLayout_CoastDown_Input04_A.addWidget(self.CD_Input04, 1, 0, 1, 1)

        self.verticalSpacer_CoastDown_Input04_A_01 = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.gridLayout_CoastDown_Input04_A.addItem(self.verticalSpacer_CoastDown_Input04_A_01, 0, 0, 1, 1)

        self.verticalSpacer_CoastDown_Input04_A_02 = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.gridLayout_CoastDown_Input04_A.addItem(self.verticalSpacer_CoastDown_Input04_A_02, 2, 0, 1, 1)


        self.CD_Main03Grid.addLayout(self.gridLayout_CoastDown_Input04_A, 4, 3, 1, 1)

        self.LabelSub_CD01 = QLabel(self.CD_Main03)
        self.LabelSub_CD01.setObjectName(u"LabelSub_CD01")
        sizePolicy3.setHeightForWidth(self.LabelSub_CD01.sizePolicy().hasHeightForWidth())
        self.LabelSub_CD01.setSizePolicy(sizePolicy3)
        self.LabelSub_CD01.setMinimumSize(QSize(105, 0))
        font7 = QFont()
        font7.setFamily(u"Segoe UI")
        font7.setPointSize(11)
        self.LabelSub_CD01.setFont(font7)

        self.CD_Main03Grid.addWidget(self.LabelSub_CD01, 1, 1, 1, 1)

        self.LabelSub_CD08 = QLabel(self.CD_Main03)
        self.LabelSub_CD08.setObjectName(u"LabelSub_CD08")
        self.LabelSub_CD08.setMinimumSize(QSize(105, 0))
        font8 = QFont()
        font8.setFamily(u"Segoe UI")
        font8.setPointSize(11)
        font8.setBold(False)
        font8.setWeight(50)
        self.LabelSub_CD08.setFont(font8)

        self.CD_Main03Grid.addWidget(self.LabelSub_CD08, 8, 1, 1, 1)

        self.horizontalSpacer_CoastDown_Main05_02 = QSpacerItem(1, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.CD_Main03Grid.addItem(self.horizontalSpacer_CoastDown_Main05_02, 1, 2, 8, 1)

        self.gridLayout_CoastDown_Input03_A = QGridLayout()
        self.gridLayout_CoastDown_Input03_A.setSpacing(0)
        self.gridLayout_CoastDown_Input03_A.setObjectName(u"gridLayout_CoastDown_Input03_A")
        self.CD_Input03 = QDoubleSpinBox(self.CD_Main03)
        self.CD_Input03.setObjectName(u"CD_Input03")
        sizePolicy3.setHeightForWidth(self.CD_Input03.sizePolicy().hasHeightForWidth())
        self.CD_Input03.setSizePolicy(sizePolicy3)
        self.CD_Input03.setMinimumSize(QSize(0, 0))
        self.CD_Input03.setMaximumSize(QSize(16777215, 16777215))
        self.CD_Input03.setFont(font6)
        self.CD_Input03.setStyleSheet(u"padding: 3px;")
        self.CD_Input03.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.CD_Input03.setProperty("showGroupSeparator", True)
        self.CD_Input03.setDecimals(5)
        self.CD_Input03.setMaximum(10.000000000000000)
        self.CD_Input03.setSingleStep(0.010000000000000)
        self.CD_Input03.setStepType(QAbstractSpinBox.DefaultStepType)
        self.CD_Input03.setValue(1.000000000000000)

        self.gridLayout_CoastDown_Input03_A.addWidget(self.CD_Input03, 1, 0, 1, 1)

        self.verticalSpacer_CoastDown_Input03_A_01 = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.gridLayout_CoastDown_Input03_A.addItem(self.verticalSpacer_CoastDown_Input03_A_01, 0, 0, 1, 1)

        self.verticalSpacer_CoastDown_Input03_A_02 = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.gridLayout_CoastDown_Input03_A.addItem(self.verticalSpacer_CoastDown_Input03_A_02, 2, 0, 1, 1)


        self.CD_Main03Grid.addLayout(self.gridLayout_CoastDown_Input03_A, 3, 3, 1, 1)

        self.gridLayout_CoastDown_RodPos05_2 = QGridLayout()
        self.gridLayout_CoastDown_RodPos05_2.setSpacing(0)
        self.gridLayout_CoastDown_RodPos05_2.setObjectName(u"gridLayout_CoastDown_RodPos05_2")
        self.verticalSpacer_CoastDown_RodPos05_2 = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.gridLayout_CoastDown_RodPos05_2.addItem(self.verticalSpacer_CoastDown_RodPos05_2, 0, 0, 1, 1)

        self.verticalSpacer_CoastDown_RodPos05_3 = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.gridLayout_CoastDown_RodPos05_2.addItem(self.verticalSpacer_CoastDown_RodPos05_3, 2, 0, 1, 1)

        self.CD_Input06 = QDoubleSpinBox(self.CD_Main03)
        self.CD_Input06.setObjectName(u"CD_Input06")
        sizePolicy3.setHeightForWidth(self.CD_Input06.sizePolicy().hasHeightForWidth())
        self.CD_Input06.setSizePolicy(sizePolicy3)
        self.CD_Input06.setMinimumSize(QSize(0, 0))
        self.CD_Input06.setMaximumSize(QSize(1221312, 16777215))
        self.CD_Input06.setFont(font6)
        self.CD_Input06.setStyleSheet(u"padding: 3px;")
        self.CD_Input06.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.CD_Input06.setProperty("showGroupSeparator", True)
        self.CD_Input06.setMaximum(100000.000000000000000)
        self.CD_Input06.setStepType(QAbstractSpinBox.DefaultStepType)
        self.CD_Input06.setValue(50.000000000000000)

        self.gridLayout_CoastDown_RodPos05_2.addWidget(self.CD_Input06, 1, 0, 1, 1)


        self.CD_Main03Grid.addLayout(self.gridLayout_CoastDown_RodPos05_2, 6, 3, 1, 1)

        self.gridLayout_CoastDown_RodPos05_4 = QGridLayout()
        self.gridLayout_CoastDown_RodPos05_4.setSpacing(0)
        self.gridLayout_CoastDown_RodPos05_4.setObjectName(u"gridLayout_CoastDown_RodPos05_4")
        self.verticalSpacer_CoastDown_RodPos05_6 = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.gridLayout_CoastDown_RodPos05_4.addItem(self.verticalSpacer_CoastDown_RodPos05_6, 0, 0, 1, 1)

        self.verticalSpacer_CoastDown_RodPos05_7 = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.gridLayout_CoastDown_RodPos05_4.addItem(self.verticalSpacer_CoastDown_RodPos05_7, 2, 0, 1, 1)

        self.CD_Input08 = QDoubleSpinBox(self.CD_Main03)
        self.CD_Input08.setObjectName(u"CD_Input08")
        sizePolicy3.setHeightForWidth(self.CD_Input08.sizePolicy().hasHeightForWidth())
        self.CD_Input08.setSizePolicy(sizePolicy3)
        self.CD_Input08.setMinimumSize(QSize(0, 0))
        self.CD_Input08.setMaximumSize(QSize(1221312, 16777215))
        self.CD_Input08.setFont(font6)
        self.CD_Input08.setStyleSheet(u"padding: 3px;")
        self.CD_Input08.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.CD_Input08.setProperty("showGroupSeparator", True)
        self.CD_Input08.setMaximum(100000.000000000000000)
        self.CD_Input08.setStepType(QAbstractSpinBox.DefaultStepType)
        self.CD_Input08.setValue(50.000000000000000)

        self.gridLayout_CoastDown_RodPos05_4.addWidget(self.CD_Input08, 1, 0, 1, 1)


        self.CD_Main03Grid.addLayout(self.gridLayout_CoastDown_RodPos05_4, 8, 3, 1, 1)

        self.LabelSub_CD06 = QLabel(self.CD_Main03)
        self.LabelSub_CD06.setObjectName(u"LabelSub_CD06")
        self.LabelSub_CD06.setMinimumSize(QSize(105, 0))
        self.LabelSub_CD06.setFont(font8)

        self.CD_Main03Grid.addWidget(self.LabelSub_CD06, 6, 1, 1, 1)

        self.horizontalSpacer_CoastDown_Main05_01 = QSpacerItem(5, 5, QSizePolicy.Fixed, QSizePolicy.Minimum)

        self.CD_Main03Grid.addItem(self.horizontalSpacer_CoastDown_Main05_01, 1, 0, 8, 1)

        self.gridLayout_CoastDown_RodPos05_5 = QGridLayout()
        self.gridLayout_CoastDown_RodPos05_5.setSpacing(0)
        self.gridLayout_CoastDown_RodPos05_5.setObjectName(u"gridLayout_CoastDown_RodPos05_5")
        self.verticalSpacer_CoastDown_RodPos05_8 = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.gridLayout_CoastDown_RodPos05_5.addItem(self.verticalSpacer_CoastDown_RodPos05_8, 0, 0, 1, 1)

        self.verticalSpacer_CoastDown_RodPos05_9 = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.gridLayout_CoastDown_RodPos05_5.addItem(self.verticalSpacer_CoastDown_RodPos05_9, 2, 0, 1, 1)

        self.CD_Input01 = QDoubleSpinBox(self.CD_Main03)
        self.CD_Input01.setObjectName(u"CD_Input01")
        sizePolicy3.setHeightForWidth(self.CD_Input01.sizePolicy().hasHeightForWidth())
        self.CD_Input01.setSizePolicy(sizePolicy3)
        self.CD_Input01.setMinimumSize(QSize(0, 0))
        self.CD_Input01.setMaximumSize(QSize(16777215, 16777215))
        self.CD_Input01.setFont(font6)
        self.CD_Input01.setStyleSheet(u"padding: 3px;")
        self.CD_Input01.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.CD_Input01.setProperty("showGroupSeparator", True)
        self.CD_Input01.setDecimals(2)
        self.CD_Input01.setMinimum(0.000000000000000)
        self.CD_Input01.setMaximum(100000.000000000000000)
        self.CD_Input01.setSingleStep(100.000000000000000)
        self.CD_Input01.setStepType(QAbstractSpinBox.DefaultStepType)
        self.CD_Input01.setValue(0.000000000000000)

        self.gridLayout_CoastDown_RodPos05_5.addWidget(self.CD_Input01, 1, 0, 1, 1)


        self.CD_Main03Grid.addLayout(self.gridLayout_CoastDown_RodPos05_5, 1, 3, 1, 1)

        self.LabelSub_CD07 = QLabel(self.CD_Main03)
        self.LabelSub_CD07.setObjectName(u"LabelSub_CD07")
        self.LabelSub_CD07.setMinimumSize(QSize(105, 0))
        self.LabelSub_CD07.setFont(font8)

        self.CD_Main03Grid.addWidget(self.LabelSub_CD07, 7, 1, 1, 1)

        self.gridLayout_CoastDown_RodPos05 = QGridLayout()
        self.gridLayout_CoastDown_RodPos05.setSpacing(0)
        self.gridLayout_CoastDown_RodPos05.setObjectName(u"gridLayout_CoastDown_RodPos05")
        self.CD_Input05 = QDoubleSpinBox(self.CD_Main03)
        self.CD_Input05.setObjectName(u"CD_Input05")
        sizePolicy3.setHeightForWidth(self.CD_Input05.sizePolicy().hasHeightForWidth())
        self.CD_Input05.setSizePolicy(sizePolicy3)
        self.CD_Input05.setMinimumSize(QSize(0, 0))
        self.CD_Input05.setMaximumSize(QSize(16777215, 16777215))
        self.CD_Input05.setFont(font6)
        self.CD_Input05.setStyleSheet(u"padding: 3px;")
        self.CD_Input05.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.CD_Input05.setProperty("showGroupSeparator", True)
        self.CD_Input05.setMaximum(100000.000000000000000)
        self.CD_Input05.setStepType(QAbstractSpinBox.DefaultStepType)
        self.CD_Input05.setValue(50.000000000000000)

        self.gridLayout_CoastDown_RodPos05.addWidget(self.CD_Input05, 1, 0, 1, 1)

        self.verticalSpacer_CoastDown_RodPos05_01 = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.gridLayout_CoastDown_RodPos05.addItem(self.verticalSpacer_CoastDown_RodPos05_01, 0, 0, 1, 1)

        self.verticalSpacer_CoastDown_RodPos05_02 = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.gridLayout_CoastDown_RodPos05.addItem(self.verticalSpacer_CoastDown_RodPos05_02, 2, 0, 1, 1)


        self.CD_Main03Grid.addLayout(self.gridLayout_CoastDown_RodPos05, 5, 3, 1, 1)

        self.LabelSub_CD05 = QLabel(self.CD_Main03)
        self.LabelSub_CD05.setObjectName(u"LabelSub_CD05")
        self.LabelSub_CD05.setMinimumSize(QSize(105, 0))
        self.LabelSub_CD05.setFont(font8)

        self.CD_Main03Grid.addWidget(self.LabelSub_CD05, 5, 1, 1, 1)

        self.gridLayout_CoastDown_RodPos05_3 = QGridLayout()
        self.gridLayout_CoastDown_RodPos05_3.setSpacing(0)
        self.gridLayout_CoastDown_RodPos05_3.setObjectName(u"gridLayout_CoastDown_RodPos05_3")
        self.verticalSpacer_CoastDown_RodPos05_4 = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.gridLayout_CoastDown_RodPos05_3.addItem(self.verticalSpacer_CoastDown_RodPos05_4, 0, 0, 1, 1)

        self.verticalSpacer_CoastDown_RodPos05_5 = QSpacerItem(20, 10, QSizePolicy.Minimum, QSizePolicy.Preferred)

        self.gridLayout_CoastDown_RodPos05_3.addItem(self.verticalSpacer_CoastDown_RodPos05_5, 2, 0, 1, 1)

        self.CD_Input07 = QDoubleSpinBox(self.CD_Main03)
        self.CD_Input07.setObjectName(u"CD_Input07")
        sizePolicy3.setHeightForWidth(self.CD_Input07.sizePolicy().hasHeightForWidth())
        self.CD_Input07.setSizePolicy(sizePolicy3)
        self.CD_Input07.setMinimumSize(QSize(0, 0))
        self.CD_Input07.setMaximumSize(QSize(1221312, 16777215))
        self.CD_Input07.setFont(font6)
        self.CD_Input07.setStyleSheet(u"padding: 3px;")
        self.CD_Input07.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.CD_Input07.setProperty("showGroupSeparator", True)
        self.CD_Input07.setMaximum(100000.000000000000000)
        self.CD_Input07.setStepType(QAbstractSpinBox.DefaultStepType)
        self.CD_Input07.setValue(50.000000000000000)

        self.gridLayout_CoastDown_RodPos05_3.addWidget(self.CD_Input07, 1, 0, 1, 1)


        self.CD_Main03Grid.addLayout(self.gridLayout_CoastDown_RodPos05_3, 7, 3, 1, 1)

        self.LabelSub_CD03 = QLabel(self.CD_Main03)
        self.LabelSub_CD03.setObjectName(u"LabelSub_CD03")
        sizePolicy3.setHeightForWidth(self.LabelSub_CD03.sizePolicy().hasHeightForWidth())
        self.LabelSub_CD03.setSizePolicy(sizePolicy3)
        self.LabelSub_CD03.setMinimumSize(QSize(105, 0))
        self.LabelSub_CD03.setFont(font6)

        self.CD_Main03Grid.addWidget(self.LabelSub_CD03, 3, 1, 1, 1)

        self.LabelSub_CD04 = QLabel(self.CD_Main03)
        self.LabelSub_CD04.setObjectName(u"LabelSub_CD04")
        sizePolicy3.setHeightForWidth(self.LabelSub_CD04.sizePolicy().hasHeightForWidth())
        self.LabelSub_CD04.setSizePolicy(sizePolicy3)
        self.LabelSub_CD04.setMinimumSize(QSize(105, 0))
        self.LabelSub_CD04.setFont(font6)

        self.CD_Main03Grid.addWidget(self.LabelSub_CD04, 4, 1, 1, 1)

        self.LabelSub_CD02 = QLabel(self.CD_Main03)
        self.LabelSub_CD02.setObjectName(u"LabelSub_CD02")
        sizePolicy3.setHeightForWidth(self.LabelSub_CD02.sizePolicy().hasHeightForWidth())
        self.LabelSub_CD02.setSizePolicy(sizePolicy3)
        self.LabelSub_CD02.setMinimumSize(QSize(105, 0))
        font9 = QFont()
        font9.setFamily(u"Segoe UI")
        font9.setPointSize(10)
        font9.setBold(False)
        font9.setWeight(50)
        self.LabelSub_CD02.setFont(font9)

        self.CD_Main03Grid.addWidget(self.LabelSub_CD02, 2, 1, 1, 1)


        self.layoutCoastdown0005.addLayout(self.CD_Main03Grid, 0, 0, 1, 1)


        self.gridLayout_Coastdown_Total.addWidget(self.CD_Main03, 2, 0, 1, 1)


        self.gridLayout_Coastdown_InputSet_Total.addLayout(self.gridLayout_Coastdown_Total, 0, 0, 1, 1)


        self.gridLayout_43.addWidget(self.Coastdown_InputSet_Total, 0, 0, 3, 1)


        self.gridLayout.addWidget(self.CoastDown_frame_MainWindow, 0, 0, 1, 1)


        self.retranslateUi(unitWidget_Coastdown)

        self.CoastDown_tabWidget.setCurrentIndex(1)


        QMetaObject.connectSlotsByName(unitWidget_Coastdown)
    # setupUi

    def retranslateUi(self, unitWidget_Coastdown):
        unitWidget_Coastdown.setWindowTitle(QCoreApplication.translate("unitWidget_Coastdown", u"Form", None))
        ___qtablewidgetitem = self.Coastdown_DB.horizontalHeaderItem(0)
        ___qtablewidgetitem.setText(QCoreApplication.translate("unitWidget_Coastdown", u"Database Name", None));
        ___qtablewidgetitem1 = self.Coastdown_DB.horizontalHeaderItem(1)
        ___qtablewidgetitem1.setText(QCoreApplication.translate("unitWidget_Coastdown", u"Avg. Temperature", None));
        ___qtablewidgetitem2 = self.Coastdown_DB.horizontalHeaderItem(2)
        ___qtablewidgetitem2.setText(QCoreApplication.translate("unitWidget_Coastdown", u"Boron Concentration", None));
        ___qtablewidgetitem3 = self.Coastdown_DB.horizontalHeaderItem(3)
        ___qtablewidgetitem3.setText(QCoreApplication.translate("unitWidget_Coastdown", u"Bank5 Position", None));
        ___qtablewidgetitem4 = self.Coastdown_DB.horizontalHeaderItem(4)
        ___qtablewidgetitem4.setText(QCoreApplication.translate("unitWidget_Coastdown", u"Bank4 Position", None));
        ___qtablewidgetitem5 = self.Coastdown_DB.horizontalHeaderItem(5)
        ___qtablewidgetitem5.setText(QCoreApplication.translate("unitWidget_Coastdown", u"Bank3 Position", None));
        ___qtablewidgetitem6 = self.Coastdown_DB.horizontalHeaderItem(6)
        ___qtablewidgetitem6.setText(QCoreApplication.translate("unitWidget_Coastdown", u"BankP Position", None));
        ___qtablewidgetitem7 = self.Coastdown_DB.verticalHeaderItem(0)
        ___qtablewidgetitem7.setText(QCoreApplication.translate("unitWidget_Coastdown", u"1", None));
        ___qtablewidgetitem8 = self.Coastdown_DB.verticalHeaderItem(1)
        ___qtablewidgetitem8.setText(QCoreApplication.translate("unitWidget_Coastdown", u"2", None));
        ___qtablewidgetitem9 = self.Coastdown_DB.verticalHeaderItem(2)
        ___qtablewidgetitem9.setText(QCoreApplication.translate("unitWidget_Coastdown", u"3", None));
        ___qtablewidgetitem10 = self.Coastdown_DB.verticalHeaderItem(3)
        ___qtablewidgetitem10.setText(QCoreApplication.translate("unitWidget_Coastdown", u"4", None));
        self.CoastDown_tabWidget.setTabText(self.CoastDown_tabWidget.indexOf(self.Coastdown_tabWidget_DB), QCoreApplication.translate("unitWidget_Coastdown", u"Database", None))
        self.label_166.setText(QCoreApplication.translate("unitWidget_Coastdown", u"    CoastDown Calculation Result      ", None))
        self.label_167.setText(QCoreApplication.translate("unitWidget_Coastdown", u"-", None))
        self.label_168.setText(QCoreApplication.translate("unitWidget_Coastdown", u"Inlet Temperature (\u2103)", None))
        self.label_169.setText(QCoreApplication.translate("unitWidget_Coastdown", u"Plant", None))
        self.label_170.setText(QCoreApplication.translate("unitWidget_Coastdown", u"N-1 Rod Worth (%\u0394\u03c1)", None))
        self.label_171.setText(QCoreApplication.translate("unitWidget_Coastdown", u"Cycle", None))
        self.label_172.setText(QCoreApplication.translate("unitWidget_Coastdown", u"0.00", None))
        self.label_173.setText(QCoreApplication.translate("unitWidget_Coastdown", u"Total Defect (%\u0394\u03c1)", None))
        self.label_174.setText(QCoreApplication.translate("unitWidget_Coastdown", u"13", None))
        self.label_175.setText(QCoreApplication.translate("unitWidget_Coastdown", u"Stuck Rod Worth (%\u0394\u03c1)", None))
        self.label_176.setText(QCoreApplication.translate("unitWidget_Coastdown", u"-", None))
        self.label_177.setText(QCoreApplication.translate("unitWidget_Coastdown", u"Stuck Rod (N-1 Condition)", None))
        self.label_178.setText(QCoreApplication.translate("unitWidget_Coastdown", u"Required Value (%\u0394\u03c1)", None))
        self.label_179.setText(QCoreApplication.translate("unitWidget_Coastdown", u"-", None))
        self.label_180.setText(QCoreApplication.translate("unitWidget_Coastdown", u"Burnup (MWD/MTU)", None))
        self.label_181.setText(QCoreApplication.translate("unitWidget_Coastdown", u"SKN01", None))
        self.label_182.setText(QCoreApplication.translate("unitWidget_Coastdown", u"-", None))
        self.label_183.setText(QCoreApplication.translate("unitWidget_Coastdown", u"Shutdown Margin (%\u0394\u03c1)", None))
        self.label_184.setText(QCoreApplication.translate("unitWidget_Coastdown", u"-", None))
        self.label_185.setText(QCoreApplication.translate("unitWidget_Coastdown", u"-", None))
        self.label_186.setText(QCoreApplication.translate("unitWidget_Coastdown", u"CEA Configuration", None))
        self.label_187.setText(QCoreApplication.translate("unitWidget_Coastdown", u"-", None))
        self.label_188.setText(QCoreApplication.translate("unitWidget_Coastdown", u"-", None))
        self.CoastDown_tabWidget.setTabText(self.CoastDown_tabWidget.indexOf(self.Coastdown_tabInput), QCoreApplication.translate("unitWidget_Coastdown", u"Input", None))
        self.CoastDown_tabWidget.setTabText(self.CoastDown_tabWidget.indexOf(self.Coastdown_tabRodPos), QCoreApplication.translate("unitWidget_Coastdown", u"Rod Input", None))
        self.CoastDown_tabWidget.setTabText(self.CoastDown_tabWidget.indexOf(self.Coastdown_tabReport), QCoreApplication.translate("unitWidget_Coastdown", u"Report", None))
        self.Coastdown_save_button.setText(QCoreApplication.translate("unitWidget_Coastdown", u"Save", None))
        self.Coastdown_run_button.setText(QCoreApplication.translate("unitWidget_Coastdown", u"Run", None))
        self.CoastDown_CalcTarget02.setText(QCoreApplication.translate("unitWidget_Coastdown", u"Power Search", None))
#if QT_CONFIG(statustip)
        self.CoastDown_CalcTarget01.setStatusTip("")
#endif // QT_CONFIG(statustip)
        self.CoastDown_CalcTarget01.setText(QCoreApplication.translate("unitWidget_Coastdown", u"ASI Control", None))
        self.LabelTitle_CD02.setText(QCoreApplication.translate("unitWidget_Coastdown", u"Calculation Object", None))
        self.LabelTitle_CD01.setText(QCoreApplication.translate("unitWidget_Coastdown", u"Input Setup", None))
        self.CoastDown_InpOpt2_NDR.setText(QCoreApplication.translate("unitWidget_Coastdown", u"NDR", None))
#if QT_CONFIG(statustip)
        self.CoastDown_InpOpt1_Snapshot.setStatusTip("")
#endif // QT_CONFIG(statustip)
        self.CoastDown_InpOpt1_Snapshot.setText(QCoreApplication.translate("unitWidget_Coastdown", u"Snapshot", None))
        self.LabelTitle_CD03.setText(QCoreApplication.translate("unitWidget_Coastdown", u"Calculation Information", None))
        self.LabelSub_CD01.setText(QCoreApplication.translate("unitWidget_Coastdown", u"Cycle Burnup\n"
"(MWD/MTU)", None))
        self.LabelSub_CD08.setText(QCoreApplication.translate("unitWidget_Coastdown", u"Bank 3\n"
"Position (cm)", None))
        self.LabelSub_CD06.setText(QCoreApplication.translate("unitWidget_Coastdown", u"Bank 5\n"
"Position (cm)", None))
        self.LabelSub_CD07.setText(QCoreApplication.translate("unitWidget_Coastdown", u"Bank 4\n"
"Position (cm)", None))
        self.LabelSub_CD05.setText(QCoreApplication.translate("unitWidget_Coastdown", u"Bank P\n"
"Position (cm)", None))
        self.LabelSub_CD03.setText(QCoreApplication.translate("unitWidget_Coastdown", u"Stopping\n"
"Criterion(Day)", None))
        self.LabelSub_CD04.setText(QCoreApplication.translate("unitWidget_Coastdown", u"Depletion Interval\n"
"(MWD/MTU)", None))
        self.LabelSub_CD02.setText(QCoreApplication.translate("unitWidget_Coastdown", u"Target ASI", None))
    # retranslateUi

