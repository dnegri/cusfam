# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'unitWidget_RPCS.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class Ui_unitWidget_RPCS(object):
    def setupUi(self, unitWidget_RPCS):
        if not unitWidget_RPCS.objectName():
            unitWidget_RPCS.setObjectName(u"unitWidget_RPCS")
        unitWidget_RPCS.resize(1964, 1417)
        self.gridLayout = QGridLayout(unitWidget_RPCS)
        self.gridLayout.setObjectName(u"gridLayout")
        self.RPCS_frame_MainWindow = QFrame(unitWidget_RPCS)
        self.RPCS_frame_MainWindow.setObjectName(u"RPCS_frame_MainWindow")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.RPCS_frame_MainWindow.sizePolicy().hasHeightForWidth())
        self.RPCS_frame_MainWindow.setSizePolicy(sizePolicy)
        self.RPCS_frame_MainWindow.setStyleSheet(u"background-color: rgb(44, 49, 60);\n"
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
        self.RPCS_frame_MainWindow.setFrameShape(QFrame.NoFrame)
        self.RPCS_frame_MainWindow.setFrameShadow(QFrame.Raised)
        self.gridLayout_RPCS_frame_MainWindow = QGridLayout(self.RPCS_frame_MainWindow)
        self.gridLayout_RPCS_frame_MainWindow.setSpacing(3)
        self.gridLayout_RPCS_frame_MainWindow.setObjectName(u"gridLayout_RPCS_frame_MainWindow")
        self.RPCS_tabWidget = QTabWidget(self.RPCS_frame_MainWindow)
        self.RPCS_tabWidget.setObjectName(u"RPCS_tabWidget")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(200)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.RPCS_tabWidget.sizePolicy().hasHeightForWidth())
        self.RPCS_tabWidget.setSizePolicy(sizePolicy1)
        font = QFont()
        font.setFamily(u"Segoe UI")
        font.setPointSize(14)
        self.RPCS_tabWidget.setFont(font)
        self.RPCS_tabWidget.setStyleSheet(u"\n"
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
        self.RPCS_Database = QWidget()
        self.RPCS_Database.setObjectName(u"RPCS_Database")
        self.gridLayout_32 = QGridLayout(self.RPCS_Database)
        self.gridLayout_32.setObjectName(u"gridLayout_32")
        self.RPCS_DB = QTableWidget(self.RPCS_Database)
        if (self.RPCS_DB.columnCount() < 7):
            self.RPCS_DB.setColumnCount(7)
        __qtablewidgetitem = QTableWidgetItem()
        self.RPCS_DB.setHorizontalHeaderItem(0, __qtablewidgetitem)
        __qtablewidgetitem1 = QTableWidgetItem()
        self.RPCS_DB.setHorizontalHeaderItem(1, __qtablewidgetitem1)
        __qtablewidgetitem2 = QTableWidgetItem()
        self.RPCS_DB.setHorizontalHeaderItem(2, __qtablewidgetitem2)
        __qtablewidgetitem3 = QTableWidgetItem()
        self.RPCS_DB.setHorizontalHeaderItem(3, __qtablewidgetitem3)
        __qtablewidgetitem4 = QTableWidgetItem()
        self.RPCS_DB.setHorizontalHeaderItem(4, __qtablewidgetitem4)
        __qtablewidgetitem5 = QTableWidgetItem()
        self.RPCS_DB.setHorizontalHeaderItem(5, __qtablewidgetitem5)
        __qtablewidgetitem6 = QTableWidgetItem()
        self.RPCS_DB.setHorizontalHeaderItem(6, __qtablewidgetitem6)
        if (self.RPCS_DB.rowCount() < 4):
            self.RPCS_DB.setRowCount(4)
        __qtablewidgetitem7 = QTableWidgetItem()
        self.RPCS_DB.setVerticalHeaderItem(0, __qtablewidgetitem7)
        __qtablewidgetitem8 = QTableWidgetItem()
        self.RPCS_DB.setVerticalHeaderItem(1, __qtablewidgetitem8)
        __qtablewidgetitem9 = QTableWidgetItem()
        self.RPCS_DB.setVerticalHeaderItem(2, __qtablewidgetitem9)
        __qtablewidgetitem10 = QTableWidgetItem()
        self.RPCS_DB.setVerticalHeaderItem(3, __qtablewidgetitem10)
        self.RPCS_DB.setObjectName(u"RPCS_DB")
        sizePolicy2 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.RPCS_DB.sizePolicy().hasHeightForWidth())
        self.RPCS_DB.setSizePolicy(sizePolicy2)
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
        self.RPCS_DB.setPalette(palette)
        self.RPCS_DB.setStyleSheet(u"QTableWidget {	\n"
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
        self.RPCS_DB.setFrameShape(QFrame.NoFrame)
        self.RPCS_DB.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.RPCS_DB.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.RPCS_DB.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.RPCS_DB.setAlternatingRowColors(False)
        self.RPCS_DB.setSelectionMode(QAbstractItemView.SingleSelection)
        self.RPCS_DB.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.RPCS_DB.setShowGrid(True)
        self.RPCS_DB.setGridStyle(Qt.SolidLine)
        self.RPCS_DB.setSortingEnabled(False)
        self.RPCS_DB.horizontalHeader().setVisible(False)
        self.RPCS_DB.horizontalHeader().setCascadingSectionResizes(True)
        self.RPCS_DB.horizontalHeader().setDefaultSectionSize(155)
        self.RPCS_DB.horizontalHeader().setProperty("showSortIndicator", False)
        self.RPCS_DB.horizontalHeader().setStretchLastSection(True)
        self.RPCS_DB.verticalHeader().setVisible(False)
        self.RPCS_DB.verticalHeader().setCascadingSectionResizes(True)
        self.RPCS_DB.verticalHeader().setDefaultSectionSize(30)
        self.RPCS_DB.verticalHeader().setHighlightSections(False)
        self.RPCS_DB.verticalHeader().setStretchLastSection(False)

        self.gridLayout_32.addWidget(self.RPCS_DB, 0, 0, 1, 1)

        self.RPCS_tabWidget.addTab(self.RPCS_Database, "")
        self.RPCS_tabInput = QWidget()
        self.RPCS_tabInput.setObjectName(u"RPCS_tabInput")
        self.RPCS_tabInput.setStyleSheet(u"background-color: rgb(51, 58, 72);")
        self.gridLayout_83 = QGridLayout(self.RPCS_tabInput)
        self.gridLayout_83.setObjectName(u"gridLayout_83")
        self.gridLayout_83.setHorizontalSpacing(7)
        self.gridLayout_83.setContentsMargins(0, 0, 0, 0)
        self.frame_RPCS_tabInput = QFrame(self.RPCS_tabInput)
        self.frame_RPCS_tabInput.setObjectName(u"frame_RPCS_tabInput")
        self.frame_RPCS_tabInput.setFrameShape(QFrame.StyledPanel)
        self.frame_RPCS_tabInput.setFrameShadow(QFrame.Raised)
        self.gridLayout_84 = QGridLayout(self.frame_RPCS_tabInput)
        self.gridLayout_84.setObjectName(u"gridLayout_84")
        self.label_235 = QLabel(self.frame_RPCS_tabInput)
        self.label_235.setObjectName(u"label_235")
        font1 = QFont()
        font1.setFamily(u"Segoe UI")
        font1.setPointSize(20)
        font1.setUnderline(False)
        self.label_235.setFont(font1)
        self.label_235.setAlignment(Qt.AlignCenter)
        self.label_235.setMargin(20)

        self.gridLayout_84.addWidget(self.label_235, 1, 0, 1, 2)

        self.label_236 = QLabel(self.frame_RPCS_tabInput)
        self.label_236.setObjectName(u"label_236")
        font2 = QFont()
        font2.setFamily(u"Segoe UI")
        font2.setPointSize(12)
        font2.setUnderline(False)
        self.label_236.setFont(font2)
        self.label_236.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_236.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_236.setMargin(8)

        self.gridLayout_84.addWidget(self.label_236, 7, 1, 1, 1)

        self.label_237 = QLabel(self.frame_RPCS_tabInput)
        self.label_237.setObjectName(u"label_237")
        self.label_237.setFont(font2)
        self.label_237.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_237.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_237.setMargin(8)

        self.gridLayout_84.addWidget(self.label_237, 6, 0, 1, 1)

        self.label_238 = QLabel(self.frame_RPCS_tabInput)
        self.label_238.setObjectName(u"label_238")
        self.label_238.setFont(font2)
        self.label_238.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_238.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_238.setMargin(8)

        self.gridLayout_84.addWidget(self.label_238, 3, 0, 1, 1)

        self.label_239 = QLabel(self.frame_RPCS_tabInput)
        self.label_239.setObjectName(u"label_239")
        self.label_239.setFont(font2)
        self.label_239.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_239.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_239.setMargin(8)

        self.gridLayout_84.addWidget(self.label_239, 10, 0, 1, 1)

        self.label_240 = QLabel(self.frame_RPCS_tabInput)
        self.label_240.setObjectName(u"label_240")
        self.label_240.setFont(font2)
        self.label_240.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_240.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_240.setMargin(8)

        self.gridLayout_84.addWidget(self.label_240, 4, 0, 1, 1)

        self.label_241 = QLabel(self.frame_RPCS_tabInput)
        self.label_241.setObjectName(u"label_241")
        self.label_241.setFont(font2)
        self.label_241.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_241.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_241.setMargin(8)

        self.gridLayout_84.addWidget(self.label_241, 5, 1, 1, 1)

        self.label_242 = QLabel(self.frame_RPCS_tabInput)
        self.label_242.setObjectName(u"label_242")
        self.label_242.setFont(font2)
        self.label_242.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_242.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_242.setMargin(8)

        self.gridLayout_84.addWidget(self.label_242, 11, 0, 1, 1)

        self.label_243 = QLabel(self.frame_RPCS_tabInput)
        self.label_243.setObjectName(u"label_243")
        self.label_243.setFont(font2)
        self.label_243.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_243.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_243.setMargin(8)

        self.gridLayout_84.addWidget(self.label_243, 4, 1, 1, 1)

        self.verticalSpacer_40 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Fixed)

        self.gridLayout_84.addItem(self.verticalSpacer_40, 0, 0, 1, 1)

        self.label_244 = QLabel(self.frame_RPCS_tabInput)
        self.label_244.setObjectName(u"label_244")
        self.label_244.setFont(font2)
        self.label_244.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_244.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_244.setMargin(8)

        self.gridLayout_84.addWidget(self.label_244, 9, 0, 1, 1)

        self.verticalSpacer_41 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_84.addItem(self.verticalSpacer_41, 14, 0, 1, 1)

        self.label_245 = QLabel(self.frame_RPCS_tabInput)
        self.label_245.setObjectName(u"label_245")
        self.label_245.setFont(font2)
        self.label_245.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_245.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_245.setMargin(8)

        self.gridLayout_84.addWidget(self.label_245, 13, 1, 1, 1)

        self.label_246 = QLabel(self.frame_RPCS_tabInput)
        self.label_246.setObjectName(u"label_246")
        self.label_246.setFont(font2)
        self.label_246.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_246.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_246.setMargin(8)

        self.gridLayout_84.addWidget(self.label_246, 8, 0, 1, 1)

        self.label_247 = QLabel(self.frame_RPCS_tabInput)
        self.label_247.setObjectName(u"label_247")
        self.label_247.setFont(font2)
        self.label_247.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_247.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_247.setMargin(8)

        self.gridLayout_84.addWidget(self.label_247, 13, 0, 1, 1)

        self.label_248 = QLabel(self.frame_RPCS_tabInput)
        self.label_248.setObjectName(u"label_248")
        self.label_248.setFont(font2)
        self.label_248.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_248.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_248.setMargin(8)

        self.gridLayout_84.addWidget(self.label_248, 6, 1, 1, 1)

        self.label_249 = QLabel(self.frame_RPCS_tabInput)
        self.label_249.setObjectName(u"label_249")
        self.label_249.setFont(font2)
        self.label_249.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_249.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_249.setMargin(8)

        self.gridLayout_84.addWidget(self.label_249, 5, 0, 1, 1)

        self.label_250 = QLabel(self.frame_RPCS_tabInput)
        self.label_250.setObjectName(u"label_250")
        self.label_250.setMaximumSize(QSize(100, 16777215))
        self.label_250.setFont(font2)
        self.label_250.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_250.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_250.setMargin(8)

        self.gridLayout_84.addWidget(self.label_250, 3, 1, 1, 1)

        self.label_251 = QLabel(self.frame_RPCS_tabInput)
        self.label_251.setObjectName(u"label_251")
        self.label_251.setFont(font2)
        self.label_251.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_251.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_251.setMargin(8)

        self.gridLayout_84.addWidget(self.label_251, 8, 1, 1, 1)

        self.label_252 = QLabel(self.frame_RPCS_tabInput)
        self.label_252.setObjectName(u"label_252")
        self.label_252.setFont(font2)
        self.label_252.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_252.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_252.setMargin(8)

        self.gridLayout_84.addWidget(self.label_252, 12, 0, 1, 1)

        self.label_253 = QLabel(self.frame_RPCS_tabInput)
        self.label_253.setObjectName(u"label_253")
        self.label_253.setFont(font2)
        self.label_253.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_253.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_253.setMargin(8)

        self.gridLayout_84.addWidget(self.label_253, 11, 1, 1, 1)

        self.label_254 = QLabel(self.frame_RPCS_tabInput)
        self.label_254.setObjectName(u"label_254")
        self.label_254.setFont(font2)
        self.label_254.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_254.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_254.setMargin(8)

        self.gridLayout_84.addWidget(self.label_254, 9, 1, 1, 1)

        self.label_255 = QLabel(self.frame_RPCS_tabInput)
        self.label_255.setObjectName(u"label_255")
        self.label_255.setFont(font2)
        self.label_255.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_255.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_255.setMargin(8)

        self.gridLayout_84.addWidget(self.label_255, 7, 0, 1, 1)

        self.label_256 = QLabel(self.frame_RPCS_tabInput)
        self.label_256.setObjectName(u"label_256")
        self.label_256.setFont(font2)
        self.label_256.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_256.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_256.setMargin(8)

        self.gridLayout_84.addWidget(self.label_256, 12, 1, 1, 1)

        self.line_9 = QFrame(self.frame_RPCS_tabInput)
        self.line_9.setObjectName(u"line_9")
        self.line_9.setFrameShape(QFrame.HLine)
        self.line_9.setFrameShadow(QFrame.Sunken)

        self.gridLayout_84.addWidget(self.line_9, 2, 0, 1, 2)

        self.label_257 = QLabel(self.frame_RPCS_tabInput)
        self.label_257.setObjectName(u"label_257")
        self.label_257.setFont(font2)
        self.label_257.setContextMenuPolicy(Qt.DefaultContextMenu)
        self.label_257.setAlignment(Qt.AlignRight|Qt.AlignTrailing|Qt.AlignVCenter)
        self.label_257.setMargin(8)

        self.gridLayout_84.addWidget(self.label_257, 10, 1, 1, 1)


        self.gridLayout_83.addWidget(self.frame_RPCS_tabInput, 0, 0, 1, 1)

        self.RPCS_tabWidget.addTab(self.RPCS_tabInput, "")
        self.RPCS_tabRodPos = QWidget()
        self.RPCS_tabRodPos.setObjectName(u"RPCS_tabRodPos")
        self.gridLayout_85 = QGridLayout(self.RPCS_tabRodPos)
        self.gridLayout_85.setObjectName(u"gridLayout_85")
        self.gridLayout_85.setContentsMargins(0, 0, 0, 0)
        self.frame_RPCS_tabRodPos = QFrame(self.RPCS_tabRodPos)
        self.frame_RPCS_tabRodPos.setObjectName(u"frame_RPCS_tabRodPos")
        self.frame_RPCS_tabRodPos.setAutoFillBackground(False)
        self.frame_RPCS_tabRodPos.setStyleSheet(u"background-color: rgb(51, 58, 72);")
        self.gridLayout_86 = QGridLayout(self.frame_RPCS_tabRodPos)
        self.gridLayout_86.setObjectName(u"gridLayout_86")
        self.RPCS_WidgetAxial = QWidget(self.frame_RPCS_tabRodPos)
        self.RPCS_WidgetAxial.setObjectName(u"RPCS_WidgetAxial")
        sizePolicy3 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy3.setHorizontalStretch(6)
        sizePolicy3.setVerticalStretch(10)
        sizePolicy3.setHeightForWidth(self.RPCS_WidgetAxial.sizePolicy().hasHeightForWidth())
        self.RPCS_WidgetAxial.setSizePolicy(sizePolicy3)
        self.RPCS_WidgetAxial.setMinimumSize(QSize(400, 0))

        self.gridLayout_86.addWidget(self.RPCS_WidgetAxial, 0, 1, 1, 1)

        self.frame_RPCS_TableWidget = QFrame(self.frame_RPCS_tabRodPos)
        self.frame_RPCS_TableWidget.setObjectName(u"frame_RPCS_TableWidget")
        self.frame_RPCS_TableWidget.setFrameShape(QFrame.StyledPanel)
        self.frame_RPCS_TableWidget.setFrameShadow(QFrame.Raised)
        self.gridlayout_RPCS_TableWidget = QGridLayout(self.frame_RPCS_TableWidget)
        self.gridlayout_RPCS_TableWidget.setObjectName(u"gridlayout_RPCS_TableWidget")

        self.gridLayout_86.addWidget(self.frame_RPCS_TableWidget, 0, 0, 1, 1)

        self.RPCS_WidgetRadial = QWidget(self.frame_RPCS_tabRodPos)
        self.RPCS_WidgetRadial.setObjectName(u"RPCS_WidgetRadial")
        sizePolicy3.setHeightForWidth(self.RPCS_WidgetRadial.sizePolicy().hasHeightForWidth())
        self.RPCS_WidgetRadial.setSizePolicy(sizePolicy3)
        self.RPCS_WidgetRadial.setMinimumSize(QSize(0, 0))

        self.gridLayout_86.addWidget(self.RPCS_WidgetRadial, 1, 1, 1, 1)

        self.RPCS_widgetChart = QWidget(self.frame_RPCS_tabRodPos)
        self.RPCS_widgetChart.setObjectName(u"RPCS_widgetChart")
        sizePolicy4 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy4.setHorizontalStretch(10)
        sizePolicy4.setVerticalStretch(10)
        sizePolicy4.setHeightForWidth(self.RPCS_widgetChart.sizePolicy().hasHeightForWidth())
        self.RPCS_widgetChart.setSizePolicy(sizePolicy4)
        self.RPCS_widgetChart.setMinimumSize(QSize(0, 0))

        self.gridLayout_86.addWidget(self.RPCS_widgetChart, 1, 0, 1, 1)


        self.gridLayout_85.addWidget(self.frame_RPCS_tabRodPos, 0, 0, 1, 1)

        self.RPCS_tabWidget.addTab(self.RPCS_tabRodPos, "")
        self.RPCS_tabReport = QWidget()
        self.RPCS_tabReport.setObjectName(u"RPCS_tabReport")
        self.RPCS_tabReport.setStyleSheet(u"QWidget{\n"
"\n"
"}")
        self.gridLayout_87 = QGridLayout(self.RPCS_tabReport)
        self.gridLayout_87.setObjectName(u"gridLayout_87")
        self.gridLayout_87.setHorizontalSpacing(5)
        self.gridLayout_87.setContentsMargins(0, 0, 0, 0)
        self.frame_RPCS_tabReport_03 = QWidget(self.RPCS_tabReport)
        self.frame_RPCS_tabReport_03.setObjectName(u"frame_RPCS_tabReport_03")
        self.frame_RPCS_tabReport_03.setStyleSheet(u"background-color: rgb(51, 58, 72);")
        self.gridLayout_88 = QGridLayout(self.frame_RPCS_tabReport_03)
        self.gridLayout_88.setObjectName(u"gridLayout_88")
        self.verticalSpacer_RPCS_tabReport_01 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_88.addItem(self.verticalSpacer_RPCS_tabReport_01, 1, 1, 1, 1)

        self.horizontalSpacer_RPCS_tabReport_02 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_88.addItem(self.horizontalSpacer_RPCS_tabReport_02, 0, 2, 1, 1)

        self.horizontalSpacer_RPCS_tabReport_01 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_88.addItem(self.horizontalSpacer_RPCS_tabReport_01, 0, 0, 1, 1)


        self.gridLayout_87.addWidget(self.frame_RPCS_tabReport_03, 0, 0, 1, 1)

        self.RPCS_tabWidget.addTab(self.RPCS_tabReport, "")

        self.gridLayout_RPCS_frame_MainWindow.addWidget(self.RPCS_tabWidget, 0, 1, 5, 1)

        self.verticalSpacer_42 = QSpacerItem(20, 4000, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_RPCS_frame_MainWindow.addItem(self.verticalSpacer_42, 3, 0, 1, 1)

        self.RPCS_Main05_CalcButton = QFrame(self.RPCS_frame_MainWindow)
        self.RPCS_Main05_CalcButton.setObjectName(u"RPCS_Main05_CalcButton")
        self.RPCS_Main05_CalcButton.setMinimumSize(QSize(0, 0))
        self.RPCS_Main05_CalcButton.setFrameShape(QFrame.StyledPanel)
        self.RPCS_Main05_CalcButton.setFrameShadow(QFrame.Raised)
        self.RPCS_Control_CalcButton_Grid = QGridLayout(self.RPCS_Main05_CalcButton)
        self.RPCS_Control_CalcButton_Grid.setObjectName(u"RPCS_Control_CalcButton_Grid")
        self.RPCS_Control_CalcButton_Grid.setContentsMargins(0, 0, 0, 0)
        self.RPCS_save_button = QPushButton(self.RPCS_Main05_CalcButton)
        self.RPCS_save_button.setObjectName(u"RPCS_save_button")
        self.RPCS_save_button.setMinimumSize(QSize(0, 50))
        self.RPCS_save_button.setFont(font)
        self.RPCS_save_button.setStyleSheet(u"QPushButton {\n"
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

        self.RPCS_Control_CalcButton_Grid.addWidget(self.RPCS_save_button, 0, 0, 1, 1)

        self.RPCS_run_button = QPushButton(self.RPCS_Main05_CalcButton)
        self.RPCS_run_button.setObjectName(u"RPCS_run_button")
        self.RPCS_run_button.setMinimumSize(QSize(220, 50))
        self.RPCS_run_button.setFont(font)
        self.RPCS_run_button.setStyleSheet(u"QPushButton {\n"
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

        self.RPCS_Control_CalcButton_Grid.addWidget(self.RPCS_run_button, 0, 1, 1, 1)


        self.gridLayout_RPCS_frame_MainWindow.addWidget(self.RPCS_Main05_CalcButton, 4, 0, 1, 1)

        self.RPCS_InputSet_Total = QFrame(self.RPCS_frame_MainWindow)
        self.RPCS_InputSet_Total.setObjectName(u"RPCS_InputSet_Total")
        self.RPCS_InputSet_Total.setFrameShape(QFrame.StyledPanel)
        self.RPCS_InputSet_Total.setFrameShadow(QFrame.Raised)
        self.gridLayout_RPCS_InputSet_Total = QGridLayout(self.RPCS_InputSet_Total)
        self.gridLayout_RPCS_InputSet_Total.setSpacing(0)
        self.gridLayout_RPCS_InputSet_Total.setObjectName(u"gridLayout_RPCS_InputSet_Total")
        self.gridLayout_RPCS_InputSet_Total.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_RPCS_Total = QGridLayout()
        self.gridLayout_RPCS_Total.setObjectName(u"gridLayout_RPCS_Total")
        self.RPCS_Main02 = QFrame(self.RPCS_InputSet_Total)
        self.RPCS_Main02.setObjectName(u"RPCS_Main02")
        self.RPCS_Main02.setFrameShape(QFrame.StyledPanel)
        self.RPCS_Main02.setFrameShadow(QFrame.Raised)
        self.gridLayout_Lifetime_Main04_Input_5 = QGridLayout(self.RPCS_Main02)
        self.gridLayout_Lifetime_Main04_Input_5.setSpacing(6)
        self.gridLayout_Lifetime_Main04_Input_5.setObjectName(u"gridLayout_Lifetime_Main04_Input_5")
        self.gridLayout_Lifetime_Main04_Input_5.setContentsMargins(9, 9, 9, 9)
        self.RPCS_Main02Grid = QGridLayout()
        self.RPCS_Main02Grid.setObjectName(u"RPCS_Main02Grid")
        self.horizontalSpacer_RPCS_RdcOpt02 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.RPCS_Main02Grid.addItem(self.horizontalSpacer_RPCS_RdcOpt02, 1, 2, 3, 1)

        self.RPCS_RdcOpt01 = QRadioButton(self.RPCS_Main02)
        self.RPCS_RdcOpt01.setObjectName(u"RPCS_RdcOpt01")
        sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy5.setHorizontalStretch(0)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.RPCS_RdcOpt01.sizePolicy().hasHeightForWidth())
        self.RPCS_RdcOpt01.setSizePolicy(sizePolicy5)
        self.RPCS_RdcOpt01.setMinimumSize(QSize(0, 0))
        font3 = QFont()
        font3.setFamily(u"Segoe UI")
        font3.setPointSize(12)
        self.RPCS_RdcOpt01.setFont(font3)
#if QT_CONFIG(whatsthis)
        self.RPCS_RdcOpt01.setWhatsThis(u"")
#endif // QT_CONFIG(whatsthis)
        self.RPCS_RdcOpt01.setAutoExclusive(False)

        self.RPCS_Main02Grid.addWidget(self.RPCS_RdcOpt01, 1, 1, 1, 1)

        self.horizontalSpacer_RPCS_RdcOpt01 = QSpacerItem(5, 5, QSizePolicy.Fixed, QSizePolicy.Minimum)

        self.RPCS_Main02Grid.addItem(self.horizontalSpacer_RPCS_RdcOpt01, 1, 0, 3, 1)

        self.RPCS_RdcOpt02 = QRadioButton(self.RPCS_Main02)
        self.RPCS_RdcOpt02.setObjectName(u"RPCS_RdcOpt02")
        sizePolicy5.setHeightForWidth(self.RPCS_RdcOpt02.sizePolicy().hasHeightForWidth())
        self.RPCS_RdcOpt02.setSizePolicy(sizePolicy5)
        self.RPCS_RdcOpt02.setMinimumSize(QSize(0, 0))
        self.RPCS_RdcOpt02.setFont(font3)
        self.RPCS_RdcOpt02.setAutoExclusive(False)

        self.RPCS_Main02Grid.addWidget(self.RPCS_RdcOpt02, 2, 1, 1, 1)

        self.LabelTitle_RPCS02 = QLabel(self.RPCS_Main02)
        self.LabelTitle_RPCS02.setObjectName(u"LabelTitle_RPCS02")
        self.LabelTitle_RPCS02.setMinimumSize(QSize(0, 0))
        font4 = QFont()
        font4.setFamily(u"Segoe UI")
        font4.setPointSize(14)
        font4.setBold(True)
        font4.setWeight(75)
        self.LabelTitle_RPCS02.setFont(font4)

        self.RPCS_Main02Grid.addWidget(self.LabelTitle_RPCS02, 0, 0, 1, 3)


        self.gridLayout_Lifetime_Main04_Input_5.addLayout(self.RPCS_Main02Grid, 0, 1, 1, 1)


        self.gridLayout_RPCS_Total.addWidget(self.RPCS_Main02, 1, 0, 1, 1)

        self.RPCS_Main01 = QFrame(self.RPCS_InputSet_Total)
        self.RPCS_Main01.setObjectName(u"RPCS_Main01")
        self.RPCS_Main01.setFrameShape(QFrame.StyledPanel)
        self.RPCS_Main01.setFrameShadow(QFrame.Raised)
        self.gridLayout_Lifetime_Main01_InputSetup_3 = QGridLayout(self.RPCS_Main01)
        self.gridLayout_Lifetime_Main01_InputSetup_3.setObjectName(u"gridLayout_Lifetime_Main01_InputSetup_3")
        self.RPCS_Main01Grid = QGridLayout()
        self.RPCS_Main01Grid.setSpacing(6)
        self.RPCS_Main01Grid.setObjectName(u"RPCS_Main01Grid")
        self.RPCS_Main01Grid.setContentsMargins(0, 0, 0, 0)
        self.RPCS_InpOpt1_Snapshot = QRadioButton(self.RPCS_Main01)
        self.RPCS_InpOpt1_Snapshot.setObjectName(u"RPCS_InpOpt1_Snapshot")
        sizePolicy5.setHeightForWidth(self.RPCS_InpOpt1_Snapshot.sizePolicy().hasHeightForWidth())
        self.RPCS_InpOpt1_Snapshot.setSizePolicy(sizePolicy5)
        self.RPCS_InpOpt1_Snapshot.setMinimumSize(QSize(0, 0))
        self.RPCS_InpOpt1_Snapshot.setFont(font3)
#if QT_CONFIG(whatsthis)
        self.RPCS_InpOpt1_Snapshot.setWhatsThis(u"")
#endif // QT_CONFIG(whatsthis)
        self.RPCS_InpOpt1_Snapshot.setAutoExclusive(False)

        self.RPCS_Main01Grid.addWidget(self.RPCS_InpOpt1_Snapshot, 1, 1, 1, 2)

        self.RPCS_InpOpt2_NDR = QRadioButton(self.RPCS_Main01)
        self.RPCS_InpOpt2_NDR.setObjectName(u"RPCS_InpOpt2_NDR")
        sizePolicy5.setHeightForWidth(self.RPCS_InpOpt2_NDR.sizePolicy().hasHeightForWidth())
        self.RPCS_InpOpt2_NDR.setSizePolicy(sizePolicy5)
        self.RPCS_InpOpt2_NDR.setMinimumSize(QSize(0, 0))
        self.RPCS_InpOpt2_NDR.setFont(font3)
        self.RPCS_InpOpt2_NDR.setAutoExclusive(False)

        self.RPCS_Main01Grid.addWidget(self.RPCS_InpOpt2_NDR, 2, 1, 1, 2)

        self.LabelTitle_RPCS01 = QLabel(self.RPCS_Main01)
        self.LabelTitle_RPCS01.setObjectName(u"LabelTitle_RPCS01")
        self.LabelTitle_RPCS01.setMinimumSize(QSize(0, 0))
        self.LabelTitle_RPCS01.setFont(font4)

        self.RPCS_Main01Grid.addWidget(self.LabelTitle_RPCS01, 0, 0, 1, 5)

        self.horizontalSpacer_RPCS_Main01_01 = QSpacerItem(1, 20, QSizePolicy.Fixed, QSizePolicy.Minimum)

        self.RPCS_Main01Grid.addItem(self.horizontalSpacer_RPCS_Main01_01, 1, 0, 2, 1)

        self.RPCS_Snapshot = QComboBox(self.RPCS_Main01)
        self.RPCS_Snapshot.setObjectName(u"RPCS_Snapshot")
        sizePolicy5.setHeightForWidth(self.RPCS_Snapshot.sizePolicy().hasHeightForWidth())
        self.RPCS_Snapshot.setSizePolicy(sizePolicy5)
        self.RPCS_Snapshot.setMinimumSize(QSize(120, 0))
        self.RPCS_Snapshot.setMaximumSize(QSize(16777215, 30))
        font5 = QFont()
        font5.setFamily(u"Segoe UI")
        font5.setPointSize(10)
        self.RPCS_Snapshot.setFont(font5)

        self.RPCS_Main01Grid.addWidget(self.RPCS_Snapshot, 1, 3, 1, 2)

        self.horizontalSpacer_RPCS_Main01_02 = QSpacerItem(1, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.RPCS_Main01Grid.addItem(self.horizontalSpacer_RPCS_Main01_02, 2, 3, 1, 1)


        self.gridLayout_Lifetime_Main01_InputSetup_3.addLayout(self.RPCS_Main01Grid, 0, 0, 1, 1)


        self.gridLayout_RPCS_Total.addWidget(self.RPCS_Main01, 0, 0, 1, 1)

        self.RPCS_Main03 = QFrame(self.RPCS_InputSet_Total)
        self.RPCS_Main03.setObjectName(u"RPCS_Main03")
        self.RPCS_Main03.setFrameShape(QFrame.StyledPanel)
        self.RPCS_Main03.setFrameShadow(QFrame.Raised)
        self.gridLayout_Lifetime_Main04_Input_7 = QGridLayout(self.RPCS_Main03)
        self.gridLayout_Lifetime_Main04_Input_7.setSpacing(6)
        self.gridLayout_Lifetime_Main04_Input_7.setObjectName(u"gridLayout_Lifetime_Main04_Input_7")
        self.gridLayout_Lifetime_Main04_Input_7.setContentsMargins(9, 9, 9, 9)
        self.RPCS_Main03Grid = QGridLayout()
        self.RPCS_Main03Grid.setObjectName(u"RPCS_Main03Grid")
        self.gridLayout_89 = QGridLayout()
        self.gridLayout_89.setObjectName(u"gridLayout_89")
        self.verticalSpacer_57 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_89.addItem(self.verticalSpacer_57, 2, 0, 1, 1)

        self.RPCS_Input02 = QDoubleSpinBox(self.RPCS_Main03)
        self.RPCS_Input02.setObjectName(u"RPCS_Input02")
        sizePolicy6 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        sizePolicy6.setHorizontalStretch(0)
        sizePolicy6.setVerticalStretch(0)
        sizePolicy6.setHeightForWidth(self.RPCS_Input02.sizePolicy().hasHeightForWidth())
        self.RPCS_Input02.setSizePolicy(sizePolicy6)
        self.RPCS_Input02.setMinimumSize(QSize(0, 0))
        self.RPCS_Input02.setMaximumSize(QSize(16777215, 16777215))
        self.RPCS_Input02.setFont(font5)
        self.RPCS_Input02.setStyleSheet(u"padding: 3px;")
        self.RPCS_Input02.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.RPCS_Input02.setProperty("showGroupSeparator", True)
        self.RPCS_Input02.setDecimals(5)
        self.RPCS_Input02.setMinimum(0.000000000000000)
        self.RPCS_Input02.setMaximum(10.000000000000000)
        self.RPCS_Input02.setSingleStep(0.010000000000000)
        self.RPCS_Input02.setStepType(QAbstractSpinBox.DefaultStepType)
        self.RPCS_Input02.setValue(1.000000000000000)

        self.gridLayout_89.addWidget(self.RPCS_Input02, 1, 0, 1, 1)

        self.verticalSpacer_58 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_89.addItem(self.verticalSpacer_58, 0, 0, 1, 1)


        self.RPCS_Main03Grid.addLayout(self.gridLayout_89, 2, 3, 1, 2)

        self.horizontalSpacer_RPCS_Main06_01 = QSpacerItem(5, 5, QSizePolicy.Fixed, QSizePolicy.Minimum)

        self.RPCS_Main03Grid.addItem(self.horizontalSpacer_RPCS_Main06_01, 1, 0, 5, 1)

        self.gridLayout_41 = QGridLayout()
        self.gridLayout_41.setObjectName(u"gridLayout_41")
        self.horizontalSpacer_RPCS_RDC_apply = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.gridLayout_41.addItem(self.horizontalSpacer_RPCS_RDC_apply, 0, 0, 1, 1)

        self.RPCS_RDC_apply = QPushButton(self.RPCS_Main03)
        self.RPCS_RDC_apply.setObjectName(u"RPCS_RDC_apply")
        self.RPCS_RDC_apply.setMinimumSize(QSize(60, 0))
        self.RPCS_RDC_apply.setFont(font3)
        self.RPCS_RDC_apply.setStyleSheet(u"QPushButton {\n"
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

        self.gridLayout_41.addWidget(self.RPCS_RDC_apply, 0, 1, 1, 1)


        self.RPCS_Main03Grid.addLayout(self.gridLayout_41, 5, 1, 1, 4)

        self.LabelSub_RPCS02 = QLabel(self.RPCS_Main03)
        self.LabelSub_RPCS02.setObjectName(u"LabelSub_RPCS02")
        sizePolicy5.setHeightForWidth(self.LabelSub_RPCS02.sizePolicy().hasHeightForWidth())
        self.LabelSub_RPCS02.setSizePolicy(sizePolicy5)
        self.LabelSub_RPCS02.setMinimumSize(QSize(105, 0))
        font6 = QFont()
        font6.setFamily(u"Segoe UI")
        font6.setPointSize(11)
        self.LabelSub_RPCS02.setFont(font6)

        self.RPCS_Main03Grid.addWidget(self.LabelSub_RPCS02, 2, 1, 1, 1)

        self.gridLayout_90 = QGridLayout()
        self.gridLayout_90.setObjectName(u"gridLayout_90")
        self.verticalSpacer_59 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_90.addItem(self.verticalSpacer_59, 2, 0, 1, 1)

        self.verticalSpacer_60 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_90.addItem(self.verticalSpacer_60, 0, 0, 1, 1)

        self.RPCS_Input01 = QDoubleSpinBox(self.RPCS_Main03)
        self.RPCS_Input01.setObjectName(u"RPCS_Input01")
        sizePolicy5.setHeightForWidth(self.RPCS_Input01.sizePolicy().hasHeightForWidth())
        self.RPCS_Input01.setSizePolicy(sizePolicy5)
        self.RPCS_Input01.setMinimumSize(QSize(0, 0))
        self.RPCS_Input01.setMaximumSize(QSize(16777215, 16777215))
        self.RPCS_Input01.setFont(font5)
        self.RPCS_Input01.setStyleSheet(u"padding: 3px;")
        self.RPCS_Input01.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.RPCS_Input01.setProperty("showGroupSeparator", True)
        self.RPCS_Input01.setDecimals(2)
        self.RPCS_Input01.setMinimum(0.000000000000000)
        self.RPCS_Input01.setMaximum(100000.000000000000000)
        self.RPCS_Input01.setSingleStep(100.000000000000000)
        self.RPCS_Input01.setStepType(QAbstractSpinBox.DefaultStepType)
        self.RPCS_Input01.setValue(0.000000000000000)

        self.gridLayout_90.addWidget(self.RPCS_Input01, 1, 0, 1, 1)


        self.RPCS_Main03Grid.addLayout(self.gridLayout_90, 1, 3, 1, 2)

        self.LabelSub_RPCS01 = QLabel(self.RPCS_Main03)
        self.LabelSub_RPCS01.setObjectName(u"LabelSub_RPCS01")
        sizePolicy5.setHeightForWidth(self.LabelSub_RPCS01.sizePolicy().hasHeightForWidth())
        self.LabelSub_RPCS01.setSizePolicy(sizePolicy5)
        self.LabelSub_RPCS01.setMinimumSize(QSize(105, 0))
        self.LabelSub_RPCS01.setFont(font6)

        self.RPCS_Main03Grid.addWidget(self.LabelSub_RPCS01, 1, 1, 1, 1)

        self.LabelSub_RPCS03 = QLabel(self.RPCS_Main03)
        self.LabelSub_RPCS03.setObjectName(u"LabelSub_RPCS03")
        sizePolicy5.setHeightForWidth(self.LabelSub_RPCS03.sizePolicy().hasHeightForWidth())
        self.LabelSub_RPCS03.setSizePolicy(sizePolicy5)
        self.LabelSub_RPCS03.setMinimumSize(QSize(105, 0))
        self.LabelSub_RPCS03.setFont(font6)

        self.RPCS_Main03Grid.addWidget(self.LabelSub_RPCS03, 3, 1, 2, 1)

        self.LabelTitle_RPCS03 = QLabel(self.RPCS_Main03)
        self.LabelTitle_RPCS03.setObjectName(u"LabelTitle_RPCS03")
        self.LabelTitle_RPCS03.setMinimumSize(QSize(0, 0))
        self.LabelTitle_RPCS03.setFont(font4)

        self.RPCS_Main03Grid.addWidget(self.LabelTitle_RPCS03, 0, 0, 1, 5)

        self.gridLayout_91 = QGridLayout()
        self.gridLayout_91.setObjectName(u"gridLayout_91")
        self.RPCS_Input03 = QDoubleSpinBox(self.RPCS_Main03)
        self.RPCS_Input03.setObjectName(u"RPCS_Input03")
        sizePolicy5.setHeightForWidth(self.RPCS_Input03.sizePolicy().hasHeightForWidth())
        self.RPCS_Input03.setSizePolicy(sizePolicy5)
        self.RPCS_Input03.setMinimumSize(QSize(0, 0))
        self.RPCS_Input03.setMaximumSize(QSize(16777215, 16777215))
        self.RPCS_Input03.setFont(font5)
        self.RPCS_Input03.setStyleSheet(u"padding: 3px;")
        self.RPCS_Input03.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.RPCS_Input03.setProperty("showGroupSeparator", True)
        self.RPCS_Input03.setDecimals(3)
        self.RPCS_Input03.setMinimum(0.000000000000000)
        self.RPCS_Input03.setMaximum(100.000000000000000)
        self.RPCS_Input03.setSingleStep(0.100000000000000)
        self.RPCS_Input03.setStepType(QAbstractSpinBox.DefaultStepType)
        self.RPCS_Input03.setValue(3.000000000000000)

        self.gridLayout_91.addWidget(self.RPCS_Input03, 1, 0, 1, 1)

        self.verticalSpacer_51 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_91.addItem(self.verticalSpacer_51, 0, 0, 1, 1)

        self.verticalSpacer_52 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_91.addItem(self.verticalSpacer_52, 2, 0, 1, 1)


        self.RPCS_Main03Grid.addLayout(self.gridLayout_91, 3, 3, 2, 2)

        self.horizontalSpacer_RPCS_Main06_02 = QSpacerItem(1, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.RPCS_Main03Grid.addItem(self.horizontalSpacer_RPCS_Main06_02, 1, 2, 4, 1)


        self.gridLayout_Lifetime_Main04_Input_7.addLayout(self.RPCS_Main03Grid, 0, 1, 1, 1)


        self.gridLayout_RPCS_Total.addWidget(self.RPCS_Main03, 2, 0, 1, 1)


        self.gridLayout_RPCS_InputSet_Total.addLayout(self.gridLayout_RPCS_Total, 0, 0, 1, 1)

        self.verticalSpacer_43 = QSpacerItem(20, 4000, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.gridLayout_RPCS_InputSet_Total.addItem(self.verticalSpacer_43, 1, 0, 1, 1)


        self.gridLayout_RPCS_frame_MainWindow.addWidget(self.RPCS_InputSet_Total, 0, 0, 3, 1)


        self.gridLayout.addWidget(self.RPCS_frame_MainWindow, 0, 0, 1, 1)


        self.retranslateUi(unitWidget_RPCS)

        self.RPCS_tabWidget.setCurrentIndex(2)


        QMetaObject.connectSlotsByName(unitWidget_RPCS)
    # setupUi

    def retranslateUi(self, unitWidget_RPCS):
        unitWidget_RPCS.setWindowTitle(QCoreApplication.translate("unitWidget_RPCS", u"Form", None))
        ___qtablewidgetitem = self.RPCS_DB.horizontalHeaderItem(0)
        ___qtablewidgetitem.setText(QCoreApplication.translate("unitWidget_RPCS", u"Database Name", None));
        ___qtablewidgetitem1 = self.RPCS_DB.horizontalHeaderItem(1)
        ___qtablewidgetitem1.setText(QCoreApplication.translate("unitWidget_RPCS", u"Avg. Temperature", None));
        ___qtablewidgetitem2 = self.RPCS_DB.horizontalHeaderItem(2)
        ___qtablewidgetitem2.setText(QCoreApplication.translate("unitWidget_RPCS", u"Boron Concentration", None));
        ___qtablewidgetitem3 = self.RPCS_DB.horizontalHeaderItem(3)
        ___qtablewidgetitem3.setText(QCoreApplication.translate("unitWidget_RPCS", u"Bank5 Position", None));
        ___qtablewidgetitem4 = self.RPCS_DB.horizontalHeaderItem(4)
        ___qtablewidgetitem4.setText(QCoreApplication.translate("unitWidget_RPCS", u"Bank4 Position", None));
        ___qtablewidgetitem5 = self.RPCS_DB.horizontalHeaderItem(5)
        ___qtablewidgetitem5.setText(QCoreApplication.translate("unitWidget_RPCS", u"Bank3 Position", None));
        ___qtablewidgetitem6 = self.RPCS_DB.horizontalHeaderItem(6)
        ___qtablewidgetitem6.setText(QCoreApplication.translate("unitWidget_RPCS", u"BankP Position", None));
        ___qtablewidgetitem7 = self.RPCS_DB.verticalHeaderItem(0)
        ___qtablewidgetitem7.setText(QCoreApplication.translate("unitWidget_RPCS", u"1", None));
        ___qtablewidgetitem8 = self.RPCS_DB.verticalHeaderItem(1)
        ___qtablewidgetitem8.setText(QCoreApplication.translate("unitWidget_RPCS", u"2", None));
        ___qtablewidgetitem9 = self.RPCS_DB.verticalHeaderItem(2)
        ___qtablewidgetitem9.setText(QCoreApplication.translate("unitWidget_RPCS", u"3", None));
        ___qtablewidgetitem10 = self.RPCS_DB.verticalHeaderItem(3)
        ___qtablewidgetitem10.setText(QCoreApplication.translate("unitWidget_RPCS", u"4", None));
        self.RPCS_tabWidget.setTabText(self.RPCS_tabWidget.indexOf(self.RPCS_Database), QCoreApplication.translate("unitWidget_RPCS", u"Database", None))
        self.label_235.setText(QCoreApplication.translate("unitWidget_RPCS", u"    RPCS Simulation Result      ", None))
        self.label_236.setText(QCoreApplication.translate("unitWidget_RPCS", u"-", None))
        self.label_237.setText(QCoreApplication.translate("unitWidget_RPCS", u"Inlet Temperature (\u2109)", None))
        self.label_238.setText(QCoreApplication.translate("unitWidget_RPCS", u"Plant", None))
        self.label_239.setText(QCoreApplication.translate("unitWidget_RPCS", u"N-1 Rod Worth (%\u0394\u03c1)", None))
        self.label_240.setText(QCoreApplication.translate("unitWidget_RPCS", u"Cycle", None))
        self.label_241.setText(QCoreApplication.translate("unitWidget_RPCS", u"0.00", None))
        self.label_242.setText(QCoreApplication.translate("unitWidget_RPCS", u"Total Defect (%\u0394\u03c1)", None))
        self.label_243.setText(QCoreApplication.translate("unitWidget_RPCS", u"13", None))
        self.label_244.setText(QCoreApplication.translate("unitWidget_RPCS", u"Stuck Rod Worth (%\u0394\u03c1)", None))
        self.label_245.setText(QCoreApplication.translate("unitWidget_RPCS", u"-", None))
        self.label_246.setText(QCoreApplication.translate("unitWidget_RPCS", u"Stuck Rod (N-1 Condition)", None))
        self.label_247.setText(QCoreApplication.translate("unitWidget_RPCS", u"Required Value (%\u0394\u03c1)", None))
        self.label_248.setText(QCoreApplication.translate("unitWidget_RPCS", u"-", None))
        self.label_249.setText(QCoreApplication.translate("unitWidget_RPCS", u"Burnup (MWD/MTU)", None))
        self.label_250.setText(QCoreApplication.translate("unitWidget_RPCS", u"SKN01", None))
        self.label_251.setText(QCoreApplication.translate("unitWidget_RPCS", u"-", None))
        self.label_252.setText(QCoreApplication.translate("unitWidget_RPCS", u"Shutdown Margin (%\u0394\u03c1)", None))
        self.label_253.setText(QCoreApplication.translate("unitWidget_RPCS", u"-", None))
        self.label_254.setText(QCoreApplication.translate("unitWidget_RPCS", u"-", None))
        self.label_255.setText(QCoreApplication.translate("unitWidget_RPCS", u"CEA Configuration", None))
        self.label_256.setText(QCoreApplication.translate("unitWidget_RPCS", u"-", None))
        self.label_257.setText(QCoreApplication.translate("unitWidget_RPCS", u"-", None))
        self.RPCS_tabWidget.setTabText(self.RPCS_tabWidget.indexOf(self.RPCS_tabInput), QCoreApplication.translate("unitWidget_RPCS", u"Input", None))
        self.RPCS_tabWidget.setTabText(self.RPCS_tabWidget.indexOf(self.RPCS_tabRodPos), QCoreApplication.translate("unitWidget_RPCS", u"Results", None))
        self.RPCS_tabWidget.setTabText(self.RPCS_tabWidget.indexOf(self.RPCS_tabReport), QCoreApplication.translate("unitWidget_RPCS", u"Report", None))
        self.RPCS_save_button.setText(QCoreApplication.translate("unitWidget_RPCS", u"Save", None))
        self.RPCS_run_button.setText(QCoreApplication.translate("unitWidget_RPCS", u"Run", None))
#if QT_CONFIG(statustip)
        self.RPCS_RdcOpt01.setStatusTip("")
#endif // QT_CONFIG(statustip)
        self.RPCS_RdcOpt01.setText(QCoreApplication.translate("unitWidget_RPCS", u"Succeccive Input", None))
        self.RPCS_RdcOpt02.setText(QCoreApplication.translate("unitWidget_RPCS", u"Excel-Base Input", None))
        self.LabelTitle_RPCS02.setText(QCoreApplication.translate("unitWidget_RPCS", u"Shutdown Input Setting", None))
#if QT_CONFIG(statustip)
        self.RPCS_InpOpt1_Snapshot.setStatusTip("")
#endif // QT_CONFIG(statustip)
        self.RPCS_InpOpt1_Snapshot.setText(QCoreApplication.translate("unitWidget_RPCS", u"Snapshot", None))
        self.RPCS_InpOpt2_NDR.setText(QCoreApplication.translate("unitWidget_RPCS", u"NDR", None))
        self.LabelTitle_RPCS01.setText(QCoreApplication.translate("unitWidget_RPCS", u"Input Setup", None))
        self.RPCS_RDC_apply.setText(QCoreApplication.translate("unitWidget_RPCS", u"Insert Input", None))
        self.LabelSub_RPCS02.setText(QCoreApplication.translate("unitWidget_RPCS", u"<html><head/><body><p>Target<br>Eigenvalue</p></body></html>", None))
        self.LabelSub_RPCS01.setText(QCoreApplication.translate("unitWidget_RPCS", u"<html><head/><body><p>Cycle Burnup<br>(MWD/MTU)</p></body></html>", None))
        self.LabelSub_RPCS03.setText(QCoreApplication.translate("unitWidget_RPCS", u"<html><head/><body><p><span style=\" font-size:10pt;\">Power Decrease<br/>Ratio (%/hour)</span></p></body></html>", None))
        self.LabelTitle_RPCS03.setText(QCoreApplication.translate("unitWidget_RPCS", u"Succeccive Input Setting", None))
    # retranslateUi

