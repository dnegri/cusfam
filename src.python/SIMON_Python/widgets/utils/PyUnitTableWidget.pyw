
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QTableView, QSizePolicy, QFrame, QAbstractItemView, QAbstractScrollArea, QHeaderView
from PyQt5.QtCore import QSize, QSizeF, QCoreApplication, Qt
from PyQt5.QtGui import QPalette, QBrush, QColor, QFont
import Definitions as df

class unitTableWidget(QTableWidget):
    def __init__(self,frame,headerItem):

        # 00. Define Widget and Store gridlayout and buttonSetting
        super().__init__(frame)
        self.frame = frame

        # 01. Make Header Item and List
        self.nHeaderItem = len(headerItem)
        self.headerItemList = headerItem
        self.tableRowHeaderItem = []
        self.tableColumnHeaderItem = []

        self.makeTableHeader()
        self.resizeTable()

    def makeTableHeader(self):

        # Set Table Row numberQt
        self.setColumnCount(self.nHeaderItem)
        # Set Table Column number
        self.setRowCount(df.tableDefaultColumnNum)

        # Define Table Row HeaderItem
        font = QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(df.tableDefaultTextSize)
        font2 = QFont()
        font2.setFamily("Segoe UI")
        font2.setPointSize(df.tableMinimumTextSize)
        for iRow in range(self.nHeaderItem):
            headerItem = QTableWidgetItem()
            headerItem.setFont(font)
            full_text = self.headerItemList[iRow].split("\n")
            for unit_text in full_text:
                if(len(unit_text)>7):
                    headerItem.setFont(font2)
            self.setHorizontalHeaderItem(iRow, headerItem)
            self.tableRowHeaderItem.append(headerItem) # Insert HeaderItem

        # Define Table Column HeaderItem
        for iColumn in range(df.tableDefaultColumnNum):
            headerItem = QTableWidgetItem()
            self.setVerticalHeaderItem(iColumn, headerItem)
            self.tableColumnHeaderItem.append(headerItem)
            #headerItem.setFont(font)

        # Define Dummy Item in Table
        for iRow in range(self.nHeaderItem):
            headerItem = QTableWidgetItem()
            self.setItem(0, iRow, headerItem)

        # Define TableWidget objectName
        self.setObjectName(u"SD_tableWidget")

        # Define TableWidget SizePolicy
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.setMaximumSize(QSize(16777215, 16777215))

        # Define Brush for define TableWidget Palette
        brush = QBrush(QColor(210, 210, 210, 255))
        brush.setStyle(Qt.SolidPattern)
        brush1 = QBrush(QColor(38, 44, 53, 255))
        brush1.setStyle(Qt.SolidPattern)
        brush5 = QBrush(QColor(210, 210, 210, 128))
        brush5.setStyle(Qt.NoBrush)
        brush6 = QBrush(QColor(210, 210, 210, 128))
        brush6.setStyle(Qt.NoBrush)
        brush7 = QBrush(QColor(210, 210, 210, 128))
        brush7.setStyle(Qt.NoBrush)

        # Define TableWidget Palette
        palette1 = QPalette()
        palette1.setBrush(QPalette.Active, QPalette.WindowText, brush)
        palette1.setBrush(QPalette.Active, QPalette.Button, brush1)
        palette1.setBrush(QPalette.Active, QPalette.Text, brush)
        palette1.setBrush(QPalette.Active, QPalette.ButtonText, brush)
        palette1.setBrush(QPalette.Active, QPalette.Base, brush1)
        palette1.setBrush(QPalette.Active, QPalette.Window, brush1)
        brush = QBrush(QColor(210, 210, 210, 128))
        brush.setStyle(Qt.NoBrush)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette1.setBrush(QPalette.Active, QPalette.PlaceholderText, brush5)
#endif
        palette1.setBrush(QPalette.Inactive, QPalette.WindowText, brush)
        palette1.setBrush(QPalette.Inactive, QPalette.Button, brush1)
        palette1.setBrush(QPalette.Inactive, QPalette.Text, brush)
        palette1.setBrush(QPalette.Inactive, QPalette.ButtonText, brush)
        palette1.setBrush(QPalette.Inactive, QPalette.Base, brush1)
        palette1.setBrush(QPalette.Inactive, QPalette.Window, brush1)
        brush6 = QBrush(QColor(210, 210, 210, 128))
        brush6.setStyle(Qt.NoBrush)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette1.setBrush(QPalette.Inactive, QPalette.PlaceholderText, brush6)
#endif
        palette1.setBrush(QPalette.Disabled, QPalette.WindowText, brush)
        palette1.setBrush(QPalette.Disabled, QPalette.Button, brush1)
        palette1.setBrush(QPalette.Disabled, QPalette.Text, brush)
        palette1.setBrush(QPalette.Disabled, QPalette.ButtonText, brush)
        palette1.setBrush(QPalette.Disabled, QPalette.Base, brush1)
        palette1.setBrush(QPalette.Disabled, QPalette.Window, brush1)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
        palette1.setBrush(QPalette.Disabled, QPalette.PlaceholderText, brush7)
#endif
        self.setPalette(palette1)

        # Define TableWidget StyleSheet
        self.setStyleSheet(u"QTableWidget {	\n"
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
"QScrollBar:handle:horizontal {\n"
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
"    bo"
                        "rder-bottom: 1px solid rgb(44, 49, 60);\n"
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
"QTableCornerButton::section{ \n"
"\n"
"	background-color: rgb(38, 44, 53);\n"
"}\n"
"")
        self.setFrameShape(QFrame.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setDefaultDropAction(Qt.IgnoreAction)
        self.setAlternatingRowColors(False)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setShowGrid(True)
        self.setGridStyle(Qt.SolidLine)
        self.setSortingEnabled(False)
        self.horizontalHeader().setVisible(False)
        self.horizontalHeader().setCascadingSectionResizes(True)
        self.horizontalHeader().setMinimumSectionSize(52)
        self.horizontalHeader().setDefaultSectionSize(52)
        self.horizontalHeader().setProperty("showSortIndicator", False)
        self.horizontalHeader().setStretchLastSection(False)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setCascadingSectionResizes(True)
        self.verticalHeader().setDefaultSectionSize(30)
        self.verticalHeader().setHighlightSections(False)
        self.verticalHeader().setStretchLastSection(False)

        # Insert Text in HeaderItem
        for iRow in range(self.nHeaderItem):
            # headerItem = self.tableRowHeaderItem[iRow]
            # headerItem.setText(QCoreApplication.translate("unitWidget_SD",self.headerItemList[iRow],None))
            self.tableRowHeaderItem[iRow].setText(QCoreApplication.translate("unitWidget_SD",self.headerItemList[iRow],None))

        for iColumn in range(df.tableDefaultColumnNum):
            a = "%d" %(iColumn+1)
            self.tableColumnHeaderItem[iColumn].setText(QCoreApplication.translate("unitWidget_SD",a,None))
            # self.tableColumnHeaderItem[iColumn] = self.verticalHeaderItem(iColumn)

    def resizeTable(self):
        self.horizontalHeader().setVisible(True)
        self.verticalHeader().setVisible(True)
        width = []
        for column in range(self.horizontalHeader().count()):
            self.horizontalHeader().setSectionResizeMode(column,QHeaderView.ResizeToContents)
            width.append(self.horizontalHeader().sectionSize(column))

        wfactor = self.horizontalHeader().width() / sum(width)
        for column in range(self.horizontalHeader().count()):
            self.horizontalHeader().setSectionResizeMode(column, QHeaderView.Stretch)
            self.horizontalHeader().resizeSection(column, width[column]*wfactor)

        self.resizeColumnsToContents()

        __sortingEnabled = self.isSortingEnabled()
        self.setSortingEnabled(False)
        self.setSortingEnabled(__sortingEnabled)