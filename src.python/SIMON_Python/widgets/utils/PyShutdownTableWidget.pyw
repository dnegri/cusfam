import widgets.utils.PyUnitTableWidget as unitTable
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QTableView, QSizePolicy, QFrame, QAbstractItemView, \
    QAbstractScrollArea, QHeaderView, QPushButton, QSpacerItem, QHBoxLayout, QWidget, QAbstractSpinBox, QFileDialog
from PyQt5.QtCore import QSize, QSizeF, QCoreApplication, Qt
from PyQt5.QtGui import QPalette, QBrush, QColor, QFont

import widgets.utils.PyTableDoubleSpinBox as unitDoubleBox
import os
import pandas as pd
import Definitions as df


class ShutdownTableWidget(unitTable.unitTableWidget):
    def __init__(self, frame, headerItem, tableItemFormat, number_input=2, number_output=6, number_rod=4):
        super().__init__(frame, headerItem )

        self.tableItemFormat = tableItemFormat
        self.number_of_input_elements = number_input
        self.number_of_output_elements = number_output
        self.number_of_rod_output_elements = number_rod

        # Initialize Dataset
        self.InputArray = []
        self.table_array = []
        self.calc_rodPosBox = []
        self.calc_rodPos = []
        self.rodPosChangedHistory = []
        self.nOutputArray = 0
        self.outputArray = []

        self.outputFlag = False

        # Make TableWidget Layout
        self.tableButtonLayout = QHBoxLayout()
        self.tableButtonLayout.setObjectName(u"horizontalLayout_SD_TableWidget")

        # Define Button for control TableWidget Dataset
        self.SD_tableWidget_button01 = QPushButton(self.frame)
        self.SD_tableWidget_button02 = QPushButton(self.frame)
        self.SD_tableWidget_button03 = QPushButton(self.frame)
        self.SD_tableWidget_button01.hide()
        self.SD_tableWidget_button02.hide()
        self.SD_tableWidget_button03.hide()
        self.makeButton()
        # self.SD_tableWidget_button03 = QPushButton(self.frame)

        # self.makeButton()
        self.tableDatasetFlag = False
        self.last_update = 0
        self.columnHeader = headerItem

        self.idx_flag = 0
        self.idx_input = 1
        self.idx_calc_value = 2
        self.idx_rod_pos = 3

        self.idx_flag_input = 0
        self.idx_flag_calc  = 1
        self.idx_flag_rod_pos = 2


    # def returnButtonLayout(self):
    #     return self.tableButtonLayout
    # 
    # def returnTableButton(self):
    #     return self.SD_tableWidget_button01, self.SD_tableWidget_button02, self.SD_tableWidget_button03

    def makeButton(self):
        # Define horizontalSpacer
        self.horizontalSpacer_SD_TableWidget = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.tableButtonLayout.addItem(self.horizontalSpacer_SD_TableWidget)

        self.SD_tableWidget_button01.setObjectName(u"SD_tableWidget_button01")
        self.SD_tableWidget_button01.setMinimumSize(QSize(110, 24))
        spinBox_DoubleSpinBox = df.styleSheet_Table_DoubleSpinBox
        self.SD_tableWidget_button01.setStyleSheet(spinBox_DoubleSpinBox)
        self.tableButtonLayout.addWidget(self.SD_tableWidget_button01)

        self.SD_tableWidget_button02.setObjectName(u"SD_tableWidget_button02")
        self.SD_tableWidget_button02.setMinimumSize(QSize(110, 24))
        self.SD_tableWidget_button02.setStyleSheet(spinBox_DoubleSpinBox)

        self.tableButtonLayout.addWidget(self.SD_tableWidget_button02)

        self.SD_tableWidget_button03.setObjectName(u"SD_tableWidget_button03")
        self.SD_tableWidget_button03.setMinimumSize(QSize(110, 24))
        self.SD_tableWidget_button03.setStyleSheet(spinBox_DoubleSpinBox)

        self.tableButtonLayout.addWidget(self.SD_tableWidget_button03)
        self.SD_tableWidget_button01.setText(QCoreApplication.translate("unitWidget_SD", u"To Excel", None))
        self.SD_tableWidget_button02.setText(QCoreApplication.translate("unitWidget_SD", u"Reset", None))
        self.SD_tableWidget_button03.setText(QCoreApplication.translate("unitWidget_SD", u"Clear Output", None))


    def insertSnapshotInputArray(self, inputArray):
        self.InputArray = inputArray
        self.calc_rodPos = []
        self.calc_rodPosBox = []
        nStep = len(self.InputArray)

        # Reset TableWidget and, Redefine Row number
        self.setRowCount(0)
        self.setRowCount(nStep)
        _translate = QCoreApplication.translate
        for iStep in range(nStep):
            for iRow in range(len(self.InputArray[iStep])):
                item = QTableWidgetItem()
                font = QFont()
                font.setPointSize(11)
                font.setBold(False)
                # font.setWeight(75)
                item.setFont(font)
                item.setTextAlignment(Qt.AlignCenter)
                if(iRow==4 or iRow==5):
                    text = ""
                else:
                    text = self.tableItemFormat[iRow] % self.InputArray[iStep][iRow]

                item.setText(_translate("Form", text))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.setItem(iStep, iRow, item)
        #self.setRowCount(nStep+1)
        #
        # for iRow in range(len(self.InputArray[nStep - 1])):
        #     item = QTableWidgetItem()
        #     font = QFont()
        #     font.setPointSize(11)
        #     font.setBold(False)
        #     # font.setWeight(75)
        #     item.setFont(font)
        #     item.setTextAlignment(Qt.AlignCenter)
        #     text = self.tableItemFormat[iRow] % self.InputArray[iStep][iRow]
        #     if (iRow == 0):
        #         text = "%.3f" % self.InputArray[nStep - 1][iRow]
        #     item.setText(_translate("Form", text))
        #     item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        #     self.setItem(nStep - 1, iRow, item)

    def addSnapshotInputArray(self, nSnapshotArray, inputArray):

        self.InputArray = inputArray
        self.calc_rodPos = []
        self.calc_rodPosBox = []
        nStep = len(self.InputArray)

        # Reset TableWidget and, Redefine Row number
        #self.setRowCount(0)
        self.setRowCount(nStep)

        self.setTableItemSetting(nSnapshotArray)
        self.setTableSpinBox(nSnapshotArray)

        _translate = QCoreApplication.translate
        for iStep in range(nSnapshotArray,nStep - 1):
            for iRow in range(len(self.InputArray[iStep])):
                item = QTableWidgetItem()
                font = QFont()
                font.setPointSize(11)
                font.setBold(False)
                # font.setWeight(75)
                item.setFont(font)
                item.setTextAlignment(Qt.AlignCenter)
                text = self.tableItemFormat[iRow] % self.InputArray[iStep][iRow]

                item.setText(_translate("Form", text))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.setItem(iStep, iRow, item)

        for iRow in range(len(self.InputArray[nStep - 1])):
            item = QTableWidgetItem()
            font = QFont()
            font.setPointSize(11)
            font.setBold(False)
            # font.setWeight(75)
            item.setFont(font)
            item.setTextAlignment(Qt.AlignCenter)
            text = self.tableItemFormat[iRow] % self.InputArray[iStep][iRow]
            if(iRow == 0):
                text = "%.3f" % self.InputArray[nStep - 1][iRow]
            item.setText(_translate("Form", text))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.setItem(nStep - 1, iRow, item)

    def add_input_array(self, table_array):
        self.table_array = table_array
        # self.calc_rodPos = []
        # self.calc_rodPosBox = []
        nStep = len(self.table_array)

        # Reset TableWidget and, Redefine Row number
        self.setRowCount(0)
        self.setRowCount(nStep)

        self.set_table_item_setting()
        self.set_table_spinbox()

        _translate = QCoreApplication.translate
        for iStep in range(nStep - 1):
            for iRow in range(self.number_of_input_elements):
                item = QTableWidgetItem()
                font = QFont()
                font.setPointSize(11)
                font.setBold(False)
                # font.setWeight(75)
                item.setFont(font)
                item.setTextAlignment(Qt.AlignCenter)
                text = self.tableItemFormat[iRow] % self.table_array[iStep][self.idx_input][iRow]

                item.setText(_translate("Form", text))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.setItem(iStep, iRow, item)

        for iRow in range(self.number_of_input_elements):
            item = QTableWidgetItem()
            font = QFont()
            font.setPointSize(11)
            font.setBold(False)
            # font.setWeight(75)
            item.setFont(font)
            item.setTextAlignment(Qt.AlignCenter)
            text = self.tableItemFormat[iRow] % self.table_array[iStep+1][self.idx_input][iRow]
            if(iRow == 0):
                text = "%.3f" % self.table_array[iStep+1][self.idx_input][iRow]
            item.setText(_translate("Form", text))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.setItem(nStep - 1, iRow, item)

    def set_table_item_setting(self):
        nRow = self.rowCount()
        nColumn = self.columnCount()

        # Setting TableWidgetItem Interaction
        for iRow in range(nRow):
            #if(self.input_array[iRow])
            if(self.table_array[iRow][self.idx_flag][self.idx_flag_rod_pos]==True):
                columnValue = nColumn - self.number_of_rod_output_elements
            else:
                columnValue = nColumn
            for iColumn in range(columnValue):#- self.number_of_rod_output_elements):
                tmp = QTableWidgetItem()
                tmp.setFlags(tmp.flags() ^ Qt.ItemIsEditable)
                tmp.setTextAlignment(Qt.AlignCenter)
                self.setItem(iRow, iColumn, tmp)

    def set_table_spinbox(self):
        nRow = self.rowCount()
        nColumn = self.columnCount()
        # Define
        for iRow in range(nRow):
            if (self.table_array[iRow][self.idx_flag][self.idx_flag_rod_pos] == True):
                unitRodPos = []
                unitData = []
                for iColumn in range(self.number_of_rod_output_elements):
                    unitBox = self.makeUnitBox(iRow, iColumn)
                    # unitBox.hide()
                    unitBox.setVisible(False)
                    unitRodPos.append(unitBox)
                    unitData.append(unitBox.value())

                self.calc_rodPosBox.append(unitRodPos)
                self.calc_rodPos.append(unitData)
            else:
                unitRodPos = [ None, None, None, None]
                unitData = [ 0.0, 0.0, 0.0, 0.0]
                self.calc_rodPosBox.append(unitRodPos)
                self.calc_rodPos.append(unitData)

    def addInputArray(self, inputArray):

        self.InputArray = inputArray
        self.calc_rodPos = []
        self.calc_rodPosBox = []
        nStep = len(self.InputArray)

        # Reset TableWidget and, Redefine Row number
        self.setRowCount(0)
        self.setRowCount(nStep)

        self.setTableItemSetting(0)
        self.setTableSpinBox(0)

        # input_length = len(self.InputArray[0])

        # ydnam limit input array is set to 2
        input_length = 2

        _translate = QCoreApplication.translate
        for iStep in range(nStep):
            for iRow in range(input_length):
                item = QTableWidgetItem()
                font = QFont()
                font.setPointSize(11)
                font.setBold(False)
                # font.setWeight(75)
                item.setFont(font)
                item.setTextAlignment(Qt.AlignCenter)
                text = self.tableItemFormat[iRow] % self.InputArray[iStep][iRow]

                item.setText(_translate("Form", text))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.setItem(iStep, iRow, item)
        #
        # for iRow in range(len(self.InputArray[nStep - 1])):
        #     item = QTableWidgetItem()
        #     font = QFont()
        #     font.setPointSize(11)
        #     font.setBold(False)
        #     # font.setWeight(75)
        #     item.setFont(font)
        #     item.setTextAlignment(Qt.AlignCenter)
        #     text = self.tableItemFormat[iRow] % self.InputArray[iStep][iRow]
        #     if (iRow == 0):
        #         text = "%.3f" % self.InputArray[nStep - 1][iRow]
        #     item.setText(_translate("Form", text))
        #     item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        #     self.setItem(nStep - 1, iRow, item)

    def setTableItemSetting(self,nSnapshotArray):
        nRow = self.rowCount()
        nColumn = self.columnCount()

        # Setting TableWidgetItem Interaction
        for iRow in range(nSnapshotArray,nRow):
            for iColumn in range(nColumn - self.number_of_rod_output_elements):
                tmp = QTableWidgetItem()
                tmp.setFlags(tmp.flags() ^ Qt.ItemIsEditable)
                tmp.setTextAlignment(Qt.AlignCenter)
                self.setItem(iRow, iColumn, tmp)


    def setTableSpinBox(self,nSnapshotArray):
        nRow = self.rowCount()
        nColumn = self.columnCount()
        # Define
        for iRow in range(nSnapshotArray,nRow):
            unitRodPos = []
            unitData = []
            for iColumn in range(self.number_of_rod_output_elements):
                unitBox = self.makeUnitBox(iRow, iColumn)
                # unitBox.hide()
                unitBox.setVisible(False)
                unitRodPos.append(unitBox)
                unitData.append(unitBox.value())

            self.calc_rodPosBox.append(unitRodPos)
            self.calc_rodPos.append(unitData)

    def makeUnitBox(self, iRow, iColumn):
        subID = ["_P", "_5", "_4", "_3"]

        # Set Font for DoubleSpinBox
        font = QFont()
        font.setFamily(u"Segoe UI")
        font.setBold(True)
        font.setPointSize(10)

        doubleSpinBoxWidget = QWidget()
        doubleSpinBoxWidget.setStyleSheet("QWidget { background-color: rgb(38, 44, 53);}")

        doubleSpinBox = unitDoubleBox.tableDoubleSpinBox()
        doubleSpinBox.saveBoxPosition(iRow, iColumn)
        # doubleSpinBox = QtWidgets.QDoubleSpinBox()
        doubleSpinBox.setObjectName(u"SD_rodPos_%03d%s" % (iRow + 1, subID[iColumn]))

        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(doubleSpinBox.sizePolicy().hasHeightForWidth())
        doubleSpinBox.setSizePolicy((sizePolicy))
        doubleSpinBox.setMinimumSize(QSize(0, 0))
        doubleSpinBox.setMaximumSize(QSize(120, 16777215))
        doubleSpinBox.setAlignment(Qt.AlignCenter)
        doubleSpinBox.setFont(font)
        # doubleSpinBox.setStyleSheet(u"padding: 3px;")
        doubleSpinBox.setButtonSymbols(QAbstractSpinBox.NoButtons)
        doubleSpinBox.setProperty("showGroupSeparator", True)
        doubleSpinBox.setDecimals(2)
        doubleSpinBox.setMinimum(0.000000000000000)
        doubleSpinBox.setMaximum(381.000000000000)
        doubleSpinBox.setSingleStep(1.000000000000000)
        doubleSpinBox.setStepType(QAbstractSpinBox.DefaultStepType)
        doubleSpinBox.setValue(381.000000000000000)
        doubleSpinBox.setReadOnly(True)

        layout = QHBoxLayout(doubleSpinBoxWidget)
        layout.addWidget(doubleSpinBox)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)

        item = QTableWidgetItem()
        item.setTextAlignment(Qt.AlignCenter)

        # (a,b) = doubleSpinBox.returnBoxPosition()
        # print(a,b)
        self.setCellWidget(iRow, iColumn + self.number_of_input_elements + self.number_of_output_elements,
                           doubleSpinBoxWidget)

        doubleSpinBox.editingFinished.connect(lambda: self.checkChangedCondition(iRow, iColumn))

        # self.SD_tableWidget_button01.clicked['bool'].connect(self.clickSaveAsExcel)
        # self.SD_tableWidget_button02.clicked['bool'].connect(self.resetPositionData)

        return doubleSpinBox

    def checkChangedCondition(self, iRow, iColumn):

        tmp = round(self.calc_rodPosBox[iRow][iColumn].value(), 2)
        tmp2 = round(self.calc_rodPos[iRow][iColumn], 2)
        # print("return!",iRow, iColumn,tmp, tmp2)

        if (tmp != tmp2):
            insertFlag = True
            unitRodPos = [iRow, iColumn]
            for idx in self.rodPosChangedHistory:
                if (idx == unitRodPos):
                    insertFlag = False
                    break
            if (insertFlag == True):
                self.rodPosChangedHistory.append(unitRodPos)
                self.rodPosChangedHistory.sort()

            # Change Font Color
            palette = QPalette()
            palette.setColor(QPalette.Text, QColor(0xFF, 0x99, 0x33))
            # self.calc_rodPosBox[iRow][iColumn].setPalette(palette)
            self.calc_rodPosBox[iRow][iColumn].setStyleSheet(u"color: rgb(255,51,51);")
            self.calc_rodPosBox[iRow][iColumn].update()

            print("color changed!")

        else:
            popupFlag = False
            unitRodPos = [iRow, iColumn]
            for idx in self.rodPosChangedHistory:
                if (idx == unitRodPos):
                    popupFlag = True
                    break
            if (popupFlag == True):
                self.rodPosChangedHistory.remove(unitRodPos)

            palette = QPalette()
            palette.setColor(QPalette.Text, QColor(0xFF, 0x99, 0x33))
            # self.calc_rodPosBox[iRow][iColumn].setPalette(palette)
            self.calc_rodPosBox[iRow][iColumn].setStyleSheet(u"color: rgb(210,210,210);")
            self.calc_rodPosBox[iRow][iColumn].update()

            print("color didn't changed!")

    def clickSaveAsExcel(self):

        if not self.outputArray:
            return False

        if len(self.outputArray) == 0:
            return False

        for output_temp in self.outputArray:
            if len(output_temp) < self.number_of_output_elements + self.number_of_rod_output_elements:
                return False

        fn, _ = QFileDialog.getSaveFileName(None, 'Export FA datas as Excel', ".",
                                            'Microsoft Excel WorkSheet(*.xlsx);;'
                                            'Microsoft Excel 97-2003 WorkSheet(*.xls)')
        if fn != '':

            base_dir = os.getcwd()
            xlxs_dir = os.path.join(base_dir, fn)
            writer = pd.ExcelWriter(xlxs_dir, engine="xlsxwriter")

            columnHeader = self.columnHeader
            # df2 = pd.DataFrame(excelDataSet,columns=columnHeader)
            # df2.to_excel(writer,
            #             sheet_name='FA_Inventory_Data',
            #             na_rep='NaN',
            #             header= False,
            #             index = False,
            #             startrow=1,
            #             startcol=1)
            # (max_row,max_col) = df2.shape
            workbook = writer.book
            worksheet = workbook.add_worksheet()  # s['FA_Inventory_Data']
            # worksheet.autofilter(0,0,max_row,max_col)

            headerFormat = workbook.add_format({'bold': True,
                                                'text_wrap': True,
                                                'align': 'center',
                                                'valign': 'vcenter',
                                                'fg_color': '#93cddd',
                                                'border': 1})

            # format00 = workbook.add_format({'align':'center','border':1})
            format01 = workbook.add_format({'align': 'center', 'border': 1, 'num_format': '0.0'})
            format02 = workbook.add_format({'align': 'center', 'border': 1, 'num_format': '0.00'})
            format03 = workbook.add_format({'align': 'center', 'border': 1, 'num_format': '0.000'})
            format05 = workbook.add_format({'align': 'center', 'border': 1, 'num_format': '0.00000'})

            formatArray = [format01, format02, format03, format01, format01,
                           format03, format03, format03, format02, format02, format02, format02 ]

            for rowIdx in range(len(columnHeader)):
                worksheet.write(0, rowIdx + 1, columnHeader[rowIdx], headerFormat)

            for colIdx in range(len(self.InputArray)):
                for rowIdx in range(self.number_of_input_elements):
                    worksheet.write(colIdx + 1, rowIdx + 1, self.InputArray[colIdx][rowIdx], formatArray[rowIdx])

            if (self.outputFlag == False):
                pass
                # for colIdx in range(len(self.inputArray)):
                #    for rowIdx in range(4, 9):
                #        worksheet.write(colIdx + 1, rowIdx + 1, '', formatArray[rowIdx])
            else:
                for colIdx in range(len(self.outputArray)):
                    #for rowIdx in range(10):
                    for rowIdx in range(self.number_of_input_elements, self.number_of_input_elements +
                                                                       self.number_of_output_elements +
                                                                       self.number_of_rod_output_elements):
                        if rowIdx - self.number_of_input_elements < 2:
                            value = self.outputArray[colIdx][rowIdx - self.number_of_input_elements]
                        elif rowIdx - self.number_of_input_elements == 2:
                            value = self.outputArray[colIdx][df.asi_o_reactivity]
                        else:
                            value = self.outputArray[colIdx][rowIdx - self.number_of_input_elements-1]
                        worksheet.write(colIdx + 1, rowIdx + 1,
                                        value,
                                        formatArray[rowIdx])
            worksheet.set_column('B:J', 12)
            writer.close()

        return True

    def resetPositionData(self):
        if (len(self.rodPosChangedHistory) == 0):
            if (self.tableDatasetFlag == True):
                print("Error! Calculation dataset didn't changed!")
            else:
                print("Error! There is No Calculation DataSet")
            return

        for iRow in range(len(self.calc_rodPos)):
            for iColumn in range(self.number_of_rod_output_elements):
                value = self.calc_rodPos[iRow][iColumn]
                self.calc_rodPosBox[iRow][iColumn].setValue(value)
                self.calc_rodPosBox[iRow][iColumn].setStyleSheet(u"color: rgb(210,210,210);")
                self.calc_rodPosBox[iRow][iColumn].update()

        self.rodPosChangedHistory = []

    def makeOutputTable(self, outputArray):
        self.outputArray = outputArray
        self.outputFlag = True
        _translate = QCoreApplication.translate
        nStep = len(outputArray)
        self.nOutputArray = nStep
        for iStep in range(nStep):
            for iColumn in range(self.number_of_output_elements):
                item = QTableWidgetItem()
                font = QFont()
                font.setPointSize(11)
                font.setBold(False)
                # font.setWeight(75)
                item.setFont(font)
                item.setTextAlignment(Qt.AlignCenter)
                text = "%.3f" % outputArray[iStep][iColumn]
                if len(outputArray[iStep]) == self.number_of_rod_output_elements:
                    text = ""
                item.setText(_translate("Form", text))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.setItem(iStep, iColumn + self.number_of_input_elements, item)

        if len(outputArray[iStep]) != self.number_of_output_elements:
            for iRow in range(nStep):
                if len(self.calc_rodPosBox[iRow]) == self.number_of_rod_output_elements:
                    for iColumn in range(self.number_of_rod_output_elements):
                        visible = True
                        number_output = self.number_of_output_elements
                        if len(outputArray[iStep]) == self.number_of_rod_output_elements:
                            number_output = 0

                        value_display = outputArray[iRow][iColumn + number_output-1]
                        self.calc_rodPosBox[iRow][iColumn].setVisible(visible)
                        self.calc_rodPosBox[iRow][iColumn].setValue(value_display)
                        self.calc_rodPos[iRow][iColumn] = value_display

    def appendOutputTable(self, outputArray, start_index):
        if start_index == 0:
            self.outputArray = outputArray
            self.nOutputArray = len(outputArray)
            self.last_update = self.nOutputArray - 1
        else:
            self.outputArray += outputArray
            self.nOutputArray = len(self.outputArray)
            self.last_update = self.nOutputArray - 1

        self.outputFlag = True
        _translate = QCoreApplication.translate

        nStep = len(outputArray)

        for iStep in range(nStep):
            for iColumn in range(self.number_of_output_elements):
                item = QTableWidgetItem()
                font = QFont()
                font.setPointSize(11)
                font.setBold(False)
                # font.setWeight(75)
                item.setFont(font)
                item.setTextAlignment(Qt.AlignCenter)

                text = "%.3f" % outputArray[iStep][iColumn]
                if iColumn == 2:
                    text = "%.3f" % outputArray[iStep][df.asi_o_reactivity]
                elif iColumn > 2:
                    text = "%.3f" % outputArray[iStep][iColumn-1]

                item.setText(_translate("Form", text))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.setItem(start_index + iStep, iColumn + self.number_of_input_elements, item)

        for iRow in range(nStep):
            if len(self.calc_rodPosBox[start_index + iRow]) == self.number_of_rod_output_elements:
                for iColumn in range(self.number_of_rod_output_elements):
                    # self.calc_rodPosBox[iRow][iColumn].show()
                    visible = True
                    value_display = outputArray[iRow][iColumn + self.number_of_output_elements-1]

                    self.calc_rodPosBox[start_index + iRow][iColumn].setVisible(visible)
                    self.calc_rodPosBox[start_index + iRow][iColumn].setValue(value_display)
                    self.calc_rodPos[start_index + iRow][iColumn] = value_display

    def clearOutputArray(self, start=0):

        self.outputArray = []
        self.nOutputArray = 0
        self.last_update = 0

        self.outputFlag = True
        _translate = QCoreApplication.translate
        nStep = len(self.InputArray)
        for iStep in range(start, nStep):
            for iColumn in range(self.number_of_output_elements):
                item = QTableWidgetItem()
                font = QFont()
                font.setPointSize(11)
                font.setBold(False)
                # font.setWeight(75)
                item.setFont(font)
                item.setTextAlignment(Qt.AlignCenter)
                text = ""
                item.setText(_translate("Form", text))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.setItem(iStep, iColumn + self.number_of_input_elements, item)

    def clearOutputRodArray(self, start=0):

        self.outputArray = []
        self.nOutputArray = 0
        self.last_update = 0

        self.outputFlag = True
        _translate = QCoreApplication.translate
        nStep = len(self.calc_rodPosBox)
        for iRow in range(start, nStep):
            if len(self.calc_rodPosBox[start + iRow]) == self.number_of_rod_output_elements:
                for iColumn in range(self.number_of_rod_output_elements):
                    # self.calc_rodPosBox[iRow][iColumn].show()
                    visible = False
                    # value_display = outputArray[iRow][iColumn+self.number_of_output_elements]

                    self.calc_rodPosBox[start + iRow][iColumn].setVisible(visible)
                    # self.calc_rodPosBox[start+iRow][iColumn].setValue(value_display)
                    # self.calc_rodPos[start+iRow][iColumn] = value_display

    def getDefinedRodPosition(self):
        return self.rodPosChangedHistory, self.calc_rodPos

    def checkModified(self):
        for iRow in range(len(self.calc_rodPosBox)):
            if(self.calc_rodPosBox[iRow][0]==None):
                continue
            for iColumn in range(len(self.calc_rodPosBox[0])):
                tmp = round(self.calc_rodPosBox[iRow][iColumn].value(), 2)
                tmp2 = round(self.calc_rodPos[iRow][iColumn], 2)
                if tmp != tmp2:
                    return True
        return False

    def getRodValues(self):
        value_array = []
        for iRow in range(len(self.calc_rodPosBox)):
            value_array.append([])
            for iColumn in range(len(self.calc_rodPosBox[0])):
                tmp = round(self.calc_rodPosBox[iRow][iColumn].value(), 2)
                value_array[-1].append(tmp)
        return value_array

    def returnButtonLayout(self):
        return self.tableButtonLayout

    def returnTableButton(self):
        return self.SD_tableWidget_button01, self.SD_tableWidget_button02, self.SD_tableWidget_button03