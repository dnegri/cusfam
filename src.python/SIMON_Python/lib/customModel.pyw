import os.path as path
import csv
from PyQt5 import QtWidgets, uic, QtCore, QtGui
from PyQt5.QtWidgets import QHBoxLayout
import data.database.PyRestartManagerWindow as restartManager
from PyQt5.QtGui import QIcon

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

class CsvWindow:
    def __init__(self):
        super().__init__()

        self.nCalcOpt = 2
        self.defaultSetLoc = 0
        self.defaultLoc = [ ]

        self.csvDataSet = []
        self.calcFlag = [ False for idx in range(self.nCalcOpt)]

        self.checkBoxGroup = []
        self.fileLocation = []

        self.tableHeight = []
        self.tableWidth  = []
        self.defaultFileSettingFlag = True
        self.allRestartFileFlag = True
        self.defaultFlag = False

    def __del__(self):
        del self.csvDataSet
        del self.calcFlag
        del self.checkBoxGroup
        del self.fileLocation
        del self.tableHeight
        del self.tableWidth

    def checkClicked(self):
        # print("okk")
        self.ui.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(True)
        self.ui.pushButtonChange.setEnabled(True)
        self.changeHeader()

    def clickAccepted(self):
        checkFlag = False
        if(self.checkBoxGroup[-1].isChecked()):
            idx = 0
            return True, self.fileLocation[self.defaultSetLoc]
        else:
            for idx in range(len(self.checkBoxGroup)-1,-1,-1):
                if(self.checkBoxGroup[idx].isChecked()):
                    checkFlag = True
                    break
        if(checkFlag==False):
            QtWidgets.QMessageBox.information(self, "Warning Message", "None of Restart File Was Selected\nPlease Select Restart File.")
            return False, ""
        else:
            inverseIdx = len(self.checkBoxGroup) -2 - idx
            return True, self.fileLocation[idx]

    def readCSV(self,fileName):
        file = open(fileName, 'r',encoding='utf-8')
        data = csv.reader(file)
        totalDataSet = []
        calcOpt = _CALC_NONE_
        for line in data:
            if(len(line) >0):
                totalDataSet.append(line)
                # print(line)
                # if(len(line[0])!=0):
                #     if(line[0]==_STR_MONITOR_):
                #         pass
                #         # calcOpt = _CALC_MONITOR_
                #     elif(line[0]==_STR_ECC_):
                #         pass
                #         # calcOpt = _CALC_ECC_
                #     elif(line[0]==_STR_SDM_):
                #         pass
                #         # calcOpt = _CALC_SDM_
                #     elif(line[0]==_STR_AO_CONTROL_):
                #         pass
                #         # calcOpt = _CALC_AO_CONTROL_
                #     elif(line[0]==_STR_LIFETIME_):
                #         pass
                #         # calcOpt = _CALC_LIFETIME_
                #     elif(line[0]==_STR_COASTDOWN_):
                #         pass
                #         # calcOpt = _CALC_COASTDOWN_
                # else:
                #     if(str(line[2])==_STR_FAIL_):
                #         pass
                #     elif(str(line[2])==_STR_SUCCESS_):
                #         # TODO SGH, MAKE SURE TO PREPARE ERROR STATUS
                #         if(calcOpt==_CALC_NONE_):
                #             pass
                #         else:
                #             self.storeParameter(calcOpt,line)

        # columnHeader = ["Calculation Option",
        #                 "Calc Index",
        #                 "Loading Opt",
        #                 "File Location & Name",
        #                 "P1",
        #                 "P2",
        #                 "P3",
        #                 "P4",
        #                 "Calculation Time"]
        # df2 = pd.DataFrame(totalDataSet,columns=columnHeader)
        # del df2['Calc Index']
        # for colNum, val in enumerate(df2.columns.values):
        #     print(colNum,val)


        # for idx in range(len(totalDataSet)-1):
        for idx in range(len(totalDataSet)-1):
            line = totalDataSet[idx]
            print (line)
            if(len(line[0])!=0):
                if(line[0]==_STR_MONITOR_):
                    calcOpt = _CALC_MONITOR_
                elif(line[0]==_STR_ECC_):
                    calcOpt = _CALC_ECC_
                elif(line[0]==_STR_SDM_):
                    calcOpt = _CALC_SDM_
                elif(line[0]==_STR_AO_CONTROL_):
                    calcOpt = _CALC_AO_CONTROL_
                elif(line[0]==_STR_LIFETIME_):
                    calcOpt = _CALC_LIFETIME_
                elif(line[0]==_STR_COASTDOWN_):
                    calcOpt = _CALC_COASTDOWN_

                if(calcOpt==_CALC_NONE_):
                    pass
                else:
                    self.storeParameter(calcOpt,line)


    def storeParameter(self,calcOpt, LineData):
        dataSet = []
        # Store Restart File Location
        self.fileLocation.append((LineData[3]))
        loc = path.dirname(LineData[3])
        name = path.basename(LineData[3])
        locationStartTime = str(LineData[-1]) #"TMP" #TODO YDNAM, insert Calculation Time in RESTART CSV FILE
        # opt = calcOpt
        bp = float(LineData[4])
        asi = float(LineData[5])
        fxy = float(LineData[6])
        rodPos = float(LineData[7])

        dataSet.append(calcOpt)
        dataSet.append(format(bp,".1f"))
        dataSet.append(format(asi,".3f"))
        dataSet.append(format(fxy,".3f"))
        dataSet.append(format(rodPos,".3f"))
        dataSet.append(locationStartTime)
        dataSet.append(name)
        dataSet.append(loc)

        if(LineData[-1]=="True"):
            self.defaultFlag = True
            self.defaultSetLoc = len(self.csvDataSet)
            self.calcFlag[_CALC_DEFAULT_] = True
            self.defaultLoc.append(self.defaultSetLoc)
            # self.csvDataSet[_CALC_DEFAULT_].insert(0, dataSet)

        self.csvDataSet.append(dataSet)

    def initRestartWindow(self):
        self.ui = restartManager.Ui_Dialog()
        if(len(self.csvDataSet)!=0):
            self.calcFlag[_CALC_ALL_]=True
        self.ui.initData(self.nCalcOpt, self.calcFlag)
        self.ui.setupUi(self)
        self.setWindowFlag(QtCore.Qt.WindowMinimizeButtonHint,True)
        self.setWindowFlag(QtCore.Qt.WindowMaximizeButtonHint,True)
        self.setWindowIcon(QIcon(":/knf/test01.ico"))
        self.setWindowTitle("ASTRA RESTART FILE MANAGER")

        for iOpt in range(self.nCalcOpt):
            if(self.calcFlag[iOpt]==True):
                pointerName = eval(_POINTER_LIST_[iOpt])
                pointerName.setColumnWidth(1,200)
                # pointerName.horizontalHeader().setSectionsMovable(True)
                pointerName.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)

        # self.ui.buttonBox.button(QtWidgets.QDialogButtonBox.Ok).setEnabled(False)
        # self.ui.pushButtonChange.setEnabled(False)
        self.ui.pushButtonChange.clicked.connect(self.test001)

    def storeData(self):
        self.sortData()
        self.initRestartWindow()
        self.storeCSV()
        if(self.defaultFlag==True):
            self.storeDefault()
        self.groupCheckButton()
        self.setWindowSize()
        self.settingInitial()

    def sortData(self):
        sortingDataSet = [[] for idx in range(6)]
        # print(sortingDataSet)
        # nRow = len(self.csvDataSet)
        # for idx in range(nRow):

        pass

    def storeCSV(self):
        nRow = len(self.csvDataSet)
        if(nRow==0):
            return
        pointerName = eval(_POINTER_A_)
        _translate = QtCore.QCoreApplication.translate
        pointerName.setRowCount(nRow)
        pointerName.setColumnCount(_NUM_PRINTOUT_)

        for restartIdx in range(nRow):
            item = QtWidgets.QTableWidgetItem()
            pointerName.setVerticalHeaderItem(restartIdx, item)
            # print(item.size())

        # Insert Restart Dataset From TableWidget
        # for restartIdx in range(nRow-1,-1,-1):
        for restartIdx in range(nRow):
            item = QtWidgets.QTableWidgetItem()
            font = QtGui.QFont()
            font.setPointSize(11)
            font.setBold(False)
            font.setWeight(75)
            item.setFont(font)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            calcOpt = self.csvDataSet[restartIdx][0]
            name = "%s" % (STRING_CALC_OPT[calcOpt])
            item.setText(_translate("Form", name))
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
            # inverseIdx = nRow - restartIdx - 1
            # pointerName.setItem(inverseIdx, 1, item)
            pointerName.setItem(restartIdx, 1, item)
            for columnIdx in range(1,_NUM_PRINTOUT_-1):
                item = QtWidgets.QTableWidgetItem()
                font = QtGui.QFont()
                font.setPointSize(11)
                font.setBold(False)
                font.setWeight(75)
                item.setFont(font)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                name = "%s" % (self.csvDataSet[restartIdx][columnIdx])
                item.setText(_translate("Form", name))
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                # inverseIdx = nRow - restartIdx - 1
                pointerName.setItem(restartIdx, columnIdx+1, item)

        # Make CheckBox inside TableWidget
        # for restartIdx in range(nRow-1,-1,-1):
        for restartIdx in range(nRow):
            checkBoxWidget = QtWidgets.QWidget()
            checkBox = QtWidgets.QCheckBox()
            checkBox.setGeometry(0,0,100,100)
            checkBox.setStyleSheet("QCheckBox::indicator"
                                   "{"
                                   "width : 16px;"
                                   "height : 16px;"
                                   "}")

            checkBox.setAutoExclusive(True)
            # Set "Ok" Button Enabled if One of Restart File Was Selected
            checkBox.clicked.connect(self.checkClicked)


            layout = QHBoxLayout(checkBoxWidget)
            layout.addWidget(checkBox)
            layout.setAlignment(QtCore.Qt.AlignCenter)
            layout.setContentsMargins(0,0,0,0)

            item = QtWidgets.QTableWidgetItem()
            item.setTextAlignment(QtCore.Qt.AlignCenter)

            pointerName.setCellWidget(restartIdx,0,checkBoxWidget)
            self.checkBoxGroup.append(checkBox)

        # Resize TableWidget
        pointerName.resizeColumnsToContents()
        pointerName.horizontalHeader().setStretchLastSection(True)

        # pointerName.horizontalHeader().setMinimumHeight(200)
        pointerName.verticalHeader().setStretchLastSection(True)
        pointerName.setColumnWidth(5, 160)
        for i in range(_NUM_PRINTOUT_):
            if(i==5):
                continue
            pointerName.horizontalHeader().setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)
        #self.setTableWidth(pointerName)

    def storeDefault(self):
        nRow = len(self.defaultLoc)
        if(self.calcFlag[_CALC_DEFAULT_]==False):
            return
        pointerName = eval(_POINTER_D_)
        _translate = QtCore.QCoreApplication.translate
        pointerName.setRowCount(nRow)
        pointerName.setColumnCount(_NUM_PRINTOUT_)

        for restartIdx in range(nRow):
            item = QtWidgets.QTableWidgetItem()
            pointerName.setVerticalHeaderItem(restartIdx, item)
            # print(item.size())

        # Insert Restart Dataset From TableWidget
        for restartIdx in range(nRow):
            defaultIdx = self.defaultLoc[restartIdx]
            item = QtWidgets.QTableWidgetItem()
            font = QtGui.QFont()
            font.setPointSize(11)
            font.setBold(False)
            font.setWeight(75)
            item.setFont(font)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            calcOpt = self.csvDataSet[defaultIdx][0]
            name = "%s" % (STRING_CALC_OPT[calcOpt])
            item.setText(_translate("Form", name))
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
            pointerName.setItem(restartIdx, 1, item)
            for columnIdx in range(1,_NUM_PRINTOUT_-1):
                defaultIdx = self.defaultLoc[restartIdx]
                item = QtWidgets.QTableWidgetItem()
                font = QtGui.QFont()
                font.setPointSize(11)
                font.setBold(False)
                font.setWeight(75)
                item.setFont(font)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                name = "%s" % (self.csvDataSet[defaultIdx][columnIdx])
                item.setText(_translate("Form", name))
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                pointerName.setItem(restartIdx, columnIdx+1, item)

        # Make CheckBox inside TableWidget
        for restartIdx in range(nRow):
            checkBoxWidget = QtWidgets.QWidget()
            checkBox = QtWidgets.QCheckBox()
            checkBox.setGeometry(0,0,100,100)
            checkBox.setStyleSheet("QCheckBox::indicator"
                                   "{"
                                   "width : 16px;"
                                   "height : 16px;"
                                   "}")

            checkBox.setAutoExclusive(True)
            # Set "Ok" Button Enabled if One of Restart File Was Selected
            checkBox.clicked.connect(self.checkClicked)


            layout = QHBoxLayout(checkBoxWidget)
            layout.addWidget(checkBox)
            layout.setAlignment(QtCore.Qt.AlignCenter)
            layout.setContentsMargins(0,0,0,0)

            item = QtWidgets.QTableWidgetItem()
            item.setTextAlignment(QtCore.Qt.AlignCenter)

            pointerName.setCellWidget(restartIdx,0,checkBoxWidget)
            self.checkBoxGroup.append(checkBox)

        # Resize TableWidget
        pointerName.resizeColumnsToContents()
        pointerName.horizontalHeader().setStretchLastSection(True)
        pointerName.verticalHeader().setStretchLastSection(True)
        pointerName.setColumnWidth(5, 160)
        for i in range(_NUM_PRINTOUT_):
            if(i==5):
                continue
            pointerName.horizontalHeader().setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)
        #self.setTableWidth(pointerName)

    def groupCheckButton(self):
        # Make CheckBox Group
        self.groupButtonBox = QtWidgets.QButtonGroup(self)
        for idx in range(len(self.checkBoxGroup)):
            self.groupButtonBox.addButton(self.checkBoxGroup[idx])



    def setTableWidth(self,pointerName):
        width = pointerName.verticalHeader().width()
        width+= pointerName.horizontalHeader().length()
        if(pointerName.verticalScrollBar().isVisible()):
            width += pointerName.verticalScrollBar().width()
        width += pointerName.frameWidth() * 2

        height = 0
        height += pointerName.verticalHeader().length() + 30
        # height+= pointerName.horizontalHeader().length()
        if(pointerName.horizontalScrollBar().isVisible()):
            height += pointerName.horizontalScrollBar().length()
        height += pointerName.frameWidth() * 2
        # pointerName.setFixedWidth(width)
        pointerName.setFixedHeight(height)

        width  = pointerName.verticalHeader().length()
        width += pointerName.frameWidth() * 2
        width += pointerName.horizontalHeader().length()

        height = pointerName.frameWidth() * 2
        height+= pointerName.verticalHeader().width()
        height += pointerName.horizontalHeader().height()

        self.tableWidth.append(width)
        self.tableHeight.append(height)

    def setWindowSize(self):
        windowWidth = max(self.tableWidth) + 50
        # print(windowWidth)
        windowHeight = 0
        for i in range(len(self.tableHeight)):
            windowHeight += self.tableHeight[i] + 79

        currentHeight = self.height()
        if(windowHeight > currentHeight):
            windowHeight = currentHeight
        self.resize(windowWidth,windowHeight)



    def test001(self):
        if(self.checkBoxGroup[-1].isChecked()):
            return
        else:
            for idx in range(len(self.checkBoxGroup)-1,-1,-1):
                if(self.checkBoxGroup[idx].isChecked()):
                    break

        nRow = len(self.defaultLoc)
        if(self.calcFlag[_CALC_DEFAULT_]==False):
            return
        pointerName = eval(_POINTER_D_)
        _translate = QtCore.QCoreApplication.translate
        pointerName.setRowCount(nRow)
        pointerName.setColumnCount(_NUM_PRINTOUT_)
        self.defaultSetLoc = idx
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setPointSize(11)
        font.setBold(False)
        font.setWeight(75)
        item.setFont(font)
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        calcOpt = self.csvDataSet[idx][0]
        name = "%s" % (STRING_CALC_OPT[calcOpt])
        item.setText(_translate("Form", name))
        item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
        pointerName.setItem(0, 1, item)
        for columnIdx in range(1,_NUM_PRINTOUT_-1):
            item = QtWidgets.QTableWidgetItem()
            font = QtGui.QFont()
            font.setPointSize(11)
            font.setBold(False)
            font.setWeight(75)
            item.setFont(font)
            item.setTextAlignment(QtCore.Qt.AlignCenter)
            name = "%s" % (self.csvDataSet[idx][columnIdx])
            item.setText(_translate("Form", name))
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
            pointerName.setItem(0, columnIdx+1, item)
        self.changeHeaderDefault()

    def settingInitial(self):
        self.checkBoxGroup[0].setChecked(True)
        self.changeHeader()
        if(self.defaultFlag==True):
            self.changeHeaderDefault()

    def changeHeader(self):
        [defaultFlag, allDataFlag] = self.ui.returnExcelFlag()
        for idx in range(len(self.checkBoxGroup) - 1, -1, -1):
            if (self.checkBoxGroup[idx].isChecked()):
                break
        if (self.csvDataSet[idx][0] == _CALC_MONITOR_):
            self.ui.tableWidgetAll.horizontalHeaderItem(5).setText("Rod Position")
        elif (self.csvDataSet[idx][0] == _CALC_ECC_):
            self.ui.tableWidgetAll.horizontalHeaderItem(5).setText(_STR_ECC_)
        elif (self.csvDataSet[idx][0] == _CALC_SDM_):
            self.ui.tableWidgetAll.horizontalHeaderItem(5).setText(_STR_SDM_)
        elif (self.csvDataSet[idx][0] == _CALC_AO_CONTROL_):
            self.ui.tableWidgetAll.horizontalHeaderItem(5).setText(_STR_AO_CONTROL_)
        elif (self.csvDataSet[idx][0] == _CALC_LIFETIME_):
            self.ui.tableWidgetAll.horizontalHeaderItem(5).setText(_STR_LIFETIME_)
        elif (self.csvDataSet[idx][0] == _CALC_COASTDOWN_):
            self.ui.tableWidgetAll.horizontalHeaderItem(5).setText(_STR_COASTDOWN_)

    def changeHeaderDefault(self):
        [ defaultFlag , allDataFlag ] = self.ui.returnExcelFlag()
        for idx in range(len(self.checkBoxGroup) - 1, -1, -1):
            if (self.checkBoxGroup[idx].isChecked()):
                break
        if(self.csvDataSet[idx][0]==_CALC_MONITOR_):
            self.ui.tableWidgetDefault.horizontalHeaderItem(5).setText("Rod Position")
        elif(self.csvDataSet[idx][0]==_CALC_ECC_):
            self.ui.tableWidgetDefault.horizontalHeaderItem(5).setText(_STR_ECC_)
        elif(self.csvDataSet[idx][0]==_CALC_SDM_):
            self.ui.tableWidgetDefault.horizontalHeaderItem(5).setText(_STR_SDM_)
        elif(self.csvDataSet[idx][0]==_CALC_AO_CONTROL_):
            self.ui.tableWidgetDefault.horizontalHeaderItem(5).setText(_STR_AO_CONTROL_)
        elif(self.csvDataSet[idx][0]==_CALC_LIFETIME_):
            self.ui.tableWidgetDefault.horizontalHeaderItem(5).setText(_STR_LIFETIME_)
        elif(self.csvDataSet[idx][0]==_CALC_COASTDOWN_):
            self.ui.tableWidgetDefault.horizontalHeaderItem(5).setText(_STR_COASTDOWN_)

