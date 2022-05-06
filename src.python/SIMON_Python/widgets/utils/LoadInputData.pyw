#import Definitions as df
import os
import csv
#from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon

import datetime
import widgets.utils.PySnapshotTableWidget as table01
from widgets.utils.PySnapshotWindow import SnapshotPopupWidget
import glob
from model import *
import utils as ut
import constants as cs
from operator import itemgetter

Sentence_END = "0 FPA FILE CATALOGED ON SYSTEM.  FILE="
Sentence_Start = "GEOMETRY EDIT"
Sentence_Rod_Pos = "COMPUTED ROD BANK POSITIONS AND DROPPED/EJECTED ROD SUMMARY"
Sentence_Information = "EDIT OF SNAPSHOT INFORMATION"
Sentence_Burnup = "= CASE"
Sentence_DateTime = "DATE AND TIME:"

Sentence_core_power = "CORE POWER AS CALCULATED BY CECOR"
Sentence_hfp = "NAME PLATE POWER AT 100 PCT. POWER"

Block_Rod05 = "ROD BANK NUMBER  1"
Block_Rod04 = "ROD BANK NUMBER  2"
Block_Rod03 = "ROD BANK NUMBER  3"
Block_Rod02 = "ROD BANK NUMBER  4"
Block_Rod01 = "ROD BANK NUMBER  5"
Block_Rod_P = "ROD BANK NUMBER  6"
Block_Rod_B = "ROD BANK NUMBER  7"
Block_Rod_A = "ROD BANK NUMBER  8"

key_rod5 = "R5"
key_rod4 = "R4"
key_rod3 = "R3"
key_rod2 = "R2"
key_rod1 = "R1"
key_rodP = "P_"
key_rodB = "B_"
key_rodA = "A_"

key_snapshot_name = "SNAPSHOT"
key_day = "EFPH"
key_brunup = "MWD/MTU"
CECOR_ROD_LENGTH = 381.0


class readSnapshotData:

    def __init__(self):

        self.inputArray = []
        self.rodPos = []
        self.burnup = []
        self.power = []
        self.operationPower = []
        self.hfp = []
        #self.unit_dictionary = {}
        self.nStep = 0
        self.currentStep = -1
        self.rodPosFlag = False
        self.burnupFlag = False
        self.powerFlag = False
        self.operationPowerFlag = False

        self.dateTime = ""

        self.cecor_read_actions = {

            #Sentence_Rod_Pos:self.get_current_rod,
            Sentence_Start:self.get_current_step,
            Sentence_Burnup:self.get_current_job_burnup,
            Sentence_core_power: self.get_current_core_power,
            Sentence_hfp: self.get_current_operation_power,
            Sentence_DateTime: self.get_current_date_time,


            Block_Rod05: self.get_current_rod,
            Block_Rod04: self.get_current_rod,
            Block_Rod03: self.get_current_rod,
            Block_Rod02: self.get_current_rod,
            Block_Rod01: self.get_current_rod,
            Block_Rod_P: self.get_current_rod,
            Block_Rod_B: self.get_current_rod,
            Block_Rod_A: self.get_current_rod,
            #
            # block_job_name: get_job_name,
            # block_snapshot_name: get_current_job_burnup,
            # block_rod3: get_current_rod,
            # block_rod4: get_current_rod,
            # block_rod5: get_current_rod,
            # block_rod6: get_current_rod,

            # block_hfp: get_current_hfp,
        }

    def get_current_date_time(self,lines, l_i, storage):
        line2 = lines[l_i+1].split()
        date = line2[0].split("/")
        tmpDate = "%s-%s-%s" %(date[2],date[0],date[1])
        self.date = tmpDate
        self.time = line2[1]

    def get_current_operation_power(self,lines, l_i, storage):
        if(self.operationPowerFlag == True):
            return
        else:
            self.operationPowerFlag = True
        line2 = lines[l_i]
        split_line2 = line2.split()

        self.operationPower.append(float(split_line2[-2]))

    def get_current_core_power(self,lines, l_i, storage):
        if(self.powerFlag == True):
            return
        else:
            self.powerFlag = True
        line2 = lines[l_i]
        split_line2 = line2.split()

        self.power.append(float(split_line2[-2]))

        # print("core power", float(split_line2[-2])/storage[storage[store_current_snapshot]+","+store_current_hfp])

        # storage[storage[store_current_snapshot]+","+store_current_power] = float(split_line2[-2])/storage[storage[store_current_snapshot]+","+store_current_hfp]


    def get_current_step(self,lines,l_i,storage):
        unit_dictionary = {
            key_rod5:0.0,
            key_rod4:0.0,
            key_rod3:0.0,
            key_rod2:0.0,
            key_rod1:0.0,
            key_rodP:0.0,
            key_rodB:0.0,
            key_rodA:0.0,
            # key_day:0.0,
            # key_brunup:0.0,
        }
        self.rodPosFlag = False
        self.burnupFlag = False
        self.powerFlag = False
        self.operationPowerFlag = False
        self.rodPos.append(unit_dictionary)
        self.currentStep += 1

        pass

    def get_current_job_burnup(self,lines, l_i, storage):
        if(self.burnupFlag == True):
            return
        else:
            self.burnupFlag = True
        "return element after snapeshot"
        unit_dictionary = {}
        tmp_EFPH = 0.0
        line = lines[l_i]
        split_line = line.split()
        r_strings = []
        for w_i, word in enumerate(split_line):
            # if key_snapshot_name in word:
            #     r_strings.append(split_line[w_i + 1])
            if key_brunup in word:
                try:
                    tmp_burnup = float(split_line[w_i -1])
                    tmp_day = float(split_line[w_i + 1][1:])
                    #r_strings.append(split_line[w_i + 1][1:])
                except:
                    pass
                    #int(split_line[w_i + 2])
                    #r_strings.append(split_line[w_i + 2])

        unit_dictionary[key_brunup] = tmp_burnup
        unit_dictionary[key_day] = tmp_day
        self.burnup.append(unit_dictionary)
        # print(tmp_EFPH)
        #print(r_strings)
        # storage[store_current_snapshot] = r_strings[0]
        # storage[store_snapshot_identifier + r_strings[0]] = r_strings[0]
        # storage[storage[store_current_snapshot]+","+store_current_burnup] = r_strings[1]


    def get_current_rod(self,lines, l_i, storage):
        # if(self.rodPosFlag == True):
        #     return
        # else:
        #     self.rodPosFlag = True

        unitRodPos = []
        tmp05 = 0.0
        tmp04 = 0.0
        tmp03 = 0.0
        tmp02 = 0.0
        tmp01 = 0.0
        tmp_P = 0.0
        tmp_B = 0.0
        tmp_A = 0.0
        "return element after snapeshot"
        line = lines[l_i]
        split_line = line.split()
        r_strings = [split_line[9],]
        #return r_strings
        if Block_Rod05 in line:
            self.rodPos[self.currentStep][key_rod5] = split_line[9]
        if Block_Rod04 in line:
            self.rodPos[self.currentStep][key_rod4] = split_line[9]
        if Block_Rod03 in line:
            self.rodPos[self.currentStep][key_rod3] = split_line[9]
        if Block_Rod02 in line:
            self.rodPos[self.currentStep][key_rod2] = split_line[9]
        if Block_Rod01 in line:
            self.rodPos[self.currentStep][key_rod1] = split_line[9]
        if Block_Rod_P in line:
            self.rodPos[self.currentStep][key_rodP] = split_line[9]
        if Block_Rod_B in line:
            self.rodPos[self.currentStep][key_rodB] = split_line[9]
        if Block_Rod_A in line:
            self.rodPos[self.currentStep][key_rodA] = split_line[9]

        # unitRodPos = [ tmp05, tmp04, tmp03, tmp02, tmp01, tmp_P, tmp_B, tmp_A ]
        # self.rodPos.append(unitRodPos)
        return r_strings




    # def get_current_rod(self,lines, l_i, storage):
    #     unitRodPos = []
    #     "return element after snapeshot"
    #     line = lines[l_i]
    #     split_line = line.split()
    #     r_strings = [split_line[9],]
    #     if Block_Rod05 in line:
    #         tmp05 = split_line[9]
    #         #storage[storage[store_current_snapshot]+","+key_rod3] = split_line[9]
    #         #storage[key_rod3] = split_line[9]
    #     if Block_Rod04 in line:
    #         tmp04 = split_line[9]
    #         storage[storage[store_current_snapshot]+","+key_rod4] = split_line[9]
    #         storage[key_rod4] = split_line[9]
    #     if Block_Rod03 in line:
    #         tmp03 = split_line[9]
    #         storage[storage[store_current_snapshot]+","+key_rod5] = split_line[9]
    #         storage[key_rod5] = split_line[9]
    #     if Block_Rod02 in line:
    #         tmp02 = split_line[9]
    #         storage[storage[store_current_snapshot]+","+key_rodP] = split_line[9]
    #         storage[key_rodP] = split_line[9]
    #     if Block_Rod01 in line:
    #         tmp01 = split_line[9]
    #         storage[storage[store_current_snapshot]+","+key_rodP] = split_line[9]
    #         storage[key_rodP] = split_line[9]
    #
    #     if Block_Rod_P in line:
    #         tmp_P = split_line[9]
    #         storage[storage[store_current_snapshot]+","+key_rodP] = split_line[9]
    #         storage[key_rodP] = split_line[9]
    #
    #     if Block_Rod_B in line:
    #         tmp_B = split_line[9]
    #         storage[storage[store_current_snapshot]+","+key_rodP] = split_line[9]
    #         storage[key_rodP] = split_line[9]
    #
    #     if Block_Rod_A in line:
    #         tmp_A = split_line[9]
    #         storage[storage[store_current_snapshot]+","+key_rodP] = split_line[9]
    #         storage[key_rodP] = split_line[9]
    #
    #     return r_strings

    def read_cecor_file(self, user=None, plantname="", cyclename=""):

        # file_path = "C:/Users/geonho/Desktop/02_cecor_y310_corefollow.out"
        # try:
        file_path = ""
        if user:
            file_paths = glob.glob("{}/*".format(user.cecor_directory))
            is_exist = False

            if len(file_paths) == 0:
                return ("Cecor output file not found".format(file_path), [])

            for file_path in file_paths:

                _, file_name = os.path.split(file_path)
                if ".out" not in file_name.lower():
                    return ("{} should contain .out extension".format(file_path), [])

                is_found = False
                for key in cs.DEFINED_PLANTS.keys():
                    if key.lower() in file_name:
                        is_found = True

                if not is_found:
                    return ("{} plant name not defined".format(file_path), [])

                if (plantname+cyclename).lower() in file_path.lower():
                    is_exist = True

            if not is_exist:
                return ("{} plant name not defined".format(file_path), [])

            # file_paths = glob.glob("{}/*.out".format(user.cecor_directory))
            # file_paths =
            # for file in output_path_list:
            #     file_path = output_path_list[0]
            # print("output_name", file_path[0])
        else:
            file_paths = ["C:/simon/cecor_files/02_cecor_y310_corefollow.out",]
        # try:
        inputArray = []
        # print(len(file_paths))
        for file_path in file_paths:
            if (plantname+cyclename).lower() in file_path.lower():
                self.clearFile()
                outputs = ut.get_cecore_output(file_path)
                counter = 0
                output_file = None
                for output in outputs:
                    counter += 1
                    output_file = output

                if counter == 0:
                    # print("cecor", len(output))
                    file1 = open(file_path, 'r')
                    lines = file1.readlines()
                    storage = {}
                    for l_i, line in enumerate(lines):
                        for k_i, key in enumerate(self.cecor_read_actions):
                            if key in line:
                                self.cecor_read_actions[key](lines, l_i, storage)
                                #print(key, self.cecor_read_actions[key](lines, l_i, storage) )

                    output_file = Cecor_Output.create(filename=file_path, modified_date=datetime.datetime.now())

                if len(inputArray) == 0:
                    inputArray = self.getSnapshotDataset(output_file)
                else:
                    inputArray = inputArray + self.getSnapshotDataset(output_file)
            # print("inputarray", len(inputArray))

        sorted(inputArray, key=itemgetter(1))

        duplicates = {}
        removed_array = []
        for subs in inputArray:
            burnup_key = "{:.1f}".format(subs[1])
            if burnup_key not in duplicates.keys():
                duplicates[burnup_key] = 0
                removed_array.append(subs)
            else:
                duplicates[burnup_key] += 1

        self.inputArray = removed_array

        return ("", removed_array)
        # except:
        #     return ("Error reading the following file: {} ".format(file_path), [])
        # except:
        #     pass

    def getSnapshotDataset(self, cecor_output=None):

        # self.clearFile()

        if len(cecor_output.table) > 0:

            read_table = cecor_output.table
            snapshotArrayStrings = read_table.split(",")
            length_array = int(snapshotArrayStrings[0])

            snapshotArray = []
            col_length = 8
            for row in range(length_array):
                snapshotArray.append([])
                for col in range(col_length):
                    if col == 0 or col == 7:
                        snapshotArray[-1].append(snapshotArrayStrings[1+row*col_length+col])
                    else:
                        snapshotArray[-1].append(float(snapshotArrayStrings[1+row*col_length+col]))
            return snapshotArray

        else:
            dataArray = []
            unitArray = []
            nStep = len(self.rodPos)

            for idx in range(nStep):
                unitArray.append("CECOR")
                #unitArray.append(self.burnup[idx][key_day])
                unitArray.append(self.burnup[idx][key_brunup])
                unitArray.append(float(self.power[idx]/self.operationPower[idx])*100.0)
                tmp05 = (100.0-float(self.rodPos[idx][key_rod5]))/100.0*CECOR_ROD_LENGTH
                tmp04 = (100.0-float(self.rodPos[idx][key_rod4]))/100.0*CECOR_ROD_LENGTH
                tmp03 = (100.0-float(self.rodPos[idx][key_rod3]))/100.0*CECOR_ROD_LENGTH
                tmpP  = (100.0-float(self.rodPos[idx][key_rodP]))/100.0*CECOR_ROD_LENGTH

                unitArray.append(tmp05)
                unitArray.append(tmp04)
                unitArray.append(tmp03)
                unitArray.append(tmpP)
                tmp = "%s %s" %(self.date,self.time)
                unitArray.append(tmp)
                dataArray.append(unitArray)
                unitArray = []

            storage_text = "{:d},".format(nStep)
            for values in dataArray:
                for i_s, value in enumerate(values):
                    if i_s == 0 or i_s == 7:
                        storage_text += "{},".format(value)
                    else:
                        storage_text += "{:.3f},".format(value)

            cecor_output.table = storage_text
            cecor_output.save()

        return dataArray

    # def getSnapshotDataset(self):
    #     dataArray = []
    #     unitArray = []
    #     nStep = len(self.rodPos)
    #
    #
    #     for idx in range(nStep):
    #         unitArray.append("Snapshot")
    #         #unitArray.append(self.burnup[idx][key_day])
    #         unitArray.append(self.burnup[idx][key_brunup])
    #         unitArray.append(float(self.power[idx]/self.operationPower[idx])*100.0)
    #         tmp05 = (100.0-float(self.rodPos[idx][key_rod5]))/100.0*CECOR_ROD_LENGTH
    #         tmp04 = (100.0-float(self.rodPos[idx][key_rod4]))/100.0*CECOR_ROD_LENGTH
    #         tmp03 = (100.0-float(self.rodPos[idx][key_rod3]))/100.0*CECOR_ROD_LENGTH
    #         tmpP  = (100.0-float(self.rodPos[idx][key_rodP]))/100.0*CECOR_ROD_LENGTH
    #
    #         unitArray.append(tmp05)
    #         unitArray.append(tmp04)
    #         unitArray.append(tmp03)
    #         unitArray.append(tmpP)
    #         tmp = "%s %s" %(self.date,self.time)
    #         unitArray.append(tmp)
    #         dataArray.append(unitArray)
    #         unitArray = []
    #
    #     return dataArray

    def getSnapshotCalcArray(self,idx):
        dataArray = []
        unitArray = []

        if len(self.inputArray) > 0:

            if len(self.inputArray) > idx + 1:
                for idx in range(idx + 1):
                    # unitArray.append("Snapshot")
                    unitArray.append(0.0)
                    unitArray.append(self.inputArray[idx][1])
                    unitArray.append(self.inputArray[idx][2])

                    unitArray.append(1.0)
                    unitArray.append(0.0)
                    unitArray.append(0.0)
                    unitArray.append(self.inputArray[idx][3])
                    unitArray.append(self.inputArray[idx][4])
                    unitArray.append(self.inputArray[idx][5])
                    unitArray.append(self.inputArray[idx][6])
                    # tmp = "%s %s" %(self.date,self.time)
                    # unitArray.append(tmp)
                    dataArray.append(unitArray)
                    unitArray = []

        else:

            if len(self.burnup) > idx + 1:
                for idx in range(idx+1):
                    #unitArray.append("Snapshot")
                    unitArray.append(self.burnup[idx][key_day])
                    unitArray.append(self.burnup[idx][key_brunup])
                    unitArray.append(float(self.power[idx]/self.operationPower[idx])*100.0)
                    tmp05 = (100.0-float(self.rodPos[idx][key_rod5]))/100.0*CECOR_ROD_LENGTH
                    tmp04 = (100.0-float(self.rodPos[idx][key_rod4]))/100.0*CECOR_ROD_LENGTH
                    tmp03 = (100.0-float(self.rodPos[idx][key_rod3]))/100.0*CECOR_ROD_LENGTH
                    tmpP  = (100.0-float(self.rodPos[idx][key_rodP]))/100.0*CECOR_ROD_LENGTH

                    unitArray.append(1.0)
                    unitArray.append(0.0)
                    unitArray.append(0.0)
                    unitArray.append(tmp05)
                    unitArray.append(tmp04)
                    unitArray.append(tmp03)
                    unitArray.append(tmpP)
                    #tmp = "%s %s" %(self.date,self.time)
                    #unitArray.append(tmp)
                    dataArray.append(unitArray)
                    unitArray = []

        return dataArray


    def clearFile(self):

        self.rodPos = []
        self.burnup = []
        self.power = []
        self.operationPower = []
        self.hfp = []

        self.nStep = 0
        self.currentStep = -1














class LoadInputData():
    def __init__(self,ui):
        self.ui = ui
        self.opt = True
        self.fileName = ""
        self.inputArray = []
        self.calcArray = []
        self.calcOpt = 0
        self.nArray = 0
        self.targetValue = 0

        self.status = 0
        self.widget = QWidget()
        self.widget.setWindowIcon(QIcon("test01.ico"))

        self.dataReadFunction = readSnapshotData()
        self.snapshotPopup = SnapshotPopupWidget(self.ui)

        # self.dataReadFunction.read_cecor_file()
        # self.inputArray = self.dataReadFunction.getSnapshotDataset()
        #
        # self.snapshotPopup = SnapshotPopupWidget(ui)
        # self.snapshotPopup.updatePopup(self.inputArray)

    def readSnapshotData(self):
        self.snapshotPopup.show()
        ret = self.snapshotPopup.exec_()
        if(ret==True):
            #print("Good")
            #self.nArray, self.inputArray = self.snapshotPopup.getSnapshotDataset()
            self.nArray = self.snapshotPopup.getSnapshotDataset()
            self.calcArray = self.dataReadFunction.getSnapshotCalcArray(self.nArray)
            # print(self.nArray)
            # print(self.inputArray)
        #else:
            #print("Bad")

        pass

    def openCSV(self):
        self.fileName, _ = QFileDialog.getOpenFileName(self.widget,
                                                       "Open Model Setup File",
                                                       ".",
                                                       "CSV File (*.csv )\nExcel File (*.xlsx)")
        if self.fileName =="":
            #print("Exit")
            return False
        else:
            basename = os.path.basename(self.fileName)
            # QMessageBox.information(self, df._PROGRAM_NAME_,
            #                         "Read Loading Pattern Information File\n" + basename)
            fs = open(basename,'r',newline="")
            rdr = csv.reader(fs)
            csvData = []
            for line in rdr:
                # if (len(line) != 0):
                #     print(line)
                csvData.append(line)

            fs.close()

            self.readCSVData(csvData)
            return True


    def readCSVData(self,csvData):
        firstLine = csvData[0]
        #print(firstLine)
        self.nArray = int(firstLine[1])
        self.targetValue= float(csvData[1][0])

        # if(self.nArray+2 != len(csvData)):
        #     print("Wrong Step Number")
        # else:
        #     print("Good!")

        inputArray = []

        #TODO SGH, Make

        for idx in range(2,self.nArray+2):
            unitArray = []
            unitArray.append(float(csvData[idx][1]))
            unitArray.append(float(csvData[idx][2]))
            unitArray.append(float(csvData[idx][3]))
            unitArray.append(float(csvData[idx][4]))
            inputArray.append(unitArray)

        #print(inputArray)
        self.inputArray = inputArray

    def returnCSV(self):
        return self.nArray, self.targetValue, self.inputArray

    def returnSnapshot(self):
        return self.nArray+1, self.calcArray

    def getSnapshotData(self):
        return self.inputArray

    def updateSnapshotData(self, inputArray):
        self.inputArray = inputArray

    def read_cecor_output(self, user, plantname, cyclename, ):
        # outputs = ut.get_cecore_output()
        error_message, self.inputArray = self.dataReadFunction.read_cecor_file(user, plantname, cyclename,)
        if len(error_message) > 0:
            return error_message
        self.snapshotPopup.updatePopup(self.inputArray)

        return error_message