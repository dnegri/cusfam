
from PyQt5.QtWidgets import QWidget, QVBoxLayout

import matplotlib as mpl
import math
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, \
    NavigationToolbar2QT as NavigationToolbar
import matplotlib.gridspec as gs
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

class trendWidget(QWidget):
    def __init__(self,fixedTime_flag,asi_flag, cbc_flag, power_flag):
        super().__init__()

        self.fixedTime_FLAG = fixedTime_flag
        self.ASI_FLAG = asi_flag
        self.CBC_FLAG = cbc_flag
        self.Power_FLAG = power_flag

        # 01. Define layout for widget
        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.verticalLayout)
        #plt.autoscale(tight=True)

        # 02. Define Graph Color Paramater
        self.graphBackgroundColor = '#272c36'
        self.graphColor = '#d2d2d2'
        self.graphLine  = '#1b1d23'
        self.gridColor  = '#cccccc'

        self.color_Rod_P = "#FFFF66"
        self.color_Rod05 = "#66FF66"
        self.color_Rod04 = "#66FFFF"
        self.color_Rod03 = "#6666FF"
        self.color_ASI   = "#FF66FF"
        self.color_CBC   = "#FF6666"
        self.color_PWR   = "#990099"
        self.color_ASI_BAND = "#CC0000"

        #self.color_Rod05 = "#66FF66"
        #self.color_Rod05 = "#66FF66"
        self.orange = '#FFA000'
        self.green  = "#BCFD60"
        self.white  = "#FFFFFF"

        self.barYAxisTickPivot = 400.0
        self.min_Time =  0.0
        self.max_Time = 35.0
        self.addTime = 0.0
        self.rdcPerHour = 3.0
        self.power_inc_flag = True


        # Set subplot adjust values for resize graph chart
        # If ASI and CBC value is activated, matplotlib use default size
        self.subplot_adjust_left   = 0.05
        self.subplot_adjust_right  = 0.93
        self.subplot_adjust_bottom = 0.089
        self.subplot_adjust_top    = 0.92
        self.subplot_adjust_wspace = 0.0
        self.subplot_adjust_hspace = 0.0
        if(self.ASI_FLAG==False):
            self.subplot_adjust_right += 0.035
        if(self.CBC_FLAG==False):
            self.subplot_adjust_right += 0.035
        if(self.ASI_FLAG==False or self.CBC_FLAG==False):
            self.subplot_adjust_left -= 0.012

        # Set Graph zorder for define drawing order for avoid matplotlib bug issue
        # Known matplotlib bug issue: graph pick_evend didn't activate when drawing sequence is faster then other
        # If ASI is defined, drawing sequence: graph02 -> graph03 -> graph01
        # If not, drawing sequence: graph03 -> graph01
        self.graph01_zorder = 10
        self.graph02_zorder = 1
        self.graph03_zorder = 5

        # 03. Define Graph Figure and contents margin
        self.fig, self.graph01 = plt.subplots(facecolor="None")
        self.fig.subplots_adjust(left=self.subplot_adjust_left,
                                 right=self.subplot_adjust_right,
                                 bottom=self.subplot_adjust_bottom,
                                 top=self.subplot_adjust_top,
                                 wspace=self.subplot_adjust_wspace,
                                 hspace=self.subplot_adjust_hspace)

        # 04-A. Define Main Graph for Rod Position plotting
        if(self.Power_FLAG==True):
            self.graph01_AxisY_Label = "Rod Position(cm), Power(%)"
        else:
            self.graph01_AxisY_Label = "Rod Position(cm)"
        self.defineRodPosGraph()
        # 04-B. Define SubGraphs for plotting ASI, CBC vaule
        # self.graph02 = ASI graph
        # self.graph03 = CBC graph
        if(self.ASI_FLAG==True):
            self.labelPadSize_graph03 = 10.0
            self.graph03_StrFormatter = '\n\n  %2d'
            self.graph02 = self.graph01.twinx()
            self.defineASI_Graph()
        else:
            self.labelPadSize_graph03 = 0.0
            self.graph03_StrFormatter = '%2d'
        if(self.CBC_FLAG==True):
            self.graph03 = self.graph01.twinx()
            self.defineCBC_Graph()




        # 05. make Graph Initial Format
        self.makePlotFormat()

        #self.insertDataSet()
        # 06.
        self.canvas = FigureCanvas(self.fig)

        self.verticalLayout.addWidget(self.canvas)
        self.fig.canvas.mpl_connect('pick_event', self.clickEvent)
        #self.graph01.set_zorder(self.graph02.get_zorder()+10)
        self.graph01.set_zorder(self.graph01_zorder)
        if(self.ASI_FLAG==True):
            self.graph02.set_zorder(self.graph02_zorder)
        if(self.CBC_FLAG==True):
            self.graph03.set_zorder(self.graph03_zorder)
        #self.graph03.set_zorder(self.graph02.get_zorder()+5)
        #self.graph03.set_zorder(self.graph03_zorder)
        # self.rax.set_zorder(self.graph02.get_zorder()+20)
        # self.check.on_clicked(self.checkFunc)

        self.canvas.draw()

        #self.insertDataSet()

    def defineRodPosGraph(self):
        self.graph01.tick_params(axis='x', which='major', colors=self.gridColor)
        self.graph01.tick_params(axis='y', which='major', colors=self.gridColor)
        self.graph01.tick_params(axis='y', which='minor', colors="None")
        self.graph01.set_facecolor("None")#self.graphBackgroundColor)
        self.graph01.spines['top'].set_color(self.gridColor)
        self.graph01.spines['left'].set_color(self.gridColor)
        self.graph01.spines['right'].set_color(self.gridColor)
        self.graph01.spines['bottom'].set_color(self.gridColor)
        self.graph01.spines['top'].set_linewidth(1.0)
        self.graph01.spines['left'].set_linewidth(1.0)
        self.graph01.spines['right'].set_linewidth(1.0)
        self.graph01.spines['bottom'].set_linewidth(1.0)

        self.graph01.yaxis.set_major_locator(ticker.FixedLocator([0, 100, 200, 300, 400]))#, 231, 281, 331, 381]))
        #self.graph01.yaxis.set_major_locator(ticker.FixedLocator([400, 300, 200, 100, 0]))#, 231, 281, 331, 381]))
        self.graph01.yaxis.set_minor_locator(ticker.FixedLocator([    20, 40, 60, 80, 120, 140, 160, 180, 220, 240, 260, 280, 320, 340, 360, 380]))
        self.graph01.yaxis.grid(which='major', color=self.graphColor,alpha=0.2, linewidth=0.5, linestyle='--')
        self.graph01.xaxis.grid(which='major', color=self.graphColor,alpha=0.2, linewidth=0.5, linestyle='--')
        #self.graph01.yaxis.grid(which='minor', color=self.graphColor, linewidth=0.3, linestyle='--')
        self.graph01.xaxis.set_major_locator(ticker.LinearLocator(8))
        #self.graph01.xaxis.set_major_locator(ticker.MultipleLocator(3.0))
        self.graph01.set_ylim(0.0, 400.0)
        #self.graph01.set_ylim(400.0, 0.0)
        self.graph01.set_xlim(self.min_Time, self.max_Time)
        self.graph01.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d'))
        #self.graph01.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
        self.graph01.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d'))
        #self.graph01.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
        #self.graph01.xaxis.set_color(self.gridColor)


        self.graph01.set_ylabel(self.graph01_AxisY_Label,font='Segoe UI',fontsize=12,fontweight='bold',labelpad=0.0,loc='center')#, position="right")
        self.graph01.set_xlabel("Time(Hour)",font='Segoe UI',fontsize=12,fontweight='bold',labelpad=1.0 )  # , position="right")
        self.graph01.yaxis.tick_left()
        self.graph01.yaxis.set_label_position("left")
        #self.graph01.yaxis.set_label_position("right")
        self.graph01.yaxis.get_label().set_color(self.graphColor)
        self.graph01.xaxis.get_label().set_color(self.graphColor)

    def defineASI_Graph(self):
        # 05-1. Define ASI chart
        self.graph02.set_ylim(-1.0, 1.0)
        self.graph02.tick_params(axis='x', which='major', colors=self.gridColor)
        self.graph02.tick_params(axis='y', which='major', colors=self.gridColor)
        self.graph02.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
        #self.graph02.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.3f'))
        #self.graph02.set_facecolor(self.graphBackgroundColor)
        # self.graph02.tick_params(axis='x', which='major', colors=self.gridColor)
        self.graph02.yaxis.set_major_locator(
            ticker.FixedLocator([-1.0, -0.5, 0.0, 0.5, 1.0]))  # , 231, 281, 331, 381]))
        # self.graph02.yaxis.set_minor_locator(ticker.FixedLocator([    20, 40, 60, 80, 120, 140, 160, 180, 220, 240, 260, 280, 320, 340, 360, 380]))
        self.graph02.tick_params(axis='x', which='major', colors=self.gridColor)
        self.graph02.tick_params(axis='y', which='major', colors=self.gridColor)
        self.graph02.set_facecolor(self.graphBackgroundColor)
        self.graph02.spines['top'].set_color(self.gridColor)
        self.graph02.spines['left'].set_color(self.gridColor)
        self.graph02.spines['right'].set_color(self.gridColor)
        self.graph02.spines['bottom'].set_color(self.gridColor)
        self.graph02.spines['top'].set_linewidth(1.0)
        self.graph02.spines['left'].set_linewidth(1.0)
        self.graph02.spines['right'].set_linewidth(1.0)
        self.graph02.spines['bottom'].set_linewidth(1.0)
        self.graph02.set_ylabel("ASI", font='Segoe UI', fontsize=12, fontweight='bold',labelpad=0.0)  # , position="right")
        # self.graph01.set_xlabel("Time(Hour)")  # , position="right")
        self.graph02.yaxis.set_label_position("right")
        self.graph02.yaxis.tick_right()
        self.graph02.yaxis.get_label().set_color(self.graphColor)

        # Insert ASI BAND
        #self.insert_ASI_BAND()

    def defineCBC_Graph(self):
        self.graph03.set_ylim(200.0, 600.0)

        #self.graph03.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('\n\n%7.1f'))
        self.graph03.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter(self.graph03_StrFormatter))
        self.graph03.yaxis.set_major_locator(ticker.FixedLocator([200.0, 300.0, 400.0, 500.0, 600.0]))  # , 231, 281, 331, 381]))
        self.graph03.tick_params(axis='x', which='major', colors=self.gridColor)
        #self.graph03.tick_params(axis='y', which='major', colors=self.gridColor)
        #self.graph03.tick_params(axis='y', which='major', colors=self.color_CBC, length=0)
        self.graph03.tick_params(axis='y', which='major', colors=self.gridColor, length=0)
        #self.graph03.tick_params(axis='y', which='major', colors="None")
        self.graph03.set_facecolor(self.graphBackgroundColor)
        self.graph03.spines['top'].set_color(self.gridColor)
        self.graph03.spines['left'].set_color(self.gridColor)
        self.graph03.spines['right'].set_color(self.gridColor)
        #self.graph03.spines['right'].set_color("None")
        self.graph03.spines['bottom'].set_color(self.gridColor)
        self.graph03.spines['top'].set_linewidth(1.0)
        self.graph03.spines['left'].set_linewidth(1.0)
        self.graph03.spines['right'].set_linewidth(1.0)
        self.graph03.spines['bottom'].set_linewidth(1.0)
        self.graph03.set_ylabel("CBC(ppm)",font='Segoe UI',fontsize=12,fontweight='bold',labelpad=self.labelPadSize_graph03)#, position="right")
        #self.graph01.set_xlabel("Time(Hour)")  # , position="right")
        self.graph03.yaxis.set_label_position("right")
        #self.graph03.yaxis.get_label().set_color(self.graphColor)
        self.graph03.yaxis.get_label().set_color(self.color_CBC)
        self.graph03.yaxis.get_label().set_color(self.gridColor)

    def makePlotFormat(self):
        self.time = []
        self.data_P = []
        self.data05 = []
        self.data04 = []
        self.data03 = []

        self.line_Rod_P = self.graph01.plot(self.time, self.data_P, marker='o', alpha=0.5, color=self.color_Rod_P, label="P", markersize=5)
        self.line_Rod05 = self.graph01.plot(self.time, self.data05, 'o-', alpha=0.5, color=self.color_Rod05, label="5", markersize=5)
        self.line_Rod04 = self.graph01.plot(self.time, self.data04, 'o-', alpha=0.5, color=self.color_Rod04, label="4", markersize=5)
        # self.line_Rod03 = self.graph01.plot(self.time, self.data03, 'o-', alpha=0.3, color=self.color_Rod03,label="Rod03")

        self.dot_P = self.graph01.plot(self.time, self.data_P, 'o', visible=False, alpha=1.0, color=self.color_Rod_P, picker=True, pickradius=10, label="dot_P", markersize=5)
        self.dot_5 = self.graph01.plot(self.time, self.data05, 'o', visible=False, alpha=1.0, color=self.color_Rod05, picker=True, pickradius=10, label="dot_5", markersize=5)
        self.dot_4 = self.graph01.plot(self.time, self.data04, 'o', visible=False, alpha=1.0, color=self.color_Rod04, picker=True, pickradius=10, label="dot_4", markersize=5)


        if(self.ASI_FLAG==True):
            self.data_ASI = []
            self.line_ASI = self.graph02.plot(self.time, self.data_ASI, 'o-', alpha=0.5, color=self.color_ASI, label="ASI", markersize=5)
            self.d_ASI = self.convt_ASI_POS(self.data_ASI)
            self.dot_ASI = self.graph01.plot(self.time, self.d_ASI, 'o', visible=False, alpha=1.0, color=self.color_ASI, picker=True, pickradius=10, label="dot_ASI", markersize=5)
            self.time_ASI_BAND      = []# 0.00,  5.00, 10.00, 15.00, 20.00, 23.33333, 23.33334, 25.00, 30.00, 35.01 ]
            self.data_ASI_BAND_UP   = []# 0.27,  0.27,  0.27,  0.27,  0.27,   0.27  ,    0.6  ,  0.60,  0.60,  0.60 ]
            self.data_ASI_BAND_DOWN = []#-0.27, -0.27, -0.27, -0.27, -0.27,  -0.27  ,   -0.6  , -0.60, -0.60, -0.60 ]
            self.line_ASI_BAND_UP   = self.graph02.plot(self.time_ASI_BAND, self.data_ASI_BAND_UP,   '--', alpha=0.5, color=self.color_ASI_BAND,label="ASI band")
            self.line_ASI_BAND_DOWN = self.graph02.plot(self.time_ASI_BAND, self.data_ASI_BAND_DOWN, '--', alpha=0.5, color=self.color_ASI_BAND,label="ASI band")
        if(self.CBC_FLAG==True):
            self.data_CBC = []
            self.line_CBC = self.graph03.plot(self.time, self.data_CBC, 'o-', alpha=0.5, color=self.color_CBC, label="CBC", markersize=5)
            self.d_CBC = self.convt_CBC_POS(self.data_CBC)
            self.dot_CBC = self.graph01.plot(self.time, self.d_CBC, 'o', visible=False, alpha=1.0, color=self.color_CBC, picker=True, pickradius=10, label="dot_CBC", markersize=5)
        if(self.Power_FLAG==True):
            self.data_PWR = []
            self.line_Power = self.graph01.plot(self.time, self.data_PWR, 'o-', alpha=0.5, color=self.color_PWR, label="Power")
            self.dot_PWR = self.graph01.plot(self.time, self.data_PWR, 'o', visible=False, alpha=1.0, color=self.color_PWR, picker=True, pickradius=10, label="dot_PWR", markersize=5)


        self.lns = self.line_Rod_P + self.line_Rod05 + self.line_Rod04
        if(self.ASI_FLAG==True):
            self.lns += self.line_ASI
            self.index_ASI_BAND = len(self.lns)-1

            #self.lns_ASI_BAND = self.line_ASI_BAND_UP + self.line_ASI_BAND_DOWN
        if(self.CBC_FLAG==True):
            self.lns += self.line_CBC
        if(self.Power_FLAG==True):
            self.lns += self.line_Power

        self.dts = self.dot_P + self.dot_5 + self.dot_4#
        if(self.ASI_FLAG==True):
           self.dts += self.dot_ASI
        if (self.CBC_FLAG == True):
            self.dts += self.dot_CBC
        if(self.Power_FLAG==True):
            self.dts += self.dot_PWR


        self.labs = [l.get_label() for l in self.lns]
        self.dot_labs = [l.get_label() for l in self.dts]

        self.legend = self.graph01.legend(handles=self.lns, labels=self.labs, loc='upper center',
                                          labelcolor=self.gridColor, facecolor=self.graphBackgroundColor,
                                          framealpha=1.0, ncol=8, bbox_to_anchor=(0.5, 1.098))

        self.legendClick = {}
        self.dot_legendClick = {}
        self.legend_Actvated_Flag = []

        index = 0
        # self.line
        for legendline, origline in zip(self.legend.get_lines(), self.lns):
            legendline.set_picker(True)
            self.legendClick[legendline] = [origline, index]
            self.legend_Actvated_Flag.append(True)
            index += 1

        index = 0
        for dot in self.dts:
            self.dot_legendClick[dot] = index
            index += 1

        for legendline in self.legend.legendHandles:
            legendline.set_alpha(0.8)
            legendline._legmarker.set_alpha(0.8)


    def updateRdcPerHour(self,rdcPerHour, power_inc_flag):
        self.rdcPerHour = rdcPerHour
        self.power_inc_flag = power_inc_flag
        try:
            deltaTime = 100.0 / self.rdcPerHour
            deltaTime = math.ceil(deltaTime/5.0) * 5.0

            if(self.power_inc_flag==True):
                bandChangeTime = 30.0 / self.rdcPerHour
                microTime = 0.00001
                self.time_ASI_BAND = [ self.addTime,
                                       self.addTime + bandChangeTime-microTime,
                                       self.addTime + bandChangeTime+microTime,
                                       self.addTime + deltaTime ]
                self.data_ASI_BAND_UP   = [  0.6,  0.6,  0.27,  0.27 ]
                self.data_ASI_BAND_DOWN = [ -0.6, -0.6, -0.27, -0.27 ]
            else:
                bandChangeTime = 70.0 / self.rdcPerHour
                microTime = 0.00001
                self.time_ASI_BAND = [ self.addTime,
                                       self.addTime + bandChangeTime-microTime,
                                       self.addTime + bandChangeTime+microTime,
                                       self.addTime + deltaTime ]
                self.data_ASI_BAND_UP   = [  0.27,  0.27,  0.60,  0.60 ]
                self.data_ASI_BAND_DOWN = [ -0.27, -0.27, -0.60, -0.60 ]
        except ZeroDivisionError:
            pass



    def insert_ASI_BAND(self):
        # self.time_ASI_BAND      = [ 0.00,  5.00, 10.00, 15.00, 20.00, 23.33333, 23.33334, 25.00, 30.00, 35.01 ]
        # self.data_ASI_BAND_UP   = [ 0.27,  0.27,  0.27,  0.27,  0.27,   0.27  ,    0.6  ,  0.60,  0.60,  0.60 ]
        # self.data_ASI_BAND_DOWN = [-0.27, -0.27, -0.27, -0.27, -0.27,  -0.27  ,   -0.6  , -0.60, -0.60, -0.60 ]
        self.line_ASI_BAND_UP[0].set_data(self.time_ASI_BAND, self.data_ASI_BAND_UP)
        self.line_ASI_BAND_DOWN[0].set_data(self.time_ASI_BAND, self.data_ASI_BAND_DOWN)

    def adjustTime(self, addTime):
        self.addTime = addTime
        self.min_Time += addTime
        self.max_Time += addTime
        self.graph01.set_xlim(self.min_Time, self.max_Time)
        self.graph01.xaxis.set_major_locator(ticker.LinearLocator(8))

    def insertTime(self, time):
        self.time = time
        maxTime = max(self.time)
        self.graph01.set_xlim(self.min_Time, maxTime)

        digit = math.floor(math.log10(maxTime))
        significand = maxTime / math.pow(10,digit)

        #divide_significand = 0
        format_digit = 0
        divide_significand = math.floor(significand/1.1) + 1.0

        if(significand>1.1 and significand <= 1.65):
            divide_significand = 1.5
            format_digit = 1
        if(significand>2.2 and significand <= 2.75):
            divide_significand = 2.5
            format_digit = 1

        unit = divide_significand * math.pow(10,digit-1)
        num = math.ceil(maxTime / unit)

        locatorPos = []
        for i in range(num+1):
            locatorPos.append(i*unit)

        if(locatorPos[-1] < maxTime):
            locatorPos.append(maxTime)


        if(format_digit==1 and digit<=1):
            if(digit==1):
                self.graph01.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
            else:
                self.graph01.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.3f'))
        self.graph01.xaxis.set_major_locator(ticker.FixedLocator(locatorPos))

    def insert_ECP_DataSet(self, data_P, data_5, data_4, dataOpt):#, data_ASI, data_CBC, data_power):
        tmp = dataOpt
        if(self.ASI_FLAG==True):
            self.data_ASI = tmp.pop(0)
            self.insert_ASI_BAND()
        if(self.CBC_FLAG==True):
            self.data_CBC = tmp.pop(0)
            self.resizeGraphCBC(self.data_CBC)
        if(self.Power_FLAG==True):
            self.data_Power = tmp.pop(0)

        nData = len(data_P)
        time = self.time[:nData]
        #print(data_P)
        #print(time)
        self.data_P = data_P
        self.data05 = data_5
        self.data04 = data_4
        # self.data_ASI = data_ASI
        # self.data_CBC = data_CBC
        # self.data_Power = data_power
        #self.resizeTimeAxes(time)

        self.line_Rod_P[0].set_data(time, self.data_P)
        self.line_Rod05[0].set_data(time, self.data05)
        self.line_Rod04[0].set_data(time, self.data04)
        self.dot_P[0].set_data(time, self.data_P)
        self.dot_5[0].set_data(time, self.data05)
        self.dot_4[0].set_data(time, self.data04)

        if(self.ASI_FLAG==True):
            self.line_ASI[0].set_data(time, self.data_ASI)
            self.d_ASI = self.convt_ASI_POS(self.data_ASI)
            self.dot_ASI[0].set_data(time, self.d_ASI)
        if(self.CBC_FLAG==True):
            self.line_CBC[0].set_data(time, self.data_CBC)
            self.d_CBC = self.convt_CBC_POS(self.data_CBC)
            self.dot_CBC[0].set_data(time, self.d_CBC)
        if(self.Power_FLAG==True):
            self.line_Power[0].set_data(time, self.data_Power)
            self.dot_PWR[0].set_data(time, self.data_Power)
        self.canvas.draw()

    def insertDataSet(self, time, data_P, data_5, data_4, dataOpt):#, data_ASI, data_CBC, data_power):
        tmp = dataOpt
        if(self.ASI_FLAG==True):
            self.data_ASI = tmp.pop(0)
            self.insert_ASI_BAND()
        if(self.CBC_FLAG==True):
            self.data_CBC = tmp.pop(0)
            self.resizeGraphCBC(self.data_CBC)
        if(self.Power_FLAG==True):
            self.data_Power = tmp.pop(0)


        #self.resizeGraphCBC(self.data_CBC)
        self.resizeTimeAxes(time)
        self.time =   time
        self.data_P = data_P
        self.data05 = data_5
        self.data04 = data_4
        # self.data_ASI = data_ASI
        # self.data_CBC = data_CBC
        # self.data_Power = data_power

        self.line_Rod_P[0].set_data(self.time, self.data_P)
        self.line_Rod05[0].set_data(self.time, self.data05)
        self.line_Rod04[0].set_data(self.time, self.data04)
        self.dot_P[0].set_data(self.time, self.data_P)
        self.dot_5[0].set_data(self.time, self.data05)
        self.dot_4[0].set_data(self.time, self.data04)

        if(self.ASI_FLAG==True):
            self.line_ASI[0].set_data(self.time, self.data_ASI)
            self.d_ASI = self.convt_ASI_POS(self.data_ASI)
            self.dot_ASI[0].set_data(self.time, self.d_ASI)
        if(self.CBC_FLAG==True):
            self.line_CBC[0].set_data(self.time, self.data_CBC)
            self.d_CBC = self.convt_CBC_POS(self.data_CBC)
            self.dot_CBC[0].set_data(self.time, self.d_CBC)
        if(self.Power_FLAG==True):
            self.line_Power[0].set_data(self.time, self.data_Power)
            self.dot_PWR[0].set_data(self.time, self.data_Power)
        self.canvas.draw()

    def clearData(self):
        self.time =   []
        self.data_P = []
        self.data05 = []
        self.data04 = []
        self.data_ASI = []
        self.data_CBC = []
        self.data_Power = []
        self.d_ASI = []
        self.d_CBC = []


        self.line_Rod_P[0].set_data(self.time, self.data_P)
        self.line_Rod05[0].set_data(self.time, self.data05)
        self.line_Rod04[0].set_data(self.time, self.data04)
        self.line_Power[0].set_data(self.time, self.data_Power)
        self.line_ASI[0].set_data(self.time, self.data_ASI)
        self.line_CBC[0].set_data(self.time, self.data_CBC)

        self.dot_P[0].set_data(self.time, self.data_P)
        self.dot_5[0].set_data(self.time, self.data05)
        self.dot_4[0].set_data(self.time, self.data04)
        self.dot_PWR[0].set_data(self.time, self.data_Power)
        self.d_ASI = self.convt_ASI_POS(self.data_ASI)
        self.dot_ASI[0].set_data(self.time, self.d_ASI)
        self.d_CBC = self.convt_CBC_POS(self.data_CBC)
        self.dot_CBC[0].set_data(self.time, self.d_CBC)

        self.min_Time = 0.0
        self.max_Time = 35.0
        self.addTime = 0.0
        self.graph01.set_xlim(self.min_Time, self.max_Time)
        self.graph01.xaxis.set_major_locator(ticker.LinearLocator(8))

        for i in range(len(self.legend_Actvated_Flag)):
            self.legend_Actvated_Flag[i] = True
            self.lns[i].set_visible(True)
            tmp = self.legend.legendHandles[i]
            tmp.set_alpha(0.8)
            tmp._legmarker.set_alpha(0.8)

        if(self.ASI_FLAG==True):
            self.line_ASI_BAND_UP[0].set_visible(True)
            self.line_ASI_BAND_DOWN[0].set_visible(True)

            #print(self.lns[i])

    def insertDataSetTest(self):
        self.time =   [ 0.0,      5.0,   10.0,   15.0,   20.0,   25.0, 30.0,   33.0]
        self.data_P = [ 381.0,  371.0,  354.7,  310.5,  278.6,  225.0, 190.5, 190.5]
        self.data05 = [ 381.0,  381.0,  381.0,  378.1,  346.6,  310.0, 273.6, 228.1]
        self.data04 = [ 381.0,  381.0,  381.0,  381.0,  381.0,  355.1, 319.4, 287.4]
        self.data03 = [ 381.0,  381.0,  381.0,  381.0,  381.0,  381.0, 381.0, 381.0]


        self.data_ASI = [  0.00,  -0.04,  -0.06,  -0.08, -0.099, -0.105, -0.100, -0.13 ]
        self.data_CBC = [ 450.0,  361.0,  278.0,  291.0,  341.6,  393.1,  432.0, 491.0]
        self.data_Power = [ 100.0,   85.0,   70.0,   55.0,   40.0,   25.0,   10.0,   0.0]


        self.barX_Pos = [  0.5,  1.5,  2.5,  3.5,  4.5,  5.5,  6.5,  7.5,  8.5,  9.5,
                          10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5,
                          20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5,
                          30.5, 31.5, 32.5 ]
        self.barY_Pos_P = [381,381.0,381.0,377.5,369.4,345.8,337.4,330.5,320.4,311.1,
                           300,285.5,277.3,257.1,240.3,219.4,205.6,190.5,190.5,190.5,
                         190.5,190.5,190.5,190.5,190.5,190.5,190.5,190.5,190.5,190.5,
                         190.5,190.5,190.5 ]
        self.barY_Pos05 = [381,381.0,381.0,381.0,381.0,381.0,381.0,381.0,381.0,381.0,
                           381,381.0,381.0,381.0,371.3,360.4,345.6,330.5,315.5,302.5,
                         261.0,254.5,240.0,230.5,223.5,226.5,224.5,224.5,224.5,224.5,
                         224.5,224.5,224.5 ]

        self.line_Rod_P[0].set_data(self.time, self.data_P)
        self.line_Rod05[0].set_data(self.time, self.data05)
        self.line_Rod04[0].set_data(self.time, self.data04)
        self.line_Power[0].set_data(self.time, self.data_Power)
        self.line_ASI[0].set_data(self.time, self.data_ASI)
        self.line_CBC[0].set_data(self.time, self.data_CBC)

        self.dot_P[0].set_data(self.time, self.data_P)
        self.dot_5[0].set_data(self.time, self.data05)
        self.dot_4[0].set_data(self.time, self.data04)
        self.dot_PWR[0].set_data(self.time, self.data_Power)
        self.d_ASI = self.convt_ASI_POS(self.data_ASI)
        self.dot_ASI[0].set_data(self.time, self.d_ASI)
        self.d_CBC = self.convt_CBC_POS(self.data_CBC)
        self.dot_CBC[0].set_data(self.time, self.d_CBC)


    # bar 차트의 y tick의 format을 변경하기 위한 함수
    def reversalOfBarTick(self, x, pos):
        return '%d' % (self.barYAxisTickPivot - x)

        #self.fig.canvas.manager.toolmanager.remove_tool('subplots')
    def clickEvent(self,event):
        objectName = event.artist
        # Case01. Line Index Click Event
        # Make Line Index
        # If Unit Spline is Activated and Line Index is selected,
        # This Code manke Unit Spline deactivat
        # If Unit Line was deactived, dot-click event also deactivated
        try:
            originLine, index = self.legendClick[objectName]
            # Check Current Spline Condition
            visible = not originLine.get_visible()
            # If Unit Line was deactived, dot-click event also deactivated
            self.legend_Actvated_Flag[index] = visible
            # Activate/Deactivate Spline
            originLine.set_visible(visible)
            # Deeping/Fadeding Legend Line/Marker
            tmp = self.legend.legendHandles[index]
            tmp.set_alpha(0.8 if visible else 0.2) #Line
            tmp._legmarker.set_alpha(0.8 if visible else 0.2) #Marker
            if(self.ASI_FLAG==True):
                if(index==self.index_ASI_BAND):
                    self.line_ASI_BAND_UP[0].set_visible(visible)
                    self.line_ASI_BAND_DOWN[0].set_visible(visible)

            self.fig.canvas.draw()


        # Case01. Line Dot Click Event
        # Make Line Index
        # If Unit Spline is Activated and Line Index is selected,
        # This Code manke Unit Spline deactivat
        # If Unit Line was deactived, dot-click event also deactivated
        except KeyError:
            #print(self.legend_Actvated_Flag)
            #print(objectName)
            #print(self.dot_legendClick)

            index = self.dot_legendClick[objectName]
            #print("%d dataset clicked" %index)
            if(self.legend_Actvated_Flag[index]==True):
                # TODO SGH, Make Connection Routines for display
                pass
                #print('%s click:, x=%d, y=%d, xdata=%f, ydata=%f' %('single', event.mouseevent.x, event.mouseevent.y, event.mouseevent.xdata, event.mouseevent.ydata))
            else:
                #print("Not Activated!")
                pass



    def resizeGraphCBC(self,data):

        if(len(data)==1):
            refactorValue = data[0]/100.0
            if(refactorValue<=3.0):
                self.graph03.set_ylim(0.0, 400.0)
                #self.graph03.yaxis.set_major_locator(ticker.FixedLocator([0.0, 100.0, 200.0, 300.0, 400.0]))
                self.graph03.yaxis.set_major_locator(ticker.LinearLocator(5))
                return
            else:
                yMin = ( refactorValue - 2 ) * 100.0
                yMax = ( refactorValue + 2 ) * 100.0
                self.graph03.set_ylim(yMin, yMax)
                self.graph03.yaxis.set_major_locator(ticker.LinearLocator(5))
                return


        minValue = min(data)
        maxValue = max(data)
        minP = math.floor(minValue/100.0)
        maxP = math.ceil(maxValue/100.0)
        if(maxP-minP <= 4):
            if(maxP <= 4):
                self.graph03.set_ylim(0.0, 400.0)
                self.graph03.yaxis.set_major_locator(ticker.LinearLocator(5))
                return
            else:
                y_min = 100.0 * (round((maxP+minP)/2.0)-2)
                y_max = y_min + 400.0
                self.graph03.set_ylim(y_min,y_max)
                self.graph03.yaxis.set_major_locator(ticker.LinearLocator(5))
                return
        else:
            y_min = 100.0 * max((round((maxP + minP) / 2.0) - 4), 0.0)
            y_max = y_min + 800.0
            self.graph03.set_ylim(y_min, y_max)
            self.graph03.yaxis.set_major_locator(ticker.LinearLocator(5))
            return

    def resizeTimeAxes(self,time):
        maxTime = max(time)
        if self.max_Time < maxTime:
            divs = math.ceil((maxTime-self.min_Time)/5.0)
            self.max_Time = 5.0 * divs + self.min_Time
            self.graph01.set_xlim(self.min_Time, self.max_Time)
            self.graph01.xaxis.set_major_locator(ticker.LinearLocator(divs+1))
            self.fig.canvas.draw()







    def convt_ASI_POS(self,asi):
        convt_ASI = []
        [ x1, x2 ] = self.graph02.get_ylim()
        [ y1, y2 ] = self.graph01.get_ylim()
        dx = (x2-x1)
        a = (y2-y1) / dx
        b = (x2*y1 - x1*y2)/dx
        for value in asi:
            convt_value = value * a + b
            convt_ASI.append(convt_value)

        return convt_ASI


    def convt_CBC_POS(self,cbc):
        convt_CBC = []
        [ x1, x2 ] = self.graph03.get_ylim()
        [ y1, y2 ] = self.graph01.get_ylim()
        dx = (x2-x1)
        a = (y2-y1) / dx
        b = (x2*y1 - x1*y2)/dx
        for value in cbc:
            convt_value = value * a + b
            convt_CBC.append(convt_value)

        return convt_CBC
