import numpy as np
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, \
    NavigationToolbar2QT as NavigationToolbar
import matplotlib.gridspec as gs
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as ticker


class AxialWidget(QWidget):
    def __init__(self):
        super().__init__()

        #self.setContentsMargins(0,0,0,0)


        self.verticalLayout_4 = QVBoxLayout()
        self.setLayout(self.verticalLayout_4)
        # Set Graph Label Color
        self.graphBackgroundColor = '#272c36'
        self.graphColor = '#d2d2d2'
        self.graphLine  = '#1b1d23'
        self.gridColor = '#cccccc'
        self.whiteColor = '#FFFFFF'
        self.orange = '#FFA000'
        self.green  = "#BCFD60"

        #self.fig = plt.Figure(facecolor=self.graphBackgroundColor)
        self.fig = plt.Figure(facecolor="None")

        # Initial Rod Position
        self.axialHeight = 381.0
        self.bRodP = 0.0
        self.bRod5 = 0.0
        self.bRod4 = 0.0
        self.bRod3 = 0.0
        self.aRodP = 0.0
        self.aRod5 = 0.0
        self.aRod4 = 0.0
        self.aRod3 = 0.0
        self.barYAxisTickPivot = 381
        self.barWidth = 0.32
        self.banks = [ "Bank P", "Bank 5", "Bank 4", "Bank 3"]


        self.fig.subplots_adjust(wspace=0, hspace=0)

        self.bar = self.fig.add_subplot()
        self.bar.set_facecolor(self.graphBackgroundColor)
        self.bar.tick_params(axis='x', colors=self.whiteColor)
        self.bar.tick_params(axis='y', colors=self.whiteColor)
        self.bar.tick_params(which='minor',axis='y', colors=self.whiteColor)
        self.bar.set_ylabel("Rod Position(cm)")
        #self.bar.yaxis.set_label_position("right")
        #self.bar.yaxis


        self.bar.spines['top'].set_color(self.whiteColor)
        self.bar.spines['left'].set_color(self.whiteColor)
        self.bar.spines['right'].set_color(self.whiteColor)
        self.bar.spines['bottom'].set_color(self.whiteColor)

        self.bar.spines['top'].set_linewidth(2.0)
        self.bar.spines['left'].set_linewidth(2.0)
        self.bar.spines['right'].set_linewidth(2.0)
        self.bar.spines['bottom'].set_linewidth(2.0)

        self.canvas = FigureCanvas(self.fig)
        self.fig.tight_layout()

        #self.toolbar = NavigationToolbar(self.canvas, self)
        #self.verticalLayout_4.addWidget(self.toolbar)
        self.verticalLayout_4.addWidget(self.canvas)

        # init graph
        self.init_graph()

        self.drawBar()
        cursor = Qt.Cross


    def setChartSize(self,width,height):
        self.canvas.setFixedSize(width,760)
        #self.canvas


    # bar 차트의 y tick의 format을 변경하기 위한 함수
    def reversalOfBarTick(self, x, pos):
        return '%d' % (self.barYAxisTickPivot - x)

    def init_graph(self):

        self.bar.cla()
        self.bar.set_xlim(-0.5,3.5)
        self.bar.set_ylim(self.barYAxisTickPivot, 0)
        
        self.bar.yaxis.tick_right()
        self.bar.xaxis.tick_top()
        self.bar.tick_params(axis='x', rotation=0)

        formatter = FuncFormatter(self.reversalOfBarTick)
        self.bar.yaxis.set_major_formatter(formatter)

        self.bar.xaxis.set_minor_locator(ticker.FixedLocator([0.5,1.5,2.5]))

        self.bar.set_ylabel("Rod Position(cm)")#, position="right")
        self.bar.yaxis.set_label_position("right")
        self.bar.yaxis.get_label().set_color(self.graphColor)
        self.bar.yaxis.set_major_locator(ticker.FixedLocator([0,31,81,131,181,231,281,331,381]))
        self.bar.yaxis.set_minor_locator(ticker.FixedLocator([   0,  11,  21,  31,  41,  51,  61,  71,  81,  91,
                                                               101, 111, 121, 131, 141, 151, 161, 171, 181, 191,
                                                               201, 211, 221, 231, 241, 251, 261, 271, 281, 291,
                                                               301, 311, 321, 331, 341, 351, 361, 371, 381]))

        self.bar.xaxis.grid(which='minor',alpha=0.7,linewidth=1)
        self.bar.yaxis.grid(which='major',alpha=1.0,linewidth=1)
        self.bar.yaxis.grid(which='minor',linestyle='--',alpha=0.2,linewidth=1)


    def drawBar(self):

        values   = [ self.bRodP, self.bRod5, self.bRod4, self.bRod3 ]
        values2  = [ self.aRodP, self.aRod5, self.aRod4, self.aRod3 ]

        x = [0,1,2,3]
        x1 = []
        x2 = []
        for i in range(4):
            x1.append(x[i]-self.barWidth/2.0)
            x2.append(x[i]+self.barWidth/2.0)


        self.bar.xaxis.set_minor_locator(ticker.FixedLocator([0.5, 1.5, 2.5]))
        self.bar01 = self.bar.bar(x1, values, alpha=1.0, color=self.orange,width=self.barWidth)
        self.bar02 = self.bar.bar(x2, values2,alpha=1.0, color=self.green ,width=self.barWidth)


        self.bar.legend(['Rod Position\nBefore Shutdown','Rod Position\nAfter Shutdown'],loc='lower center', bbox_to_anchor=(0.5,-0.1,0.0,-1.5),ncol=2,frameon=False, labelcolor=self.gridColor)#,fancybox=False)

        self.bar.set_xticks([0,1,2,3])
        self.bar.set_xticklabels(self.banks)
        self.canvas.draw()

    def RedrawBar01(self,a,b,c):
        self.bar.tick_params(axis='x', colors=self.whiteColor)
        self.bar.tick_params(axis='y', colors=self.whiteColor)
        dataset = [self.axialHeight - a, self.axialHeight - b,self.axialHeight-c,self.axialHeight-381.0]
        for i,j in zip(self.bar01,dataset):
            i.set_height(j)
        self.canvas.draw()

    def RedrawBar02(self,a,b,c):
        self.bar.tick_params(axis='x', colors=self.whiteColor)
        self.bar.tick_params(axis='y', colors=self.whiteColor)
        dataset = [self.axialHeight - a, self.axialHeight - b,self.axialHeight-c,self.axialHeight-381.0]
        for i,j in zip(self.bar02,dataset):
            i.set_height(j)
        self.canvas.draw()
        
    """
    def showLifetimeSummary(self, lifetimeSummary):
        self.ti_power.setText(str(lifetimeSummary.power))
        self.ti_inletTemp_c.setText(str(lifetimeSummary.inletTemp_F))
        self.ti_inletTemp_f.setText(str(lifetimeSummary.inletTemp_C))
        self.ti_burnup.setText(str(lifetimeSummary.burnup))
        self.ti_time_efpd.setText(str(lifetimeSummary.time_EFPD))
        self.ti_time_efph.setText(str(lifetimeSummary.time_EFPH))
        self.ti_keff.setText(str(lifetimeSummary.keff))
        self.ti_boron_ppm.setText(str(lifetimeSummary.boron))
        self.ti_xeWorth_pcm.setText(str(lifetimeSummary.xeWorth))
        self.ti_smWorth_pcm.setText(str(lifetimeSummary.smWorth))
        self.ti_fqWithU.setText(str(lifetimeSummary.fqWithU))
        self.ti_fqLimit.setText(str(lifetimeSummary.fqLimit))
        self.ti_fdhWithU.setText(str(lifetimeSummary.fdhWithU))
        self.ti_fdhLimit.setText(str(lifetimeSummary.fdhLimit))
        self.ti_ao.setText(str(lifetimeSummary.ao))
        self.ti_d_bank.setText(str(lifetimeSummary.dBank))
        self.ti_c_bank.setText(str(lifetimeSummary.cBank))
        self.ti_b_bank.setText(str(lifetimeSummary.bBank))
        self.ti_a_bank.setText(str(lifetimeSummary.aBank))

        # QTableWidget의 cell 높이 설정
        height = self.tb_lifetimeSum.height()
        for i in range(self.tb_lifetimeSum.rowCount()):
            self.tb_lifetimeSum.setRowHeight(i, (height-15)/(self.tb_lifetimeSum.rowCount()))
        # QTableWidget의 cell 높이 설정
        self.tb_lifetimeSum.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
    """