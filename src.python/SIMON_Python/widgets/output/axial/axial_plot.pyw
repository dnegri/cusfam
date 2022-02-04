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
        #self.setContentsMargins(125,125,125,125)

        self.verticalLayout_4 = QVBoxLayout()
        self.setLayout(self.verticalLayout_4)
        # 그래프 색상 설정
        self.graphBackgroundColor = '#272c36'
        self.graphColor = '#d2d2d2'
        self.graphLine  = '#1b1d23'
        self.gridColor = '#999999'
        self.whiteColor = '#FFFFFF'
        self.orange = '#FFA000'

        #self.fig = plt.Figure(facecolor=self.graphBackgroundColor)
        self.fig = plt.Figure(facecolor="None")

#        self.axe = plt.Axes()

        self.fig.subplots_adjust(wspace=0, hspace=0)
        grid = gs.GridSpec(8, 8)
        #grid = gs.GridSpec(1, 1)

        # self.axial = self.fig.subplots(2,1)
        # self.axe = self.fig.add_axes()#grid[:,:-2])
        self.axial = self.fig.add_subplot(grid[:, :-2])

        # plt.xticks([0.1,0.2,0.3,0.4])
        # plt.minorticks_on()
        # plt.show()

        # self.axial.minorticks_on()
        # self.axial.xaxis.minorticks_on()
        # self.axial.xaxis.set_major_locator(AutoMinorLocator(4))
        # self.axial.xaxis.set_minor_locator(AutoMinorLocator(4))


        #self.axial.set_xlim
        self.axial.set_xlabel('Test')
        self.axial.set_ylabel('Test')
        self.axial.yaxis.set_major_locator(plt.MultipleLocator(10))
        self.axial.yaxis.set_minor_locator(plt.MultipleLocator(2))
        self.axial.set_facecolor(self.graphBackgroundColor)

        # self.axial.yaxis.yticks([0,10,20,30,40,50,60,70,80,90,100])
        self.axial.tick_params(axis='x', which='major', colors=self.gridColor)
        self.axial.tick_params(axis='y', which='major', colors=self.gridColor)

        self.axial.spines['top'].set_color(self.gridColor)
        self.axial.spines['left'].set_color(self.gridColor)
        self.axial.spines['right'].set_color(self.gridColor)
        self.axial.spines['bottom'].set_color(self.gridColor)

        self.bar = self.fig.add_subplot(grid[:, -2:])
        self.bar.set_facecolor(self.graphBackgroundColor)
        self.bar.tick_params(axis='x', colors=self.gridColor)
        self.bar.tick_params(axis='y', colors=self.gridColor)


        self.bar.spines['top'].set_color(self.gridColor)
        self.bar.spines['left'].set_color(self.gridColor)
        self.bar.spines['right'].set_color(self.gridColor)
        self.bar.spines['bottom'].set_color(self.gridColor)
        # self.axial.setContentsMargins(-10,-10,-10,-10)
        # bar의  시작점 y=0을 옯길 수 없음, 따라서 입력값의 101-x를 계산하여 전시함
        # tick 값 또한 101-x를 계산하여 전시
        # 101의 값이 self.barYAxisTickPivot
        #self.barYAxisTickPivot = 382
        self.barYAxisTickPivot = 381

        self.canvas = FigureCanvas(self.fig)

        #self.toolbar = NavigationToolbar(self.canvas, self)
        #self.verticalLayout_4.addWidget(self.toolbar)
        self.verticalLayout_4.addWidget(self.canvas)

        # init graph
        self.init_graph()

        # # Mapping slot
        # self.button_layout = QHBoxLayout()
        # self.pushButton_5 = QPushButton()
        # self.button_layout.addWidget(self.pushButton_5)
        # self.pushButton_4 = QPushButton()
        # self.button_layout.addWidget(self.pushButton_4)
        # self.pushButton_3 = QPushButton()
        # self.button_layout.addWidget(self.pushButton_3)
        #
        # self.verticalLayout_4.addLayout(self.button_layout)
        # self.pushButton_5.clicked.connect(self.slotPB5)
        # self.pushButton_4.clicked.connect(self.slotPB4)
        # self.pushButton_3.clicked.connect(self.slotPB3)

    # bar 차트의 y tick의 format을 변경하기 위한 함수
    def reversalOfBarTick(self, x, pos):
        return '%d' % (self.barYAxisTickPivot - x)

    def init_graph(self):
        self.axial.cla()
        self.bar.cla()

        # self.axial.set_xlabel("Axial power profile", color=self.graphColor)
        # self.axial.set_ylabel("Core height(%)", color=self.graphColor)
        #
        # self.axial.set_xlim(0.0, 2.0)
        # self.axial.set_ylim(0, 100)

        self.axial.tick_params(axis='x', which='major', colors=self.whiteColor, grid_linewidth=1, grid_color=self.gridColor)
        self.axial.tick_params(axis='y', which='major', colors=self.whiteColor, grid_linewidth=1, grid_color=self.gridColor)
        #self.axial.set_ylim(0, 382)
        self.axial.set_ylim(0, 381)
        self.axial.set_xlim(0.0, 2.0)

        self.bar.set_ylim(self.barYAxisTickPivot, 0)
        self.bar.yaxis.tick_right()
        self.bar.xaxis.tick_top()
        self.bar.tick_params(axis='x', rotation=0)

        formatter = FuncFormatter(self.reversalOfBarTick)
        self.bar.yaxis.set_major_formatter(formatter)

        #self.bar.yaxis.set_major_locator(ticker.FixedLocator([0, 51, 101, 151, 201, 251, 301, 351, 382]))
        self.bar.yaxis.set_major_locator(ticker.FixedLocator([0, 31, 81, 131, 181, 231, 281, 331, 381]))
        self.clearAxial()


        axialHeight = [0]
        axialPower = [0]

        #self.drawAxial(axialPower, axialHeight)

        data = {' P': self.barYAxisTickPivot-1, 'R5': self.barYAxisTickPivot-1, 'R4': self.barYAxisTickPivot-1, 'R3': self.barYAxisTickPivot-1}
        self.drawBar(data)

        self.canvas.draw()

    def slotPB3(self):
        self.init_graph()

    def slotPB5(self):
        axialPower = [0.3422, 0.4775, 0.7518, 0.8785, 0.9692, 1.0539, 1.1206, 1.1502, 1.1642, 1.1686, 1.166, 1.159,
                      1.1473, 1.1311, 1.1122, 1.0912, 1.0663, 1.0367, 1.0042, 0.9656, 0.9033, 0.8325, 0.7556, 0.6386,
                      0.3964, 0.2662]

        axialHeight = [98.81, 96.81, 95.62, 93.62, 91, 87.38, 82.38, 77.62, 72.62, 67.38, 62.38, 57.62, 52.62, 47.38,
                       42.38, 37.62, 32.62, 27.38, 22.38, 17.62, 12.62, 9, 6.38, 4.38, 3.19, 1.19]

        self.drawAxial(axialPower, axialHeight)

    def clearAxial(self):
        self.axial.cla()
        self.axial.tick_params(axis='x', colors=self.whiteColor)
        self.axial.tick_params(axis='y', colors=self.whiteColor)

    def drawAxial(self, x, y):
        self.axial.cla()
        self.axial.plot(x, y, color='orange')

        #self.axial.fill_betweenx(y, min(x), x, facecolor='cornflowerblue')

        # set axial tick arrange
        x_tick_min = min(x)
        x_tick_max = max(x)
        #print("x_tick_min = " + str(x_tick_min) + "     x_tick_max" +str(x_tick_max))
        if x_tick_min == 0 and x_tick_max == 0:
            self.axial.set_xlim(0.0, 2.0)
        else:
            #self.axial.set_xlim(x_tick_min, x_tick_max * 1.1)
            x_tick_range = round(x_tick_max * 1.1,3)
            self.axial.set_xlim(0.0, x_tick_range)

        y_tick_min = min(y)
        y_tick_max = max(y)
        self.axial.set_ylim(y_tick_min, y_tick_max)

        self.axial.yaxis.grid(which='major', color=self.graphColor, linewidth=0.01, linestyle='-')
        # self.axial.yaxis.grid(which='minor', color=self.graphColor, linewidth=1, linestyle='--')
        self.axial.xaxis.grid(which='major', color=self.graphColor, linewidth=0.01, linestyle='--')
        # self.axial.xaxis.grid(which='minor', color=self.graphColor, linewidth=1, linestyle='--')
        self.axial.yaxis.set_major_locator(plt.MultipleLocator(20))
        self.axial.yaxis.set_minor_locator(AutoMinorLocator(2))
        self.axial.xaxis.set_major_locator(plt.MultipleLocator(0.4))
        self.axial.xaxis.set_minor_locator(AutoMinorLocator(2))

        self.axial.set_facecolor(self.graphBackgroundColor)

        # self.axial.yaxis.yticks([0,10,20,30,40,50,60,70,80,90,100])
        self.axial.tick_params(axis='x', which='major', colors=self.whiteColor, grid_linewidth=1, grid_color=self.gridColor)
        self.axial.tick_params(axis='y', which='major', colors=self.whiteColor, grid_linewidth=1, grid_color=self.gridColor)

        self.canvas.draw()


    def slotPB4(self):
        data = {' P': 100, '05': 80, '04': 60, '03': 0}
        self.drawBar(data)

    def drawBar(self, data):
        self.bar.cla()
        banks = list(data.keys())
        values = list(data.values())

        self.bar.set_ylim(self.barYAxisTickPivot, 0)
        self.bar.yaxis.tick_right()
        self.bar.xaxis.tick_top()
        self.bar.tick_params(axis='x', rotation=0, colors=self.whiteColor)
        self.bar.tick_params(axis='y', rotation=0, colors=self.whiteColor)

        formatter = FuncFormatter(self.reversalOfBarTick)
        self.bar.yaxis.set_major_formatter(formatter)

        #self.bar.yaxis.set_major_locator(ticker.FixedLocator([0, 51, 101, 151, 201, 251, 301, 351, 382]))
        self.bar.yaxis.set_major_locator(ticker.FixedLocator([0, 31, 81, 131, 181, 231, 281, 331, 381]))

        # bar의  시작점 y=0을 옯길 수 없음, 따라서 입력값의 101-x를 계산하여 전시함
        # tick 값 또한 101-x를 계산하여 전시
        for i in range(len(values)):
            values[i] = self.barYAxisTickPivot - values[i]

        self.bar.bar(banks, values, color='orange')

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