import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtCore import QSize, QObject, QCoreApplication
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QTableWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, \
    NavigationToolbar2QT as NavigationToolbar

import Definitions as df
import matplotlib.colors as mcol
import matplotlib.cm as cm
import widgets.utils.PyUnitButton as ts

class RadialWidget(QWidget):
    def __init__(self, frame, layout, core_type):
        super().__init__()

        _BLANK_ASM_ = " "
        #self.bClassRCS = [[_BLANK_ASM_ for col in range(df._OPR1000_XPOS_)] for row in range(df._OPR1000_YPOS_)]

        if(core_type==df.type_opr1000):
            self.bClassRCS = [[_BLANK_ASM_ for col in range(df._OPR1000_XPOS_)] for row in range(df._OPR1000_YPOS_)]
        elif (core_type == df.type_apr1400):
            self.bClassRCS = [[_BLANK_ASM_ for col in range(df._APR1400_XPOS_)] for row in range(df._ARP1400_YPOS_)]

        self.verticalLayout_6 = QGridLayout()
        self.setLayout(self.verticalLayout_6)

        # 그래프 색상 설정
        self.graphBackgroundColor = '#272c36'
        self.graphColor = '#d2d2d2'
        self.graphLine = '#1b1d23'
        self.gridColor = '#999999'

        self.core_type = core_type
        self.guide_box_pos = [[7,7],[8,8]]

        self.textColor_Black = '#000000'
        self.textColor_White = '#FFFFFF'

        self.funcType = eval("ts.outputButton")

        self.data = np.empty((8, 8), dtype=float)

        self.globalButtonInfo = {}
        self.settingLP(frame, layout)
        #self.data[self.data == 0.0] = np.nan

    def showAssemblyLoading(self):
        pass

    def swapAssembly(self):
        pass

    def settingLP(self, frame, gridLayout):
        _translate = QCoreApplication.translate

        #if(self.core_type==df)
        xID = df.OPR1000_xPos_Quart
        yID = df.OPR1000_yPos_Quart
        if(self.core_type==df.type_opr1000):
            xID = df.OPR1000_xPos_Quart
            yID = df.OPR1000_yPos_Quart
        elif(self.core_type==df.type_apr1400):
            xID = df.APR1400_xPos_Quart
            yID = df.APR1400_yPos_Quart

        for yPos in range(len(yID)):
            for xPos in range(len(xID)):
                flag = False
                if(self.core_type==df.type_opr1000):
                    flag = df.OPR1000MAP_RADIAL_QUART[xPos][yPos]
                elif (self.core_type == df.type_apr1400):
                    flag = df.APR1400MAP_RADIAL_QUART[xPos][yPos]

                #if(df.OPR1000MAP_RADIAL_QUART[xPos][yPos]==True):
                if(flag==True):
                    bName ="Core_%s%s" % (xID[xPos], yID[yPos])
                    self.globalButtonInfo[bName] = [df._POSITION_CORE_, 0, xPos, yPos]
                    # generate button geometry
                    buttonCore = self.funcType( bName, self.showAssemblyLoading,self.swapAssembly, frame) # type: QPushButton
                    buttonCore.setEnabled(True)
                    sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
                    sizePolicy.setHorizontalStretch(0)
                    sizePolicy.setVerticalStretch(0)
                    sizePolicy.setHeightForWidth(buttonCore.sizePolicy().hasHeightForWidth())
                    buttonCore.setSizePolicy(sizePolicy)
                    buttonCore.setMinimumSize(QSize(20, 20))
                    buttonCore.setMaximumSize(QSize(16777215, 16777215))
                    buttonCore.setBaseSize(QSize(60, 60))
                    #buttonCore.setText(df.OPR1000MAP_BP[xPos][yPos])
                    buttonCore.setIconSize(QSize(16, 16))
                    #buttonCore.setFlat(False)
                    buttonCore.setObjectName(bName)
                    buttonCore.setStyle(QStyleFactory.create("Windows"))
                    buttonCore.setDisabled(True)
                    #if xPos == 7 and yPos == 7:
                    if xPos == self.guide_box_pos[self.core_type][0] and yPos == self.guide_box_pos[self.core_type][1]:
                        # buttonCore.setText("Avg.Power \nFxy   \nFq   \nFr")
                        buttonCore.setText("Pow \nFxy")
                        if(self.core_type==df.type_opr1000):
                            gridLayout.addWidget(buttonCore, yPos+1, xPos+1, 1, 1)
                        elif(self.core_type==df.type_apr1400):
                            gridLayout.addWidget(buttonCore, yPos+1, xPos+1, 1, 1)
                        self.bClassRCS[xPos][yPos] = buttonCore
                        buttonCore.setStyleSheet( "background-color: #272c36;"
                                                  "border-radius = 0px;"
                                                  "color: white;"
                                                  "padding: 5px;"
                                                  "border:1px solid rgb(150,150,150);"
                                                  "Text-align:right;")
                    else:
                        style_sheet = "background-color: #272c36;" \
                                      "border-radius: 0px;" \
                                      "border-width: 0.5px;" \
                                      "border-style: solid;" \
                                      "color: white;"
                        if xPos == 0 and yPos == 0:
                            style_sheet += "border-color: rgb(150,150,150);"
                        elif xPos == 0:
                            style_sheet += "border-color: rgb({},{},{}) rgb(150,150,150) rgb(150,150,150) rgb(150,150,150);".format(
                                df.rgbSet[-3][0], df.rgbSet[-3][1], df.rgbSet[-3][2])
                        elif yPos == 0:
                            style_sheet += "border-color: rgb(150,150,150) rgb(150,150,150) rgb(150,150,150) rgb({},{},{});".format(
                                df.rgbSet[-3][0], df.rgbSet[-3][1], df.rgbSet[-3][2])
                        else:
                            style_sheet += "border-color: rgb({},{},{}) rgb(150,150,150) rgb(150,150,150) rgb({},{},{});".format(
                                df.rgbSet[-3][0], df.rgbSet[-3][1], df.rgbSet[-3][2], df.rgbSet[-3][0], df.rgbSet[-3][1], df.rgbSet[-3][1])

                        gridLayout.addWidget(buttonCore, yPos+1, xPos+1, 1, 1)
                        self.bClassRCS[xPos][yPos] = buttonCore
                        buttonCore.setStyleSheet(style_sheet)


    @staticmethod
    def colormap(predicted_row_col, vmin, vmax):
        # Make a normalizer that will map the time values from
        # [start_time,end_time+1] -> [0,1].
        if vmax-vmin == 0:
            vmax += 0.00001
        cnorm = mcol.Normalize(vmin=vmin, vmax=vmax)

        # Turn these into an object that can be used to map time values to colors and
        # can be passed to plt.colorbar().
        cpick = cm.ScalarMappable(norm=cnorm, cmap="coolwarm")
        r, g, b, a = cpick.to_rgba(predicted_row_col)
        return r, g, b
    #
    #
    # def slotPB_avg(self):
    #     self.data[:, :] = [
    #         [0.000, 0.000, 0.000, 0.000, 0.000, 0.100, 0.100, 0.100, 0.100, 0.100, 0.000, 0.000, 0.000, 0.000, 0.000],
    #         [0.000, 0.000, 0.000, 0.100, 0.100, 0.110, 0.110, 0.110, 0.110, 0.110, 0.100, 0.100, 0.000, 0.000, 0.000],
    #         [0.000, 0.000, 0.100, 0.110, 0.110, 0.120, 0.120, 0.120, 0.120, 0.120, 0.110, 0.110, 0.100, 0.000, 0.000],
    #         [0.000, 0.100, 0.110, 0.120, 0.120, 0.130, 0.130, 0.130, 0.130, 0.130, 0.120, 0.120, 0.110, 0.100, 0.000],
    #         [0.000, 0.100, 0.110, 0.120, 0.130, 0.140, 0.140, 0.140, 0.140, 0.140, 0.130, 0.120, 0.110, 0.100, 0.000],
    #         [0.100, 0.110, 0.120, 0.130, 0.140, 0.140, 0.150, 0.150, 0.150, 0.140, 0.140, 0.130, 0.120, 0.110, 0.100],
    #         [0.100, 0.110, 0.120, 0.130, 0.140, 0.150, 0.150, 0.160, 0.150, 0.150, 0.140, 0.130, 0.120, 0.110, 0.100],
    #         [0.100, 0.110, 0.120, 0.130, 0.140, 0.150, 0.160, 0.160, 0.160, 0.150, 0.140, 0.130, 0.120, 0.110, 0.100],
    #         [0.100, 0.110, 0.120, 0.130, 0.140, 0.150, 0.150, 0.160, 0.150, 0.150, 0.140, 0.130, 0.120, 0.110, 0.100],
    #         [0.100, 0.110, 0.120, 0.130, 0.140, 0.140, 0.150, 0.150, 0.150, 0.140, 0.140, 0.130, 0.120, 0.110, 0.100],
    #         [0.000, 0.100, 0.110, 0.120, 0.130, 0.140, 0.140, 0.140, 0.140, 0.140, 0.130, 0.120, 0.110, 0.100, 0.000],
    #         [0.000, 0.100, 0.110, 0.120, 0.120, 0.130, 0.130, 0.130, 0.130, 0.130, 0.120, 0.120, 0.110, 0.100, 0.000],
    #         [0.000, 0.000, 0.100, 0.110, 0.110, 0.120, 0.120, 0.120, 0.120, 0.120, 0.110, 0.110, 0.100, 0.000, 0.000],
    #         [0.000, 0.000, 0.000, 0.100, 0.100, 0.110, 0.110, 0.110, 0.110, 0.110, 0.100, 0.100, 0.000, 0.000, 0.000],
    #         [0.000, 0.000, 0.000, 0.000, 0.000, 0.100, 0.100, 0.100, 0.100, 0.100, 0.000, 0.000, 0.000, 0.000, 0.000]
    #     ]
    #
    #     #self.data[self.data == 0.0] = np.nan
    #     self.drawGraph(self.fig, self.canvas, self.data)
    #
    # def initGraph(self, fig, canvas):
    #
    #     #labael 정의
    #     xLabel = ["R", "P", "N", "M", "L", "K", "J", "H", "G", "F", "E", "D", "C", "B", "A", ]
    #     yLabel = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]
    #
    #     fig.clf()
    #     self.axe = fig.add_subplot(1, 1, 1)
    #     self.axe.set_facecolor(self.graphBackgroundColor)
    #
    #     self.axe.spines['top'].set_color(self.gridColor)
    #     self.axe.spines['left'].set_color(self.gridColor)
    #     self.axe.spines['right'].set_color(self.gridColor)
    #     self.axe.spines['bottom'].set_color(self.gridColor)
    #
    # def drawGraph(self, fig, canvas, data):
    #
    #     xLabel = ["R", "P", "N", "M", "L", "K", "J", "H", "G", "F", "E", "D", "C", "B", "A", ]
    #     yLabel = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]
    #     #Figure 초기화
    #     fig.clf()
    #     axe = fig.add_subplot(1, 1, 1)
    #     axe.set_facecolor(self.graphBackgroundColor)
    #
    #
    #     #axe.set_title('TEST', color=self.gridColor)
    #
    #     # Setting colorbar and colorbar style
    #     im = axe.imshow(data, cmap=plt.cm.plasma)#, colors = self.gridColor)
    #     # Setting tick and ticklabel color
    #     im.axes.tick_params(color=self.gridColor, labelcolor=self.gridColor)
    #
    #     cb = fig.colorbar(im)
    #
    #     # set imshow outline
    #     for spline in im.axes.spines.values():
    #         spline.set_color(self.gridColor)
    #     # set colorbar label and label color
    #     #cb.set_label('TTT',color=self.gridColor)
    #     # set colorbar label tick color
    #     cb.ax.yaxis.set_tick_params(color=self.gridColor)
    #     # set colorbar edgecolor
    #     cb.outline.set_edgecolor(color=self.gridColor)
    #     #axe.imshow['top'].set_color(self.gridColor)
    #
    #
    #     # self.fig.xlabel('aaa',fontsize=10)
    #     # set colorbar ticklabel and color
    #     plt.setp(plt.getp(cb.ax.axes,'yticklabels'),color=self.gridColor)
    #
    #
    #     # We want to show all ticks...
    #     axe.set_xticks(np.arange(len(xLabel)))
    #     axe.set_yticks(np.arange(len(yLabel)))
    #     # ... and label them with the respective list entries
    #     axe.set_xticklabels(xLabel)#, minor=True)
    #     axe.set_yticklabels(yLabel)#, minor=True)
    #     #axe.grid(color="#CCCCCC",linestyle='-',linewidth=1)
    #     axe.tick_params(axis='x', colors=self.graphColor)#,which="minor")
    #     axe.tick_params(axis='y', colors=self.graphColor)
    #
    #     # axe.xlabel('TEST', fontsize=12)
    #     axe.xaxis.tick_top()
    #     # x,y = np.meshgrid(np.linespace(0,1,11),np.linespace(0,0.6,7))
    #     #xx,yy = np.meshgrid(np.arange(15),np.arange(15))
    #     #z = self.data[xx,yy]#(xx+1) * (yy+1)
    #     # mesh = np.meshgrid(xx, yy)#, shading='auto', alpha=0.5)
    #     #plt.scatter(xx,yy)
    #     #fig.colorbar(mesh)
    #     #fig.colorbar
    #
    #     #radal 그래프 x, y축 가이드 line
    #     # axe.annotate('', xy=(0, 4.5), xytext=(0, -0.5), arrowprops=dict(color=self.graphColor, arrowstyle='-'))
    #     # axe.annotate('', xy=(1, 2.5), xytext=(1, -0.5), arrowprops=dict(color=self.graphColor, arrowstyle='-'))
    #     # axe.annotate('', xy=(2, 1.5), xytext=(2, -0.5), arrowprops=dict(color=self.graphColor, arrowstyle='-'))
    #     # axe.annotate('', xy=(2.5, 0.5), xytext=(2.5, 13.5), arrowprops=dict(color=self.gridColor, arrowstyle='-'))
    #     # axe.annotate('', xy=(4, 0.5), xytext=(4, -0.5), arrowprops=dict(color=self.graphColor, arrowstyle='-'))
    #     # axe.annotate('', xy=(10, 0.5), xytext=(10, -0.5), arrowprops=dict(color=self.graphColor, arrowstyle='-'))
    #     # axe.annotate('', xy=(11, 0.5), xytext=(11, -0.5), arrowprops=dict(color=self.graphColor, arrowstyle='-'))
    #     # axe.annotate('', xy=(12, 1.5), xytext=(12, -0.5), arrowprops=dict(color=self.graphColor, arrowstyle='-'))
    #     # axe.annotate('', xy=(13, 2.5), xytext=(13, -0.5), arrowprops=dict(color=self.graphColor, arrowstyle='-'))
    #     # axe.annotate('', xy=(14, 4.5), xytext=(14, -0.5), arrowprops=dict(color=self.graphColor, arrowstyle='-'))
    #     #
    #     # axe.annotate('', xy=(4.5, 0), xytext=(-0.5, 0), arrowprops=dict(color=self.graphColor, arrowstyle='-'))
    #     # axe.annotate('', xy=(2.5, 1), xytext=(-0.5, 1), arrowprops=dict(color=self.graphColor, arrowstyle='-'))
    #     # axe.annotate('', xy=(1.5, 2), xytext=(-0.5, 2), arrowprops=dict(color=self.graphColor, arrowstyle='-'))
    #     # axe.annotate('', xy=(0.5, 3), xytext=(-0.5, 3), arrowprops=dict(color=self.graphColor, arrowstyle='-'))
    #     # axe.annotate('', xy=(0.5, 4), xytext=(-0.5, 4), arrowprops=dict(color=self.graphColor, arrowstyle='-'))
    #     # axe.annotate('', xy=(0.5, 10), xytext=(-0.5, 10), arrowprops=dict(color=self.graphColor, arrowstyle='-'))
    #     # axe.annotate('', xy=(0.5, 11), xytext=(-0.5, 11), arrowprops=dict(color=self.graphColor, arrowstyle='-'))
    #     # axe.annotate('', xy=(1.5, 12), xytext=(-0.5, 12), arrowprops=dict(color=self.graphColor, arrowstyle='-'))
    #     # axe.annotate('', xy=(2.5, 13), xytext=(-0.5, 13), arrowprops=dict(color=self.graphColor, arrowstyle='-'))
    #     # axe.annotate('', xy=(4.5, 14), xytext=(-0.5, 14), arrowprops=dict(color=self.graphColor, arrowstyle='-'))
    #
    #     # GridLine Maker
    #     #axe.annotate('', xy=( 2.5, 0.5), xytext=(2.5, 13.5), arrowprops=dict(color=self.gridColor, arrowstyle='-'))
    #     axe.annotate('', xy=( -0.46,  4.45), xytext=( -0.46,   9.55), arrowprops=dict(color=self.gridColor, arrowstyle='-'))
    #     axe.annotate('', xy=(  0.5 ,  2.45), xytext=(  0.5 ,  11.55), arrowprops=dict(color=self.gridColor, arrowstyle='-'))
    #     axe.annotate('', xy=(  1.5 ,  1.45), xytext=(  1.5 ,  12.55), arrowprops=dict(color=self.gridColor, arrowstyle='-'))
    #     axe.annotate('', xy=(  2.5 ,  0.45), xytext=(  2.5 ,  13.55), arrowprops=dict(color=self.gridColor, arrowstyle='-'))
    #     axe.annotate('', xy=(  3.5 ,  0.45), xytext=(  3.5 ,  13.55), arrowprops=dict(color=self.gridColor, arrowstyle='-'))
    #     axe.annotate('', xy=(  4.5 , -0.5 ), xytext=(  4.5 ,  14.55), arrowprops=dict(color=self.gridColor, arrowstyle='-'))
    #     axe.annotate('', xy=(  5.5 , -0.5 ), xytext=(  5.5 ,  14.55), arrowprops=dict(color=self.gridColor, arrowstyle='-'))
    #     axe.annotate('', xy=(  6.5 , -0.5 ), xytext=(  6.5 ,  14.55), arrowprops=dict(color=self.gridColor, arrowstyle='-'))
    #     axe.annotate('', xy=(  7.5 , -0.5 ), xytext=(  7.5 ,  14.55), arrowprops=dict(color=self.gridColor, arrowstyle='-'))
    #     axe.annotate('', xy=(  8.5 , -0.5 ), xytext=(  8.5 ,  14.55), arrowprops=dict(color=self.gridColor, arrowstyle='-'))
    #     axe.annotate('', xy=(  9.5 , -0.5 ), xytext=(  9.5 ,  14.55), arrowprops=dict(color=self.gridColor, arrowstyle='-'))
    #     axe.annotate('', xy=( 10.5 ,  0.45), xytext=( 10.5 ,  13.55), arrowprops=dict(color=self.gridColor, arrowstyle='-'))
    #     axe.annotate('', xy=( 11.5 ,  0.45), xytext=( 11.5 ,  13.55), arrowprops=dict(color=self.gridColor, arrowstyle='-'))
    #     axe.annotate('', xy=( 12.5 ,  1.45), xytext=( 12.5 ,  12.55), arrowprops=dict(color=self.gridColor, arrowstyle='-'))
    #     axe.annotate('', xy=( 13.5 ,  2.45), xytext=( 13.5 ,  11.55), arrowprops=dict(color=self.gridColor, arrowstyle='-'))
    #     axe.annotate('', xy=( 14.49,  4.45), xytext=( 14.49,   9.55), arrowprops=dict(color=self.gridColor, arrowstyle='-'))
    #
    #
    #     axe.annotate('', xy=(  4.45, -0.46), xytext=(  9.55, -0.46), arrowprops=dict(color=self.gridColor, arrowstyle='-'))
    #     axe.annotate('', xy=(  2.45,  0.5 ), xytext=( 11.55,  0.5 ), arrowprops=dict(color=self.gridColor, arrowstyle='-'))
    #     axe.annotate('', xy=(  1.45,  1.5 ), xytext=( 12.55,  1.5 ), arrowprops=dict(color=self.gridColor, arrowstyle='-'))
    #     axe.annotate('', xy=(  0.45,  2.5 ), xytext=( 13.55,  2.5 ), arrowprops=dict(color=self.gridColor, arrowstyle='-'))
    #     axe.annotate('', xy=(  0.45,  3.5 ), xytext=( 13.55,  3.5 ), arrowprops=dict(color=self.gridColor, arrowstyle='-'))
    #     axe.annotate('', xy=( -0.50,  4.5 ), xytext=( 14.55,  4.5 ), arrowprops=dict(color=self.gridColor, arrowstyle='-'))
    #     axe.annotate('', xy=( -0.50,  5.5 ), xytext=( 14.55,  5.5 ), arrowprops=dict(color=self.gridColor, arrowstyle='-'))
    #     axe.annotate('', xy=( -0.50,  6.5 ), xytext=( 14.55,  6.5 ), arrowprops=dict(color=self.gridColor, arrowstyle='-'))
    #     axe.annotate('', xy=( -0.50,  7.5 ), xytext=( 14.55,  7.5 ), arrowprops=dict(color=self.gridColor, arrowstyle='-'))
    #     axe.annotate('', xy=( -0.50,  8.5 ), xytext=( 14.55,  8.5 ), arrowprops=dict(color=self.gridColor, arrowstyle='-'))
    #     axe.annotate('', xy=( -0.50,  9.5 ), xytext=( 14.55,  9.5 ), arrowprops=dict(color=self.gridColor, arrowstyle='-'))
    #     axe.annotate('', xy=(  0.45, 10.5 ), xytext=( 13.55, 10.5 ), arrowprops=dict(color=self.gridColor, arrowstyle='-'))
    #     axe.annotate('', xy=(  0.45, 11.5 ), xytext=( 13.55, 11.5 ), arrowprops=dict(color=self.gridColor, arrowstyle='-'))
    #     axe.annotate('', xy=(  1.45, 12.5 ), xytext=( 12.55, 12.5 ), arrowprops=dict(color=self.gridColor, arrowstyle='-'))
    #     axe.annotate('', xy=(  2.45, 13.5 ), xytext=( 11.55, 13.5 ), arrowprops=dict(color=self.gridColor, arrowstyle='-'))
    #     axe.annotate('', xy=(  4.45, 14.49), xytext=(  9.55, 14.49), arrowprops=dict(color=self.gridColor, arrowstyle='-'))
    #
    #
    #     # Rotate the tick labels and set their alignment.
    #     # plt.setp(self.radial.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    #
    #     # Loop over data dimensions and create text annotations.
    #
    #     for i in range(15):
    #         for j in range(15):
    #             if not np.isnan(self.data[i, j]):
    #                 if(self.data[i,j]>=1.000):
    #                     unitColor = self.textColor_Black
    #                 else:
    #                     unitColor = self.textColor_White
    #                 text = axe.text(j, i, '{:.3f}'.format(self.data[i, j]), ha="center", va="center", color=unitColor, fontsize=7)
    #     fig.tight_layout()
    #     canvas.draw()
    #
    def drawGraphB(self, data):
        min_value = np.min(data[data!=0.0])
        #min_value = np.min(data)
        max_value = np.max(data)
        x_length = len(df.OPR1000_xPos_Quart)
        y_length = len(df.OPR1000_xPos_Quart)
        if(self.core_type==df.type_opr1000):
            x_length = len(df.OPR1000_xPos_Quart)
            y_length = len(df.OPR1000_xPos_Quart)
        elif(self.core_type==df.type_apr1400):
            pass
            # x_length = len(df.APR1400_xPos_Quart)
            # y_length = len(df.APR1400_yPos_Quart)

        #for xPos in range(len(df.OPR1000_xPos_Quart)):
        for xPos in range(x_length):
            #for yPos in range(len(df.OPR1000_yPos_Quart)):
            for yPos in range(y_length):

                if xPos == self.guide_box_pos[self.core_type][0] and yPos == self.guide_box_pos[self.core_type][1]:
                    continue
                if(data[xPos, yPos]!=0.0):
                    r, g, b = self.colormap(data[xPos, yPos], min_value, max_value)
                else:
                    continue
                #
                # if (self.data[xPos, yPos] >= 1.000):
                #     unitColor = self.textColor_Black
                # else:
                #     unitColor = self.textColor_White
                # text = axe.text(j, i, '{:.3f}'.format(self.data[i, j]), ha="center", va="center", color=unitColor,
                #                 fontsize=7)
                if(data[xPos,yPos]>=1.0):
                    r_t, g_t, b_t = (255, 255, 255)
                else:
                    r_t, g_t, b_t = (0, 0, 0)
                #if df.OPR1000MAP_RADIAL_QUART[xPos][yPos]:
                if(self.core_type==df.type_opr1000):
                    flag = df.OPR1000MAP_RADIAL_QUART[xPos][yPos]
                elif(self.core_type==df.type_apr1400):
                    flag = df.APR1400MAP_RADIAL_QUART[xPos][yPos]
                if flag==True:
                    #print(r, g, b)
                    style_sheet = "background-color: rgb({},{},{});"\
                                    "border-radius: 0px;"\
                                    "border-width: 0.5px;"\
                                    "border-style: solid;"\
                                    "color: rgb({},{},{});".format(
                                        int(r * 255), int(g * 255), int(b * 255)
                                        , r_t, g_t, b_t)

                    if xPos == 0 and yPos == 0:
                        style_sheet += "border-color: rgb(150,150,150);"
                    elif xPos == 0:
                        style_sheet += "border-color: rgb({},{},{}) rgb(150,150,150) rgb(150,150,150) rgb(150,150,150);".format(
                                        int(r * 255), int(g * 255), int(b * 255))
                    elif yPos == 0:
                        style_sheet += "border-color: rgb(150,150,150) rgb(150,150,150) rgb(150,150,150) rgb({},{},{});".format(
                                        int(r * 255), int(g * 255), int(b * 255))
                    else:
                        style_sheet += "border-color: rgb({},{},{}) rgb(150,150,150) rgb(150,150,150) rgb({},{},{});".format(
                                        int(r * 255), int(g * 255), int(b * 255), int(r * 255), int(g * 255), int(b * 255))

                    self.bClassRCS[xPos][yPos].setStyleSheet(style_sheet)
                    self.bClassRCS[xPos][yPos].setText("{:.3f}\n{:.3f}".format(data[xPos, yPos], data[xPos, yPos]))

    def slotPB_fdh(self):
        self.data[:, :] = [
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.365, 0.409, 0.392, 0.412, 0.363, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.627, 1.119, 0.915, 1.047, 0.965, 1.042, 0.920, 1.113, 0.631, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.595, 1.143, 1.316, 1.388, 1.261, 1.492, 1.257, 1.380, 1.326, 1.138, 0.595, 0.000, 0.000],
            [0.000, 0.631, 1.138, 1.135, 1.326, 1.576, 1.249, 1.540, 1.233, 1.585, 1.323, 1.135, 1.143, 0.627, 0.000],
            [0.000, 1.113, 1.326, 1.323, 1.437, 1.627, 1.896, 1.790, 1.896, 1.657, 1.437, 1.326, 1.316, 1.119, 0.000],
            [0.363, 0.920, 1.380, 1.585, 1.657, 1.753, 1.592, 1.641, 1.617, 1.753, 1.627, 1.576, 1.388, 0.915, 0.365],
            [0.412, 1.042, 1.257, 1.233, 1.896, 1.617, 1.583, 1.964, 1.583, 1.592, 1.896, 1.249, 1.261, 1.047, 0.409],
            [0.392, 0.965, 1.492, 1.540, 1.790, 1.641, 1.964, 1.563, 1.964, 1.641, 1.790, 1.540, 1.492, 0.965, 0.392],
            [0.409, 1.047, 1.261, 1.249, 1.896, 1.592, 1.583, 1.964, 1.583, 1.617, 1.896, 1.233, 1.257, 1.042, 0.412],
            [0.365, 0.915, 1.388, 1.576, 1.627, 1.753, 1.617, 1.641, 1.592, 1.753, 1.657, 1.585, 1.380, 0.920, 0.363],
            [0.000, 1.119, 1.316, 1.326, 1.437, 1.657, 1.896, 1.790, 1.896, 1.627, 1.437, 1.323, 1.326, 1.113, 0.000],
            [0.000, 0.627, 1.143, 1.135, 1.323, 1.585, 1.233, 1.540, 1.249, 1.576, 1.326, 1.135, 1.138, 0.631, 0.000],
            [0.000, 0.000, 0.595, 1.138, 1.326, 1.380, 1.257, 1.492, 1.261, 1.388, 1.316, 1.143, 0.595, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.631, 1.113, 0.920, 1.042, 0.965, 1.047, 0.915, 1.119, 0.627, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.363, 0.412, 0.392, 0.409, 0.365, 0.000, 0.000, 0.000, 0.000, 0.000]]

        # self.data[self.data == 0.0] = np.nan
        self.drawGraph(self.fig, self.canvas, self.data)
        #print("clicked~~~!!!!")

    def slot_astra_data(self, data):
        data = np.array(data)
        self.data[:, :] = data[-9:-1, -9:-1]
        self.drawGraphB(self.data)

    def clear_data(self):
        self.data[:, :] = [
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        ]

        self.drawGraphB(self.data)
