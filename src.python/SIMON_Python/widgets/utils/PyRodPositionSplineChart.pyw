# import sys
# import platform
# from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import (QCoreApplication, QPropertyAnimation, QDate, QDateTime, QMetaObject, QObject, QPoint, QRectF, QSize, QTime, QUrl, Qt, QEvent)
# from PyQt5.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont, QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter, QPixmap, QRadialGradient)
from PyQt5.QtGui import QBrush, QColor, QFont, QPainter, QPen
# from PyQt5 import QtWidgets
# from PyQt5.QtWidgets import QApplication, QMainWindow
# import sys

import PyQt5.QtChart as QtCharts
from PyQt5.QtCore import Qt, QPointF
# from PyQt5.QtWidgets import *
# from model import *
# import datetime
# import os
# import glob
# from ui_main_rev9 import Ui_MainWindow
import Definitions as df


class UnitSplineChart:

    def __init__(self, unit):
        #background - color: rgb(51, 58, 72);
        self.set0 = QtCharts.QBoxSet("Rod Position")
        self.series01 = QtCharts.QLineSeries()
        self.series02 = QtCharts.QLineSeries()
        self.series03 = QtCharts.QLineSeries()
        self.series04 = QtCharts.QLineSeries()
        self.seriesCBC = QtCharts.QLineSeries()
        self.seriesASIBand = QtCharts.QLineSeries()
        self.seriesASIBandB = QtCharts.QLineSeries()

        # self.series01 = QtCharts.QBarSet("Rod Position")
        self.chart = QtCharts.QChart()
        # self.axisX = QtCharts.QBarCategoryAxis()
        self.axisX = QtCharts.QValueAxis()
        self.axisY = QtCharts.QValueAxis()
        self.axisY_CBC = QtCharts.QValueAxis()
        self.chartView = QtCharts.QChartView()

        #self.chartView.setContentsMargins(-10,-10,-10,-10)
        self.chart.setContentsMargins(-25,-25,-25,-25)


        self.font_Title = QFont()
        self.font_AxisX_Title = QFont()
        self.font_AxisX_Label = QFont()
        self.font_AxisY_Title = QFont()
        self.font_AxisY_Label = QFont()

        self.axisBrush = QBrush(Qt.white)

        self.chart.addSeries(self.series01)
        self.chart.addSeries(self.series02)
        self.chart.addSeries(self.series03)
        self.chart.addSeries(self.series04)
        self.chart.addSeries(self.seriesCBC)
        self.chart.addSeries(self.seriesASIBand)
        self.chart.addSeries(self.seriesASIBandB)

        #self.chart.setPlotAreaBackgroundBrush(QBrush(QColor(151, 158, 172)))
        #self.chartView.back
        self.chart.setPlotAreaBackgroundVisible(True)
        self.chart.setBackgroundBrush(QBrush(QColor(151, 158, 172)))
        self.chartView.setBackgroundBrush(QBrush(QColor(151, 158, 172)))
        self.chartView.chart().setBackgroundBrush(QBrush(QColor(151, 158, 172)))

        self.setChartFont()
        self.setBrushStyle()

        self.create_bar()

        self.series01.attachAxis(self.axisX)
        self.series01.attachAxis(self.axisY)
        self.series02.attachAxis(self.axisX)
        self.series02.attachAxis(self.axisY)
        self.series03.attachAxis(self.axisX)
        self.series03.attachAxis(self.axisY)
        self.series04.attachAxis(self.axisX)
        self.series04.attachAxis(self.axisY)

        self.seriesCBC.attachAxis(self.axisY_CBC)
        self.seriesCBC.attachAxis(self.axisX)

        self.seriesASIBand.attachAxis(self.axisY_CBC)
        self.seriesASIBand.attachAxis(self.axisX)

        self.seriesASIBandB.attachAxis(self.axisY_CBC)
        self.seriesASIBandB.attachAxis(self.axisX)

        #self.series01.clicked['QPointF'].connect(self.test0001)

        #self.pSeries01 = self.series01.points()
        #self.pSeries02 = self.series02.points()
        #self.pSeries03 = self.series03.points()

        # self.replaceRodPosition(3,list0,listn)

        # self.replaceRodPosition(1,list0,list)

    def setXPoints(self, x_20, x_100):
        self.x_20 = x_20
        self.x_100 = x_100

    def drawASIBand(self, x_20, x_100, band_20_up, band_20_down):
        self.seriesASIBand.clear()
        self.seriesASIBandB.clear()
        self.seriesASIBand.append(QPointF(0, band_20_down))
        self.seriesASIBand.append(QPointF(x_20, band_20_down))
        self.seriesASIBand.append(QPointF(x_20, band_20_up))
        self.seriesASIBand.append(QPointF(x_100, band_20_up))
        self.seriesASIBandB.append(QPointF(0, -1*band_20_down))
        self.seriesASIBandB.append(QPointF(x_20, -1*band_20_down))
        self.seriesASIBandB.append(QPointF(x_20, -1*band_20_up))
        self.seriesASIBandB.append(QPointF(x_100, -1*band_20_up))


    def setChartFont(self):
        self.font_Title.setBold(True)
        self.font_Title.setPixelSize(24)
        self.font_Title.setFamily("Segoe UI")

        self.font_AxisX_Title.setBold(True)
        self.font_AxisX_Title.setPixelSize(15)
        self.font_AxisX_Title.setFamily("Segoe UI")

        self.font_AxisX_Label.setPixelSize(12)
        self.font_AxisX_Label.setFamily("Segoe UI")

        self.font_AxisY_Title.setBold(True)
        self.font_AxisY_Title.setPixelSize(15)
        self.font_AxisY_Title.setFamily("Segoe UI")

        self.font_AxisY_Label.setPixelSize(12)
        self.font_AxisY_Label.setFamily("Segoe UI")

    def setBrushStyle(self):
        pen = QPen()
        pen.setWidth(3)
        self.series01.setPen(pen)
        self.series01.setColor(QColor(df.RGB_ORANGE[0], df.RGB_ORANGE[1], df.RGB_ORANGE[2]))
        self.series01.setPointsVisible(True)
        self.series01.setName('P')

        pen2 = QPen()
        pen2.setWidth(3)
        self.series02.setPen(pen2)
        self.series02.setColor(QColor(df.RGB_GREEN[0], df.RGB_GREEN[1], df.RGB_GREEN[2]))
        self.series02.setPointsVisible(True)
        self.series02.setName('R5')

        pen3 = QPen()
        pen3.setWidth(3)
        self.series03.setPen(pen3)
        self.series03.setColor(QColor(df.RGB_BLUE[0], df.RGB_BLUE[1], df.RGB_BLUE[2]))
        self.series03.setPointsVisible(True)
        self.series03.setName('R4')

        pen4 = QPen()
        pen4.setWidth(3)
        self.series04.setPen(pen3)
        self.series04.setColor(QColor(df.RGB_GREY01[0], df.RGB_GREY01[1], df.RGB_GREY01[2]))
        self.series04.setPointsVisible(True)
        self.series04.setName('R3')

        pen4 = QPen()
        pen4.setWidth(3)
        pen4.setStyle(Qt.DotLine)
        self.seriesCBC.setPen(pen4)
        self.seriesCBC.setPointsVisible(True)
        self.seriesCBC.setColor(QColor(df.RGB_PURPLE[0], df.RGB_PURPLE[1], df.RGB_PURPLE[2]))
        self.seriesCBC.setName('ASI')

        pen4 = QPen()
        pen4.setWidth(3)
        pen4.setStyle(Qt.DotLine)
        self.seriesASIBand.setPen(pen4)
        self.seriesASIBand.setPointsVisible(True)
        self.seriesASIBand.setColor(QColor(df.RGB_RED[0], df.RGB_RED[1], df.RGB_RED[2]))
        self.seriesASIBand.setName('ASI Band top')

        pen4 = QPen()
        pen4.setWidth(3)
        pen4.setStyle(Qt.DotLine)
        self.seriesASIBandB.setPen(pen4)
        self.seriesASIBandB.setPointsVisible(True)
        self.seriesASIBandB.setColor(QColor(df.RGB_RED[0], df.RGB_RED[1], df.RGB_RED[2]))
        self.seriesASIBandB.setName('ASI Band bottom')

    #    def linkStyle(self):
    #        self.chart.setTitleFont(self.font_Title)

    def create_bar(self):
        # The QBarSet class represents a set of bars in the bar chart.
        # It groups several bars into a bar set

        # self.series02.setPointLabelsVisible(True)
        # self.series02.setPointLabelsColor(Qt.white)
        # self.series02.setPointLabelsClipping(False)
        # self.series02.setPointLabelsFormat("(@xPoint,@yPoint)")

        # Define Chart as 'chart'
        # self.chart.addSeries(self.series01)
        # self.chart.addSeries(self.series02)
        #self.chart.setTitle("Rod Position and ASI")
        #self.chart.setTitleFont(self.font_Title)
        #self.chart.setAnimationOptions(QtCharts.QChart.SeriesAnimations)

        # categories = ["Bank 5", "Bank 4", "Bank 3", "Bank P"]
        self.axisX.setTitleFont(self.font_AxisX_Title)
        self.axisX.setTitleBrush(self.axisBrush)
        self.axisX.setLabelsFont(self.font_AxisX_Label)
        self.axisX.setLabelFormat("%.2f")
        self.axisX.setLabelsBrush(self.axisBrush)
        self.axisX.setTitleText("Time After Power Change(Hour)")
        self.axisX.setTickType(QtCharts.QValueAxis.TicksFixed)
        self.axisX.setMax(35.0)
        self.axisX.setMin(0.0)
        self.axisX.setTickCount(8)
        self.axisX.setTickInterval(1.0)
        #self.axisX.setC
        self.chart.addAxis(self.axisX, Qt.AlignBottom)
        # self.series01.attachAxis(self.axisX)
        # self.series02.attachAxis(self.axisX)
        # self.series03.attachAxis(self.axisX)

        self.axisY.setMin(0)
        # self.axisY.setReverse(True)
        self.axisY.setTitleFont(self.font_AxisY_Title)
        self.axisY.setTitleBrush(self.axisBrush)
        self.axisY.setLabelsFont(self.font_AxisY_Label)
        self.axisY.setLabelFormat("%.2f")

        self.axisY.setLabelsBrush(self.axisBrush)
        self.axisY.setMax(400.0)
        self.axisY.setTitleText("Rod Position(cm)")
        self.axisY.setTickType(QtCharts.QValueAxis.TicksDynamic)
        # self.axisY.append(0.0)
        self.axisY.setTickInterval(100.0)
        self.axisY.setTickAnchor(0.0)
        self.axisY.setMinorTickCount(4)
        self.chart.addAxis(self.axisY, Qt.AlignLeft)
        # self.series01.attachAxis(self.axisY)
        # self.series02.attachAxis(self.axisY)
        # self.series03.attachAxis(self.axisY)

        self.axisY_CBC.setMin(-1.0)
        # self.axisY.setReverse(True)
        self.axisY_CBC.setTitleFont(self.font_AxisY_Title)
        self.axisY_CBC.setTitleBrush(self.axisBrush)
        self.axisY_CBC.setLabelsFont(self.font_AxisY_Label)
        self.axisY_CBC.setLabelFormat("%.3f")

        self.axisY_CBC.setLabelsBrush(self.axisBrush)
        self.axisY_CBC.setMax(1.0)
        self.axisY_CBC.setTitleText("ASI")
        self.axisY_CBC.setTickType(QtCharts.QValueAxis.TicksDynamic)
        # self.axisY.append(0.0)
        self.axisY_CBC.setTickInterval(0.5)
        # self.axisY_CBC.setTickAnchor(0.0)
        # self.axisY_CBC.setMinorTickCount(4)
        self.chart.addAxis(self.axisY_CBC, Qt.AlignRight)

        # self.series02.attachAxis(self.axisY)

        # if(unit==df.RodPosUnit_percent):
        #     self.axisY.setMax(100.0)
        #     self.axisY.setTitleText("Unit : %")
        #     self.axisY.setTickType(QtCharts.QValueAxis.TicksFixed)
        #     self.axisY.setTickCount(11)
        #     self.axisY.setTickInterval(10.0)
        #     self.axisY.setMinorTickCount(4)
        # elif(unit==df.RodPosUnit_cm):
        #     self.axisY.setMax(381.0)
        #     self.axisY.setTitleText("Unit : cm")
        #     self.axisY.setTickType(QtCharts.QValueAxis.TicksDynamic)
        #     #self.axisY.append(0.0)
        #     self.axisY.setTickInterval(50.0)
        #     self.axisY.setTickAnchor(0.0)
        #     self.axisY.setMinorTickCount(4)
        # elif(unit==df.RodPosUnit_inch):
        #     self.axisY.setMax(150.0)
        #     self.axisY.setTitleText("Unit : inch")
        #     self.axisY.setTickType(QtCharts.QValueAxis.TicksFixed)
        #     self.axisY.setTickCount(16)
        #     self.axisY.setTickInterval(10.0)
        #     self.axisY.setMinorTickCount(4)

        self.chart.legend().setVisible(True)
        self.chart.legend().setLabelBrush(self.axisBrush)
        self.chart.legend().setAlignment(Qt.AlignTop)
        #self.chart.legend().detachFromChart()
        size = self.chartView.size()
        #size = self.chart.size()
        chartXLen = (size.width() *0.7)
        chartYLen = (size.height()*0.9)
        legendX = 200
        #legendX   = int(size.width() * 0.1)
        tmp = self.chart.pos()
        a = self.chart.x()
        b = self.chart.y()
        #print(chartXLen,chartYLen)
        #print(a,b)
        self.chart.legend().setReverseMarkers(False)
        #self.chart.legend().setGeometry(QRectF(chartXLen,chartYLen,legendX,0))
        # self.chart.legend().setGeometry(QRectF(50,0,legendX,0))
        self.chart.legend().setContentsMargins(0,0,0,0)

        # chart.setPlotArea(a)
        self.chart.setTitleBrush(self.axisBrush)
        self.chart.setBackgroundBrush(QBrush(QColor("transparent")))#55, 58, 69)))

        self.chartView = QtCharts.QChartView(self.chart)
        self.chartView.setRenderHint(QPainter.Antialiasing)

    def appendRodPosition(self, nPos, rodPos, posCBC):

        if (nPos == 1):
            self.series01.append(rodPos[0])

        elif (nPos == 2):
            self.series01.append(rodPos[0])
            self.series02.append(rodPos[1])

        elif (nPos == 3):
            self.series01.append(rodPos[0])
            self.series02.append(rodPos[1])
            self.series03.append(rodPos[2])
        elif (nPos == 4):
            self.series01.append(rodPos[0])
            self.series02.append(rodPos[1])
            self.series03.append(rodPos[2])
            self.series04.append(rodPos[3])

        self.seriesCBC.append(posCBC)


    def clear(self):
        self.series01.clear()
        self.series02.clear()
        self.series03.clear()
        self.series04.clear()
        self.seriesCBC.clear()

    def replaceRodPosition(self, nPos, rodPos, posCBC):
        self.clear()
        unitAxis = self.series01.attachedAxes()
        if (len(unitAxis) != 0):
            self.series01.detachAxis(self.axisY)
            self.series01.detachAxis(self.axisX)
        unitAxis = self.series02.attachedAxes()
        if (len(unitAxis) != 0):
            self.series02.detachAxis(self.axisY)
            self.series02.detachAxis(self.axisX)
        unitAxis = self.series03.attachedAxes()
        if (len(unitAxis) != 0):
            self.series03.detachAxis(self.axisY)
            self.series03.detachAxis(self.axisX)
        unitAxis = self.series04.attachedAxes()
        if (len(unitAxis) != 0):
            self.series04.detachAxis(self.axisY)
            self.series04.detachAxis(self.axisX)
        unitAxis = self.seriesCBC.attachedAxes()
        if (len(unitAxis) != 0):
            self.seriesCBC.detachAxis(self.axisY_CBC)
            self.seriesCBC.detachAxis(self.axisX)
        # self.chart.removeAllSeries()

        if (nPos == 1):
            self.series01.append(rodPos[0])
            # self.chart.addSeries(self.series01)
            self.series01.attachAxis(self.axisX)
            self.series01.attachAxis(self.axisY)

        elif (nPos == 2):
            self.series01.append(rodPos[0])
            self.series02.append(rodPos[1])
            # self.chart.addSeries(self.series01)
            # self.chart.addSeries(self.series02)
            self.series01.attachAxis(self.axisX)
            self.series01.attachAxis(self.axisY)
            self.series02.attachAxis(self.axisX)
            self.series02.attachAxis(self.axisY)

        elif (nPos == 3):
            self.series01.append(rodPos[0])
            self.series02.append(rodPos[1])
            self.series03.append(rodPos[2])
            # self.chart.addSeries(self.series01)
            # self.chart.addSeries(self.series02)
            # self.chart.addSeries(self.series03)
            self.series01.attachAxis(self.axisX)
            self.series01.attachAxis(self.axisY)
            self.series02.attachAxis(self.axisX)
            self.series02.attachAxis(self.axisY)
            self.series03.attachAxis(self.axisX)
            self.series03.attachAxis(self.axisY)

        elif (nPos == 4):
            self.series01.append(rodPos[0])
            self.series02.append(rodPos[1])
            self.series03.append(rodPos[2])
            self.series04.append(rodPos[3])
            # self.chart.addSeries(self.series01)
            # self.chart.addSeries(self.series02)
            # self.chart.addSeries(self.series03)
            self.series01.attachAxis(self.axisX)
            self.series01.attachAxis(self.axisY)
            self.series02.attachAxis(self.axisX)
            self.series02.attachAxis(self.axisY)
            self.series03.attachAxis(self.axisX)
            self.series03.attachAxis(self.axisY)
            self.series04.attachAxis(self.axisX)
            self.series04.attachAxis(self.axisY)

        # minCBC = min()
        # self.axisY_CBC.setMin(800)
        # self.axisY_CBC.setMax(1200.0)

        self.seriesCBC.append(posCBC)
        # self.chart.addSeries((self.seriesCBC))

        self.seriesCBC.attachAxis(self.axisY_CBC)
        self.seriesCBC.attachAxis(self.axisX)

        # self.chartView = QtCharts.QChartView(self.chart)
        # self.chartView.setRenderHint(QPainter.Antialiasing)
        # self.chart.legend().setVisible(True)
        # self.chart.legend().setLabelBrush(self.axisBrush)
        # self.chart.legend().setAlignment(Qt.AlignBottom)

        # chart.setPlotArea(a)
        # self.chart.setTitleBrush(self.axisBrush)
        # self.chart.setBackgroundBrush(QBrush(QColor(55, 58, 69)))

        # self.chartView = QtCharts.QChartView(self.chart)
        # self.chartView.setRenderHint(QPainter.Antialiasing)
        # chart.setPlotArea(a)
        # self.chart.setTitleBrush(self.axisBrush)
        # self.chart.setBackgroundBrush(QBrush(QColor(55, 58, 69)))
        #
        # self.chartView = QtCharts.QChartView(self.chart)
        # self.chartView.setRenderHint(QPainter.Antialiasing)

    def returnChart(self):
        return self.chartView

#
# class ECP_BarChart(UnitBarChart):
#     def __init__(self,pos5,pos4,pos3,posP,unit):
#
#         self.set0 = QtCharts.QBarSet("Rod Position Before Shutdown")
#         self.set1 = QtCharts.QBarSet("Rod Position After Shutdown")
#         self.series01 = QtCharts.QBarSeries()
#         self.chart = QtCharts.QChart()
#         self.axisX = QtCharts.QBarCategoryAxis()
#         self.axisY = QtCharts.QValueAxis()
#         self.chartView = QtCharts.QChartView()
#
#         self.font_Title = QFont()
#         self.font_AxisX_Label = QFont()
#         self.font_AxisY_Title = QFont()
#         self.font_AxisY_Label = QFont()
#
#         self.axisBrush = QBrush(Qt.white)
#
#         self.setChartFont()
#         self.setBrushStyle()
#
#         self.create_bar(pos5,pos4,pos3,posP,unit)
#
#     def setBrushStyle(self):
#         self.set0.setBrush(QBrush(QColor(df.RGB_ORANGE[0], df.RGB_ORANGE[1], df.RGB_ORANGE[2])))
#         self.set1.setBrush(QBrush(QColor(df.RGB_GREEN[0] , df.RGB_GREEN[1] , df.RGB_GREEN[2])))
#
#     def replaceRodPositionBeforeShutdown(self, posB5, posB4, posB3, posBP, \
#                                                posA5, posA4, posA3, posAP, \
#                                                 unit, nSet):
#
#         self.series01.remove(self.set0)
#         if (nSet == 2):
#             self.series01.remove(self.set1)
#
#         self.set0 = QtCharts.QBarSet("Rod Position Before Shutdown")
#         self.set0 << posB5 << posB4 << posB3 << posBP
#         self.set0.setBrush(QBrush(QColor(df.RGB_ORANGE[0], df.RGB_ORANGE[1], df.RGB_ORANGE[2])))
#         self.series01.append(self.set0)
#
#         if (nSet == 2):
#             self.set1 = QtCharts.QBarSet("Rod Position After Shutdown")
#             self.set1 << posA5 << posA4 << posA3 << posAP
#             self.set1.setBrush(QBrush(QColor(df.RGB_GREEN[0], df.RGB_GREEN[1], df.RGB_GREEN[2])))
#             self.series01.append(self.set1)
#
#         if(unit==df.RodPosUnit_percent):
#             self.axisY.setMax(100.0)
#             self.axisY.setMin(0)
#             self.axisY.setLabelFormat("%.2f")
#             self.axisY.setTickType(QtCharts.QValueAxis.TicksFixed)
#             self.axisY.setTickCount(11)
#             self.axisY.setTickInterval(10.0)
#             self.axisY.setMinorTickCount(4)
#             self.axisY.setTitleText("Unit : %")
#         elif(unit==df.RodPosUnit_cm):
#             self.axisY.setMax(381.0)
#             self.axisY.setMin(0)
#             self.axisY.setLabelFormat("%.2f")
#             self.axisY.setTickType(QtCharts.QValueAxis.TicksDynamic)
#             self.axisY.setTickInterval(50.0)
#             self.axisY.setTickAnchor(0.0)
#             self.axisY.setMinorTickCount(4)
#             self.axisY.setTitleText("Unit : cm")
#         elif(unit==df.RodPosUnit_inch):
#             self.axisY.setMax(150.0)
#             self.axisY.setMin(0)
#             self.axisY.setLabelFormat("%.2f")
#             self.axisY.setTickType(QtCharts.QValueAxis.TicksFixed)
#             self.axisY.setTickCount(16)
#             self.axisY.setTickInterval(10.0)
#             self.axisY.setMinorTickCount(4)
#             self.axisY.setTitleText("Unit : inch")
#
#     def replaceRodPositionAfterShutdown(self, posB5, posB4, posB3, posBP, \
#                                               posA5, posA4, posA3, posAP, \
#                                                unit, nSet, afterFlag ):
#         new_nSet = nSet
#         # 01. Make New Bar Chart
#         if (nSet == 1 and afterFlag == True):
#             new_nSet = 2
#             self.set1 = QtCharts.QBarSet("Rod Position After Shutdown")
#             self.set1 << posA5 << posA4 << posA3 << posAP
#             self.set1.setBrush(QBrush(QColor(df.RGB_GREEN[0], df.RGB_GREEN[1], df.RGB_GREEN[2])))
#             self.series01.append(self.set1)
#
#         # 02. Delete Bar Chart for Rod Position After Shutdown
#         elif (nSet == 2 and afterFlag == False):
#             new_nSet = 1
#             self.series01.remove(self.set1)
#
#         # 03. Change Bar Chart Value for Rod Position After Shutdown
#         elif (nSet == 2 and afterFlag == True):
#             new_nSet = 2
#             self.series01.remove(self.set1)
#             self.series01.remove(self.set0)
#
#             self.set0 = QtCharts.QBarSet("Rod Position Before Shutdown")
#             self.set0 << posB5 << posB4 << posB3 << posBP
#             self.set0.setBrush(QBrush(QColor(df.RGB_ORANGE[0], df.RGB_ORANGE[1], df.RGB_ORANGE[2])))
#
#             self.set1 = QtCharts.QBarSet("Rod Position After Shutdown")
#             self.set1 << posA5 << posA4 << posA3 << posAP
#             self.set1.setBrush(QBrush(QColor(df.RGB_GREEN[0], df.RGB_GREEN[1], df.RGB_GREEN[2])))
#             self.series01.append(self.set0)
#             self.series01.append(self.set1)
#
#         return new_nSet
    #def test0001(self,b):
    #    self.pSeries01 = self.series01.points()
    #    print("Goooood!")
    #    #print(self.pSeries01.length())
    #    a = len(self.pSeries01)
    #    for iPoint in range(a):
    #        if self.pSeries01[iPoint]==b:
    #            print(iPoint)