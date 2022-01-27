from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import (QCoreApplication, QPropertyAnimation, QDate, QDateTime, QMetaObject, QObject, QPoint, QRect, QSize, QTime, QUrl, Qt, QEvent)
from PyQt5.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont, QFontDatabase, QIcon, QKeySequence, QLinearGradient, QPalette, QPainter, QPixmap, QRadialGradient)
from PyQt5.QtWidgets import *

from splash_screen import Ui_SplashScreen

from PyQt5.QtCore import QPointF, pyqtSignal, QThread, QObject, QProcess


## ==> SPLASHSCREEN WINDOW
class SplashScreen(QMainWindow):

    killed = pyqtSignal(str)

    def __init__(self):
        QMainWindow.__init__(self)
        self.setWindowModality(Qt.ApplicationModal)
        self.ui = Ui_SplashScreen()
        self.ui.setupUi(self)
        self.ui.kill_button.clicked.connect(self.killProcess)

        ## ==> SET INITIAL PROGRESS BAR TO (0) ZERO
        self.progressBarValue(0)
        self.max_counter = 1
        self.each_time = 1.0
        self.time_jumper = 0
        self.time_counter = 0

        ## ==> REMOVE STANDARD TITLE BAR
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint) # Remove title bar
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground) # Set background to transparent

        ## ==> APPLY DROP SHADOW EFFECT
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(20)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(0)
        self.shadow.setColor(QColor(0, 0, 0, 120))
        self.ui.circularBg.setGraphicsEffect(self.shadow)

        # ## QTIMER ==> START
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.progress_nano)
        # # TIMER IN MILLISECONDS
        #self.timer.start(15)

        self.counter = 0
        self.jumper = 10
        self.is_nano = False
        ## SHOW ==> MAIN WINDOW
        ########################################################################
        self.show()
        ## ==> END ##

    def killProcess(self):
        self.killed.emit("True")
        self.close()

    def init_progress(self, progress_length, each_time, is_nano=False):

        self.is_nano = is_nano
        self.counter = 0
        self.jumper = 10

        self.time_jumper = 1
        self.time_counter = 0

        self.max_counter = progress_length
        self.each_time = each_time

        if self.is_nano:
            self.timer.stop()
            self.timer.start(self.each_time/10)

    def progress_nano(self):
        if self.time_counter >= 100:
            if self.counter >= 100:
                self.time_counter = self.counter-1
            else:
                self.time_counter = 0
            self.time_jumper = self.jumper-1

        value = self.time_counter

        interval = 100/self.max_counter

        # HTML TEXT PERCENTAGE
        htmlText = """<p><span style=" font-size:68pt;">{VALUE}</span><span style=" font-size:58pt; vertical-align:super;">%</span></p>"""

        if value >= self.counter+interval: value = self.counter
        # REPLACE VALUE
        newHtml = htmlText.replace("{VALUE}", str(self.jumper))

        if value > self.time_jumper:
            # APPLY NEW PERCENTAGE TEXT
            self.ui.labelPercentage.setText(newHtml)
            self.time_jumper += 1

        # SET VALUE TO PROGRESS BAR
        # fix max value error if > than 100
        self.progressBarValue(value)

        # INCREASE COUNTER
        self.time_counter += interval/100

    ## DEF TO LOANDING
    ########################################################################
    def progress(self):
        value = self.counter

        # HTML TEXT PERCENTAGE
        htmlText = """<p><span style=" font-size:68pt;">{VALUE}</span><span style=" font-size:58pt; vertical-align:super;">%</span></p>"""

        # REPLACE VALUE
        newHtml = htmlText.replace("{VALUE}", str(self.jumper))

        if value > self.jumper:
            # APPLY NEW PERCENTAGE TEXT
            self.ui.labelPercentage.setText(newHtml)
            self.jumper += 10

        # SET VALUE TO PROGRESS BAR
        # fix max value error if > than 100
        if value >= 100: value = 1.000

        self.progressBarValue(value)

        # CLOSE SPLASH SCREE AND OPEN APP
        if self.counter > 100:
            # CLOSE SPLASH SCREEN
            self.close()

        # INCREASE COUNTER
        self.counter += 100/self.max_counter

        # STOP TIMER
        if self.is_nano:
            self.timer.stop()
            self.time_counter = self.counter
            self.time_jumper = self.jumper
            self.timer.start(self.each_time/100)


    ## DEF PROGRESS BAR VALUE
    ########################################################################
    def progressBarValue(self, value):

        # PROGRESSBAR STYLESHEET BASE
        styleSheet = """
        QFrame{
        	border-radius: 150px;
        	background-color: qconicalgradient(cx:0.5, cy:0.5, angle:90, stop:{STOP_1} rgba(255, 0, 127, 0), stop:{STOP_2} rgba(85, 170, 255, 255));
        }
        """

        # GET PROGRESS BAR VALUE, CONVERT TO FLOAT AND INVERT VALUES
        # stop works of 1.000 to 0.000
        progress = (100 - value) / 100.0

        # GET NEW VALUES
        stop_1 = str(progress - 0.001)
        stop_2 = str(progress)

        # SET VALUES TO NEW STYLESHEET
        newStylesheet = styleSheet.replace("{STOP_1}", stop_1).replace("{STOP_2}", stop_2)

        # APPLY STYLESHEET WITH NEW VALUES
        self.ui.circularProgress.setStyleSheet(newStylesheet)

    def close(self):
        if self.is_nano:
            self.timer.stop()
        super().close()

