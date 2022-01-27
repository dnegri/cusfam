from PyQt5.QtWidgets import QPushButton, QGraphicsColorizeEffect
from PyQt5.QtCore import Qt, QMimeData, pyqtSignal, QPropertyAnimation
from PyQt5.QtGui import QDrag, QPixmap, QPainter, QColor

from PyQt5.QtCore import QModelIndex
import Definitions as df

_BLANK_ASM_ = " "

class startButton(QPushButton):

    clickButton = pyqtSignal(int)

    def __init__(self, calcOpt, functionStart, parent):

        super().__init__(parent)

        self.calcOpt = calcOpt
        self.setCheckable(True)
        self.checkStatus = False
        self.functionStart  = functionStart
        self.clickButton.connect(self.functionStart)

        self.setAcceptDrops(True)
        self.isDropAble = True

    def __del__(self):
        del self.functionStart

    def mouseReleaseEvent(self,e):
        self.clickButton.emit(self.calcOpt)

