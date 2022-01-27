from PySide2.QtWidgets import QPushButton, QGraphicsColorizeEffect
from PySide2.QtCore import Qt, QMimeData, Signal, QPropertyAnimation
from PySide2.QtGui import QDrag, QPixmap, QPainter, QColor

from PySide2.QtCore import QModelIndex
import Define as df

_BLANK_ASM_ = " "

class unitButton(QPushButton):

    clickButton = Signal(str)
    changeButton = Signal(str,str)


    def __init__(self, bName, funcClick, changeData, parent):#, title, assembly_id, parent):

        super().__init__(parent)
        # a = self().keyPressEvent()
        self.bName = bName
        self.RGB = [255,255,255]
        self.setCheckable(True)
        self.checkStatus = False
        self.funcClick  = funcClick
        self.changeData = ""
        self.clickButton.connect(self.funcClick)
        self.swapFlag = False
        # a = Qt.TabFocus
        # self.keyPressEvent()
        if(changeData!= df._BLANK_ASM_):
            self.swapFlag = True
            self.changeData = changeData
            self.changeButton.connect(self.changeData)
        self.setAcceptDrops(True)
        self.isDropAble = True

    def __del__(self):
        del self.funcClick
        del self.changeData
        del self.RGB

    def rename(self,newName):
        self.bName = newName

    def stroeRGB(self,rgb):
        self.RGB = rgb

    def mouseReleaseEvent(self,e):
        # if(self.checkStatus==False):
        #     print("Checked")
        #     self.checkStatus = True
        #     self.clickButton.emit(self.bName)
        #
        # else:
        #     print("UnChecked")
        #     self.checkStatus = False
        # super().mouseReleaseEvent(e)
        self.clickButton.emit(self.bName)
        # print(self.xpos, self.ypos)

    def mouseMoveEvent(self, e):
        # super().mouseMoveEvent(e)
        mimeData = QMimeData()
        mimeData.setText(self.bName)
        drag = QDrag(self)
        pixmap = QPixmap(self.size())
        painter = QPainter(pixmap)
        painter.drawPixmap(self.rect(), self.grab())
        painter.end()
        drag.setPixmap(pixmap)
        drag.setMimeData(mimeData)
        drag.exec_(Qt.MoveAction)

    # def setAssemblyID(self, assembly_id):
    #     self.isDropAble = True
    #     self.assembly_id = assembly_id

    def dragEnterEvent(self, e):
    #     #print(e.mimeData())
        if(self.swapFlag==True):
            e.accept()
        # if self.isDropAble:
        #     print("accepted")
    def dragMoveEvent(self, e):
        pass
    def dropEvent(self, e):
    #     #self.setText(e.mimeData().text())
    #     #if "empty" in e.mimeData().text():
        if self.isDropAble:
            self.changeButton.emit(e.mimeData().text(), self.bName)

class outputButton(QPushButton):
    clickButton = Signal(str)
    # changeButton = pyqtSignal(str,str)
    def __init__(self, bName, funcClick,dummy, parent):
        super().__init__(parent)
        self.bName = bName
        self.setCheckable(True)
        self.checkStatus = False
        self.funcClick  = funcClick
        self.clickButton.connect(self.funcClick)
        self.swapFlag = False
        # self.setAcceptDrops(True)
        # self.isDropAble = True
    def __del__(self):
        del self.funcClick

    def rename(self,newName):
        self.bName = newName

    def mouseReleaseEvent(self,e):
        # if(self.checkStatus==False):
        #     print("Checked")
        #     self.checkStatus = True
        #     self.clickButton.emit(self.bName)
        #
        # else:
        #     print("UnChecked")
        #     self.checkStatus = False
        self.clickButton.emit(self.bName)
        # print(self.xpos, self.ypos)

