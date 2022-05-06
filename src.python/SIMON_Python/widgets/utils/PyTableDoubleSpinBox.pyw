from PyQt5.QtWidgets import QDoubleSpinBox
from PyQt5.QtCore import QSize, QSizeF
from PyQt5.QtGui import QPalette, QBrush, QColor

class tableDoubleSpinBox(QDoubleSpinBox):
    def __init__(self):
        super().__init__()

        self.row = 0
        self.column = 0
        self.setStyleSheet("QDoubleSpinBox { background-color: rgb(38, 44, 53);}")
        # pal = QPalette()
        # pal.setColor(QPalette.background(), QBrush(QColor(155, 158, 169)))
        # self.setPalette(pal)

    def saveBoxPosition(self,row,column):
        self.row =  row
        self.column = column


    def returnRow(self):
        return self.row
    def returnColumn(self):
        return self.column

    def returnBoxPosition(self):
        return (self.row, self.column)