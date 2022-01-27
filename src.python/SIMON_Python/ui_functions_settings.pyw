################################################################################
##
## BY: WANDERSON M.PIMENTA
## PROJECT MADE WITH: Qt Designer and PySide2
## V: 1.0.0
##
## This project can be used freely for all uses, as long as they maintain the
## respective credits only in the Python scripts, any information in the visual
## interface (GUI) can be modified without any implication.
##
## There are limitations on Qt licenses if you want to use your products
## commercially, I recommend reading them on the official website:
## https://doc.qt.io/qtforpython/licenses.html
##
################################################################################

## ==> GUI FILE
from main import *

class UIFunctionsSettings(MainWindow):

    ########################################################################
    ## START - GUI FUNCTIONS
    ########################################################################

    def setAllComponents(self):
        self.ui.username_button.clicked.connect(self.setUsername)
        #self.setWorkingUI()
        #self.setPlantFileUI()
        #self.setRestartUI()
        #self.setRestartUI()

    def setUsernameUI(self):
        self.ui.username_button.clicked.connect(self.setUsername)

    def setWorkingUI(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Working Directory")
        self.ui.working_button.clicked.connect(self.Button)

    def setPlantFileUI(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Working Directory")
        self.ui.plant_button.clicked.connect(self.Button)

    def setRestartUI(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Working Directory")
        self.ui.restart_button.clicked.connect(self.Button)

    def setSnapshotUI(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Working Directory")
        self.ui.snapshot_button.clicked.connect(self.Button)

    def setUsername(self):
        if self.ui.username_edit.text() != "":
            user, created = model.User.get_or_create(username=self.ui.username_edit.text())
            if created:
                self.ui.username_label.setText("User Login: " + self.ui.username_edit.text())
            self.current_user = user


    ## ==> DYNAMIC MENUS
    ########################################################################
    def addNewMenu(self, name, objName, icon, isTopMenu):
        font = QFont()
        font.setFamily(u"Segoe UI")
        button = QPushButton(str(count),self)
        button.setObjectName(objName)
        sizePolicy3 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(button.sizePolicy().hasHeightForWidth())
        button.setSizePolicy(sizePolicy3)
        button.setMinimumSize(QSize(0, 70))
        button.setLayoutDirection(Qt.LeftToRight)
        button.setFont(font)
        button.setStyleSheet(Style.style_bt_standard.replace('ICON_REPLACE', icon))
        button.setText(name)
        button.setToolTip(name)
        button.clicked.connect(self.Button)

        if isTopMenu:
            self.ui.layout_menus.addWidget(button)
        else:
            self.ui.layout_menu_bottom.addWidget(button)

    ########################################################################
    ## END - GUI DEFINITIONS
    ########################################################################
