# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'annotation_tool.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(820, 730)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(820, 730))
        MainWindow.setMaximumSize(QtCore.QSize(820, 730))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("images/microglialcell.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setMaximumSize(QtCore.QSize(820, 730))
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(2, 0, 808, 671))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.vLayout_All = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.vLayout_All.setContentsMargins(6, 6, 6, 6)
        self.vLayout_All.setObjectName("vLayout_All")
        self.gBox_Data = QtWidgets.QGroupBox(self.verticalLayoutWidget_2)
        self.gBox_Data.setStyleSheet("QGroupBox {\n"
"    border: 1px solid gray;\n"
"    border-radius: 9px;\n"
"    margin-top: 0.5em;\n"
"    font: 75 10pt \"Ubuntu\";\n"
"\n"
"}\n"
"\n"
"QGroupBox::title {\n"
"    subcontrol-origin: margin;\n"
"    left: 10px;\n"
"    padding: 0 3px 0 3px;\n"
"\n"
"}\n"
"\n"
"")
        self.gBox_Data.setObjectName("gBox_Data")
        self.layoutControl = QtWidgets.QHBoxLayout(self.gBox_Data)
        self.layoutControl.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.layoutControl.setContentsMargins(9, 7, 6, 6)
        self.layoutControl.setSpacing(5)
        self.layoutControl.setObjectName("layoutControl")
        self.hLayout_DataControls = QtWidgets.QHBoxLayout()
        self.hLayout_DataControls.setObjectName("hLayout_DataControls")
        spacerItem = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.hLayout_DataControls.addItem(spacerItem)
        self.lblDataPath = QtWidgets.QLabel(self.gBox_Data)
        font = QtGui.QFont()
        font.setFamily("Noto Sans")
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        self.lblDataPath.setFont(font)
        self.lblDataPath.setAutoFillBackground(False)
        self.lblDataPath.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.lblDataPath.setObjectName("lblDataPath")
        self.hLayout_DataControls.addWidget(self.lblDataPath)
        self.txtDataPath = QtWidgets.QTextEdit(self.gBox_Data)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.txtDataPath.sizePolicy().hasHeightForWidth())
        self.txtDataPath.setSizePolicy(sizePolicy)
        self.txtDataPath.setMinimumSize(QtCore.QSize(0, 24))
        self.txtDataPath.setMaximumSize(QtCore.QSize(16777215, 24))
        font = QtGui.QFont()
        font.setFamily("Noto Sans")
        font.setPointSize(10)
        self.txtDataPath.setFont(font)
        self.txtDataPath.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.txtDataPath.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.txtDataPath.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.txtDataPath.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.txtDataPath.setReadOnly(True)
        self.txtDataPath.setObjectName("txtDataPath")
        self.hLayout_DataControls.addWidget(self.txtDataPath)
        self.lblGoTo = QtWidgets.QLabel(self.gBox_Data)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.lblGoTo.setFont(font)
        self.lblGoTo.setObjectName("lblGoTo")
        self.hLayout_DataControls.addWidget(self.lblGoTo)
        self.leBatchNo = QtWidgets.QLineEdit(self.gBox_Data)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.leBatchNo.sizePolicy().hasHeightForWidth())
        self.leBatchNo.setSizePolicy(sizePolicy)
        self.leBatchNo.setMaximumSize(QtCore.QSize(50, 25))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.leBatchNo.setFont(font)
        self.leBatchNo.setInputMethodHints(QtCore.Qt.ImhDigitsOnly|QtCore.Qt.ImhPreferNumbers)
        self.leBatchNo.setAlignment(QtCore.Qt.AlignCenter)
        self.leBatchNo.setObjectName("leBatchNo")
        self.hLayout_DataControls.addWidget(self.leBatchNo)
        self.hLayout_DataControls.setStretch(1, 12)
        self.hLayout_DataControls.setStretch(2, 73)
        self.hLayout_DataControls.setStretch(3, 5)
        self.hLayout_DataControls.setStretch(4, 10)
        self.layoutControl.addLayout(self.hLayout_DataControls)
        self.line = QtWidgets.QFrame(self.gBox_Data)
        self.line.setStyleSheet("#line {\n"
"    border: 1px solid red;\n"
"}\n"
"")
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.layoutControl.addWidget(self.line)
        self.btnStartNext = QtWidgets.QPushButton(self.gBox_Data)
        self.btnStartNext.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnStartNext.sizePolicy().hasHeightForWidth())
        self.btnStartNext.setSizePolicy(sizePolicy)
        self.btnStartNext.setMinimumSize(QtCore.QSize(0, 30))
        self.btnStartNext.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("Noto Sans")
        font.setPointSize(10)
        self.btnStartNext.setFont(font)
        self.btnStartNext.setStyleSheet("QPushButton { \n"
"    background-color: rgb(49, 49, 49);\n"
"    color: rgb(255, 255, 255);\n"
"    border-radius: 9px;\n"
"}\n"
"QPushButton:pressed { \n"
"    background-color: rgb(172, 172, 172);\n"
"    color: rgb(49, 49, 49);\n"
"    border: 1px solid gray;\n"
"}\n"
"QPushButton:hover:!pressed\n"
"{\n"
"  border: 1px solid red;\n"
"}")
        self.btnStartNext.setObjectName("btnStartNext")
        self.layoutControl.addWidget(self.btnStartNext)
        self.btnExport = QtWidgets.QPushButton(self.gBox_Data)
        self.btnExport.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnExport.sizePolicy().hasHeightForWidth())
        self.btnExport.setSizePolicy(sizePolicy)
        self.btnExport.setMinimumSize(QtCore.QSize(0, 30))
        self.btnExport.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setFamily("Noto Sans")
        font.setPointSize(10)
        self.btnExport.setFont(font)
        self.btnExport.setStyleSheet("QPushButton { \n"
"    background-color: rgb(49, 49, 49);\n"
"    color: rgb(255, 255, 255);\n"
"    border-radius: 9px;\n"
"}\n"
"QPushButton:pressed { \n"
"    background-color: rgb(172, 172, 172);\n"
"    color: rgb(49, 49, 49);\n"
"    border: 1px solid gray;\n"
"}\n"
"QPushButton:hover:!pressed\n"
"{\n"
"  border: 1px solid red;\n"
"}")
        self.btnExport.setCheckable(False)
        self.btnExport.setObjectName("btnExport")
        self.layoutControl.addWidget(self.btnExport)
        self.layoutControl.setStretch(0, 70)
        self.layoutControl.setStretch(2, 15)
        self.layoutControl.setStretch(3, 15)
        self.vLayout_All.addWidget(self.gBox_Data)
        self.hLayout_Controls = QtWidgets.QHBoxLayout()
        self.hLayout_Controls.setObjectName("hLayout_Controls")
        self.vLayout_Rendering = QtWidgets.QVBoxLayout()
        self.vLayout_Rendering.setObjectName("vLayout_Rendering")
        self.gridGroupBox = QtWidgets.QGroupBox(self.verticalLayoutWidget_2)
        self.gridGroupBox.setStyleSheet("QGroupBox {\n"
"    border: 1px solid gray;\n"
"    border-radius: 9px;\n"
"    margin-top: 0.5em;\n"
"    font: 75 10pt \"Ubuntu\";\n"
"\n"
"}\n"
"\n"
"QGroupBox::title {\n"
"    subcontrol-origin: margin;\n"
"    left: 10px;\n"
"    padding: 0 3px 0 3px;\n"
"\n"
"}\n"
"\n"
"")
        self.gridGroupBox.setObjectName("gridGroupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.gridGroupBox)
        self.gridLayout.setObjectName("gridLayout")
        self.hLayout_Rendering = QtWidgets.QHBoxLayout()
        self.hLayout_Rendering.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.hLayout_Rendering.setContentsMargins(-1, 0, -1, -1)
        self.hLayout_Rendering.setObjectName("hLayout_Rendering")
        spacerItem1 = QtWidgets.QSpacerItem(8, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.hLayout_Rendering.addItem(spacerItem1)
        self.cbxResizing = QtWidgets.QCheckBox(self.gridGroupBox)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.cbxResizing.setFont(font)
        self.cbxResizing.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.cbxResizing.setObjectName("cbxResizing")
        self.hLayout_Rendering.addWidget(self.cbxResizing)
        self.lblVoxelX = QtWidgets.QLabel(self.gridGroupBox)
        font = QtGui.QFont()
        font.setBold(False)
        font.setItalic(True)
        font.setWeight(50)
        self.lblVoxelX.setFont(font)
        self.lblVoxelX.setObjectName("lblVoxelX")
        self.hLayout_Rendering.addWidget(self.lblVoxelX)
        self.leVoxelX = QtWidgets.QLineEdit(self.gridGroupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.leVoxelX.sizePolicy().hasHeightForWidth())
        self.leVoxelX.setSizePolicy(sizePolicy)
        self.leVoxelX.setMaximumSize(QtCore.QSize(50, 25))
        self.leVoxelX.setAlignment(QtCore.Qt.AlignCenter)
        self.leVoxelX.setObjectName("leVoxelX")
        self.hLayout_Rendering.addWidget(self.leVoxelX)
        self.lblVoxelY = QtWidgets.QLabel(self.gridGroupBox)
        font = QtGui.QFont()
        font.setItalic(True)
        self.lblVoxelY.setFont(font)
        self.lblVoxelY.setObjectName("lblVoxelY")
        self.hLayout_Rendering.addWidget(self.lblVoxelY)
        self.leVoxelY = QtWidgets.QLineEdit(self.gridGroupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.leVoxelY.sizePolicy().hasHeightForWidth())
        self.leVoxelY.setSizePolicy(sizePolicy)
        self.leVoxelY.setMaximumSize(QtCore.QSize(50, 25))
        self.leVoxelY.setAlignment(QtCore.Qt.AlignCenter)
        self.leVoxelY.setObjectName("leVoxelY")
        self.hLayout_Rendering.addWidget(self.leVoxelY)
        self.lblVoxelZ = QtWidgets.QLabel(self.gridGroupBox)
        font = QtGui.QFont()
        font.setItalic(True)
        self.lblVoxelZ.setFont(font)
        self.lblVoxelZ.setObjectName("lblVoxelZ")
        self.hLayout_Rendering.addWidget(self.lblVoxelZ)
        self.leVoxelZ = QtWidgets.QLineEdit(self.gridGroupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.leVoxelZ.sizePolicy().hasHeightForWidth())
        self.leVoxelZ.setSizePolicy(sizePolicy)
        self.leVoxelZ.setMaximumSize(QtCore.QSize(50, 25))
        self.leVoxelZ.setAlignment(QtCore.Qt.AlignCenter)
        self.leVoxelZ.setObjectName("leVoxelZ")
        self.hLayout_Rendering.addWidget(self.leVoxelZ)
        spacerItem2 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.hLayout_Rendering.addItem(spacerItem2)
        self.gridLayout.addLayout(self.hLayout_Rendering, 5, 0, 1, 1)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setContentsMargins(-1, 0, -1, -1)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.btnRenderings = QtWidgets.QPushButton(self.gridGroupBox)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        self.btnRenderings.setFont(font)
        self.btnRenderings.setFlat(True)
        self.btnRenderings.setObjectName("btnRenderings")
        self.horizontalLayout_3.addWidget(self.btnRenderings)
        self.rbtnMIP = QtWidgets.QRadioButton(self.gridGroupBox)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.rbtnMIP.setFont(font)
        self.rbtnMIP.setCheckable(True)
        self.rbtnMIP.setChecked(True)
        self.rbtnMIP.setObjectName("rbtnMIP")
        self.horizontalLayout_3.addWidget(self.rbtnMIP)
        self.rbtnTranslucent = QtWidgets.QRadioButton(self.gridGroupBox)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.rbtnTranslucent.setFont(font)
        self.rbtnTranslucent.setObjectName("rbtnTranslucent")
        self.horizontalLayout_3.addWidget(self.rbtnTranslucent)
        self.rbtnAdditive = QtWidgets.QRadioButton(self.gridGroupBox)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.rbtnAdditive.setFont(font)
        self.rbtnAdditive.setObjectName("rbtnAdditive")
        self.horizontalLayout_3.addWidget(self.rbtnAdditive)
        self.line_2 = QtWidgets.QFrame(self.gridGroupBox)
        self.line_2.setStyleSheet("#line_2 {\n"
"    border: 1px solid red;\n"
"}\n"
"")
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.horizontalLayout_3.addWidget(self.line_2)
        self.btnColormaps = QtWidgets.QPushButton(self.gridGroupBox)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.btnColormaps.setFont(font)
        self.btnColormaps.setFlat(True)
        self.btnColormaps.setObjectName("btnColormaps")
        self.horizontalLayout_3.addWidget(self.btnColormaps)
        spacerItem3 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem3)
        self.gridLayout.addLayout(self.horizontalLayout_3, 0, 0, 1, 1)
        self.line_3 = QtWidgets.QFrame(self.gridGroupBox)
        self.line_3.setStyleSheet("#line_3 {\n"
"    border: 1px solid gray;\n"
"}\n"
"")
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.gridLayout.addWidget(self.line_3, 1, 0, 1, 1)
        self.vLayout_Rendering.addWidget(self.gridGroupBox)
        self.hLayout_Controls.addLayout(self.vLayout_Rendering)
        self.gBoxLabels = QtWidgets.QGroupBox(self.verticalLayoutWidget_2)
        self.gBoxLabels.setStyleSheet("QGroupBox {\n"
"    border: 1px solid gray;\n"
"    border-radius: 9px;\n"
"    margin-top: 0.5em;\n"
"    font: 75 10pt \"Ubuntu\";\n"
"\n"
"}\n"
"\n"
"QGroupBox::title {\n"
"    subcontrol-origin: margin;\n"
"    left: 10px;\n"
"    padding: 0 3px 0 3px;\n"
"\n"
"}\n"
"\n"
"")
        self.gBoxLabels.setObjectName("gBoxLabels")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.gBoxLabels)
        self.horizontalLayout_2.setContentsMargins(6, 8, 6, 6)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.btnRamified = QtWidgets.QPushButton(self.gBoxLabels)
        self.btnRamified.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnRamified.sizePolicy().hasHeightForWidth())
        self.btnRamified.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Noto Sans")
        font.setPointSize(9)
        font.setBold(False)
        font.setUnderline(False)
        font.setWeight(50)
        font.setStrikeOut(False)
        self.btnRamified.setFont(font)
        self.btnRamified.setStyleSheet("QPushButton { \n"
"    background-color: rgb(49, 49, 49);\n"
"    color: rgb(255, 255, 255);\n"
"    border-radius: 9px;\n"
"    padding: 4px;\n"
"}\n"
"QPushButton:pressed { \n"
"    background-color: rgb(172, 172, 172);\n"
"    color: rgb(49, 49, 49);\n"
"    border: 1px solid gray;\n"
"}\n"
"\n"
"QPushButton:hover:!pressed\n"
"{\n"
"  border: 1px solid red;\n"
"}\n"
"\n"
"QPushButton:disabled {\n"
"    background-color:rgb(172, 172, 172);\n"
"}")
        self.btnRamified.setIconSize(QtCore.QSize(34, 24))
        self.btnRamified.setCheckable(True)
        self.btnRamified.setChecked(False)
        self.btnRamified.setObjectName("btnRamified")
        self.horizontalLayout_2.addWidget(self.btnRamified)
        self.btnAmoeboid = QtWidgets.QPushButton(self.gBoxLabels)
        self.btnAmoeboid.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnAmoeboid.sizePolicy().hasHeightForWidth())
        self.btnAmoeboid.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Noto Sans")
        font.setPointSize(9)
        font.setBold(False)
        font.setUnderline(False)
        font.setWeight(50)
        font.setStrikeOut(False)
        self.btnAmoeboid.setFont(font)
        self.btnAmoeboid.setStyleSheet("QPushButton { \n"
"    background-color: rgb(49, 49, 49);\n"
"    color: rgb(255, 255, 255);\n"
"    border-radius: 9px;\n"
"    padding: 4px;\n"
"}\n"
"QPushButton:pressed { \n"
"    background-color: rgb(172, 172, 172);\n"
"    color: rgb(49, 49, 49);\n"
"    border: 1px solid gray;\n"
"}\n"
"\n"
"QPushButton:hover:!pressed\n"
"{\n"
"  border: 1px solid red;\n"
"}\n"
"\n"
"QPushButton:disabled {\n"
"    background-color:rgb(172, 172, 172);\n"
"}")
        self.btnAmoeboid.setCheckable(True)
        self.btnAmoeboid.setObjectName("btnAmoeboid")
        self.horizontalLayout_2.addWidget(self.btnAmoeboid)
        self.btnGarbage = QtWidgets.QPushButton(self.gBoxLabels)
        self.btnGarbage.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnGarbage.sizePolicy().hasHeightForWidth())
        self.btnGarbage.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Noto Sans")
        font.setPointSize(9)
        font.setBold(False)
        font.setUnderline(False)
        font.setWeight(50)
        font.setStrikeOut(False)
        self.btnGarbage.setFont(font)
        self.btnGarbage.setStyleSheet("QPushButton { \n"
"    background-color: rgb(49, 49, 49);\n"
"    color: rgb(255, 255, 255);\n"
"    border-radius: 9px;\n"
"    padding: 4px;\n"
"}\n"
"QPushButton:pressed { \n"
"    background-color: rgb(172, 172, 172);\n"
"    color: rgb(49, 49, 49);\n"
"    border: 1px solid gray;\n"
"}\n"
"\n"
"QPushButton:hover:!pressed\n"
"{\n"
"  border: 1px solid red;\n"
"}\n"
"\n"
"QPushButton:disabled {\n"
"    background-color:rgb(172, 172, 172);\n"
"}")
        self.btnGarbage.setCheckable(True)
        self.btnGarbage.setFlat(False)
        self.btnGarbage.setObjectName("btnGarbage")
        self.horizontalLayout_2.addWidget(self.btnGarbage)
        self.btnUncertain = QtWidgets.QPushButton(self.gBoxLabels)
        self.btnUncertain.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btnUncertain.sizePolicy().hasHeightForWidth())
        self.btnUncertain.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Noto Sans")
        font.setPointSize(9)
        font.setBold(False)
        font.setUnderline(False)
        font.setWeight(50)
        font.setStrikeOut(False)
        self.btnUncertain.setFont(font)
        self.btnUncertain.setStyleSheet("QPushButton { \n"
"    background-color: rgb(49, 49, 49);\n"
"    color: rgb(255, 255, 255);\n"
"    border-radius: 9px;\n"
"    padding: 4px;\n"
"}\n"
"QPushButton:pressed { \n"
"    background-color: rgb(172, 172, 172);\n"
"    color: rgb(49, 49, 49);\n"
"    border: 1px solid gray;\n"
"}\n"
"\n"
"QPushButton:hover:!pressed\n"
"{\n"
"  border: 1px solid red;\n"
"}\n"
"\n"
"QPushButton:disabled {\n"
"    background-color:rgb(172, 172, 172);\n"
"}")
        self.btnUncertain.setCheckable(True)
        self.btnUncertain.setFlat(False)
        self.btnUncertain.setObjectName("btnUncertain")
        self.horizontalLayout_2.addWidget(self.btnUncertain)
        self.hLayout_Controls.addWidget(self.gBoxLabels)
        self.hLayout_Controls.setStretch(1, 35)
        self.vLayout_All.addLayout(self.hLayout_Controls)
        self.lblStats = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.lblStats.setFont(font)
        self.lblStats.setObjectName("lblStats")
        self.vLayout_All.addWidget(self.lblStats)
        self.gLayout_Core = QtWidgets.QGridLayout()
        self.gLayout_Core.setContentsMargins(0, 0, 0, 0)
        self.gLayout_Core.setSpacing(5)
        self.gLayout_Core.setObjectName("gLayout_Core")
        self.line_6 = QtWidgets.QFrame(self.verticalLayoutWidget_2)
        self.line_6.setStyleSheet("#line_4, #line_6 {\n"
"    border: 1px solid gray;\n"
"}\n"
"")
        self.line_6.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.gLayout_Core.addWidget(self.line_6, 1, 1, 1, 1)
        self.line_4 = QtWidgets.QFrame(self.verticalLayoutWidget_2)
        self.line_4.setStyleSheet("#line_4, #line_6 {\n"
"    border: 1px solid gray;\n"
"}\n"
"")
        self.line_4.setFrameShadow(QtWidgets.QFrame.Plain)
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setObjectName("line_4")
        self.gLayout_Core.addWidget(self.line_4, 1, 0, 1, 1)
        self.batchList = QtWidgets.QListWidget(self.verticalLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.batchList.sizePolicy().hasHeightForWidth())
        self.batchList.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Sans Serif")
        font.setPointSize(7)
        self.batchList.setFont(font)
        self.batchList.setIconSize(QtCore.QSize(15, 15))
        self.batchList.setTextElideMode(QtCore.Qt.ElideMiddle)
        self.batchList.setObjectName("batchList")
        self.gLayout_Core.addWidget(self.batchList, 2, 1, 1, 1)
        self.gGLWidget = QtWidgets.QOpenGLWidget(self.verticalLayoutWidget_2)
        self.gGLWidget.setObjectName("gGLWidget")
        self.gLayout_Core.addWidget(self.gGLWidget, 2, 0, 1, 1)
        self.gLayout_Core.setColumnStretch(0, 65)
        self.vLayout_All.addLayout(self.gLayout_Core)
        self.gBox_LogWindow = QtWidgets.QGroupBox(self.verticalLayoutWidget_2)
        self.gBox_LogWindow.setStyleSheet("QGroupBox {\n"
"    border: 1px solid gray;\n"
"    border-radius: 9px;\n"
"    margin-top: 0.5em;\n"
"    font: 75 10pt \"Ubuntu\";\n"
"\n"
"}\n"
"\n"
"QGroupBox::title {\n"
"    subcontrol-origin: margin;\n"
"    left: 10px;\n"
"    padding: 0 3px 0 3px;\n"
"\n"
"}\n"
"\n"
"")
        self.gBox_LogWindow.setObjectName("gBox_LogWindow")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.gBox_LogWindow)
        self.verticalLayout_4.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.logwin = QtWidgets.QTextEdit(self.gBox_LogWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.logwin.sizePolicy().hasHeightForWidth())
        self.logwin.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Cantarell")
        font.setPointSize(8)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.logwin.setFont(font)
        self.logwin.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.logwin.setReadOnly(True)
        self.logwin.setObjectName("logwin")
        self.verticalLayout_4.addWidget(self.logwin)
        self.verticalLayout_4.setStretch(0, 25)
        self.vLayout_All.addWidget(self.gBox_LogWindow)
        self.vLayout_All.setStretch(3, 75)
        self.vLayout_All.setStretch(4, 25)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setStyleSheet("QStatusBar{\n"
"    padding-left:8px;\n"
"    background:rgb(49, 49, 49);\n"
"    font-size:9px;\n"
"    color: rgb(255, 255, 255);\n"
"}")
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 820, 22))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.menubar.setFont(font)
        self.menubar.setStyleSheet("        QMenuBar {\n"
"            background-color: rgb(49,49,49);\n"
"            color: rgb(255,255,255);\n"
"            border: 1px solid #000;\n"
"        }\n"
"\n"
"        QMenuBar::item {\n"
"            background-color: rgb(49,49,49);\n"
"            color: rgb(255,255,255);\n"
"        }\n"
"\n"
"        QMenuBar::item::selected {\n"
"            background-color: rgb(30,30,30);\n"
"        }\n"
"\n"
"        QMenu {\n"
"            background-color: rgb(49,49,49);\n"
"            color: rgb(255,255,255);\n"
"            border: 1px solid #000;           \n"
"        }\n"
"\n"
"        QMenu::item::selected {\n"
"            background-color: rgb(30,30,30);\n"
"        }")
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.menuFile.setFont(font)
        self.menuFile.setObjectName("menuFile")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        font = QtGui.QFont()
        font.setPointSize(9)
        self.menuHelp.setFont(font)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.actionLoad_Path = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        self.actionLoad_Path.setFont(font)
        self.actionLoad_Path.setObjectName("actionLoad_Path")
        self.actionQuit = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        self.actionQuit.setFont(font)
        self.actionQuit.setObjectName("actionQuit")
        self.actionTo_File = QtWidgets.QAction(MainWindow)
        self.actionTo_File.setObjectName("actionTo_File")
        self.actionAbout_MDAT = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        self.actionAbout_MDAT.setFont(font)
        self.actionAbout_MDAT.setObjectName("actionAbout_MDAT")
        self.menuFile.addAction(self.actionLoad_Path)
        self.menuFile.addAction(self.actionQuit)
        self.menuHelp.addAction(self.actionAbout_MDAT)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Microglia Dataset Annotation Tool"))
        self.gBox_Data.setTitle(_translate("MainWindow", "Data Loader/Exporter"))
        self.lblDataPath.setText(_translate("MainWindow", "Data Path"))
        self.txtDataPath.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Noto Sans\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:9pt;\">/data_ssd/brain_mapping/training_data/cc/011019</span></p></body></html>"))
        self.lblGoTo.setText(_translate("MainWindow", "Go to"))
        self.leBatchNo.setText(_translate("MainWindow", "0"))
        self.btnStartNext.setText(_translate("MainWindow", "Load (S)"))
        self.btnStartNext.setShortcut(_translate("MainWindow", "S"))
        self.btnExport.setText(_translate("MainWindow", "Export (E)"))
        self.btnExport.setShortcut(_translate("MainWindow", "E"))
        self.gridGroupBox.setTitle(_translate("MainWindow", "Rendering/Resizing"))
        self.cbxResizing.setText(_translate("MainWindow", "Resize? (Z)"))
        self.cbxResizing.setShortcut(_translate("MainWindow", "Z"))
        self.lblVoxelX.setText(_translate("MainWindow", "VX"))
        self.leVoxelX.setText(_translate("MainWindow", "1.65"))
        self.lblVoxelY.setText(_translate("MainWindow", "VY"))
        self.leVoxelY.setText(_translate("MainWindow", "1.65"))
        self.lblVoxelZ.setText(_translate("MainWindow", "VZ"))
        self.leVoxelZ.setText(_translate("MainWindow", "2.0"))
        self.btnRenderings.setText(_translate("MainWindow", "(M)ethod"))
        self.btnRenderings.setShortcut(_translate("MainWindow", "M"))
        self.rbtnMIP.setText(_translate("MainWindow", "mip"))
        self.rbtnTranslucent.setText(_translate("MainWindow", "translucent"))
        self.rbtnAdditive.setText(_translate("MainWindow", "additive"))
        self.btnColormaps.setText(_translate("MainWindow", "(C)olormaps"))
        self.btnColormaps.setShortcut(_translate("MainWindow", "C"))
        self.gBoxLabels.setTitle(_translate("MainWindow", "Labels"))
        self.btnRamified.setText(_translate("MainWindow", "Ramified\n"
"(R)"))
        self.btnRamified.setShortcut(_translate("MainWindow", "R"))
        self.btnAmoeboid.setText(_translate("MainWindow", "Amoeboid\n"
"(A)"))
        self.btnAmoeboid.setShortcut(_translate("MainWindow", "A"))
        self.btnGarbage.setText(_translate("MainWindow", "Garbage\n"
"(G)"))
        self.btnGarbage.setShortcut(_translate("MainWindow", "G"))
        self.btnUncertain.setText(_translate("MainWindow", "Uncertain\n"
"(Q)"))
        self.btnUncertain.setShortcut(_translate("MainWindow", "Q"))
        self.lblStats.setText(_translate("MainWindow", "  Set [ None ] | Bath # [ 0 ] | R: 0 (0.0%), A: 0 (0.0%), G: 0 (0.0%)"))
        self.gBox_LogWindow.setTitle(_translate("MainWindow", "Log"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.actionLoad_Path.setText(_translate("MainWindow", "Load Path"))
        self.actionLoad_Path.setToolTip(_translate("MainWindow", "Load Path"))
        self.actionLoad_Path.setShortcut(_translate("MainWindow", "Ctrl+L"))
        self.actionQuit.setText(_translate("MainWindow", "Quit"))
        self.actionQuit.setShortcut(_translate("MainWindow", "Ctrl+Q"))
        self.actionTo_File.setText(_translate("MainWindow", "to File..."))
        self.actionAbout_MDAT.setText(_translate("MainWindow", "About MDAT"))

