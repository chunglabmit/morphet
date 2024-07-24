# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'prediction/prediction_tool_ui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1041, 884)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(10, 0, 1021, 651))
        self.tabWidget.setStyleSheet("QTabWidget::pane  {border: 1px solid gray; border-radius: 9px; margin-top: 0.5em; font: 75 10pt \"Ubuntu\";}")
        self.tabWidget.setObjectName("tabWidget")
        self.tabPrediction = QtWidgets.QWidget()
        self.tabPrediction.setObjectName("tabPrediction")
        self.layoutWidget = QtWidgets.QWidget(self.tabPrediction)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 10, 1001, 591))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gLayoutTAB = QtWidgets.QGridLayout(self.layoutWidget)
        self.gLayoutTAB.setContentsMargins(0, 0, 0, 0)
        self.gLayoutTAB.setObjectName("gLayoutTAB")
        self.gLayoutCanvas = QtWidgets.QGridLayout()
        self.gLayoutCanvas.setObjectName("gLayoutCanvas")
        self.gGLWidget = QtWidgets.QOpenGLWidget(self.layoutWidget)
        self.gGLWidget.setObjectName("gGLWidget")
        self.gLayoutCanvas.addWidget(self.gGLWidget, 0, 0, 1, 1)
        self.gLayoutTAB.addLayout(self.gLayoutCanvas, 0, 0, 1, 1)
        self.gLayoutParams = QtWidgets.QGridLayout()
        self.gLayoutParams.setObjectName("gLayoutParams")
        self.tabWidget_sub = QtWidgets.QTabWidget(self.layoutWidget)
        self.tabWidget_sub.setObjectName("tabWidget_sub")
        self.tabControl = QtWidgets.QWidget()
        self.tabControl.setObjectName("tabControl")
        self.gBoxDataPath = QtWidgets.QGroupBox(self.tabControl)
        self.gBoxDataPath.setGeometry(QtCore.QRect(7, 0, 312, 351))
        self.gBoxDataPath.setMaximumSize(QtCore.QSize(400, 10000))
        self.gBoxDataPath.setStyleSheet("QGroupBox {\n"
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
"}\n"
"\n"
"")
        self.gBoxDataPath.setObjectName("gBoxDataPath")
        self.txtDataPath = QtWidgets.QTextEdit(self.gBoxDataPath)
        self.txtDataPath.setGeometry(QtCore.QRect(10, 40, 291, 24))
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
        font.setItalic(True)
        self.txtDataPath.setFont(font)
        self.txtDataPath.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.txtDataPath.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.txtDataPath.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.txtDataPath.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.txtDataPath.setReadOnly(True)
        self.txtDataPath.setObjectName("txtDataPath")
        self.tblParams = QtWidgets.QTableWidget(self.gBoxDataPath)
        self.tblParams.setGeometry(QtCore.QRect(10, 90, 291, 251))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tblParams.sizePolicy().hasHeightForWidth())
        self.tblParams.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(7)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.tblParams.setFont(font)
        self.tblParams.setGridStyle(QtCore.Qt.DotLine)
        self.tblParams.setRowCount(30)
        self.tblParams.setColumnCount(3)
        self.tblParams.setObjectName("tblParams")
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setPointSize(9)
        item.setFont(font)
        self.tblParams.setItem(0, 0, item)
        self.tblParams.horizontalHeader().setVisible(False)
        self.tblParams.horizontalHeader().setCascadingSectionResizes(False)
        self.tblParams.horizontalHeader().setDefaultSectionSize(70)
        self.tblParams.horizontalHeader().setHighlightSections(True)
        self.tblParams.horizontalHeader().setMinimumSectionSize(50)
        self.tblParams.verticalHeader().setVisible(False)
        self.tblParams.verticalHeader().setCascadingSectionResizes(False)
        self.tblParams.verticalHeader().setDefaultSectionSize(13)
        self.tblParams.verticalHeader().setMinimumSectionSize(13)
        self.tblParams.verticalHeader().setSortIndicatorShown(True)
        self.tblParams.verticalHeader().setStretchLastSection(False)
        self.lblDataPath = QtWidgets.QLabel(self.gBoxDataPath)
        self.lblDataPath.setGeometry(QtCore.QRect(10, 20, 141, 18))
        self.lblDataPath.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.lblDataPath.setFrameShadow(QtWidgets.QFrame.Plain)
        self.lblDataPath.setTextFormat(QtCore.Qt.RichText)
        self.lblDataPath.setWordWrap(False)
        self.lblDataPath.setObjectName("lblDataPath")
        self.lblTableParams = QtWidgets.QLabel(self.gBoxDataPath)
        self.lblTableParams.setGeometry(QtCore.QRect(10, 70, 141, 18))
        self.lblTableParams.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.lblTableParams.setFrameShadow(QtWidgets.QFrame.Plain)
        self.lblTableParams.setTextFormat(QtCore.Qt.RichText)
        self.lblTableParams.setWordWrap(False)
        self.lblTableParams.setObjectName("lblTableParams")
        self.gBoxRanges = QtWidgets.QGroupBox(self.tabControl)
        self.gBoxRanges.setGeometry(QtCore.QRect(7, 355, 312, 121))
        self.gBoxRanges.setMaximumSize(QtCore.QSize(16777215, 300))
        self.gBoxRanges.setStyleSheet("QGroupBox {\n"
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
"}\n"
"\n"
"")
        self.gBoxRanges.setObjectName("gBoxRanges")
        self.gridLayoutWidget_3 = QtWidgets.QWidget(self.gBoxRanges)
        self.gridLayoutWidget_3.setGeometry(QtCore.QRect(0, 20, 311, 91))
        self.gridLayoutWidget_3.setObjectName("gridLayoutWidget_3")
        self.gLayoutRanges = QtWidgets.QGridLayout(self.gridLayoutWidget_3)
        self.gLayoutRanges.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.gLayoutRanges.setContentsMargins(0, 0, 0, 0)
        self.gLayoutRanges.setObjectName("gLayoutRanges")
        self.hLayoutXR = QtWidgets.QHBoxLayout()
        self.hLayoutXR.setObjectName("hLayoutXR")
        spacerItem = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.hLayoutXR.addItem(spacerItem)
        self.label = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.label.setObjectName("label")
        self.hLayoutXR.addWidget(self.label)
        self.gLayoutRanges.addLayout(self.hLayoutXR, 0, 0, 1, 1)
        self.hLayoutZR = QtWidgets.QHBoxLayout()
        self.hLayoutZR.setObjectName("hLayoutZR")
        spacerItem1 = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.hLayoutZR.addItem(spacerItem1)
        self.label_2 = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.label_2.setObjectName("label_2")
        self.hLayoutZR.addWidget(self.label_2)
        self.gLayoutRanges.addLayout(self.hLayoutZR, 2, 0, 1, 1)
        self.hLayoutYR = QtWidgets.QHBoxLayout()
        self.hLayoutYR.setObjectName("hLayoutYR")
        spacerItem2 = QtWidgets.QSpacerItem(10, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.hLayoutYR.addItem(spacerItem2)
        self.label_3 = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.label_3.setObjectName("label_3")
        self.hLayoutYR.addWidget(self.label_3)
        self.gLayoutRanges.addLayout(self.hLayoutYR, 1, 0, 1, 1)
        self.gBox_Control = QtWidgets.QGroupBox(self.tabControl)
        self.gBox_Control.setGeometry(QtCore.QRect(7, 480, 312, 61))
        self.gBox_Control.setStyleSheet("QGroupBox {\n"
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
        self.gBox_Control.setObjectName("gBox_Control")
        self.btnLoadVolume = QtWidgets.QPushButton(self.gBox_Control)
        self.btnLoadVolume.setGeometry(QtCore.QRect(12, 21, 90, 34))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.btnLoadVolume.setFont(font)
        self.btnLoadVolume.setObjectName("btnLoadVolume")
        self.btnRunPrediction = QtWidgets.QPushButton(self.gBox_Control)
        self.btnRunPrediction.setGeometry(QtCore.QRect(112, 21, 90, 34))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.btnRunPrediction.setFont(font)
        self.btnRunPrediction.setObjectName("btnRunPrediction")
        self.btnUpdate = QtWidgets.QPushButton(self.gBox_Control)
        self.btnUpdate.setGeometry(QtCore.QRect(212, 21, 90, 34))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.btnUpdate.setFont(font)
        self.btnUpdate.setObjectName("btnUpdate")
        self.tabWidget_sub.addTab(self.tabControl, "")
        self.tabAlignment = QtWidgets.QWidget()
        self.tabAlignment.setObjectName("tabAlignment")
        self.alignTree = QtWidgets.QTreeWidget(self.tabAlignment)
        self.alignTree.setGeometry(QtCore.QRect(6, 8, 311, 531))
        self.alignTree.setObjectName("alignTree")
        self.alignTree.header().setDefaultSectionSize(200)
        self.tabWidget_sub.addTab(self.tabAlignment, "")
        self.gLayoutParams.addWidget(self.tabWidget_sub, 0, 0, 1, 1)
        self.gLayoutParams.setRowStretch(0, 68)
        self.gLayoutTAB.addLayout(self.gLayoutParams, 0, 1, 1, 1)
        self.gLayoutTAB.setColumnStretch(0, 67)
        self.gLayoutTAB.setColumnStretch(1, 33)
        self.tabWidget.addTab(self.tabPrediction, "")
        self.tabAnalysis = QtWidgets.QWidget()
        self.tabAnalysis.setObjectName("tabAnalysis")
        self.gridLayoutWidget_4 = QtWidgets.QWidget(self.tabAnalysis)
        self.gridLayoutWidget_4.setGeometry(QtCore.QRect(10, 10, 961, 601))
        self.gridLayoutWidget_4.setObjectName("gridLayoutWidget_4")
        self.gLayoutPlot = QtWidgets.QGridLayout(self.gridLayoutWidget_4)
        self.gLayoutPlot.setContentsMargins(0, 0, 0, 0)
        self.gLayoutPlot.setObjectName("gLayoutPlot")
        self.tabWidget.addTab(self.tabAnalysis, "")
        self.tabAnalysis2 = QtWidgets.QWidget()
        self.tabAnalysis2.setObjectName("tabAnalysis2")
        self.gridLayoutWidget_5 = QtWidgets.QWidget(self.tabAnalysis2)
        self.gridLayoutWidget_5.setGeometry(QtCore.QRect(10, 10, 1001, 601))
        self.gridLayoutWidget_5.setObjectName("gridLayoutWidget_5")
        self.gLayoutPlot2 = QtWidgets.QGridLayout(self.gridLayoutWidget_5)
        self.gLayoutPlot2.setContentsMargins(0, 0, 0, 0)
        self.gLayoutPlot2.setObjectName("gLayoutPlot2")
        self.tabWidget.addTab(self.tabAnalysis2, "")
        self.gBox_logWindow = QtWidgets.QGroupBox(self.centralwidget)
        self.gBox_logWindow.setGeometry(QtCore.QRect(10, 650, 1021, 171))
        self.gBox_logWindow.setStyleSheet("QGroupBox {\n"
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
        self.gBox_logWindow.setObjectName("gBox_logWindow")
        self.logwin = QtWidgets.QTextEdit(self.gBox_logWindow)
        self.logwin.setGeometry(QtCore.QRect(10, 20, 1001, 141))
        font = QtGui.QFont()
        font.setPointSize(8)
        self.logwin.setFont(font)
        self.logwin.setObjectName("logwin")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1041, 20))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionLoad_Path = QtWidgets.QAction(MainWindow)
        self.actionLoad_Path.setObjectName("actionLoad_Path")
        self.actionQuit = QtWidgets.QAction(MainWindow)
        self.actionQuit.setObjectName("actionQuit")
        self.menuFile.addAction(self.actionLoad_Path)
        self.menuFile.addAction(self.actionQuit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        self.tabWidget_sub.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Prediction Viewer (Powered by Chung lab)"))
        self.gBoxDataPath.setTitle(_translate("MainWindow", "Parameters"))
        self.txtDataPath.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Noto Sans\'; font-size:10pt; font-weight:400; font-style:italic;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" color:#7a7a7a;\">Please load path using Menu (Ctrl+L)</span></p></body></html>"))
        __sortingEnabled = self.tblParams.isSortingEnabled()
        self.tblParams.setSortingEnabled(False)
        self.tblParams.setSortingEnabled(__sortingEnabled)
        self.lblDataPath.setText(_translate("MainWindow", "[ Working Directory ]"))
        self.lblTableParams.setText(_translate("MainWindow", "[ Pamameter Table ]"))
        self.gBoxRanges.setTitle(_translate("MainWindow", "Ranges"))
        self.label.setText(_translate("MainWindow", "X"))
        self.label_2.setText(_translate("MainWindow", "Z"))
        self.label_3.setText(_translate("MainWindow", "Y"))
        self.gBox_Control.setTitle(_translate("MainWindow", "Controls"))
        self.btnLoadVolume.setText(_translate("MainWindow", "Load Volume"))
        self.btnRunPrediction.setText(_translate("MainWindow", "Run Prediction"))
        self.btnUpdate.setText(_translate("MainWindow", "Update"))
        self.tabWidget_sub.setTabText(self.tabWidget_sub.indexOf(self.tabControl), _translate("MainWindow", "Control"))
        self.alignTree.headerItem().setText(0, _translate("MainWindow", "REGION"))
        self.alignTree.headerItem().setText(1, _translate("MainWindow", "ID"))
        self.tabWidget_sub.setTabText(self.tabWidget_sub.indexOf(self.tabAlignment), _translate("MainWindow", "Alignment"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabPrediction), _translate("MainWindow", "Visualization"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabAnalysis), _translate("MainWindow", "Analysis (Prediction)"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabAnalysis2), _translate("MainWindow", "Analysis (Atlas Alignment)"))
        self.gBox_logWindow.setTitle(_translate("MainWindow", "Logs"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionLoad_Path.setText(_translate("MainWindow", "Load Path"))
        self.actionLoad_Path.setShortcut(_translate("MainWindow", "Ctrl+L"))
        self.actionQuit.setText(_translate("MainWindow", "Quit"))
        self.actionQuit.setShortcut(_translate("MainWindow", "Ctrl+Q"))

