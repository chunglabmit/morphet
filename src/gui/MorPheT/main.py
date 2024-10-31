import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
import os.path
sys.path.append("../../")
import tqdm
import pickle
from datetime import datetime
from shutil import copyfile
import time
from itertools import cycle
from functools import partial
#import seaborn as sns
import json
from skimage import io
import pandas as pd
from copy import deepcopy
from glob import glob
import imagesize
import ast
import subprocess
import tifffile
import multiprocessing
import zarr
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from argparse import Namespace
from dask.array.image import imread as dimread
import vispy.io as vio

# UI
# Ensure using PyQt5 backend
import matplotlib
matplotlib.use('QT5Agg')
#import qdarkstyle
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import (QProcess, QThread, QUrl, pyqtSignal, pyqtSlot)
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtGui import QPixmap


import pyqtgraph as pg
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
from pyqtgraph.Qt import QtCore
import pyqtgraph.opengl as gl
import MorPheT as CUI
import numpy as np
from vispy import app, scene
from vispy.visuals.transforms import STTransform
from vispy.color import get_colormaps, BaseColormap

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
#from matplotlib.backends.qt_compat import is_pyqt5
#if is_pyqt5():
from matplotlib.backends.backend_qt5agg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
#else:
#    from matplotlib.backends.backend_qt4agg import (
#        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)


# torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils

# Phathom
from phathom import io

# Internal
from ccd.cell_center_detection import BM_CCD
import utils.util as bmUtil
from utils.util import PRT
from utils.const import RenderingMethod, StainChannel, \
                        Phase, Dataset, AtlasConst, ColorMap, \
                        BrainAge, LearningMethod
from utils.data.preprocessing import BMPreprocessing as BMPrep
from utils.data.microgliaDB import MDBLabel
from analysis.predictionAnalyzer import PredictionAnalyzer
from analysis.morphologyPredictor import build_args, MorphologyPredictor
from analysis.alignmentAnalyzer import AlignmentAnalyzer
from utils.data.tif2zarr import Tif2Zarr
from align.alignment import BMAligner, MouseAtlas
from annotate import AnnotationApp
from evaluator import EvaluatorApp


class Worker(QtCore.QObject):
    madeProgress = pyqtSignal([int, int])
    finished = pyqtSignal([np.ndarray, np.ndarray])
    finished_PHE = pyqtSignal()

    def __init__(self, func, args):
        super(Worker, self).__init__()
        self.func = func
        self.args = args


    @pyqtSlot()
    def run(self):
        self.args['progSignal'] = self.madeProgress
        mapped, mapped_n = self.func(**self.args)

        # emit the finished signal - we're done
        self.finished.emit(mapped, mapped_n)


    @pyqtSlot()
    def run_PHE(self):
        self.args['progSignal'] = self.madeProgress
        self.func(**self.args)

        # emit the finished signal - we're done
        self.finished_PHE.emit()


class UserInput(object):
    OWNER = "owner"
    MARKER = "marker"


class Objectives(object):
    O4X = "4x"
    O10X = "10x"
    O15X = "15x"

    VS = { O4X: [ 2.0, 1.8, 1.8 ],
           O10X: [ 2.0, 0.65, 0.65 ],
           O15X: [ 2.0, 0.41, 0.41 ]
          }


class ImageWindow(QtWidgets.QDialog):
    def __init__(self, widget, parent=None):
        super(ImageWindow, self).__init__(parent)

        self.app = parent
        self.setupUi(widget)
        self.dpath = None
        self.ext = None
        self.current_z = None
        self.point = None


    def setupUi(self, widget):
        self.plotWidget = widget
        self.layout = QVBoxLayout()


        self.btnPrev = QtWidgets.QToolButton(self)
        self.btnPrev.setText("Prev")
        self.btnPrev.clicked.connect(lambda:self.update_image(True))

        self.btnNext = QtWidgets.QToolButton(self)
        self.btnNext.setText("Next")
        self.btnNext.clicked.connect(lambda:self.update_image(False))

        self.bLayout = QHBoxLayout()
        self.bLayout.setAlignment(QtCore.Qt.AlignHCenter)
        self.bLayout.addWidget(self.btnPrev)
        self.bLayout.addWidget(self.btnNext)

        self.layout.addLayout(self.bLayout)
        self.layout.addWidget(self.plotWidget)

        self.setLayout(self.layout)


    def update_image(self, toPrev):
        if self.dpath is None:
            return

        self.current_z = self.current_z - 1 if toPrev else self.current_z + 1
        print("new z: ", self.current_z)

        imgpath = self.imgfiles[self.current_z]
        img = tifffile.imread(imgpath)
        self.im.set_data(img)
        self.fig.canvas.draw()


    def show(self):
        self.exec_()


    def setImage(self, dpath, ext, z, point, ps=40):
        self.dpath = dpath
        self.ext = ext
        self.current_z = z
        self.point = point

        self.fig = self.plotWidget.getFigure()
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)

        self.imgfiles = sorted(glob("%s/*.%s"%(self.dpath, self.ext)))
        imgpath = self.imgfiles[z]

        img = tifffile.imread(imgpath)
        yy, xx = img.shape
        hh = 800
        ww = int(xx * hh / float(yy))
        #self.setGeometry(500, 300, ww, 800)
        self.setGeometry(500, 300, 800, 800)

        x, y = self.point
        self.im = self.ax.imshow(img, cmap=ColorMap.CM_VOLUME, vmin=100, vmax=10000)
        self.ax.annotate("", xy=(x, y), xytext=(x+70, y-70),
                        arrowprops=dict(facecolor='red', edgecolor='r',
                        lw=3, arrowstyle='->'))

        self.fig.tight_layout()
        self.plotWidget.draw()




class TextDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(TextDialog, self).__init__(parent)

        self.setupUi()

    def setupUi(self):
        self.buttonBox = QtWidgets.QDialogButtonBox(self)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.textBrowser = QtWidgets.QTextBrowser(self)
        self.verticalLayout = QtWidgets.QVBoxLayout(self)
        self.verticalLayout.addWidget(self.textBrowser)
        self.verticalLayout.addWidget(self.buttonBox)

        for btn in self.buttonBox.buttons():
            btn.clicked.connect(self.close)


    def set_windowTitle(self, t):
        self.setWindowTitle(t)

    def set_text(self, t):
        self.textBrowser.setText(t)
        self.resize(QtCore.QSize(500, 700))
#        w.setStyleSheet('color: blue')
#        w.setFont(QtWidgets.QFont("Monospace"))
#        w.setWordWrapMode(QtWidgets.QTextOption.NoWrap)

class MorPheTApp(QtWidgets.QMainWindow, CUI.Ui_MorPheT):
    BM_CANVAS_NO = 0

    def __init__(self, **args):
        """init"""
        #phase = args.pop('phase')
        super(MorPheTApp, self).__init__(**args)

        # setup UI
        self.setupUi(self)

        # load AnnotationApp
        self.annApp = AnnotationApp(self)
        self.evalApp = EvaluatorApp(self)

        # load dialogs
        self.textDialog = TextDialog(self)

        # initialize variables
        self.appStartTime = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        self.checkIcon = QtGui.QIcon('./images/check_icon.png')
        self.user_params = {UserInput.OWNER: [self.ledit_owner, None],
                            UserInput.MARKER: [self.ledit_marker, None]
                            }
        self.objectiveButtons = [ self.rbtn_4x, self.rbtn_10x, self.rbtn_15x ]

        self.data_root = "/mnt/cephfs/MYK"
        self.selected_data = ""
        self.file_ext = None
        self.pre_params = {}
        self.app_params = None
        self.params = None
        self.curr_xr = None
        self.curr_yr = None
        self.curr_zr = None
        self.curr_sc = None

        # Cell Center Detection
        self.settings_on_startup = True
        self.ccd_params = None
        self.ccd = None
        self.ccd_s = None
        self.dpi = 84

        # Alignment
        self.aligner = None
        self.moving_tif = None
        self.aligned_point_json = None
        self.rescaled_aligned_point_json = None
        self.mAtlas = None      # rescaled to original resolution
        self.mAtlas_s = None    # non-rescaled
        self.axAlignMapped = None

        # Prediction & Analysis
        self.model_params = None    # dictionary of models
        self.model_id = None        # Model identifier
        self.usv_model_id = None    # Unsupervised Model identifier
        self.mp = None              # MorphologyPredictor
        self.analyzer = None        # PredictionAnalyzer


        self.bmPrep = BMPrep()
        self.base_marker_color = [1.0, 1.0, 0.0, 1.0]

        self.expandToggle = "<"
        self.shrinkToggle = "="
        self.dtab_expanded = False
        self.mtab_expanded = False

        self.worker_PHE = None

    def setup(self):
        # link gui item connections
        self.link_actions()

        # create plot widgets and add to grid layout
        self.plotWidget = self.create_a_plotWidget()
        self.plotWidget_pred = self.create_a_plotWidget()
        self.plotWidget_atlas = self.create_a_plotWidget()
        self.plotWidget_atlas_an = self.create_a_plotWidget()

        self.plotImageWindow = self.create_a_plotWidget()
        self.imageWindow = ImageWindow(self.plotImageWindow)

        self.gLayoutPlot.addWidget(self.plotWidget, 0, 0)   # (row, col, rowspan, colspan, alignment)
#        self.gLayoutPrediction.addWidget(self.plotWidget_pred, 0, 0)
        self.gLayoutAlign1.addWidget(self.plotWidget_atlas, 0, 0)
        self.gLayoutAlign1.setGeometry(QtCore.QRect(0, 0, 10000, 351))
        self.gLayoutAlign2.addWidget(self.plotWidget_atlas_an, 0, 0)
        self.gLayoutAlign2.setGeometry(QtCore.QRect(0, 0, 100, 200))

        # load app config
        configf = "app_config.json";
        if not os.path.isfile(configf):
            self.log("app_config.json file not found! It could result in fatal errors.",
                     PRT.ERROR)
        with open(configf) as fp:
            self.app_params = json.load(fp)
        self.app_params = Namespace(**self.app_params)
        self.log("App Config Params: {}".format(self.app_params))

        self.update_model_info()

        self.webViews = [self.webView_left, self.webView_right]


    def update_model_info(self):
        mpf = self.app_params.mpjson
        with open(mpf) as fp:
            self.model_params = json.load(fp)

        models = self.model_params.keys()
        # convert String class ID to Integer
        for m in models:
            mm = self.model_params[m]
            for k in mm['class'].keys():
                mm['class'][int(k)] = mm['class'].pop(k)

        self.log("Found %d models from %s"%(len(models), self.app_params.mpjson))
        sv_idx = 0
        usv_idx = 0
        for key in models:
            m = self.model_params[key]
            if m['learning'] == LearningMethod.SUPERVISED:
                tblObj = self.tblModelParams
                idx = sv_idx
                sv_idx += 1
            else:
                tblObj = self.tblUSVModelParams
                idx = usv_idx
                usv_idx += 1

            tblObj.insertRow(idx)
            tblObj.setItem(idx, 0, QtWidgets.QTableWidgetItem(key))
            tblObj.setItem(idx, 1, QtWidgets.QTableWidgetItem(m['net_type']))
            tblObj.setItem(idx, 2, QtWidgets.QTableWidgetItem(m['train_data']))
            tblObj.setItem(idx, 3, QtWidgets.QTableWidgetItem(str(m['num_class'])))
            tblObj.setItem(idx, 4, QtWidgets.QTableWidgetItem(str(m['version'])))
            tblObj.setItem(idx, 5, QtWidgets.QTableWidgetItem(m['model_file']))
            tblObj.resizeColumnsToContents()


    def create_a_plotWidget(self):
        pw = MatplotlibWidget(dpi=self.dpi)
        pw.fig.set_facecolor('black')
        pw.toolbar.setStyleSheet("margin-top:0px;background-color:#d4d4d4;QToolBar { border: 1px;background-color:#ffffff; }")
        pw.toolbar.setMaximumHeight(23)

        return pw


    def link_actions(self):
        """link action functions to target UI items"""
        self.txt_dpath.mousePressEvent = self.on_data_directory_selected

        self.ledit_owner.returnPressed.connect(lambda:self.set_param(UserInput.OWNER))
        self.ledit_marker.returnPressed.connect(lambda:self.set_param(UserInput.MARKER))

        self.cbox_subdirs.currentIndexChanged.connect(self.on_subdir_changed)
        self.cbox_alignChnList.currentIndexChanged.connect(self.on_alignChn_changed)

        for lb in self.objectiveButtons:
            lb.clicked.connect(self.on_objective_changed)

        self.btnConfigure.clicked.connect(self.configure)
        self.btnLock.clicked.connect(self.configLockToggle)
        self.btnGenZarr.clicked.connect(self.generate_zarr)
        self.btnCCD_setup.clicked.connect(self.CCD_setup)
        self.btnCCD_run.clicked.connect(self.CCD_run)
        self.btnCCD_runSubVol.clicked.connect(self.CCD_run_subvol)
        self.btnRescale.clicked.connect(self.ALGN_rescale)
        self.btnAlign.clicked.connect(self.ALGN_align)
        self.btnAlignView.clicked.connect(self.ALGN_view)
        self.btnAlignAnalysis.clicked.connect(self.ALGN_analysis)
        self.btnVizView.clicked.connect(self.VIZ_view)
        self.btnUpdate2.clicked.connect(self.VIZ_save_to_png)
#        self.btnAlignValidation.clicked.connect(self.ALGN_generate_url)

        self.tblParams.itemChanged.connect(self.on_param_table_changed)
        self.cBoxAtlasTif.currentIndexChanged.connect(self.on_atlas_tif_changed)
        self.treeAlign.clicked.connect(self.treeItemClicked)

        self.btnPhePrediction.clicked.connect(self.PHE_predict)
        self.tblModelParams.clicked.connect(self.PHE_on_model_selected)
        self.tblUSVModelParams.clicked.connect(self.PHE_on_usv_model_selected)
        self.tblURLs.clicked.connect(self.WEB_on_url_selected)

        # User Input related
        self.flayout = QtWidgets.QFormLayout()
        self.leUrlText = QtWidgets.QLineEdit()
#        self.flayout.addRow(self.btnUrlAdd, self.leUrlText)
        self.flayout.addRow(self.leUrlText)
        self.btnUrlAdd.clicked.connect(self.get_url_from_user)
        self.btnUrlDelete.clicked.connect(self.delete_url_from_table)

        self.action_ViewAllMetadata.triggered.connect(self.ACT_view_metadata)
        self.action_Quit.triggered.connect(self.ACT_quit)

#        self.rbtnMIP.toggled.connect(lambda:self.rendering_changed(self.rbtnMIP))

#        self.actionQuit.triggered.connect(self.quit)
#        self.btnUpdate.clicked.connect(self.update_volume_and_prediction)
#        self.tabWidget.currentChanged.connect(self.tab_changed)
        self.tabMetadata.tabBarClicked.connect(partial(self.tab_clicked, self.tabMetadata))
        self.tabModules.tabBarClicked.connect(partial(self.tab_clicked, self.tabModules))


    def tab_changed(self, i):
        self.log("Changing Tab -> [%s]"%self.tabMetadata.tabText(i))


    def tab_clicked(self, t, i):
#        idx = self.tabSteps.currentIndex()
        if t == self.tabModules:
            if i == 0:
                # shrink!
                self.tabModules.setMinimumSize(1255, 30)
                self.tabModules.setMaximumSize(1255, 30)
                self.mtab_expanded = False
            else:
                # expand
                self.tabModules.setMinimumSize(1255, 741)
                self.tabModules.setMaximumSize(1255, 741)
                self.mtab_expanded = True

        else:   # handle metadata tab size
            if self.expandToggle in t.tabText(i):
                # expand window
                if t == self.tabMetadata:
                    self.tabMetadata.setMinimumSize(317, 741)
                    self.tabMetadata.setMaximumSize(317, 741)
                    self.tabMetadata.setTabText(i, " %s "%self.shrinkToggle)
                    self.dtab_expanded = True
            elif self.shrinkToggle in t.tabText(i):
                # shrink window
                if t == self.tabMetadata:
                    self.tabMetadata.setMinimumSize(27, 30)
                    self.tabMetadata.setMaximumSize(27, 30)
                    self.tabMetadata.setTabText(i, " %s "%self.expandToggle)
                    self.dtab_expanded = False

        MAX_HEIGHT = 950
        MIN_HEIGHT = 241
        MIN_WIDTH = 1300
        MAX_WIDTH = 1600
        # determin main window size
        if (self.mtab_expanded and self.dtab_expanded) or (self.dtab_expanded and not self.mtab_expanded):
            self.resize_window(MAX_WIDTH, MAX_HEIGHT)
        else:
            if self.mtab_expanded:
                self.resize_window(MIN_WIDTH, MAX_HEIGHT)
            else:
                self.resize_window(MIN_WIDTH, MIN_HEIGHT)


    def resize_window(self, w, h):
        self.setMaximumSize(w, h)
        self.centralwidget.setMaximumSize(w, h)
        self.setMinimumSize(w, h)
        self.centralwidget.setMinimumSize(w, h)
        self.resize(w, h)
        self.centralwidget.resize(w, h)


    def ACT_view_metadata(self):
        """load params.json and view on TextView"""
        text = ""
        params_f = os.path.join(self.data_root, self.app_params.pjson)
        try:
            with open(params_f, "r") as f:
                for line in f:
                    text += line
        except FileNotFoundError:
            self.log("No params.json file found!!", PRT.ERROR)
            return

        self.textDialog.set_windowTitle(params_f)
        self.textDialog.set_text(text)
        self.textDialog.exec_()


    def ACT_quit(self):
        """quit the app"""
        for i in range(2, 0, -1):
            if i > 1:
                self.log("Closing in %d seconds..."%i, PRT.WARNING)
            elif i == 1:
                self.log("Goodbye!", PRT.ERROR)

            time.sleep(1)

        sys.exit(0) # why it doesn't fully kill the application??


    def get_marker_p(self):
        """return marker and its parameter dictionary"""
        marker = self.params['marker']
        return marker, self.params[marker]


    def VIZ_view(self):
        # visualize volume
        self.VIZ_show_volume()
        if self.chkBox_refreshTree.isChecked():
            # load alignment if exists
            self.VIZ_alignTree()


    def VIZ_alignTree(self):
        if self.aligner is None:
            self.log("Run Alignment (STEP 5) first to load Alignment Tree!", PRT.WARNING)
            return

        self.log("Updating Alignment Tree...")

        try:
            self.atlas_map = self.ALGN_align_by_coords(self.mp.mRawPoints, self.mp.mPoints)
            self.log("Aligining...(Done)")
            self.update_tree(self.atlas_map)

            self.log("Updating Alignment Tree...(Done)")
    #        self.update_alignment_analysis()
        except:
            self.log("Updating Alignment Tree...(Failed)", PRT.WARNING)


    def get_cc_npy_csv(self):
        cc_csv = None
        cc_npy = None
        marker, marker_dict = self.get_marker_p()

        try:
            cc_csv = os.path.join(self.data_root, marker_dict['cc_csv'])
            cc_npy = os.path.join(self.data_root, marker_dict['cc_npy'])
        except KeyError:
            self.log("No CC files are available!", PRT.WARNING)

        return cc_npy, cc_csv


    def create_morphPredictor(self, shuffle=True):
        # get placeholder size
        cw = self.gGLWidget.frameGeometry().width()
        ch = self.gGLWidget.frameGeometry().height()

        cc_npy, cc_csv = self.get_cc_npy_csv()
        self.ccd.update_cell_centers_file(cc_npy)

        # load parameters from .json file
        params_f = os.path.join(self.data_root, self.app_params.pjson)

        params, \
        argsByChn, \
        gargs, \
        ranges, \
        cc_csv, \
        full_inference, \
        clim = build_args(channel=self.params['marker'],
                          checkpoint=None, params_f=params_f)
        print("gargs: ", gargs)
        print("margs: ", argsByChn)

        mp = MorphologyPredictor(title="MorphologyPredictor", keys='interactive',
                                 size=(cw, ch), show=True, logWindow=self.logwin,
                                 gargs=gargs, margs=argsByChn, ranges=ranges,
                                 in_memory=True, cc_csv=cc_csv,
                                 app='PyQt5', parent=self.gGLWidget)
#        mp.setup_dg(ranges=ranges, ccd=self.ccd)
        mp.load_data(clim=clim, shuffle=shuffle)

        return mp


    def VIZ_show_volume(self):
        self.log("Visualizing volume...")
        if self.params is None or self.ccd_params is None or self.ccd is None:
            self.log("Run STEP 1 and 2 first!", PRT.ERROR)
            return

        if self.mp is None:
            self.mp = self.create_morphPredictor()

        clim = [self.ccd_params.clim_0, self.ccd_params.clim_1]
        self.log("clim: {}".format(clim))
        cc_npy, _ = self.get_cc_npy_csv()
        self.ccd.update_cell_centers_file(cc_npy)
        oc = True if cc_npy is not None else False
        zr_0, zr_1 = self.params['zr']
        #new_zr = [zr_0-self.ccd_params.damp, zr_1+self.ccd_params.damp]
        ranges = [self.params['zr'], self.params['yr'], self.params['xr']]
        self.mp.load_data(clim=clim, ccd=self.ccd, overlayCenters=oc, ranges=ranges,
                          marker_color=self.base_marker_color, z_damp=self.ccd_params.damp)
        if cc_npy:
            self.log("len(mp.mPoints): %d"%len(self.mp.mPoints))
        if self.mp.is_prediction_run:
            self.mp.ranges = ranges
            marker, _ = self.get_marker_p()
            isRaw = True if "RAW" in marker else False
            labels = [ "%s"%i for i in range(self.mp.trParams.num_clusters)]
            print("labels: ", labels)
            self.mp.run(self.params['full_inference'], savePredictionArray=False,
                        savePredictionLabel=True, markers=labels, isRaw=isRaw,
                        postfix=self.model_id, z_damp=self.ccd_params.damp)

        # Update Analysis section
        self.log("Visualizing volume...(Done)")


    def VIZ_save_to_png(self):
        """snapshot current view of the vispy 3-d view to a png file"""

        # check if snapshot directory exists
        bmUtil.CHECK_DIR(self.app_params.snapshot_dir)
        marker, _ = self.get_marker_p()
        savef = os.path.join(self.app_params.snapshot_dir, "BMCanvas_%s_%s_%03d.png"%(marker, self.appStartTime, self.BM_CANVAS_NO))
        img = self.mp.render()
        vio.write_png(savef, img)
        self.log("Snapshot saved to %s"%savef, PRT.STATUS2)
        self.BM_CANVAS_NO += 1


    def on_atlas_tif_changed(self):
        # load Atlas image snapshots
        # z-stack is always 228

        selected_atlas_tif = self.cBoxAtlasTif.currentText()
        if "-- select" in selected_atlas_tif:
            return

        tifp = os.path.join(self.app_params.atlas_dir, selected_atlas_tif)
        self.draw_alignment_vol(tifp)


    def draw_alignment_vol(self, tpath, isRef=True, normalize=False, dtype=np.int,
                           vrange=None):
        try:
            vol = dimread(tpath)[0].compute()
        except Exception:
            # TODO: add message
            self.log("Error loading TIF [ %s ]"%tpath, PRT.ERROR)
            return

        self.log("alignment vol.shape: {}".format(vol.shape)) # (228, 528, 320)
        z0 = 24 if isRef else 0
        z1 = len(vol)
        step = int((z1 - z0) / 10)
        self.log("step: %d"%step)

        vol_sel = np.asarray(vol[z0:z1:step, :, :], dtype=dtype)
        npimg = self.stitch_slices(vol_sel, ncol=len(vol_sel), normalize=normalize, vrange=vrange)

        # plot!
        fig = self.plotWidget_atlas.getFigure()

        if isRef:
            ax = fig.add_subplot(211)
            title = 'ATLAS TIF (Reference)'
        else:
            ax = fig.add_subplot(212)
            title = 'MOVING TIF (Data)'

        if vrange:
            ax.imshow(npimg, cmap=ColorMap.CM_VOLUME, vmin=vrange[0], vmax=vrange[1])
        else:
            ax.imshow(npimg, cmap=ColorMap.CM_VOLUME)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        self.style_ax(ax, title, title_loc='left')

        fig.tight_layout()
        self.plotWidget_atlas.draw()


    def stitch_slices(self, vol, ncol=16, vrange=None, normalize=False, scale_each=False):
        """stitch 2D slices into one big image

        Params
        ------
        vol: 3D volumetric images
        ncol: number of columns (in input view)
        normalize: normalize each slice
        scale_each: scale each slice
        vrange: tuple of min and max values of vol

        Return
        ------
        stitched 2D numpy array

        """

        # expand channel dim
        tensor = torch.from_numpy(vol).unsqueeze(1)
        x = vutils.make_grid(tensor, normalize=normalize, scale_each=scale_each, nrow=ncol,
                             range=vrange)
        npimg = np.transpose(x.numpy(), (1, 2, 0))

        return npimg


    def update_aligner(self):
        """Update BMAligner with the current values from GUI"""
        self.aligner.flip_x = True if self.chkBox_flipX.isChecked() else False
        self.aligner.flip_y = True if self.chkBox_flipY.isChecked() else False
        self.aligner.flip_z = True if self.chkBox_flipZ.isChecked() else False

        self.aligner.xidx = int(self.cbBox_xIdx.currentText())
        self.aligner.yidx = int(self.cbBox_yIdx.currentText())
        self.aligner.zidx = int(self.cbBox_zIdx.currentText())

        self.log("ALIGNER: xIdx: %d, yIdx: %d, zIdx: %d"%(self.aligner.xidx,
                                                          self.aligner.yidx,
                                                          self.aligner.zidx), PRT.STATUS)

        # quick handling for a whole brain
        selected_atlas_tif = self.cBoxAtlasTif.currentText()
        print("selected tif: ", selected_atlas_tif)
        # TODO: REMOVE THE CONDITION
#        if '25_whole' in selected_atlas_tif:
#            self.aligner.xidx=0
#            self.aligner.zidx=2

        self.aligner.print_flip_info()

        print("subdir: ", self.aligner.subdir)
        # update subdir
        self.aligned_on = self.cbox_alignChnList.currentText()
        self.aligner.subdir = self.params[self.aligned_on]['tif_rel_path']
        print("subdir (updated): ", self.aligner.subdir)

        # save to params.json
        p_ai = self.params[AtlasConst.ALGN_INFO] if AtlasConst.ALGN_INFO in self.params.keys() else {}
        p_ai[AtlasConst.FLIP_X] = self.aligner.flip_x
        p_ai[AtlasConst.FLIP_Y] = self.aligner.flip_y
        p_ai[AtlasConst.FLIP_Z] = self.aligner.flip_z
        p_ai[AtlasConst.XIDX] = self.aligner.xidx
        p_ai[AtlasConst.YIDX] = self.aligner.yidx
        p_ai[AtlasConst.ZIDX] = self.aligner.zidx
        p_ai[AtlasConst.ANN_TIF] = self.get_atlas_ann_file()
        p_ai[AtlasConst.ALGND_ON] = self.cbox_alignChnList.currentText()
        self.params[AtlasConst.ALGN_INFO] = p_ai
        self.save_params_to_json(self.app_params.pjson, self.params)



    def get_url_from_user(self):
        text, ok = QtWidgets.QInputDialog.getText(self, "URL from User", "Enter URL")

        if ok:
            self.insert_url_to_table(text)


    def insert_url_to_table(self, u):
        row_idx = self.tblURLs.rowCount()
        self.tblURLs.insertRow(row_idx)
        self.tblURLs.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(u))


    def delete_url_from_table(self):
        row_idx = self.tblURLs.selectionModel().selectedRows()
        for idx in sorted(row_idx):
            self.tblURLs.removeRow(idx.row())

        self.enable(self.btnAlignView)
        self.btnAlignView.setText("View")


    def ALGN_view(self):
        if self.aligner is None or self.moving_tif is None:
            self.log("Run Align first!", PRT.ERROR)
            return

        atlas_tif = os.path.join(self.app_params.atlas_dir,
                                 self.cBoxAtlasTif.currentText())

        self.log("Generating view links....", PRT.STATUS)
        self.disable(self.btnAlignView)
        self.btnAlignView.setText("Viewing...")

        ref_viewer, \
        moving_viewer = self.aligner.launch_viewer(self.moving_tif, atlas_tif,
                                                self.get_atlas_ann_file(),
                                                self.aligned_point_json)

        self.insert_url_to_table(ref_viewer)
        self.insert_url_to_table(moving_viewer)


    def ALGN_analysis(self):
#        if self.ccd is None or self.ccd_params is None:
#            self.log("Run [ STEP 3 ] first to load ROI volumes!", PRT.ERROR)
#            return
        if self.aligner is None or self.moving_tif is None:
            self.log("Run Rescale first!", PRT.ERROR)
            return

        self.log("Start Analyzing the alignment....", PRT.STATUS)
        self.disable(self.btnAlignAnalysis)
        self.btnAlignAnalysis.setText("Please wait...")

        # get a slice
        moving_vol = tifffile.imread(self.moving_tif)
        md, mh, mw = moving_vol.shape

        #z_loc = int(md / 3)
        z_real_loc = self.ccd_params.zr_0 + self.ccd_params.g_slice_no
        z_rel_loc = z_real_loc / int(self.ledit_rz.text())
        z_loc = int(md * z_rel_loc)
        if self.aligner.flip_z:
            z_loc = md - z_loc
        print("z_real_loc: {}, z_rel_loc: {}, z_moving_loc: {}".format(z_real_loc, z_rel_loc, z_loc))
        self.log("z_real_loc: {}, z_rel_loc: {}, z_moving_loc: {}".format(z_real_loc, z_rel_loc, z_loc))

        zr = [self.ccd_params.zr_0, self.ccd_params.zr_1]

        # create Worker Thread
        worker = Worker(self.map_alignment,
                        {'points': None, 'rescaled': False, 'vol': moving_vol, 'z_loc': z_loc})
        self.workerThread = QThread()
        worker.moveToThread(self.workerThread)

        worker.finished.connect(self.map_alignment_on_finished)
        worker.madeProgress.connect(self.update_pBar_worker)

        self.workerThread.started.connect(worker.run)
        self.workerThread.start()
        self.log("Worker started...")

        # plot!
        fig = self.plotWidget_atlas_an.getFigure()
        fig.canvas.mpl_connect('motion_notify_event', self.test_onHover)
        ax = fig.add_subplot(121)
        ax.imshow(moving_vol[z_loc], cmap=ColorMap.CM_VOLUME,
                  vmin=self.ccd_params.vol_clim[0], vmax=self.ccd_params.vol_clim[1])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        self.style_ax(ax, "Slice [ z=%d ]"%z_loc)

        self.test_vol = moving_vol[z_loc]

        fig.tight_layout()
        self.plotWidget_atlas_an.draw()


    def test_onHover(self, event):
        if self.axAlignMapped is None:
            return

        if event.inaxes == self.axAlignMapped:
            x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
            value = self.alignedMap[y][x]
            name = self.alignedMap_n[y][x]
#            self.ledit_alignedLabel.setText("[ %d, %s ]"%(value, name))
            labels = self.mAtlas_s.get_parent_labels(value)
            self.update_aligned_labels(labels)


    def update_aligned_labels(self, labels):
        if len(labels) == 1:
            return

        nrows = self.tblAlignedLabel.rowCount()
        l_reversed = labels[::-1]
        for i in range(nrows):
            if i < len(labels) - 1:
                space = " "*i
                self.tblAlignedLabel.item(i, 0).setText("%s%s"%(space, l_reversed[i+1]))
            else:
                self.tblAlignedLabel.item(i, 0).setText("")


    @pyqtSlot(int, int)
    def update_pBar_worker(self, val, total):
        self.update_progressbar(self.pBar, val, total)
        self.statusbar.showMessage("Thread Progress...[ %d / %d ]"%(val, total))


    @pyqtSlot(np.ndarray, np.ndarray)
    def map_alignment_on_finished(self, arr, arr_n):
        self.log("Ailgnment Mapping finished!", PRT.STATUS)
        fig = self.plotWidget_atlas_an.getFigure()
        ax = fig.add_subplot(122)
        ax.imshow(arr, cmap=ColorMap.CM_VOLUME)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        self.style_ax(ax, "Alignment")
        self.axAlignMapped = ax
        self.alignedMap = arr
        self.alignedMap_n = arr_n

        fig.tight_layout()
        self.plotWidget_atlas_an.draw()

        # quit the thread
        self.log("Quitting thread")
        self.workerThread.quit()  # this will quit **as soon as thread event loop unblocks**
        self.workerThread.wait()

        # re-enable the analysis button
        self.btnAlignAnalysis.setText("Analyze (C-8)")
        self.enable(self.btnAlignAnalysis)


    def map_alignment(self, points, rescaled=False, vol=None, z_loc=None, progSignal=None):
        """map to aligned coordinates from original coordinate using MouseAtlas()"""
        if not self.mAtlas or not self.mAtlas_s:
            self.log("Run [ STEP 5.2 ] first to create MouseAtlas()!", PRT.ERROR)
        atlas = self.mAtlas if rescaled else self.mAtlas_s
        mapped = []
        mapped_n = []
        print("rescaled? ", rescaled)

        if points is None:
            assert vol is not None
            assert z_loc is not None

            mh, mw = vol[z_loc].shape
            mapped = np.asarray(vol[z_loc])
            mapped_n = np.asarray(vol[z_loc], dtype="S110")
            for i in range(mw):
                for j in range(mh):
                    label_no, _, label = atlas.get_label([z_loc, j, i], debug=False)
                    mapped[j][i] = label_no
                    mapped_n[j][i] = label

                if progSignal:
                    progSignal.emit(i+1, mw)
        else:
            for idx, p in enuerate(points):
                _, label = atlas.get_label(p, debug=False)
                mapped.append(p)

        return mapped, mapped_n



    def ALGN_align(self):
        # check packages
        #if not self.check_pkg_for_alignment():
        #    self.log("Check depencencies on alignment-related packages!", PRT.ERROR)
        #    return False

        if self.aligner is None or self.moving_tif is None:
            self.log("Run Rescale first!", PRT.ERROR)
            return False

        self.log("Start aligning....", PRT.STATUS)
        self.disable(self.btnAlign)
        self.btnAlign.setText("Please wait...")

        atlas_tif = os.path.join(self.app_params.atlas_dir, self.cBoxAtlasTif.currentText())
        try:
            atlas_point_file = self.get_atlas_point_file()
        except AssertionError:
            self.log("Can't find file: coords_*.json! Check it and try again.", PRT.ERROR)
            self.btnAlign.setText("Run Alignment")
            self.enable(self.btnAlign)
            return False

        self.aligned_point_json = os.path.join(self.data_root,
                                        "%s_alignment.json"%self.cbox_alignChnList.currentText())

        # 1. ALGINMENT
        if os.path.isfile(self.aligned_point_json):
            self.log("Alignment is already done! [ %s ]"%self.aligned_point_json, PRT.WARNING)
        else:
            cmd = "sitk-align"
            args = [ "--moving-file", self.moving_tif,
                     "--fixed-file", atlas_tif,
                     "--fixed-point-file", atlas_point_file,
                     "--alignment-point-file", self.aligned_point_json,
                     "--xyz" ]

            self.log("cmd: {}, args: {}".format(cmd, args), PRT.STATUS)
            process = QProcess(self)
            process.finished.connect(self.ALGN_on_finished)
            process.startDetached(cmd, args)

            cnt = 0
            while not os.path.isfile(self.aligned_point_json):
                txt = "Alignment In Progress "
                num_dots = cnt % 10
                for i in range(num_dots):
                    txt += "."
                self.btnAlign.setText(txt)
                cnt += 1
                QtWidgets.QApplication.processEvents()
                time.sleep(2)

        # 2. Rescale Coordinates back to original
        paramf = os.path.join(self.data_root, self.app_params.pjson)
        self.rescaled_aligned_point_json = \
                    self.aligner.rescale_coords_to_original(self.aligned_point_json, self.moving_tif)

        # update params.json
        p_ai = self.params[AtlasConst.ALGN_INFO]
        p_ai[AtlasConst.ALGND_ON] = self.cbox_alignChnList.currentText()
        p_ai[AtlasConst.ALGND_JSON] = self.aligned_point_json
        p_ai[AtlasConst.RS_ALGND_JSON] = self.rescaled_aligned_point_json
        self.params[AtlasConst.ALGN_INFO] = p_ai
        self.save_params_to_json(self.app_params.pjson, self.params)


        self.mAtlas = self.create_mouse_atlas(self.rescaled_aligned_point_json,
                                              rl_file=self.get_atlas_rl_file(),
                                              ann_tif=self.get_atlas_ann_file())
        self.mAtlas_s = self.create_mouse_atlas(self.aligned_point_json,
                                                rl_file=self.get_atlas_rl_file(),
                                                ann_tif=self.get_atlas_ann_file())

        # initialize tree
        self.log("Initializing ATLAS Tree...", PRT.STATUS2)
        self.init_tree()
        self.log("Initializing ATLAS Tree... (Done)", PRT.STATUS2)

        self.btnAlign.setText("Run Alignment")
        self.btnAlign.setIcon(self.checkIcon)
        self.enable(self.btnAlign)
        self.enable(self.btnAlignView)
        self.enable(self.btnAlignAnalysis)




    def ALGN_on_finished(self, exitCode, exitStatus):
        #TODO: DOESN'T WORK!
        print("exitCode: ", exitCode)
        print("exitStatus: ", exitStatus)
        #self.log("cmd [ {} ] failed due to error: {}".format(cmd, e.output), PRT.ERROR)
        self.btnAlign.setText("Alignment Done")


    def get_atlas_point_file(self):
        atif = self.cBoxAtlasTif.currentText()
        regx = "_".join(atif.split("_")[1:]).split(".")[0]
        self.log("regx: %s"%regx)
        pf = glob("%s/*%s*.json"%(self.app_params.atlas_dir, regx))
        assert len(pf) == 1
        return pf[0]


    def get_atlas_ann_file(self):
        atif = self.cBoxAtlasTif.currentText()
        regx = "_".join(atif.split("_")[1:]).split(".")[0]
        self.log("regx: %s"%regx)
        pf = glob("%s/annotation_%s*"%(self.app_params.atlas_dir, regx))
        assert len(pf) == 1
        return pf[0]


    def get_atlas_rl_file(self, basef="AllBrainRegions.csv"):
        """retrieve AllBrainRebions.csv file"""
        sample_age = self.cbox_age.currentText()
        basef = "AllBrainRegions_New.csv" if BrainAge.ADULT in sample_age else "AllBrainRegions_Dev_New.csv"
        f = os.path.join(self.app_params.atlas_dir, basef)
        assert os.path.isfile(f)
        self.log("ATLAS_rl_file: %s"%basef)
        return f

    def create_aligner(self):
        try:
            marker = self.cbox_alignChnList.currentText()
            self.log("creating Aligner() with marker [ %s ]"%marker)
            self.aligner = BMAligner(self.data_root, self.selected_data,
                                     atlas_root=self.app_params.atlas_dir,
                                     marker=marker,
                                     qtLogWindow=self.logwin,
                                     qtProgressBar_r=self.pBar,
                                     qtProgressBar_a=self.pBar)

        except TypeError:
            self.log("Load Configuration (STEP 1) first!", PRT.ERROR)
            return

    def ALGN_rescale(self):
        if self.ccd_params is None:
            self.log("Set up CCD Params by running STEP2-1 first!", PRT.ERROR)
            return

        if self.aligner is None:
            self.create_aligner()

        atlas_tif = self.cBoxAtlasTif.currentText()
        if "-- select" in atlas_tif:
            self.log("Select Atlas TIF first!", PRT.ERROR)
            return

        self.update_aligner()

        self.moving_tif = self.aligner.rescale_image(atlas_tif)
        if not self.moving_tif:
            self.log("Atlas TIF %s doesn't exist!"%atlas_tif, PRT.ERROR)
            return

        # save moving_tif info to params file
        self.params[AtlasConst.ALGN_INFO][AtlasConst.DS_TIF] = self.moving_tif
        self.save_params_to_json(self.app_params.pjson, self.params)


        self.log("vol_clim: {}".format(self.ccd_params.vol_clim), PRT.STATUS2)
        self.draw_alignment_vol(self.moving_tif, isRef=False, normalize=True, dtype=np.float,
                                vrange=self.ccd_params.vol_clim)
        self.update_progressbar(self.pBar, 1, 1)

        # enable alignment button
        self.btnRescale.setIcon(self.checkIcon)
        self.enable(self.btnAlign)


    def on_param_table_changed(self, item):
        if not self.tblParams.isEnabled():
            return

        ridx = item.row()
        key = self.tblParams.verticalHeaderItem(ridx).text()
        cdict = vars(self.ccd_params)

        try:
            val = ast.literal_eval(self.tblParams.item(ridx, 0).text())
            cdict[key] = val
        except:
            self.log("Failed to update CCD params! Try it again.", PRT.ERROR)
            return

        if not self.CCD_verify_subset_range():
            self.log("Incorrect (xr, yr, or zr) range!", PRT.ERROR)

        self.log("Updated {} value to [ {} ]".format(key, val))



    def check_pkg_for_alignment(self):
        # rescaling tools
        cmds = [ "rescale-image-for-alignment",
                 "sitk-align",
                 "rescale-alignment-file"
               ]

        for c in cmds:
            if not self.is_cmd_there(c):
                self.log("%s is required to do alignment!"%c, PRT.ERROR)
                return False

        self.log("All of required packages are there! [ {} ]".format(cmds), PRT.STATUS2)
        return True


    def is_cmd_there(self, cmd):
        try:
            subprocess.check_output(cmd)
        except FileNotFoundError as e:
            self.log("Command [ %s ] not available"%cmd, PRT.ERROR)
            return False
        except subprocess.CalledProcessError as e:
            pass
#            self.log("cmd [ {} ] failed due to error: {}".format(cmd, e.output), PRT.ERROR)
#            return False

        return True


    def CCD_run(self):
        marker, marker_dict = self.get_marker_p()
        # check if CCD is already done
        if 'cc_npy' in marker_dict.keys():
            self.log("CCD is already done for the marker [ %s ]!"%marker, PRT.WARNING)
            return

        self.disable(self.btnCCD_run)
        self.log("Running CCD on entire data", PRT.STATUS2)
        zarr_path = os.path.join(self.data_root,
                                 self.params[marker]['zarr_rel_path'])
        try:
            zarr_d = io.zarr.open(zarr_path)
            zarr_prob_out = None
            # TODO: take gen_prob_map from GUI
            gen_prob_map = True
            if gen_prob_map:
                zarr_prob_out_path = os.path.join(self.data_root,
                                                  "%s_prob"%self.params[marker]['zarr_rel_path'])
                zarr_prob_out = io.zarr.new_zarr(zarr_prob_out_path, zarr_d.shape, zarr_d.chunks, np.float32)
        except AttributeError:
            self.log("Create ZARR first!", PRT.ERROR)
            self.enable(self.btnCCD_run)
            return


        zarr_s = None
        if self.ccd_params.subtract_chn is not None:
            zarr_s_path = os.path.join(self.data_root,
                                       self.params[self.ccd_params.subtract_chn]['zarr_rel_path'])
            zarr_s = io.zarr.open(zarr_s_path)
            self.log("loaded Zarr_s for subtraction...")

        print("ccd_params: {}".format, self.ccd_params)
        assert self.ccd is not None

        centers = self.ccd.calc_centroids_parallel(zarr_d,
                                                   sigma=self.ccd_params.sigma,
                                                   steepness=self.ccd_params.steepness,
                                                   offset=self.ccd_params.offset,
                                                   I0=self.ccd_params.mean,
                                                   stdev=self.ccd_params.std,
                                                   prob_thresh=self.ccd_params.threshold,
                                                   min_dist=self.ccd_params.min_dist,
                                                   min_intensity=self.ccd_params.min_intensity,
                                                   chunk_size=zarr_d.chunks,
                                                   overlap=self.ccd_params.overlap,
                                                   nb_workers=self.ccd_params.num_cpu,
                                                   normalize=self.ccd_params.normalize,
                                                   normByPercentile=self.ccd_params.norm_percentile,
                                                   clip=self.ccd_params.clip,
                                                   zarr_subtract=zarr_s,
                                                   norm_min=self.ccd_params.norm_min,
                                                   norm_max=self.ccd_params.norm_max,
                                                   prob_output=zarr_prob_out,
                                                   pbar=self.pBar)


        # get CCD Stats
        n = 0
        n_none = 0
        n_nn = 0
        nc = len(centers)
        for c in centers:
            if c is not None:
                n += c.shape[0]
                n_nn += 1
            else:
                n_none += 1

        self.log("Total: %d (None: %.2f%%, Not-None:%.2f%%)"%(nc, n_none/float(nc)*100., n_nn/float(nc)*100.))
        self.log("Total # of centers: %d"%n)
        centers_list = [c for c in centers if c is not None]
        self.log("len(centers_list): %d"%len(centers_list))
        centers_vstack = np.vstack(centers_list)

        # SAVE
        cc_npy_fname = 'cell_centers_%s.npy'%marker
        cc_csv_fname = 'cell_centers_%s.csv'%marker

        # save to npy
        np.save(os.path.join(self.data_root, cc_npy_fname), centers_vstack)

        # build dictionary
        cdict_all = []
        for idx, c in enumerate(centers_vstack):
            cdict = {}
            cdict['id'] = idx
            cdict['z'] = c[0]
            cdict['y'] = c[1]
            cdict['x'] = c[2]
            cdict['label'] = -1
            cdict_all.append(cdict)

        # save to csv
        df = pd.DataFrame(cdict_all)
        df = df[['id', 'z', 'y', 'x', 'label']]

        df.to_csv(os.path.join(self.data_root, cc_csv_fname), sep=',', index=False)
        # update params
        marker_dict['cc_npy'] = cc_npy_fname
        marker_dict['cc_csv'] = cc_csv_fname

        # save back
        self.save_params_to_json(self.app_params.pjson, self.params,
                                   "params.json updated with ccd file info!",
                                   PRT.STATUS2)

        self.enable(self.btnCCD_run)


    def CCD_setup(self):
        self.disable(self.btnCCD_runSubVol)
        self.disable(self.btnCCD_run)
        self.disable(self.tblParams)

        # check if params are loaded
        if self.params is None:
            self.log("Load Configuration (STEP 1) first!", PRT.ERROR)
            return

        # update XR, YR, ZR from params
        if not self.CCD_verify_subset_range():
            self.log("Specify correct subset Range (XR, YR, ZR) first!", PRT.ERROR)
            return

        self.ccd_params = self.get_ccd_params()
        self.ccd_params.norm_percentile = [self.ccd_params.normPct_0, self.ccd_params.normPct_1]
        self.ccd_params.vol_clim = (self.ccd_params.clim_0, self.ccd_params.clim_1)

        xr = [self.ccd_params.xr_0, self.ccd_params.xr_1]
        yr = [self.ccd_params.yr_0, self.ccd_params.yr_1]
        zr = [self.ccd_params.zr_0, self.ccd_params.zr_1]
        sc = self.ccd_params.subtract_chn
        if xr == self.curr_xr and yr == self.curr_yr and zr == self.curr_zr and sc == self.curr_sc:
            self.log("Same Volume, only updating parameters!", PRT.STATUS2)
        else:
            # Create BM_CCD & Load Subset
            try:
                # update overlapping parameters in params.json from ccd_params
                self.params['xr'] = xr
                self.params['yr'] = yr
                self.params['zr'] = zr

                self.statusbar.showMessage("Reading Files...")
                self.ccd = BM_CCD(self.params, self.params['marker'], self.ccd_params.g_slice_no,
                                  damp=self.ccd_params.damp,
                                  qtLogWindow=self.logwin, qtProgressBarSetup=self.pBar,
                                  qtProgressBarRun=self.pBar)

                if self.ccd_params.subtract_chn:
                    self.ccd_s = BM_CCD(self.params, self.ccd_params.subtract_chn,
                                        self.ccd_params.g_slice_no, damp=self.ccd_params.damp,
                                        qtLogWindow=self.logwin, qtProgressBarSetup=self.pBar,
                                        qtProgressBarRun=self.pBar)

                self.curr_xr = [self.ccd_params.xr_0, self.ccd_params.xr_1]
                self.curr_yr = [self.ccd_params.yr_0, self.ccd_params.yr_1]
                self.curr_zr = [self.ccd_params.zr_0, self.ccd_params.zr_1]
                self.curr_sc = sc
                self.statusbar.showMessage("Reading Files...(Done)")
            except TypeError:
                self.log("Load configuration first!", PRT.ERROR)
                return

        # enable buttons and table
        self.enable(self.btnCCD_runSubVol)
        self.enable(self.btnCCD_run)
        self.enable(self.tblParams)


    def CCD_run_subvol(self):
        if self.ccd_params is None:
            self.log("Run Setup to load CCD params first!", PRT.ERROR)
            return

        # update CCD params
        self.ccd.update_params(slice_no=self.ccd_params.g_slice_no, damp=self.ccd_params.damp,
                               xr=self.curr_xr, yr=self.curr_yr, zr=self.curr_zr)

        if self.ccd_s:
            self.ccd_s.update_params(slice_no=self.ccd_params.g_slice_no, damp=self.ccd_params.damp)

        clip = self.ccd_params.clip
        self.log("clip: {}, norm_percentile: {}".format(clip, self.ccd_params.norm_percentile))

        # Preprocess
        self.statusbar.showMessage("Preprocessing...")
        self.reset_progressbar(self.pBar)
        self.ccd.preprocess(normByPercentile=self.ccd_params.norm_percentile, clip=clip,
                            norm_min=self.ccd_params.norm_min, norm_max=self.ccd_params.norm_max)
        if self.ccd_s:
            self.update_progressbar(self.pBar, 1, 2)
            self.ccd_s.preprocess(normByPercentile=self.ccd_params.norm_percentile, clip=clip)

        self.update_progressbar(self.pBar, 1, 1)
        if not self.preprocess_plot(self.ccd, self.ccd_s):
            return

        # Detect Centers (subset)
        self.log("Detecting centers...")
        self.statusbar.showMessage("Detecting centers...")
        self.reset_progressbar(self.pBar)
        self.detect_centers_on_subset(self.ccd, self.ccd_s)
        self.update_progressbar(self.pBar, 1, 1)

        # save CCD params to JSON
        ccd_params_dict = vars(self.ccd_params)
        ccdp_f = self.get_ccd_param_file()
        if ccdp_f is None:
            ccdp_f = 'ccd_phathom_params_%s.json'%self.params['marker']

        self.save_params_to_json(ccdp_f, ccd_params_dict)
        self.log("saved ccd params to %s"%(ccdp_f), PRT.STATUS2)

        # update params.json
        self.params[self.params['marker']]['ccd_phathom_param_json'] = ccdp_f
        params = self.get_params_from_json(self.app_params.pjson)
        params[self.params['marker']]['ccd_phathom_param_json'] = ccdp_f
        self.save_params_to_json(self.app_params.pjson, params)


    def get_params_from_json(self, jsonfile):
        # check if the file already exists
        paramf = os.path.join(self.data_root, jsonfile)
        if os.path.exists(paramf):
            backupf = "%s.%s"%(jsonfile,
                               datetime.now().strftime('%Y-%m-%d_%H%M%S'))
            backupf = os.path.join(self.backup_dir, backupf)
            self.log("%s exists! Generate a backup [ %s ]."%(jsonfile, backupf),
                     PRT.STATUS)
            copyfile(paramf, backupf)

            with open(paramf) as fp:
                params = json.load(fp)

        else:
            params = {}

        return params


    def save_params_to_json(self, jsonfile, p, msg="A JSON file is updated.",
                              loglvl=PRT.LOG):
        paramf = os.path.join(self.data_root, jsonfile)

        with open(paramf, "w") as fp:
            json.dump(p, fp, indent=4, separators=(',', ':'), sort_keys=True)
            fp.write('\n')

        self.statusbar.showMessage("Saved %s..."%paramf)
        self.log(msg, loglvl)


    def get_ccd_param_file(self):
        ccdpf_key = 'ccd_phathom_param_json'
        paramsByChn = self.params[self.params['marker']]
        if ccdpf_key in paramsByChn.keys():
            return str(paramsByChn[ccdpf_key])
        else:
            return None


    def get_ccd_params(self):
        nrows = self.tblParams.rowCount()

        # check if previosly saved param file exists
        ccdpf = self.get_ccd_param_file()
        if self.settings_on_startup and ccdpf is not None:
            self.log("On STARTUP: Loading CCD params from %s"%ccdpf)
            p = self.get_params_from_json(ccdpf)
            # update table accordingly
            for i in range(nrows):
                key = self.tblParams.verticalHeaderItem(i).text()
                if key in ['xr_0', 'xr_1', 'yr_0', 'yr_1', 'zr_0', 'zr_1']:
                    # use Rrange Information from params.json!!
                    value = ast.literal_eval(self.tblParams.item(i, 0).text())
                    p[key] = value
                else:
                    self.tblParams.item(i, 0).setText(str(p[key]))
            self.settings_on_startup = False
        else:
            p = {}
            for i in range(nrows):
                key = self.tblParams.verticalHeaderItem(i).text()
                value = ast.literal_eval(self.tblParams.item(i, 0).text())
                self.log("{}: {}".format(key, value))
                p[key] = value



        return Namespace(**p)


    def detect_centers_on_subset(self, ccd, ccd_s=None):
        # TODO: mean, stdev from table
        to_subtract = ccd_s.vol if ccd_s else None
        ccd.detect_centers(sigma=self.ccd_params.sigma,
                           steepness=self.ccd_params.steepness,
                           offset=self.ccd_params.offset,
                           threshold=self.ccd_params.threshold,
                           min_dist=self.ccd_params.min_dist,
                           mean=self.ccd_params.mean, stdev=self.ccd_params.std,
                           arr_to_subtract=to_subtract, viz=False)

        self.ccd_subset_plot(ccd, ccd_s=ccd_s)


    def CCD_verify_subset_range(self):
        nrows = self.tblParams.rowCount()
        for i in range(nrows):
            key = self.tblParams.verticalHeaderItem(i).text()
            if key in [ "xr_0", "xr_1", "yr_0", "yr_1", "zr_0", "zr_1" ]:
                value = ast.literal_eval(self.tblParams.item(i, 0).text())
                if value is None:
                    return False
                else:
                    key_s, idx = key.split('_')
                    idx = int(idx)
                    self.params[key_s][idx] = int(value)

        xr0, xr1 = self.params['xr']
        yr0, yr1 = self.params['yr']
        zr0, zr1 = self.params['zr']

        try:
            if xr0 >= 0 and xr1 <= self.params['dw']:
                if yr0 >= 0 and yr1 <= self.params['dh']:
                    if zr0 >= 0 and zr1 <= self.params['dd']:
                        if xr0 < xr1 and yr0 < yr1 and zr0 < zr1:
                            return True
        except:
            return False

        return False


    def reset_progressbar(self, pbar):
        pbar.setValue(0)


    def update_progressbar(self, pbar, at, total):
        val = np.ceil(float(at)/float(total) * 100.)
        pbar.setValue(val)
        QtWidgets.QApplication.processEvents()


    def generate_zarr(self):
        try:
            p = self.params
            p['data_root'] = p['d_root']
            p['channel'] = p['marker']
            p['batchwise'] = True
            p['file_ext'] = self.file_ext
            p['zarr_chunk_size'] = [int(self.ledit_zz.text()), int(self.ledit_zy.text()), int(self.ledit_zx.text())]
        except TypeError:
            self.log("Load Configuration (STEP 1) first!", PRT.ERROR)
            return

        t2z = Tif2Zarr(Namespace(**p), self.logwin, self.pBar)
        t2z.convert()


    def validate_configuration(self):
        if self.data_root == "/":
            return False, "Data Path not selected."
        if self.selected_data == "":
            return False, "Data not selected."

        if self.file_ext is None:
            return False, "File extension is not set."

        if self.ledit_rz.text() == 'NaN':
            return False, "Ranges are not set."

        if self.ledit_owner.text() == "":
            return False, "Owner is not set."

        if self.ledit_marker.text() == "":
            return False, "Marker is not set."

        for key in self.user_params.keys():
            if self.user_params[key][1] is None:
                return False, "Param [ %s ] not set."%key

        if not bool(self.pre_params):
            return False, "pre_params are not set."

        return True, "Good!"


    def configLockToggle(self, lock=None):
        if lock is None:
            lock = self.btnLock.isChecked()

        if lock:
            self.disable(self.btnConfigure)
            self.btnLock.setChecked(True)
        else: # unlock
            self.enable(self.btnConfigure)
            # enable other textEdits
            for key in self.user_params:
                self.reset_param(self.user_params[key][0])
            self.btnLock.setChecked(False)


    def configure(self):
        # check if ready
        v, vm = self.validate_configuration()
        if not v:
            self.log("Complete configuration first! [Err: %s]"%vm, PRT.ERROR)
            return

        # generate param.json
        self.generate_param_json()

        # lock configuration
        self.configLockToggle(lock=True)


    def step2(self):
        # disable Step 1
        self.gridGboxS1.setEnabled(False)

        # enable Step 2
        self.gridGboxS2.setEnabled(True)


    def generate_param_json(self):
        zarr_rel_path = "%s_zarr"%self.selected_data
        dd = int(self.ledit_rz.text())
        dh = int(self.ledit_ry.text())
        dw = int(self.ledit_rx.text())

        self.params = self.get_params_from_json(self.app_params.pjson)
        params = self.params
        params['marker'] = self.pre_params['marker']
        params[self.pre_params['marker']] = {
                        "tif_rel_path": self.selected_data,
                        "zarr_rel_path": zarr_rel_path,
                        "shape": [dd, dh, dw],
                        "clim": [400, 1000] # default
                        }
        marker = self.pre_params['marker']
        try:
            if marker not in params['channels']:
                params['channels'].append(marker)
        except:
            params['channels'] = [ marker ]

        params['age'] = self.cbox_age.currentText()
        params['d_root'] = self.data_root
        params['d_raw'] = self.data_root
        params['dd'] = dd
        params['dh'] = dh
        params['dw'] = dw
        params['file_ext'] = self.file_ext
        params['full_inference'] = True
        params['inf_rel_path'] = 'inference'
        params['inf_patch_size'] = [16, 32, 32]
        params['name'] = self.data_root.split('/')[-1]
        params['owner'] = self.ledit_owner.text()
        params['voxel_size'] = [ float(self.ledit_vz.text()), float(self.ledit_vy.text()),
                                 float(self.ledit_vx.text()) ]
        params['zarr_chunk_size'] = [ float(self.ledit_zz.text()), float(self.ledit_zy.text()),
                                 float(self.ledit_zx.text()) ]
        params['xr'] = [0, dw]
        params['yr'] = [0, dh]
        params['zr'] = [int(dd/2-5), int(dd/2+5)]

        # save parameters to json
        self.log("Params: {}".format(params))
        self.log("Saving %s to %s"%(self.app_params.pjson, self.data_root))

        self.save_params_to_json(self.app_params.pjson, params)

        # save params for the next step
        self.params = params

        # update table params whichever set by here (for now, ranges)
        self.update_subset_range(self.params)


    def WEB_on_url_selected(self, item):
        # get text from selected row
        row = item.row()
        vidx = int(row%2)
        url = self.tblURLs.item(row, 0).text()
        QtWidgets.QApplication.clipboard().setText(url)
        self.log("Loading & Copy URL: %s"%url, PRT.STATUS)
        url = QUrl.fromUserInput(url)
        if not url.isValid():
            self.log("Invalid URL!", PRT.ERROR)
            return

        self.webViews[vidx].load(url)
        self.webViews[vidx].show()


    def set_param(self, p):
        "set parameter and related QLineEdit actions"
        ledit, ledit_action = self.user_params[p]
        self.pre_params[p] = ledit.text()
        if ledit_action is None:
            self.user_params[p][1] = ledit.addAction(self.checkIcon, QtWidgets.QLineEdit.TrailingPosition)
            self.user_params[p][1].triggered.connect(lambda:self.reset_param(ledit))

        # disable LineEdit
        ledit.setReadOnly(True)

        self.log("successfully stored parameter [ {} ] = {}".format(p, self.pre_params[p]))


    def on_subdir_changed(self):
        self.selected_data = self.cbox_subdirs.currentText()

        try:
            # get z
            files = glob("%s/*"%os.path.join(self.data_root, self.cbox_subdirs.currentText()))
            # get x, y
            if not os.path.isfile(files[0]):
                raise Exception

            x, y = imagesize.get(files[0])
            if x < 0 or y < 0:
                raise Exception

            # set file extension
            self.file_ext = files[0].split('.')[-1]
            self.txt_fileExt.setText(self.file_ext)

        except:
            self.log("Wrong directory! Please select one with image files!", PRT.ERROR)
            self.set_data_resolution("NaN", "NaN", "NaN")
            return

        self.log("Data Resolution (z, y, x): [ %d, %s, %s ]"%(len(files), y, x))
        self.set_data_resolution("%d"%len(files), "%s"%y, "%s"%x)


    def on_alignChn_changed(self):
#        idx = self.tabSteps.currentIndex()
#        self.log("Current Tab -> [%s]"%self.tabSteps.tabText(idx))
#        if self.tabSteps.tabText(idx) == "Alignment":
        self.create_aligner()



    def update_objective(self, voxel_size):
        for key, value in Objectives.VS.items():
            if value == voxel_size:
                for ob in self.objectiveButtons:
                    if ob.text() == key:
                        ob.setChecked(True)
                        break
                break


    def on_objective_changed(self):
        self.log("on_objective_changed!!!")
        sender = self.sender()
        objective = str(sender.text())
        self.log("objective changed to: %s"%objective)

        if objective == Objectives.O4X:
            self.set_voxel_size(Objectives.VS[Objectives.O4X])

        elif objective == Objectives.O10X:
            self.set_voxel_size(Objectives.VS[Objectives.O10X])

        elif objective == Objectives.O15X:
            self.set_voxel_size(Objectives.VS[Objectives.O15X])


    def set_voxel_size(self, vs, from_pjson=False):
        """set voxel size"""
        self.log("voxel_size: {}".format(vs))
        if from_pjson:
            scale = 1.0
        else:
            # check if downsampled
            scale = 2.0 if self.chkBox_downsampled.isChecked() else 1.0

        self.ledit_vz.setText("%.2f"%vs[0])
        self.ledit_vy.setText("%.2f"%(vs[1] * scale))
        self.ledit_vx.setText("%.2f"%(vs[2] * scale))


    def set_data_resolution(self, z, y, x):
        """set data resolution"""
        self.ledit_rz.setText(z)
        self.ledit_ry.setText(y)
        self.ledit_rx.setText(x)


    def reset_param(self, ledit):
        ledit.setReadOnly(False)
        for key in self.user_params:
            if self.user_params[key][0] == ledit:
                if self.user_params[key][1] is not None:
                    QtWidgets.QLineEdit.removeAction(ledit, self.user_params[key][1])
                    self.user_params[key][1] = None



    def updateVoxelSize(self, isInit=False):
        """set voxel size from GUI input"""
        try:
            vx = float(str(self.leVoxelX.text()))
            vy = float(str(self.leVoxelY.text()))
            vz = float(str(self.leVoxelZ.text()))
        except ValueError:
            self.log("Voxel Size type is Wrong! Check and try again...")
            return

        self.voxelSize = [vz, vy, vx]
        self.log("Voxel Size (vz, vy, vx): {}".format(self.voxelSize), PRT.STATUS2)

        if not isInit:
            self.update_volume_with_resizing()


    def toggle_voxelResizing(self):
        """toggle resizing option with voxelsize"""
        self.resizeVoxel = not self.resizeVoxel
        self.log("Voxel-Resizing %s"%("enabled" if self.resizeVoxel else "disabled"), PRT.STATUS2)

        # update volume
        self.update_volume_with_resizing()


    def toggle_colormap(self):
        """toggle colormap style of volume visualization"""
        self.canvas.set_volume_style(cmapToggle=True)


    def rendering_changed(self, rbtn):
        """change volume rendering method"""
        if rbtn.isChecked():
            self.rendering = str(rbtn.text())
            self.canvas.set_volume_style(method=self.rendering)

#        indexOfChecked = [self.ButtonGroup.buttons()[x].isChecked() for x in range(len(self.ButtonGroup.buttons()))].index(True)


    def quit(self):
        """quit the app"""

        sys.exit(0) # why it doesn't fully kill the application??


#    def init(self):
#        """initialize everything (e.g. GLViewer, RawData)
#           NOTE: This should be called after show() call of the class
#        """

#    def init_canvas(self, width, height):
#        """init canvas with vispy"""


    def init_raw_data(self):
        """initialize rawData() class, which would hold data itself and loading pipeline using torch's DataLoader"""

        ext = 'npy'
        self.rawdata = RawData(self.data_root, ext, self.phase)
        self.log("Loaded data files with RawData() successfully", PRT.LOG)


    def disable(self, item):
        """disable item (e.g. button), or list of items"""

        if isinstance(item, (list,)):
            for i in item:
                i.setEnabled(False)
        else:
            item.setEnabled(False)


    def enable(self, item):
        """enable item (e.g. button), or list of items"""

        if isinstance(item, (list,)):
            for i in item:
                i.setEnabled(True)
        else:
            item.setEnabled(True)


    def on_data_directory_selected(self, caller=None):
        """connect with menubar's Load option"""

        new_data_root = str(QtWidgets.QFileDialog.getExistingDirectory(None, 'Select a folder:', self.data_root,
                                                                   QtWidgets.QFileDialog.ShowDirsOnly))
        if new_data_root == "":
            self.log("cancelled.")
            return

        if self.data_root == new_data_root:
            self.log("selected the same path as current, do nothing!", PRT.WARNING)
            return

        self.data_root = new_data_root
        self.log("got new data_path: {}".format(self.data_root), PRT.LOG)
        self.txt_dpath.setPlainText(self.data_root)

        # load sub-directories
        subdirs = next(os.walk(self.data_root))[1]
        self.cbox_subdirs.clear()
        self.cbox_subdirs.addItems(subdirs)

        # check backup directory
        self.backup_dir = os.path.join(self.data_root, "morphet_backup")
        bmUtil.CHECK_DIR(self.backup_dir)


        # check if params.json is available
        paramf = os.path.join(self.data_root, self.app_params.pjson)
        # Load metadata to GUI
        if os.path.exists(paramf):
            self.log("%s exists! loading parameters..."%self.app_params.pjson)
            with open(paramf) as fp:
                params = json.load(fp)
            self.log("params: {}".format(params))

            # update GUI
            chn = params['marker']
            chn_params = params[chn]
            sd = chn_params['tif_rel_path']
            index = self.cbox_subdirs.findText(sd, QtCore.Qt.MatchFixedString)
            if index >= 0:
                 self.cbox_subdirs.setCurrentIndex(index)
            self.set_data_resolution(str(params['dd']), str(params['dh']), str(params['dw']))

            # set voxel size and objective
            self.set_voxel_size(params['voxel_size'], from_pjson=True)
            self.update_objective(params['voxel_size'])

            self.ledit_owner.setText(params['owner'])
            self.ledit_marker.setText(chn)
            index = self.cbox_age.findText(params['age'], QtCore.Qt.MatchFixedString)
            if index >= 0:
                 self.cbox_age.setCurrentIndex(index)
            self.set_param(UserInput.OWNER)
            self.set_param(UserInput.MARKER)

            # upate params
            self.params = params

            # check XR, YR, ZR
            self.update_subset_range(params)

            # check statistics
            if 'max' in list(chn_params.keys()):
                for i in range(self.tblZarrStat.rowCount()):
                    key = self.tblZarrStat.verticalHeaderItem(i).text()
                    if key in ['min', 'max', 'std', 'mean']:
                        self.tblZarrStat.item(i, 0).setText("%.2f"%chn_params[key])

            # as it already exists, lock configuration button
            self.configLockToggle(lock=True)

            # load channel list
            self.cbox_alignChnList.clear()
            self.cbox_alignChnList.addItems(self.params['channels'])

            # load Alignment metadata if exists
            if AtlasConst.ALGN_INFO in self.params.keys():
                algn_dict = self.params[AtlasConst.ALGN_INFO]
                index = self.cbox_alignChnList.findText(algn_dict[AtlasConst.ALGND_ON],
                                                        QtCore.Qt.MatchFixedString)
                # update channel box
                if index >= 0:
                     self.cbox_alignChnList.setCurrentIndex(index)

                # update reference tif selection box
                ann_tif_f = os.path.basename(algn_dict[AtlasConst.ANN_TIF])
                af_tif_f = ann_tif_f.replace(AtlasConst.ANNOTATION, AtlasConst.AUTOFLUORESCENCE)
                index = self.cBoxAtlasTif.findText(af_tif_f, QtCore.Qt.MatchFixedString)
                if index >= 0:
                    self.cBoxAtlasTif.setCurrentIndex(index)

                self.chkBox_flipX.setChecked(algn_dict[AtlasConst.FLIP_X])
                self.chkBox_flipY.setChecked(algn_dict[AtlasConst.FLIP_Y])
                self.chkBox_flipZ.setChecked(algn_dict[AtlasConst.FLIP_Z])

                # update axis info
                index = self.cbBox_xIdx.findText(str(algn_dict[AtlasConst.XIDX]), QtCore.Qt.MatchFixedString)
                if index >= 0:
                     self.cbBox_xIdx.setCurrentIndex(index)
                index = self.cbBox_yIdx.findText(str(algn_dict[AtlasConst.YIDX]), QtCore.Qt.MatchFixedString)
                if index >= 0:
                     self.cbBox_yIdx.setCurrentIndex(index)
                index = self.cbBox_zIdx.findText(str(algn_dict[AtlasConst.ZIDX]), QtCore.Qt.MatchFixedString)
                if index >= 0:
                     self.cbBox_zIdx.setCurrentIndex(index)


    def update_subset_range(self, p):
        nrows = self.tblParams.rowCount()
        for i in range(nrows):
            key = self.tblParams.verticalHeaderItem(i).text()
            if key in [ "xr_0", "xr_1", "yr_0", "yr_1", "zr_0", "zr_1" ]:
                key_s, idx = key.split('_')
                idx = int(idx)
                if p[key_s] is not None:
                    v = str(p[key_s][int(idx)])
                    self.tblParams.item(i, 0).setText(v)
            elif key in ["clim_0", "clim_1"]:
                key_s, idx = key.split('_')
                idx = int(idx)
                v = str(p[p['marker']]['clim'][int(idx)])
                self.tblParams.item(i, 0).setText(v)


    def PHE_on_model_selected(self, item):
        if self.model_params is None:
            self.log("Model Params are not loaded!", PRT.ERROR)
            return

        row = item.row()
        model_id = self.tblModelParams.item(row, 0).text()
        mparams = self.model_params[model_id]
        if 'desc' in mparams.keys():
            self.modelDesc.setText(self.model_params[model_id]['desc'])
        else:
            self.modelDesc.setText("No description Available.")

        self.model_id = model_id


    def PHE_on_usv_model_selected(self, item):
        if self.model_params is None:
            self.log("Model Params are not loaded!", PRT.ERROR)
            return

        row = item.row()
        model_id = self.tblUSVModelParams.item(row, 0).text()
        self.usv_model_id = model_id


    def PHE_predict(self):
        if self.worker_PHE is not None:
            self.log("A Prediction thread was running! trying to abort...", PRT.ERROR)
            self.worker_PHE.finished_PHE.emit()

        if self.params is None or self.ccd_params is None or self.ccd is None:
            self.log("Run STEP 1 and 2 first!", PRT.ERROR)
            return

        saveSamples = True if self.chkBox_pheSaveSample.isChecked() else False
        saveLabels = True if self.chkBox_pheSaveResult.isChecked() else False
        runSubClustering = True if self.chkBox_pheSubCluster.isChecked() else False
        if runSubClustering:
            if self.usv_model_id is None:
                self.log("Unsupervised model is not selected!", PRT.ERROR)
                return False

            # get unsupervised model
            usv_model = self.model_params[self.usv_model_id]['model_file']
        else:
            usv_model = None

        marker, _ = self.get_marker_p()
        isRaw = True if "RAW" in marker else False

        # check MorphologyPredictor
        if self.mp is None:
            self.mp = self.create_morphPredictor(shuffle=False)

        # setup model
        mparams = self.PHE_setup_analyzer_with_model(marker)
        if not mparams:
            self.log("Something's wrong during model setup!", PRT.ERROR)
            return

        labels = [ "%s"%i for i in range(self.mp.trParams.num_clusters)]
        print("labels: ", labels)

        in_data_size = self.params['inf_patch_size']
        print("in_data_size: ", in_data_size)

        # Run Prediction
        # create Worker Thread
        self.worker_PHE = Worker(self.mp.run,
                                {'full_inference': self.params['full_inference'],
                                 'savePredictionArray': saveSamples,
                                 'savePredictionLabel': saveLabels,
                                 'postfix': self.model_id,
                                 'markers': labels,
                                 'isRaw': isRaw,
                                 'usv_model_id': self.usv_model_id,
                                 'usv_model': usv_model,
                                 'rescale_input': True,
                                 'in_data_size': in_data_size
                                })
        self.workerThread_PHE = QThread()
        self.worker_PHE.moveToThread(self.workerThread_PHE)

        self.worker_PHE.finished_PHE.connect(self.PHE_predict_on_finished)
        self.worker_PHE.madeProgress.connect(self.update_pBar_worker)

        self.workerThread_PHE.started.connect(self.worker_PHE.run_PHE)
        self.workerThread_PHE.start()
        self.log("Worker started...")

        # non-threaded version
        #self.mp.run(self.params['full_inference'], savePredictionArray=saveSamples,
        #            savePredictionLabel=saveLabels,
        #            markers=labels,
        #            isRaw=isRaw,
        #            postfix=self.model_id,
        #            usv_model=usv_model
        #            )

        # Update Analysis section
#        self.PHE_update_analysis(mparams['class'])

    def PHE_predict_on_finished(self):
        self.log("Model inference finished!", PRT.STATUS)
        # quit the thread
        self.log("Quitting thread")
        self.workerThread_PHE.quit()  # this will quit **as soon as thread event loop unblocks**
        self.workerThread_PHE.wait()


    def PHE_setup_analyzer_with_model(self, marker):
        if self.model_params is None:
            self.log("Model Params are not loaded!", PRT.ERROR)
            return False

        if self.ccd is None:
            self.log("Run CCD to load centers first!", PRT.ERROR)
            return False

        # 1. get selected model from model table
        row_id = self.tblModelParams.currentRow()
        if row_id == -1:
            self.log("Select model first from the Model Table!", PRT.ERROR)
            return False
        self.model_id = self.tblModelParams.item(row_id, 0).text()
        self.log("Selected model: [ %s ]"%self.model_id)
        mparams = self.model_params[self.model_id]

        # 2. setup analyzer
        if self.analyzer is None:
            self.analyzer = PredictionAnalyzer(self.data_root,
                                               marker,
                                               labels=mparams['class'])
        else:
            print("update path")
            self.analyzer.update_data_root(self.data_root)
        self.analyzer.config_params()

        # 3. setup model
        # TODO: Read from model_params.json
        #patch_sz = self.params['inf_patch_size']
        args_infer = ['morphologyPredictor(MorPheT)', '-ph', Phase.REALTIME,
                        '-bs', '4', '-e', '2',
                        '-ts', Phase.REALTIME, '-nc', str(mparams['num_class']),
                        '-ds', Dataset.REALTIME, '-us', 'False', '-usv', 'False',
                        '-dw', str(mparams['net_input_dim'][2]),
                        '-dh', str(mparams['net_input_dim'][1]),
                        '-dd', str(mparams['net_input_dim'][0]),
                        '-aw', mparams['model_file'],
                        '-dp', self.mp.get_save_path(), '-ie', 'True',
                        '-mt', mparams['net_type'],
                        '-cl', str(mparams['clip_data']),
                        '-nt', mparams['norm_type'],
                        '--debug'
                        ]
        print("args_infer: ", args_infer)
        self.mp.setup_model(args_infer, self.params['full_inference'])

        return mparams


    def PHE_update_analysis(self, labels):
        self.log("Updating Analysis...")
        if self.analyzer is None:
            return

        self.analyzer.update_dataframe()
        self.analyzer.load_predictions([self.model_id])

        fig = self.plotWidget_pred.getFigure()
        fig.clf()

        # Analysis Figures
        # 1. Pie Chart
        self.draw_pie(labels)

        # 2. MaxProj
#        maxProj = self.bmPrep._max_proj(self.mp.mVol)
#        ax2 = fig.add_subplot(232)
#        ax2.imshow(maxProj, clim=[800, 3500])
#        ax2.axis('off')

        # 3. Heatmap
        self.draw_heatmap()

        # Draw
        self.plotWidget_pred.draw()

        self.log("Drawing Analysis Figures. (Done)", PRT.STATUS)
        self.log("Updating Analysis...(Done)")


    def update_alignment_analysis(self):
        self.log("Alignment Analysis (BEGIN)", PRT.STATUS)
        xs = []
        ys_all = []
        ys_0 = []
        ys_1 = []
        ys_2 = []
        for rname in self.atlas_map:
            pts = self.get_all_points_from_atlas_map(rname)
            xs.append(rname)
            ys_all.append(len(pts))
            ys_0.append(len(self.atlas_map[rname]['Class 0']))
            ys_1.append(len(self.atlas_map[rname]['Class 1']))
            ys_2.append(len(self.atlas_map[rname]['Class 2']))

        # all
        ax = self.plotWidget2.getFigure().add_subplot(211)
        ax.bar(xs, ys_all)
        ax.tick_params(axis='x', rotation=90, labelsize=7)
        ax.get_figure().tight_layout()

        # stacked
        ax2 = self.plotWidget2.getFigure().add_subplot(212)
        ax2.tick_params(axis='x', rotation=90, labelsize=7)
        ax2.bar(xs, ys_0, label='Class 0')
        ax2.bar(xs, ys_1, bottom=ys_0, label='Class 1')
        ax2.bar(xs, ys_2, bottom=np.array(ys_0)+np.array(ys_1), label='Class 2')
        ax2.legend()
        ax2.get_figure().tight_layout()

        # Draw
        self.plotWidget2.draw()

        self.log("Alignment Analysis (END)", PRT.STATUS)


    def draw_pie(self, labels):
        """draw chart
        :param labels: label information
        """
        self.log("Drawing Analysis Figures.", PRT.STATUS)
        nc = len(labels)
        ax = self.plotWidget_pred.getFigure().add_subplot(231)
        sizes = [len(self.mp.prediction_points[i]) for i in range(nc)]
        up_to = len(sizes) if nc is None else nc
        p, tx, autotexts = ax.pie(sizes[:up_to],
                                  labels=list(labels.values())[:up_to],
                                  colors = ['lightcoral', 'gold', 'yellowgreen', 'lightskyblue'],
                                  autopct='%1.1f%%',
                                  shadow=True,
                                  wedgeprops={"edgecolor":"k",'linewidth': 1, 'antialiased': True},
                                  startangle=90)
        for i, a in enumerate(autotexts):
            t = a.get_text()
            a.set_text("{}\n({:,})".format(t, sizes[i]))

        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax.set_title("[{} | {}]".format(self.params['age'], self.model_id))

        # draw global pie
        ax2 = self.plotWidget_pred.getFigure().add_subplot(234)
        self.analyzer.draw_pie(self.model_id, nc=nc, ax=ax2)



    def draw_heatmap(self):
        ramifieds = self.mp.prediction_points[0]
        amoeboids = self.mp.prediction_points[1]
        garbages = self.mp.prediction_points[2]
        print(len(ramifieds), len(amoeboids), len(garbages))
        d, h, w = self.mp.mVol.shape

        num_tiles = 10
        if h < 1500 or w < 1500:
            bh = int(float(h)/10.)
            bw = int(float(w)/10.)
        else:
            bh, bw = [150, 150]

        heatmap_r = np.zeros((int(h/bh+1), int(w/bw+1)))
        heatmap_a = np.zeros((int(h/bh+1), int(w/bw+1)))
        heatmap_g = np.zeros((int(h/bh+1), int(w/bw+1)))
        print(heatmap_r.shape, heatmap_a.shape)

        for r in ramifieds:
            by, bx = [int(r[1]/bh), int(r[2]/bw)]
            heatmap_r[by][bx] += 1

        for a in amoeboids:
            by, bx = [int(a[1]/bh), int(a[2]/bw)]
            heatmap_a[by][bx] += 1

        for g in garbages:
            by, bx = [int(g[1]/bh), int(g[2]/bw)]
            heatmap_g[by][bx] += 1

        print(np.max(heatmap_r), np.max(heatmap_a), np.max(heatmap_g))

        vmin=0
        vmax=max(max(np.max(heatmap_r), np.max(heatmap_a)), np.max(heatmap_g))

        ax1 = self.plotWidget_pred.getFigure().add_subplot(233)
        sns.heatmap(heatmap_r, cmap='Reds', vmin=vmin, vmax=vmax, ax=ax1,
                    xticklabels=False, yticklabels=False)
        ax1.set_title("Heatmap of Class 0")
#        ax1.get_figure().tight_layout()

        ax2 = self.plotWidget_pred.getFigure().add_subplot(235)
        sns.heatmap(heatmap_a, cmap='Greens', vmin=vmin, vmax=vmax, ax=ax2,
                    xticklabels=False, yticklabels=False)
        ax2.set_title("Heatmap of Class 1")

#        ax3 = self.plotWidget_pred.getFigure().add_subplot(235)
#        sns.heatmap(heatmap_g, cmap='Blues', vmin=vmin, vmax=vmax, ax=ax3,
#                    xticklabels=False, yticklabels=False)
#        ax3.set_title("Heatmap of Class 2")

        ax4 = self.plotWidget_pred.getFigure().add_subplot(236)
        sns.heatmap(heatmap_r, cmap='Reds', vmin=vmin, vmax=vmax, alpha=0.5, ax=ax4,
                    xticklabels=False, yticklabels=False, cbar=False)
        sns.heatmap(heatmap_a, cmap='Greens', vmin=vmin, vmax=vmax, alpha=0.5, ax=ax4,
                    xticklabels=False, yticklabels=False, cbar=False)
        sns.heatmap(heatmap_g, cmap='Blues', vmin=vmin, vmax=vmax, alpha=0.5, ax=ax4,
                    xticklabels=False, yticklabels=False, cbar=False)
        ax4.set_title("Heatmap Overlay")


    def create_mouse_atlas(self, alignment_json, rl_file, ann_tif):
        ma = MouseAtlas(alignment_json, rl_file, ann_tif)
        self.log("Loaded Mouse ATLAS annotations with [ %s ]."%alignment_json, PRT.STATUS)

        return ma


    def init_tree(self):
        root_parent_id = -1
        # check if alreacy initialized
        try:
            top = self.treeAlign.topLevelItem(0)
            if top.text(1) in [str(AtlasConst.ADULT_MOUSE_ROOT_ID),
                               str(AtlasConst.DEV_MOUSE_ROOT_ID)]:
                # tree already initialized
                self.log("ATLAS Tree is already initialized!", PRT.STATUS2)
            else:
                self.insert_node(root_parent_id, self.treeAlign)
        except:
            self.insert_node(root_parent_id, self.treeAlign)


    def update_tree(self, selected):
        for idx, s in enumerate(list(selected.keys())):
            if s in [AtlasConst.ADULT_MOUSE_ROOT_ID,
                     AtlasConst.DEV_MOUSE_ROOT_ID]:
                continue
            if s in ['']: # s is name not ID
                continue

            try:
                item = self.treeAlign.findItems(str(s), QtCore.Qt.MatchRecursive, 0)[0]
            except IndexError:
                print("idx: ", idx, "s: ", s)
                self.log("Error while updating Alignment Tree..!", PRT.ERROR)
                return

            item.setCheckState(0, QtCore.Qt.Checked)
            parent = item.parent()
            while parent is not None:
                self.treeAlign.expandItem(parent)
                parent = parent.parent()


    def treeItemClicked(self, item):
        marker = self.params['marker']
        item = self.treeAlign.currentItem()
        if item.checkState(0) == QtCore.Qt.Checked:
            rid = int(item.text(1))
            rname = str(item.text(0))
            if rname in list(self.atlas_map.keys()):
                self.log("Region Selected: [%d] %s"%(rid, rname))
                pts = self.get_all_points_from_atlas_map(rname)
                if "RAW" in marker:
                    pts[:, 0] *= 2
                    pts[:, 2] *= 2

                self.mp.overlay_marker(pts, rname, "m", color=self.base_marker_color)


    def get_all_points_from_atlas_map(self, name):
        l = self.atlas_map[name]
        pts = []
        for k in l:
            pts += l[k]

        return np.array(pts)


    def insert_node(self, pid, parent):
        df_child = self.mAtlas.df[self.mAtlas.df['parent_structure_id'] == pid]
        for index, row in df_child.iterrows():
            nid = row['id']
            node = QtWidgets.QTreeWidgetItem(parent)
            node.setText(0, "%s"%(row['acronym']))
            node.setText(1, "%d"%(nid))
            node.setFlags(node.flags() | QtCore.Qt.ItemIsUserCheckable)
            node.setCheckState(0, QtCore.Qt.Unchecked)
            self.insert_node(nid, node)


    def ALGN_align_by_coords(self, g_pts, l_pts):
        """run alignment on the list of coordinates

        Params
        ------
        :g_pts: global coordinates
        :l_pts: local coordinates
        """

        zr, yr, xr = self.get_current_zyx_range()
        label_map = {}
        total = len(l_pts)
        _, marker_dict = self.get_marker_p()
        raw_dim = marker_dict['shape']
        ds_dim = dimread(self.get_atlas_ann_file())[0].shape

        al_info = {}
        al_info['zidx'] = self.aligner.zidx
        al_info['yidx'] = self.aligner.yidx
        al_info['xidx'] = self.aligner.xidx
        al_info['flip_z'] = self.aligner.flip_z
        al_info['flip_y'] = self.aligner.flip_y
        al_info['flip_x'] = self.aligner.flip_x

        _, cc_csv = self.get_cc_npy_csv()
        df = pd.read_csv(cc_csv)
        if AtlasConst.REGION in df.columns:
            # region extraction is already done
            df = df[['z', 'y', 'x', AtlasConst.REGION]]
            # replace NaN with empty string
            df = df.fillna('')
        else:
            df = None

        for idx, p in enumerate(l_pts):
            pg = g_pts[idx]
            if df is not None:
                z, y, x = pg
                try:
                    label = df[((df['z']==z) & (df['y']==y) & (df['x']==x))][AtlasConst.REGION].values[0]
                except:
                    print("x, y, z: %d, %d %d"%(x, y, z))
                    label = ''
            else:
                #_, label_raw, _ = self.mAtlas.get_label(list(pg), debug=False)
                #print("pg: ", pg, "label (rescaled): ", label_raw)
                # convert coord downsampled space
                pg_n = AlignmentAnalyzer._raw_to_ds(pg, raw_dim, ds_dim, al_info)
                _, label_ds, _ = self.mAtlas_s.get_label(list(pg_n), debug=False)
                #print("pg_n: ", pg_n, "label_ds: ", label_ds)
                label = label_ds

            if label not in list(label_map.keys()):
                label_map[label] = {'Class 0':[]}
            # append
            label_map[label]['Class 0'].append(p)
            self.update_progressbar(self.pBar, idx, total)
            self.statusbar.showMessage("Aligning Coords...[ %d / %d ]"%(idx, total))

        return label_map


    def get_full_size(self):
        dd = self.params['dd']
        dh = self.params['dh']
        dw = self.params['dw']

        return dd, dh, dw


    def get_current_zyx_range(self):
        xr = [ self.ccd_params.xr_0, self.ccd_params.xr_1 ]
        yr = [ self.ccd_params.yr_0, self.ccd_params.yr_1 ]
        zr = [ self.ccd_params.zr_0, self.ccd_params.zr_1 ]

        return [zr, yr, xr]



    def update_volume_with_resizing(self):
        currItem = self.batchList.currentItem()
        if currItem is None:
            # nothing to do
            return
        self.show_volume(currItem)


    def show_volume(self, current):
        """show volume with current Item"""
        try:
            key = str(current.text())
            d = self.batchData[key].data.numpy().copy()
            if self.resizeVoxel:
                d = self.bmPrep._resize(d, self.voxelSize)
            self.canvas.set_volume(d)
        except:
            # not ready
            return


    def log(self, msg, flag=PRT.LOG):
        """log wrapper"""

        # show on statusbar
        if flag in [PRT.STATUS, PRT.STATUS2]:
            self.statusbar.showMessage("[ STATUS ] %s"%msg)

        # show on logWindow
        self.logwin.append(PRT.html(self.__class__.__name__, msg, flag))
        self.logwin.moveCursor(QtGui.QTextCursor.End)
        QtWidgets.QApplication.processEvents()


    def _print(self):
        print("anything!!")


    def on_mouse_move(self, event):
        print("on_mouse_move")


    def preprocess_plot(self, ccd, ccd_s=None):
        self.log("Plotting preprocess result...", PRT.STATUS2)
        datatype = self.params['marker']

        a_slice_s = None

        # RAW
        try:
            a_slice = ccd.get_a_slice(raw=True)
            if ccd_s:
                a_slice_s = ccd_s.get_a_slice(raw=True)
                datatype_s = ccd_s.channel
        except AttributeError or IndexError:
            self.log("g_slice_no seems not right, check and retry!", PRT.ERROR)
            return False

        if a_slice is None:
            self.log("g_slice_no seems not right, check and retry!", PRT.ERROR)
            return False

        fig = self.plotWidget.getFigure()
        fig.clf()

        ax = fig.add_subplot(421) if ccd_s else fig.add_subplot(431)
        #ax = fig.add_subplot(111)
        ax.imshow(a_slice, cmap=ColorMap.CM_VOLUME)
        self.style_ax(ax, '%s-raw'%datatype)

        #saved = "/media/ssdshare2/general/MYK/data/analysis/human/CCD"
        #fig.savefig(os.path.join(saved, "CCD_raw.pdf"), dpi=300)

        a_slice = ccd.get_a_slice()
        ax2 = fig.add_subplot(422) if ccd_s else fig.add_subplot(432)
        #ax2 = fig.add_subplot(111)
        ax2.imshow(a_slice, cmap=ColorMap.CM_VOLUME)
        self.style_ax(ax2, '%s-processed'%datatype)
        #fig.savefig(os.path.join(saved, "CCD_preprocessed.pdf"), dpi=300)

        if ccd_s:
            ax = fig.add_subplot(423)
            ax.imshow(a_slice, cmap=ColorMap.CM_VOLUME)
            self.style_ax(ax, '%s-raw'%datatype_s)
            a_slice_s = ccd_s.get_a_slice()
            ax = fig.add_subplot(424)
            ax.imshow(a_slice, cmap=ColorMap.CM_VOLUME)
            self.style_ax(ax, '%s-processed'%datatype_s)

        fig.tight_layout()
        self.plotWidget.draw()

        return True


    def ccd_subset_plot(self, ccd, reverse=True, ccd_s=None):
        self.log("Plotting CCD result...", PRT.STATUS2)

        fig = self.plotWidget.getFigure()
        print("ccd_s: ", ccd_s)

        if ccd_s:
            ax1 = fig.add_subplot(425)
            ax2 = fig.add_subplot(426)
            ax3 = fig.add_subplot(427)
            ax4 = fig.add_subplot(428)
        else:
            #ax1 = fig.add_subplot(323)
            #ax2 = fig.add_subplot(324)
            ax3 = fig.add_subplot(433)
            ax4 = fig.add_subplot(4, 3, (4, 12))
            #ax3 = fig.add_subplot(111)
            #ax4 = fig.add_subplot(111)

        #pos0 = ax1.imshow(ccd.m_probs_c[ccd.slice_no], cmap=ColorMap.CM_VOLUME)
        #self.style_ax(ax1, 'Curvature')
        #pos = ax2.imshow(ccd.m_probs_i[ccd.slice_no], cmap=ColorMap.CM_VOLUME)
        #self.style_ax(ax2, 'Intensity')

        pos = ax3.imshow(ccd.m_probs[ccd.slice_no], cmap=ColorMap.CM_VOLUME)
        self.style_ax(ax3, 'All')

        #saved = "/media/ssdshare2/general/MYK/data/analysis/human/CCD"
        #fig.savefig(os.path.join(saved, "CCD_All.pdf"), dpi=300)


        self.log("len(ccd.centroids), len(ccd.cs): %d, %d"%(len(ccd.centroids), len(ccd.cs)))
        new_cs = ccd.get_centroids_in_range(ccd.centroids, ccd.slice_no, ccd.damp)

        # get damp area and calculate max-proj
        start = max(0, ccd.slice_no - ccd.damp)
        end = min(self.ccd_params.zr_1 - self.ccd_params.zr_0 + 1, ccd.slice_no + ccd.damp)

        vol = []
        for i in range(start, end, 1):
            vol.append(ccd.get_a_slice(i))
        vol = np.array(vol)
        maxProj = self.bmPrep._max_proj(vol)
        pos = ax4.imshow(maxProj, cmap=ColorMap.CM_VOLUME)
        xidx, yidx = (2, 1) if reverse else (1, 2)
        if len(new_cs):
            ax4.scatter(new_cs[:, xidx], new_cs[:, yidx], alpha=0.6, s=5, color='red')
        self.style_ax(ax4, 'CCD result')

        #fig.savefig(os.path.join(saved, "CCD_Result.pdf"), dpi=300)

        fig.subplots_adjust(right=0.95, bottom=0.11, hspace=0.25)
        cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.02])
        cb = fig.colorbar(pos, cax=cbar_ax,
                          fraction=0.046, pad=0.04, orientation='horizontal')
        cb.ax.xaxis.set_tick_params(color='w')

        self.plotWidget.draw()


    def style_ax(self, ax, title, title_loc='center'):
        fig = self.plotWidget.getFigure()
        ax.set_title(title, color='w', loc=title_loc)
        ax.tick_params(colors='w')#, grid_color='w', grid_alpha=0.5)


    def test_plot(self):
        self.log("Testing Plot")
        labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
        sizes = [15, 30, 45, 10]
        explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
        ax = self.plotWidget.getFigure().add_subplot(231)
        ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                        shadow=True, startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax2 = self.plotWidget.getFigure().add_subplot(232)
        ax2.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                        shadow=True, startangle=90)
        ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax2.get_figure().tight_layout()
        self.plotWidget.getFigure().tight_layout()

        ax3 = self.plotWidget.getFigure().add_subplot(233)
        data = np.random.randn(50, 20)
        sns.heatmap(data, xticklabels=2, yticklabels=False, ax=ax3)
        ax3.get_figure().tight_layout()

        self.plotWidget.draw()

        self.log("Drawing Pie (Done)")


if __name__ == "__main__":
#    multiprocessing.set_start_method('spawn', force=True)
    args = sys.argv
    app = QtWidgets.QApplication(sys.argv)
#    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    ifa = MorPheTApp()
    ifa.setup()
    ifa.show()
    sys.exit(app.exec_())
