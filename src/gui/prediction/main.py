import sys
import os.path
import tqdm
import pickle
from datetime import datetime
from itertools import cycle
import seaborn as sns
import json
from skimage import io
import pandas as pd
from copy import deepcopy

# UI
import qdarkstyle
from PyQt5 import QtGui, QtWidgets
from qrangeslider import QRangeSlider
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
from pyqtgraph.Qt import QtCore
import pyqtgraph.opengl as gl
import prediction_tool_ui as PTUI
import numpy as np
from vispy import app, scene
from vispy.visuals.transforms import STTransform
from vispy.color import get_colormaps, BaseColormap

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.qt_compat import is_pyqt5
if is_pyqt5():
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)


# torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Internal
import utils.util as bmUtil
from utils.util import PRT
from utils.const import RenderingMethod, StainChannel, Phase, Dataset, AtlasConst, BrainAge
from utils.data.preprocessing import BMPreprocessing as BMPrep
from utils.data.microgliaDB import MDBLabel
from analysis.predictionAnalyzer import PredictionAnalyzer
from analysis.morphologyPredictor import build_args, MorphologyPredictor
from nuggt.utils.warp import Warper
from align.alignment import MouseAtlas


class ItemDelegate(QtGui.QStyledItemDelegate):
    """customized ItemDelegaet for ListWidget, where icon resides right side of the item"""

    def paint(self, painter, option, index):
        option.decorationPosition = QtGui.QStyleOptionViewItem.Right
        super(ItemDelegate, self).paint(painter, option, index)


class PredictionApp(QtGui.QMainWindow, PTUI.Ui_MainWindow):
    def __init__(self, **args):
        """init"""
        #phase = args.pop('phase')
        super(PredictionApp, self).__init__(**args)

        self.chn = StainChannel.IBA1
        #self.chn = StainChannel.ANTIGFP
        #self.chn = StainChannel.EGFP_ANTIGFP
        #self.chn = StainChannel.CFOS

        # setup UI
        self.setupUi(self)
        self.data_path = "/"
        self.mp = None          # MorphologyPredictor
        self.analyzer = None    # PredictionAnalyzer
        self.model_id = None        # Model identifier
        self.params = None      # Data Parameter (loaded from params.json)
        self.dpi = 96
        self.bmPrep = BMPrep()
        self.labels = {0: "Class 0", 1: "Class 1", 2: "Class 2"}

        # slider variables
        self.rsX = None
        self.rsY = None
        self.rsZ = None

        # setup ATLAS
        self.ATLAS_DIR = "/media/share5/MYK/ATLAS/mouse/"

        self.init_status()


    def init_status(self):
        self.disable(self.btnRunPrediction)
        self.disable(self.btnUpdate)


    def setup(self):
#        # create analyzer, load params
#        self.setup_analyzer()

        # link gui item connections
        self.link_actions()
        self.plotWidget = MatplotlibWidget()
        self.gLayoutPlot.addWidget(self.plotWidget, 0, 0, 1, 1)
        self.plotWidget2 = MatplotlibWidget()
        self.gLayoutPlot2.addWidget(self.plotWidget2, 0, 0, 1, 1)


    def link_actions(self):
        """link action functions to target UI items"""
        self.txtDataPath.mousePressEvent = self.data_path_text_clicked

        self.actionLoad_Path.triggered.connect(self.load_data_directory)
        self.actionQuit.triggered.connect(self.quit)
        self.btnLoadVolume.clicked.connect(self.load_volume_to_UI)
        self.btnRunPrediction.clicked.connect(self.run_prediction)
        self.btnUpdate.clicked.connect(self.update_volume_and_prediction)
        self.tabWidget.currentChanged.connect(self.tab_changed)
        self.alignTree.clicked.connect(self.treeItemClicked)


    def get_atlas_ann_file(self):
        ann_tif = self.params[AtlasConst.ALGN_INFO][AtlasConst.ANN_TIF]
        assert os.path.isfile(ann_tif)
        return ann_tif

    def get_atlas_rl_file(self, basef="AllBrainRegions.csv"):
        """retrieve AllBrainRebions.csv file"""

        f = os.path.join(self.ATLAS_DIR, basef)
        assert os.path.isfile(f)
        return f


    def tab_changed(self, i):
        self.log("Changing Tab -> [%s]"%self.tabWidget.tabText(i))


    def load_volume_to_UI(self):
        if self.params is None:
            self.log("Set Working Directory First!", PRT.ERROR)
            return

        if self.mp is None:
            # setup morphologyPredictor, init Canvas
            self.setup_predictor()

            # enable sliders
            self.enable(self.rsX)
            self.enable(self.rsY)
            self.enable(self.rsZ)

        self.enable(self.btnRunPrediction)
        self.disable(self.btnLoadVolume)


    def updateXRange(self, s):
        range = s.getRange()
        self.updateRange(axis=2, range=range)
        self.log("X Ranges updated to: {}".format(range))

    def updateYRange(self, s):
        range = s.getRange()
        self.updateRange(axis=1, range=range)
        self.log("Y Ranges updated to: {}".format(range))

    def updateZRange(self, s):
        range = s.getRange()
        self.updateRange(axis=0, range=range)
        self.log("Z Ranges updated to: {}".format(range))

    def updateRange(self, axis, range):
        # update ranges on MorphologyPredictor
        self.mp.set_range(axis, range)
        if not self.btnUpdate.isEnabled():
            self.enable(self.btnUpdate)

    def update_volume_and_prediction(self):
        self.mp.load_data(clim=self.params[self.chn]['clim'])
        self.run_prediction()
        self.disable(self.btnUpdate)

    def data_path_text_clicked(self, caller=None):
        self.load_data_directory()


    @staticmethod
    def validate_batch_no(text, vrange):
        """validate batch number see if it's integer, and ranged correctly

        :param vrange: required range [ min, max ]
        """
        vmin, vmax = vrange

        try:
            val = int(text)
        except ValueError:
            return False

        if val < vmin or val > vmax:
            return False

        return True


    def get_batch_no(self, text):
        """load batch by number passed"""

        vrange = [0, self.rawdata.total_num_batches-1]
        if self.validate_batch_no(text, vrange):
            val = int(text)
            return val
        else:
            return None


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

    def toggle_rendering(self):
        """toggle next rendering method radio button"""
        toggleNext = False
        for idx, rbtn in enumerate(self.rButtons+self.rButtons):
            if toggleNext:
                rbtn.setChecked(True)
                break
            if rbtn.isChecked():
                toggleNext = True

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
        self.rawdata = RawData(self.data_path, ext, self.phase)
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

        try:
            if isinstance(item, (list,)):
                for i in item:
                    i.setEnabled(True)
            else:
                item.setEnabled(True)
        except AttributeError:
            print("AttributeError() for %s"%item)


    def update_rawData(self):
        """update rawData class object, if anything dependent attribute has changed, or first time creation"""

        try:
            if self.rawdata is None:
                self.init_raw_data()
            else:
                self.rawdata.reset(self.data_path)
        except:
            self.log("Failed to load data files with datapath (%s). Try again with correct datapath."%self.data_path, PRT.ERROR)
            self.rawdata  = None


    def label_selected(self):
        """triggered when label is selected for the data
            (e.g. one of the three buttons is clicked)"""

        sender = self.sender()
        sendername = str(sender.text())

        # get filename
        currItem = self.batchList.currentItem()
        key = str(currItem.text())

        # check
        currItem.setIcon(self.checkIcon)

        for lb in self.lblButtons:
            label = str(lb.text()).split('\n')[0]
            label_no = self.mdbLabel.get_id(label)
            if lb == sender:
                self.log("You've selected [ {} ]".format(label), PRT.STATUS)
                self.batchLabel[key] = label_no
                self.mdbLabel.add(label_no)
            else:
                if lb.isChecked():
                    self.mdbLabel.subtract(label_no)
                    lb.toggle()

        self.stat_labels()
        # proceed to the next sample
        self.move_to_next()     # this function will also update button states


    def stat_labels(self):
        """get labels stats"""
        msg = self.mdbLabel.get_count_msg()
        msg_b = "  Set [ %s ] | Batch # [ %d ] | %s"%(self.rawdata.phase, self.current_bIdx, msg)
        self.lblStats.setText(msg_b)


    def move_to_next(self):
        """move on to the next sample on the list"""
        row = self.batchList.currentRow()
        if row + 1 >= self.rawdata.batch_size:
            # reach to the end
            self.log("Reached to the end of the batch, exporting labels...", PRT.STATUS)
            self.export_labels()
            self.log("Loading the next batch...", PRT.STATUS)
            self.start_new_batch()
        else:
            self.batchList.setCurrentRow(row+1)
            self.batchList.scrollToItem(self.batchList.currentItem())


    def update_label_buttons(self):
        """update label buttons based on the labels annotated"""

        label = None
        key = str(self.batchList.currentItem().text())
        if key in self.batchLabel.keys():
            label = self.batchLabel[key]

        if label is None:
            self.reset_label_buttons()
        else:
            label_name = self.mdbLabel.get_name(label)

            for lb in self.lblButtons:
                btnlabel = str(lb.text()).split('\n')[0]
                if label_name == btnlabel:
                    if not lb.isChecked():
                        lb.toggle()
                    self.disable(lb)
                else:
                    if lb.isChecked():
                        lb.toggle()
                        self.enable(lb)


    def reset_label_buttons(self):
        """reset label buttons clicked status"""

        for lb in self.lblButtons:
            self.enable(lb)
            if lb.isChecked():
                lb.toggle()


    def load_data_directory(self):
        """connect with menubar's Load option"""

        new_data_path = str(QtGui.QFileDialog.getExistingDirectory(None, 'Select a folder:', self.data_path,
                                                      QtGui.QFileDialog.ShowDirsOnly))

        if self.data_path == new_data_path:
            self.log("selected the same path as current, do nothing!", PRT.WARNING)
            return

        if new_data_path == "":
            self.log("cancelled.")
            return

        self.data_path = new_data_path
        self.log("got new data_path: {}".format(self.data_path), PRT.LOG)
        self.txtDataPath.setPlainText(self.data_path)

        self.setup_analyzer()
        # reset all values
        #self.run()


    def run_prediction(self):
        assert self.mp is not None
        isRaw = True if "RAW" in self.chn else False
        self.mp.run(self.full_inference, savePredictionArray=False, savePredictionLabel=True,
                    markers=["Class 0", "Class 1", "Class 2"], isRaw=isRaw)
        self.disable(self.btnRunPrediction)
        self.log("Updating Analysis...")
        self.update_analysis()
        self.log("Updating Analysis...(Done)")
        if self.atlas:
            self.log("Aligining...")
            self.atlas_map = self.align()
            self.log("Aligining...(Done)")
            self.update_alignment_analysis()
            self.update_tree(self.atlas_map)


    def update_analysis(self, nc=2):
        if self.analyzer is None:
            return

        self.analyzer.update_dataframe()
        self.analyzer.load_predictions([self.model_id])

        # Analysis Figures
        # 1. Pie Chart
        #self.draw_pie(nc)

        # 2. MaxProj
        maxProj = self.bmPrep._max_proj(self.mp.mVol)
        ax2 = self.plotWidget.getFigure().add_subplot(232)
        ax2.imshow(maxProj, clim=[800, 3500])
        ax2.axis('off')

        # 3. Heatmap
        self.draw_heatmap()

        # Draw
        self.plotWidget.draw()

        self.log("Drawing Analysis Figures. (Done)", PRT.STATUS)


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


    def draw_pie(self, nc=3):
        self.log("Drawing Analysis Figures.", PRT.STATUS)
        ax = self.plotWidget.getFigure().add_subplot(231)

        sizes = [len(self.mp.prediction_points[i]) for i in range(nc)]
        up_to = len(sizes) if nc is None else nc
        p, tx, autotexts = ax.pie(sizes[:up_to],
                                  labels=list(self.labels.values())[:up_to],
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


    def draw_heatmap(self):
        ramifieds = self.mp.prediction_points[0]
        amoeboids = self.mp.prediction_points[1]
        #garbages = self.mp.prediction_points[2]
        garbages = []
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

        ax1 = self.plotWidget.getFigure().add_subplot(233)
        sns.heatmap(heatmap_r, cmap='Reds', vmin=vmin, vmax=vmax, ax=ax1,
                    xticklabels=False, yticklabels=False)
        ax1.set_title("Heatmap of Class 0")
#        ax1.get_figure().tight_layout()

        ax2 = self.plotWidget.getFigure().add_subplot(234)
        sns.heatmap(heatmap_a, cmap='Greens', vmin=vmin, vmax=vmax, ax=ax2,
                    xticklabels=False, yticklabels=False)
        ax2.set_title("Heatmap of Class 1")

        ax3 = self.plotWidget.getFigure().add_subplot(235)
        sns.heatmap(heatmap_g, cmap='Blues', vmin=vmin, vmax=vmax, ax=ax3,
                    xticklabels=False, yticklabels=False)
        ax3.set_title("Heatmap of Class 2")

        ax4 = self.plotWidget.getFigure().add_subplot(236)
        sns.heatmap(heatmap_r, cmap='Reds', vmin=vmin, vmax=vmax, alpha=0.5, ax=ax4,
                    xticklabels=False, yticklabels=False, cbar=False)
        sns.heatmap(heatmap_a, cmap='Greens', vmin=vmin, vmax=vmax, alpha=0.5, ax=ax4,
                    xticklabels=False, yticklabels=False, cbar=False)
        sns.heatmap(heatmap_g, cmap='Blues', vmin=vmin, vmax=vmax, alpha=0.5, ax=ax4,
                    xticklabels=False, yticklabels=False, cbar=False)
        ax4.set_title("Heatmap Overlay")


    def setup_predictor(self):
        #chkpt = self.params[self.chn]['inf_dpath']
        chkpt = self.params['inf_rel_path']
        params, \
        argsByChn, \
        gargs, \
        ranges, \
        cc_csv, \
        full_inference, \
        clim = build_args(self.chn, chkpt, params_dict=self.params)

        # get placeholder size
        cw = self.gGLWidget.frameGeometry().width()
        ch = self.gGLWidget.frameGeometry().height()
        self.mp = MorphologyPredictor(title="MorphologyPredictor", keys='interactive',
                                      size=(cw, ch),
                                      show=True, logWindow=self.logwin,
                                      gargs=gargs, margs=argsByChn, ranges=ranges,
                                      in_memory=False, cc_csv=cc_csv,
                                      app='PyQt5', parent=self.gGLWidget)

        self.mp.load_data(clim=clim)
        print("clim: ", clim)
        print("mp.mVol.shape: ", self.mp.mVol.shape)
        print("len(mp.mPoints): ", len(self.mp.mPoints))


        # setup model
        # TODO: pass as argumnet

        if False:
            self.EXPR="20190118-152411-bumblebee"
            self.model_id = "BFMC-20190118-152411"
            pretrained = '/data_ssd/weights/models/%s/BFMC_Phase2_ALL_00279.pth'%(self.EXPR)
            num_clusters = '3'
        elif True:
            self.EXPR = "20190312-170708-bumblebee"
            self.model_id = "BMTR-20190312-170708"

            #EXPR="20180919-154658-bumblebee"
            #EXPR="20180830-140211-bumblebee"
            #self.EXPR="20181022-185608-bumblebee"
            #self.model_id = "BMTR-20181022-185608"
            #pretrained = '/data_ssd/weights/models/%s/BMTR_AE_00099.pth'%(self.EXPR)
            pretrained = '/data_ssd2/weights/models/%s/BMTR_AE_00200.pth'%(self.EXPR)
            num_clusters = '3'
            args_infer = ['morphologyPredictor', '-ph', Phase.REALTIME, '-bs', '10', '-e', '2',
                          '-ts', Phase.REALTIME, '-nc', num_clusters, '-ds', Dataset.MICROGLIA, '-us', 'False',
                          '-aw', pretrained, '-dp', self.mp.get_save_path(), '--debug']
        elif False:
            self.EXPR = "20200705-130651-bumblebee"
            self.model_id = "TRAP-20200705-130651"
            pretrained = '/home/mykim/cbm/src/train/weights/models/%s/TRAP_AE_00099.pth'%(self.EXPR)
            num_clusters = '2'

            args_infer = ['morphologyPredictor', '-ph', Phase.REALTIME, '-bs', '10', '-e', '2',
                          '-ts', Phase.REALTIME, '-nc', num_clusters, '-ds', Dataset.TRAP, '-us', 'False',
                          '-dw', '16', '-dh', '16', '-dd', '8',
                          '-aw', pretrained, '-dp', self.mp.get_save_path(), '--debug']

        self.mp.setup_model(args_infer, full_inference)
        self.full_inference = full_inference


    def setup_analyzer(self, paramf="params.json"):
        if self.analyzer is None:
            # create Analyzer object
            self.analyzer = PredictionAnalyzer(self.data_path, self.chn, labels=self.labels)
        else:
            print("update path")
            # update data_path and reconfig
            self.analyzer.update_data_root(self.data_path)

        # configure parameters
        self.analyzer.config_params()

        # load params and update table
        self.load_params()
        self.config_range_sliders()

        # make sure Load Volume button is enabled and others are disabled
        self.enable(self.btnLoadVolume)
        self.init_status()

        # setup atlas aligner
        try:
            self.load_atlas_annotations()
        except KeyError:
            self.log("No Atlas alignment available!", PRT.WARNING)
            self.atlas = None


    def load_atlas_annotations(self):
        alignment_json = self.params[AtlasConst.ALGN_INFO][AtlasConst.RS_ALGND_JSON]
        if self.params['age'] == BrainAge.ADULT:
            rl_file = self.get_atlas_rl_file('AllBrainRegions_New.csv')
        else:
            rl_file = self.get_atlas_rl_file('AllBrainRegions_Dev_New.csv')

        ann_file = self.get_atlas_ann_file()
        self.atlas = MouseAtlas(alignment_json, rl_file, ann_file)
        self.log("Loaded ATLAS annotations.")

        # init tree
        self.init_tree()
        self.log("Init Tree (Done).")


    def init_tree(self):
        root_parent_id = -1
        self.insert_node(root_parent_id, self.alignTree)


    def update_tree(self, selected):
        print("selected.keys: ", selected.keys())
        for idx, s in enumerate(list(selected.keys())):
            if s == 0:
                continue
            if s == "":
                continue
            item = self.alignTree.findItems(str(s), QtCore.Qt.MatchRecursive, 0)[0]
            item.setCheckState(0, QtCore.Qt.Checked)
            parent = item.parent()
            while parent is not None:
                self.alignTree.expandItem(parent)
                parent = parent.parent()


    def treeItemClicked(self, item):
        item = self.alignTree.currentItem()
        if item.checkState(0) == QtCore.Qt.Checked:
            rid = int(item.text(1))
            rname = str(item.text(0))
            if rname in list(self.atlas_map.keys()):
                self.log("Region Selected: [%d] %s"%(rid, rname))
                pts = self.get_all_points_from_atlas_map(rname)
                if "RAW" in self.chn:
                    pts[:, 0] *= 2
                    pts[:, 2] *= 2


                self.mp.overlay_marker(pts, rname, "m", color=[1.0, 0.0, 1.0, 0.7])


    def get_all_points_from_atlas_map(self, name):
        l = self.atlas_map[name]
        pts = []
        for k in l:
            pts += l[k]

        return np.array(pts)


    def insert_node(self, pid, parent):
        df_child = self.atlas.df[self.atlas.df['parent_structure_id'] == pid]

        for index, row in df_child.iterrows():
            nid = row['id']
            node = QtGui.QTreeWidgetItem(parent)
            node.setText(0, "%s"%(row['acronym']))
            #node.setText(1, "%d"%(index))
            node.setText(1, "%d"%(nid))
            node.setFlags(node.flags() | QtCore.Qt.ItemIsUserCheckable)
            node.setCheckState(0, QtCore.Qt.Unchecked)
            #self.insert_node(index, node)
            self.insert_node(nid, node)


    def align(self):
        ramifieds = self.mp.prediction_points[0]
        amoeboids = self.mp.prediction_points[1]
        #garbages = self.mp.prediction_points[2]
        garbages = []
        pts = ramifieds + amoeboids + garbages
        zr, yr, xr = self.get_initial_range()
        label_map = {}
        for idx, p in enumerate(pts):
            pg = self.mp.global_coordinates(deepcopy([p]), zr, yr, xr)[0]
            label_no, label, _ = self.atlas.get_label(list(pg))
            if label not in list(label_map.keys()):
                label_map[label] = {'Class 0':[], 'Class 1':[], 'Class 2': []}

            # append
            if idx < len(ramifieds):
                label_map[label]['Class 0'].append(p)
            elif idx < len(ramifieds) + len(amoeboids):
                label_map[label]['Class 1'].append(p)
            else:
                label_map[label]['Class 2'].append(p)

        print("label_map.keys: ", len(list(label_map.keys())))
        return label_map


    def createRangeSlider(self, addTo, min=0, max=100, range=None):
        rs = QRangeSlider()
        rs.show()
        rs.setMin(min)
        rs.setMax(max)
        rs.setStart(min)
        rs.setEnd(max)
        if range is not None:
            rs.setStart(range[0])
            rs.setEnd(range[1])

        rs.setBackgroundStyle('background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #222, stop:1 #333);')
        rs.setSpanStyle('background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #282, stop:1 #393);')
        addTo.addWidget(rs)
        return rs


    def get_full_size(self):
        dd = self.params['dd']
        dh = self.params['dh']
        dw = self.params['dw']

        return dd, dh, dw


    def get_initial_range(self):
        xr = self.params['xr']
        yr = self.params['yr']
        zr = self.params['zr']

        return zr, yr, xr


    def config_range_sliders(self):
        dd, dh, dw = self.get_full_size()
        zr, yr, xr = self.get_initial_range()
        self.log("Setting Ranges: xr: {}, yr: {}, zr: {}".format(xr, yr, zr))
        if self.rsX is None:
            self.rsX = self.createRangeSlider(self.hLayoutXR, max=dw, range=xr)
            self.rsX.startValueChanged.connect(lambda:self.updateXRange(self.rsX))
            self.rsX.endValueChanged.connect(lambda:self.updateXRange(self.rsX))
        if self.rsY is None:
            self.rsY = self.createRangeSlider(self.hLayoutYR, max=dh, range=yr)
            self.rsY.startValueChanged.connect(lambda:self.updateYRange(self.rsY))
            self.rsY.endValueChanged.connect(lambda:self.updateYRange(self.rsY))
        if self.rsZ is None:
            self.rsZ = self.createRangeSlider(self.hLayoutZR, max=dd, range=zr)
            self.rsZ.startValueChanged.connect(lambda:self.updateZRange(self.rsZ))
            self.rsZ.endValueChanged.connect(lambda:self.updateZRange(self.rsZ))

        self.disable(self.rsX)
        self.disable(self.rsY)
        self.disable(self.rsZ)


    def load_params(self):
        # load params to tableViewer
        params = self.analyzer.get_param_dict()
        print(params)

        #tableItem 	= QTableWidgetItem()
        table = self.tblParams

        # initiate table
        #table.setWindowTitle("QTableWidget Example @pythonspot.com")
        #table.resize(400, 250)
        keys = list(params.keys())
        cnt = 0
        for k in sorted(keys):
            if isinstance(params[k], dict):
                for sub_k in list(params[k]):
                    cnt += 1
            else:
                cnt += 1
        table.setRowCount(cnt)
        table.setColumnCount(3)
        hhdr = table.horizontalHeader()
        #hhdr.resizeSection(0, 70)
        hhdr.setResizeMode(0, QtGui.QHeaderView.Fixed)
        hhdr.setResizeMode(1, QtGui.QHeaderView.ResizeToContents)
        hhdr.setResizeMode(2, QtGui.QHeaderView.Stretch)

        idx = 0
        for k in sorted(keys):
            # set data
            v = params[k]
            if isinstance(v, dict):
                table.setItem(idx, 0, QtGui.QTableWidgetItem(k))
                for sub_k in sorted(list(v.keys())):
                    table.setItem(idx, 1, QtGui.QTableWidgetItem(sub_k))
                    sub_v = str(v[sub_k])
                    table.setItem(idx, 2, QtGui.QTableWidgetItem(sub_v))
                    idx += 1
            else:
                v = str(v)
                table.setItem(idx, 0, QtGui.QTableWidgetItem(k))
                table.setItem(idx, 2, QtGui.QTableWidgetItem(v))
                idx += 1

        # show table
        table.show()

        self.params = params


    def start_new_batch(self, batchno=False):
        """load next batch"""
        self.current_bIdx = 0
        #self.mdbLabel.reset_counter()

        # disable button
        self.btnStartNext.setText("Next (S)")
        self.btnStartNext.setShortcut("S")
        self.enable(self.btnExport)
        self.enable(self.lblButtons)

        # start with a batch
        self.run(batchno)


    def run(self, batchno=False):
        """actual entrypoint for data annotation process"""

        if self.rawdata is None:
            self.log("Load Data Directory first!", PRT.ERROR)
            return

        if batchno:
            val = self.get_batch_no(str(self.leBatchNo.text()))

            if val is None:
                self.log("invalid batch number...try again with DIGITS ONLY ranged between (0, %d)..."%(self.rawdata.total_num_batches-1), PRT.ERROR)
                return

            self.log("loading batch No [ %d ]..."%val, PRT.STATUS)

        QtWidgets.QApplication.processEvents()
        while True:
            current_bIdx, current_batch = self.rawdata.get_a_batch()
            if not batchno:
                break
            elif val == current_bIdx:
                break

        self.current_bIdx = current_bIdx
        self.update_batch_list(current_batch)
        self.log("loaded a batch (%d samples) (bIdx: %d)"%(len(current_batch[0]), current_bIdx), PRT.STATUS)


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


    def show_current_sample(self, current, prev):
        """show sample data selected from the ListWidget"""
        try:
            key = str(current.text())
        except AttributeError:
            # not ready
            return

        self.show_volume(current)
        self.update_label_buttons()
        self.check_if_already_labeled(key)


    def check_if_already_labeled(self, filename):
        """when a sample is selected, check if already labeled by looking at button states

        :param filename: list item's text as filename
        """
        for lb in self.lblButtons:
            if not lb.isEnabled():
                # already labeled!
                label = str(lb.text()).split('\n')[0]
                self.log("Sample [ %s ] is already labeled as [ %s ]"%(filename, label), PRT.WARNING)
                break


    def update_batch_list(self, batch):
        """update batch list on ListWidget, and numpy data batch map"""

        # clear data holders
        self.batchList.clear()
        self.batchData.clear()
        self.batchLabel.clear()

        savef = self.get_export_filename()
        if os.path.isfile(savef):
            # the latest annotation file already exists
            self.batchLabel = pickle.load(open(savef, "rb"))
            self.log("Annotation file for current batch (%d) already exists!"%self.current_bIdx, PRT.WARNING)
            self.log("Loading annotations from %s..."%savef, PRT.LOG)

        fnames, data = batch
        for idx, f in enumerate(fnames):
            d = data[idx]
            s = f.split('/')
            subset = s[7]
            basef = s[-1]
            key = '/'.join([subset, basef])
            item = QtGui.QListWidgetItem(key)
            # set check icon if already labeled
            if bool(self.batchLabel) and key in self.batchLabel.keys():
                label_no = self.batchLabel[key]
                self.mdbLabel.add(label_no)
                item.setIcon(self.checkIcon)


            self.batchList.addItem(item)
            self.batchData[key] = d

            if idx == 0:
                self.batchList.setCurrentItem(item)

        self.update_label_buttons()
        self.stat_labels()



    def export_labels(self):
        """export label map to disk (a pickle file)"""

        savef = self.get_export_filename()
        if os.path.isfile(savef):
            # file already exists, backup the original file
            ctime = datetime.now().strftime('%Y-%m-%d_%H%M%S')
            backupf = "{}_backup_{}.p".format(savef.split('.')[0], ctime)
            os.rename(savef, backupf)

        assert not os.path.isfile(savef)
        with open(savef, 'wb') as fp:
            pickle.dump(self.batchLabel, fp)

        self.log("exported current batch's labels to %s"%savef, PRT.LOG)


    def get_export_filename(self):
        """build export filename"""
        save_path = '/'.join([self.data_path, self.rawdata.phase, 'labels'])
        bmUtil.CHECK_DIR(save_path)
        savef = '/'.join([save_path, "annotations_batch_%05d.p"%self.current_bIdx])

        return savef


    def log(self, msg, flag=PRT.LOG):
        """log wrapper"""

        # show on statusbar
        self.statusbar.showMessage("[ STATUS ] %s"%msg)
        # show on logWindow
        self.logwin.append(PRT.html(self.__class__.__name__, msg, flag))
        self.logwin.moveCursor(QtGui.QTextCursor.End)
        QtWidgets.QApplication.processEvents()


    def _print(self):
        print("anything!!")


    def on_mouse_move(self, event):
        print("on_mouse_move")


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
    args = sys.argv
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("fusion");
#    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    ifa = PredictionApp()
    ifa.show()
    ifa.setup()
    sys.exit(app.exec_())
