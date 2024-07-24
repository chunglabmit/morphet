import sys
import os
import pickle
import json
from datetime import datetime

# UI
from PyQt5 import QtGui

import vispy.app
from pyqtgraph.Qt import QtCore
import pyqtgraph.opengl as gl
import numpy as np
import tifffile
#from vispy import app, scene
#from vispy.visuals.transforms import STTransform
#from vispy.color import get_colormaps, BaseColormap

# torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Internal
#from utils.data.dataset import MicrogliaDataset
import utils.util as bmUtil
from utils.util import PRT
from utils.const import RenderingMethod
from utils.viz.bmCanvas import BMCanvas
from rawData import RawData
from utils.data.preprocessing import BMPreprocessing
from utils.data.dbLabel import MDBLabel, TDBLabel



class ItemDelegate(QtGui.QStyledItemDelegate):
    """customized ItemDelegaet for ListWidget, where icon resides right side of the item"""

    def paint(self, painter, option, index):
        option.decorationPosition = QtGui.QStyleOptionViewItem.Right
        super(ItemDelegate, self).paint(painter, option, index)


class AnnotationApp():
#    def __init__(self, **args):
#        """init"""
#        super(ATApp, self).__init__(**args)
    def __init__(self, qtApp):
        self.app = qtApp

        # set phase for DataSet
        self.phase = 'train'

        self.label_json = 'labels.json'
        self.label_dict = None
        self.lparams = None

        # member variables
        self.rawdata = None
        self.reset_rawdata = False
        # Temporary
#        self.data_path = None
        self.data_path = str(self.app.txtAnnDataPath.toPlainText())

        self.lblButtons = [self.app.btnRamified, self.app.btnIntermediate, self.app.btnAmoeboid,
                           self.app.btnGarbage, self.app.btnUncertain]
        self.lblButtons_inactive = []
        for lb in self.lblButtons_inactive:
            self.app.disable(lb)

        self.rButtons = [self.app.rbtnMIP, self.app.rbtnTranslucent,
                         self.app.rbtnAdditive]
        self.batchData = {}
        self.batchLabel = {}

        # TODO: get as an argument
        self.dbLabel = MDBLabel()
        #self.dbLabel = TDBLabel()

        self.bmPrep = BMPreprocessing()
        self.resizeVoxel = True if self.app.cbxResizing.isChecked() else False

        self.delegate = ItemDelegate()
        self.app.batchList.setItemDelegate(self.delegate)
        self.rendering = RenderingMethod.MIP
        self.clim_min, self.clim_max = (0, 1000)

        # link gui item connections
        self.link_actions()

        self.annotator_initialized = False


#        self.load_default_dir()


    def load_default_dir(self):
        if self.data_path is None:
            return

        self.log("got default data_path: {}".format(self.data_path))
        # load sub-directories
        subdirs = next(os.walk(self.data_path))[1]
        self.app.cbox_subdir_phase.clear()
        self.app.cbox_subdir_phase.addItems(subdirs)


    def init_annotator(self):
        self.updateVoxelSize(isInit=True)
        self.data_path = str(self.app.txtAnnDataPath.toPlainText())
        self.log("retrieved (Default) data path as %s"%self.data_path, PRT.LOGW)

        # init Canvas (GLViewer)
        cw = self.app.gGLWidget_2.frameGeometry().width()
        ch = self.app.gGLWidget_2.frameGeometry().height()
        # initialize canvas with BMCanvas
        self.canvas = BMCanvas(keys='interactive', size=(cw, ch),
                               show=True, logWindow=self.app.logwin,
                               parent=self.app.gGLWidget_2)

        # setup data
        self.init_raw_data()
        self.annotator_initialized = True


    def link_actions(self):
        """link action functions to target UI items"""

        self.app.txtAnnDataPath.mousePressEvent = self.load_ann_data_directory
        self.app.cbox_subdir_phase.currentIndexChanged.connect(self.on_phase_selected)
        self.app.btnStartNext.clicked.connect(self.start_new_batch)
        self.app.batchList.currentItemChanged.connect(self.show_current_sample)
        self.app.btnExport.clicked.connect(self.export_labels)
        for lb in self.lblButtons:
            lb.clicked.connect(self.label_selected)
        self.app.rbtnMIP.toggled.connect(lambda:self.rendering_changed(self.app.rbtnMIP))
        self.app.rbtnAdditive.toggled.connect(lambda:self.rendering_changed(self.app.rbtnAdditive))
        self.app.rbtnTranslucent.toggled.connect(lambda:self.rendering_changed(self.app.rbtnTranslucent))
        self.app.btnColormaps.clicked.connect(self.toggle_colormap)
        self.app.btnRenderings.clicked.connect(self.toggle_rendering)
        self.app.leBatchNo.returnPressed.connect(lambda:self.start_new_batch(True))
        self.app.cbxResizing.stateChanged.connect(self.toggle_voxelResizing)
        self.app.leVoxelX.returnPressed.connect(self.updateVoxelSize)
        self.app.leVoxelY.returnPressed.connect(self.updateVoxelSize)
        self.app.leVoxelZ.returnPressed.connect(self.updateVoxelSize)

        self.app.btnClimMin_dec.clicked.connect(lambda:self.update_clim_min(True))
        self.app.btnClimMin_inc.clicked.connect(self.update_clim_min)
        self.app.btnClimMax_dec.clicked.connect(lambda:self.update_clim_max(True))
        self.app.btnClimMax_inc.clicked.connect(self.update_clim_max)
        self.app.btnShowSlice.clicked.connect(lambda:self.refresh_volume(True))
        self.app.btnShowPlane.clicked.connect(lambda:self.load_XYPlane())


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


    def reconfigure_label_buttons(self):
        if self.label_dict is not None:
#        self.lblButtons = [self.app.btnRamified, self.app.btnIntermediate, self.app.btnAmoeboid,
#               self.app.btnGarbage, self.app.btnUncertain]
            labels = list(self.label_dict.keys())
            for idx, lb in enumerate(self.lblButtons):
                try:
                    new_label = labels[idx]
                    lb.setText(new_label)
                    sc = new_label.split(self.dbLabel.SHORTCUT_WILDCARD)[-1][0]
                    lb.setShortcut(sc)

                except IndexError:
                    lb.setText("")
                    lb.setShortcut("")


    def on_phase_selected(self):
        # TODO: ignore the top item as it's a guidance text

        self.phase = self.app.cbox_subdir_phase.currentText()
        self.reset_rawdata = True

        # reset label statistics
        self.dbLabel = MDBLabel(self.label_dict)

        # reconfigure label buttons
        self.reconfigure_label_buttons()

        self.app.btnStartNext.setText("&Load")
        self.app.btnStartNext.setShortcut("L")
        self.app.enable(self.app.btnStartNext)
        self.app.disable(self.app.btnExport)
        self.app.disable(self.lblButtons)


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
            vx = float(str(self.app.leVoxelX.text()))
            vy = float(str(self.app.leVoxelY.text()))
            vz = float(str(self.app.leVoxelZ.text()))
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


    def init_raw_data(self):
        """initialize rawData() class, which would hold data itself and loading pipeline using torch's DataLoader"""

        ext = 'npy'
        self.rawdata = RawData(self.data_path, ext, self.phase)
        self.log("Loaded data files with RawData() successfully")


    def update_rawData(self):
        """update rawData class object, if anything dependent attribute has changed, or first time creation"""

        try:
            if self.rawdata is None:
                self.init_raw_data()
            else:
                self.rawdata.reset(self.data_path, self.phase)
        except:
            self.log("Failed to load data files with datapath (%s). Try again with correct datapath."%self.data_path, PRT.ERROR)
            self.rawdata  = None


    def get_id_of_button(self, btn):
        # remove "&" which was used for defining as shortcut key before returning
        return str(btn.text()).replace('&', '')


    def label_selected(self):
        """triggered when label is selected for the data
            (e.g. one of the three buttons is clicked)"""
        sender = self.app.sender()
        sendername = self.get_id_of_button(sender)

        # get filename
        currItem = self.app.batchList.currentItem()
        key = str(currItem.text())

        # check
        currItem.setIcon(self.app.checkIcon)

        for lb in self.lblButtons:
            label = self.get_id_of_button(lb)
            label_no = self.dbLabel.get_id(label)
            if lb == sender:
                self.log("You've selected [ {} ]".format(label), PRT.STATUS)
                self.batchLabel[key] = label_no
                self.dbLabel.add(label_no)
                # disable button
                self.app.disable(lb)
            else:
                #if lb.isChecked():
                if not lb.isEnabled(): # previous label
                    self.dbLabel.subtract(label_no)
                    self.app.enable(lb)

        self.stat_labels()
        # proceed to the next sample
        self.move_to_next()     # this function will also update button states


    def stat_labels(self):
        """get labels stats"""
        msg = self.dbLabel.get_count_msg()
        msg_b = "  Set [ %s ] | Batch # [ %d ] | %s"%(self.rawdata.phase, self.current_bIdx, msg)
        self.app.lblStats.setText(msg_b)


    def move_to_next(self):
        """move on to the next sample on the list"""
        row = self.app.batchList.currentRow()
        if row + 1 >= self.rawdata.batch_size:
            # reach to the end
            self.log("Reached to the end of the batch, exporting labels...", PRT.STATUS)
            self.export_labels()
            self.log("Loading the next batch...", PRT.STATUS)
            self.start_new_batch()
        else:
            self.app.batchList.setCurrentRow(row+1)
            self.app.batchList.scrollToItem(self.app.batchList.currentItem())


    def update_label_buttons(self):
        """update label buttons based on the labels annotated"""

        key = str(self.app.batchList.currentItem().text())
        label = self.batchLabel[key] if key in self.batchLabel.keys() else None

        if label is None:
            self.reset_label_buttons()
        else:
            label_name = self.dbLabel.get_name(label)
            for lb in self.lblButtons:
                btnlabel = self.get_id_of_button(lb)
                if label_name == btnlabel:
                    self.app.disable(lb)
                else:
                    self.app.enable(lb)


    def reset_label_buttons(self):
        """reset label buttons clicked status"""
        for lb in self.lblButtons:
            self.app.enable(lb)


    def load_ann_data_directory(self, caller=None):
        """connect with menubar's Load option"""

        new_data_path = QtGui.QFileDialog.getExistingDirectory(None, 'Select a folder:', self.data_path,
                                                      QtGui.QFileDialog.ShowDirsOnly)
        if new_data_path == "":
            self.log("cancelled.")
            return

        if self.data_path == new_data_path:
            self.log("selected the same path as current, do nothing!", PRT.WARNING)
            return

        self.data_path = new_data_path
        self.log("got new data_path: {}".format(self.data_path))
        self.app.txtAnnDataPath.setPlainText(self.data_path)

        # check if labels.json exists
        lfile = os.path.join(self.data_path, self.label_json)
        if os.path.isfile(lfile):
            # load label dict
            with open(lfile) as fp:
                self.lparams = json.load(fp)

            self.label_dict = self.lparams['labels']
            self.clim_min, self.clim_max = self.lparams['clim']

        self.log("found labels.json: {}".format(lfile))

        # load sub-directories
        subdirs = next(os.walk(self.data_path))[1]
        subdirs.insert(0, "--select sub directory--")
        self.app.cbox_subdir_phase.clear()
        self.app.cbox_subdir_phase.addItems(subdirs)

        self.app.disable(self.app.btnStartNext)
        self.app.disable(self.app.btnExport)


    def start_new_batch(self, batchno=False):
        """initialize first if not initialized"""
        self.log("Starting New Batch!", PRT.WARNING)
        if not self.annotator_initialized:
            self.init_annotator()

        if self.reset_rawdata:
            self.init_raw_data()
            self.reset_rawdata = False

        """load next batch"""
        self.current_bIdx = 0
        #self.dbLabel.reset_counter()

        self.app.btnStartNext.setText("Next (&S)")
        self.app.btnStartNext.setShortcut("S")
        self.app.enable(self.app.btnExport)
        self.app.enable(self.lblButtons)

        # start with a batch
        self.run(batchno)


    def run(self, batchno=False):
        """actual entrypoint for data annotation process"""

        if self.rawdata is None:
            self.log("Load Data Directory first!", PRT.ERROR)
            return

        if batchno:
            val = self.get_batch_no(str(self.app.leBatchNo.text()))

            if val is None:
                self.log("invalid batch number...try again with DIGITS ONLY ranged between (0, %d)..."%(self.rawdata.total_num_batches-1), PRT.ERROR)
                return

            self.log("loading batch No [ %d ]..."%val, PRT.STATUS)

        QtGui.QApplication.processEvents()
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
        currItem = self.app.batchList.currentItem()
        if currItem is None:
            # nothing to do
            return
        self.show_volume(currItem)


    def show_volume(self, current, slice=False):
        """show volume with current Item"""
        try:
            key = str(current.text())
            d = self.batchData[key].data.numpy().copy()
            d = np.squeeze(d)
            if d.shape[0] == 128:
                # center crop # TEMPORARY CHANGE
                d = d[32:-32, 32:-32, 32:-32]
            if self.resizeVoxel:
                d = self.bmPrep._resize(d, self.voxelSize)

            if slice:
                print("show slice: ", d[8].shape)
#                d = self.bmPrep._max_proj(d)
                d = d[8]
                d = np.expand_dims(d, axis=0)
                print("show slice: ", d.shape)

                self.canvas.set_volume(d, clim=(self.clim_min, self.clim_max))
                print("show slice: ", d.shape, ", (Done)")
            else:
                self.canvas.set_volume(d, clim=(self.clim_min, self.clim_max))
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
                label = self.get_id_of_button(lb)
                self.log("Sample [ %s ] is already labeled as [ %s ]"%(filename, label), PRT.STATUS)
                break


    def update_batch_list(self, batch):
        """update batch list on ListWidget, and numpy data batch map"""

        # clear data holders
        self.app.batchList.clear()
        self.batchData.clear()
        self.batchLabel.clear()

        savef = self.get_export_filename()
        if os.path.isfile(savef):
            # the latest annotation file already exists
            self.batchLabel = pickle.load(open(savef, "rb"))
            self.log("Annotation file for current batch (%d) already exists!"%self.current_bIdx, PRT.WARNING)
            self.log("Loading annotations from %s..."%savef)

        fnames, data, data_raw = batch
        for idx, f in enumerate(fnames):
            d = data[idx]
            s = f.split('/')
            subset = s[-2]
            basef = s[-1]
            key = '/'.join([subset, basef])
            item = QtGui.QListWidgetItem(key)
            # set check icon if already labeled
            if bool(self.batchLabel) and key in self.batchLabel.keys():
                label_no = self.batchLabel[key]
                self.dbLabel.add(label_no)
                item.setIcon(self.app.checkIcon)


            self.app.batchList.addItem(item)
            self.batchData[key] = d

            if idx == 0:
                self.app.batchList.setCurrentItem(item)

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

        self.log("exported current batch's labels to %s"%savef)


    def get_export_filename(self):
        """build export filename"""
        save_path = '/'.join([self.data_path, self.rawdata.phase, 'labels'])
        dateid = self.data_path.split('/')[-1]
        bmUtil.CHECK_DIR(save_path)
        savef = '/'.join([save_path, "%s_%s_annotations_batch_%05d.p"%(dateid, self.rawdata.phase, self.current_bIdx)])

        return savef


    def log(self, msg, flag=PRT.LOGW):
        """log wrapper"""

        # show on statusbar
        self.app.statusbar.showMessage("[ STATUS ] %s"%msg)
        # show on logWindow
        self.app.logwin.append(PRT.html(self.__class__.__name__, msg, flag))
        self.app.logwin.moveCursor(QtGui.QTextCursor.End)
        QtGui.QApplication.processEvents()


    def load_XYPlane(self):
        item = self.app.batchList.currentItem()
        key = str(item.text())
        z, y, x = key.split('-')[1:4]
        z = int(z)
        point = [int(x.split('.')[0]), int(y)]

#        tiffpath = os.path.join(self.lparams['dpath'], "img_%04d.%s"%(z, self.lparams['dext']))
#        img = tifffile.imread(tiffpath)

        self.app.imageWindow.setImage(self.lparams['dpath'], self.lparams['dext'], z, point)
        self.app.imageWindow.show()



    def refresh_volume(self, slice=False):
        item = self.app.batchList.currentItem()
        self.show_volume(item, slice=slice)


    def update_clim_max(self, decrease=False):
        if decrease:
            self.clim_max -= 30
        else:
            self.clim_max += 30
        print("setting clim_min: {}, clim_max: {}".format(self.clim_min, self.clim_max))
        self.refresh_volume()


    def update_clim_min(self, decrease=False):
        if decrease:
            self.clim_min -= 30
        else:
            self.clim_min += 30

        print("setting clim_min: {}, clim_max: {}".format(self.clim_min, self.clim_max))
        self.refresh_volume()
