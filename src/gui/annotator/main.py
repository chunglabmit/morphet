import sys
import os.path
import tqdm
import pickle
from datetime import datetime
from itertools import cycle

# UI
from PyQt5 import QtGui, QtWidgets
import qdarkstyle
import vispy.app
#from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph.Qt import QtCore
import pyqtgraph.opengl as gl
import annotation_tool as ATUI
import numpy as np
from vispy import app, scene
from vispy.visuals.transforms import STTransform
from vispy.color import get_colormaps, BaseColormap

# torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Internal
from utils.data.dataset import MicrogliaDataset
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


class ATApp(QtGui.QMainWindow, ATUI.Ui_MainWindow):
    def __init__(self, **args):
        """init"""
        phase = args.pop('phase')
        super(ATApp, self).__init__(**args)

        # set phase for DataSet
        self.phase = 'train' if phase is None else phase

        # setup UI
        self.setupUi(self)

        # member variables
        self.rawdata = None
        self.data_path = str(self.txtDataPath.toPlainText())
        self.lblButtons = [self.btnRamified, self.btnAmoeboid, self.btnUncertain]
        self.lblButtons_inactive = [self.btnGarbage]
        for lb in self.lblButtons_inactive:
            self.disable(lb)

        self.rButtons = [self.rbtnMIP, self.rbtnTranslucent, self.rbtnAdditive]
        self.batchData = {}
        self.batchLabel = {}

        # TODO: get as an argument
        #self.dbLabel = MDBLabel()
        self.dbLabel = TDBLabel()

        self.bmPrep = BMPreprocessing()
        self.resizeVoxel = True if self.cbxResizing.isChecked() else False
        self.updateVoxelSize(isInit=True)

        self.delegate = ItemDelegate()
        self.batchList.setItemDelegate(self.delegate)
        self.checkIcon = QtGui.QIcon('./images/check.png')
        self.rendering = RenderingMethod.MIP

        # setup data
        self.update_rawData()
        self.log("retrieved (Default) data path as %s"%self.data_path, PRT.LOGW)

        # link gui item connections
        self.link_actions()


    def link_actions(self):
        """link action functions to target UI items"""

        self.actionLoad_Path.triggered.connect(self.load_data_directory)
        self.actionQuit.triggered.connect(self.quit)
        self.btnStartNext.clicked.connect(self.start_new_batch)
        self.batchList.currentItemChanged.connect(self.show_current_sample)
        self.btnExport.clicked.connect(self.export_labels)
        for lb in self.lblButtons:
            lb.clicked.connect(self.label_selected)
        self.rbtnMIP.toggled.connect(lambda:self.rendering_changed(self.rbtnMIP))
        self.rbtnAdditive.toggled.connect(lambda:self.rendering_changed(self.rbtnAdditive))
        self.rbtnTranslucent.toggled.connect(lambda:self.rendering_changed(self.rbtnTranslucent))
        self.btnColormaps.clicked.connect(self.toggle_colormap)
        self.btnRenderings.clicked.connect(self.toggle_rendering)
        self.leBatchNo.returnPressed.connect(lambda:self.start_new_batch(True))
        self.cbxResizing.stateChanged.connect(self.toggle_voxelResizing)
        self.leVoxelX.returnPressed.connect(self.updateVoxelSize)
        self.leVoxelY.returnPressed.connect(self.updateVoxelSize)
        self.leVoxelZ.returnPressed.connect(self.updateVoxelSize)


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


    def init(self):
        """initialize everything (e.g. GLViewer, RawData)
           NOTE: This should be called after show() call of the class
        """
        cw = self.gGLWidget.frameGeometry().width()
        ch = self.gGLWidget.frameGeometry().height()
        # initialize canvas with BMCanvas
        self.canvas = BMCanvas(keys='interactive', size=(cw, ch),
                               show=True, logWindow=self.logwin, parent=self.gGLWidget)


    def init_GLViewer(self):
        """initialize GLViewer for plotting volumetric data
        DEPRECATED
        """

        self.dataView = gl.GLViewWidget(self.centralwidget)
        self.dataView.opts['distance'] = 200
        self.dataView.setWindowTitle('glviewWidget test')
        self.dataView.show()
        self.dataView.setGeometry(self.gGLWidget.frameGeometry())
        self.dataView.setObjectName("dataViewer")

        g = gl.GLGridItem()
        self.dataView.addItem(g)


    def init_raw_data(self):
        """initialize rawData() class, which would hold data itself and loading pipeline using torch's DataLoader"""

        ext = 'npy'
        self.rawdata = RawData(self.data_path, ext, self.phase)
        self.log("Loaded data files with RawData() successfully")


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
            label_no = self.dbLabel.get_id(label)
            if lb == sender:
                self.log("You've selected [ {} ]".format(label), PRT.STATUS)
                self.batchLabel[key] = label_no
                self.dbLabel.add(label_no)
            else:
                if lb.isChecked():
                    self.dbLabel.subtract(label_no)
                    lb.toggle()

        self.stat_labels()
        # proceed to the next sample
        self.move_to_next()     # this function will also update button states


    def stat_labels(self):
        """get labels stats"""
        msg = self.dbLabel.get_count_msg()
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
            label_name = self.dbLabel.get_name(label)

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

        self.data_path = new_data_path
        self.log("got new data_path: {}".format(self.data_path))
        self.txtDataPath.setPlainText(self.data_path)
        self.update_rawData()

        # reset all values
        self.run()


    def start_new_batch(self, batchno=False):
        """load next batch"""
        self.current_bIdx = 0
        #self.dbLabel.reset_counter()

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
            d = np.squeeze(d)
            if d.shape[0] == 128:
                # center crop # TEMPORARY CHANGE
                d = d[32:-32, 32:-32, 32:-32]
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
        self.statusbar.showMessage("[ STATUS ] %s"%msg)
        # show on logWindow
        self.logwin.append(PRT.html(self.__class__.__name__, msg, flag))
        self.logwin.moveCursor(QtGui.QTextCursor.End)
        QtWidgets.QApplication.processEvents()


    def _print(self):
        print("anything!!")


    def on_mouse_move(self, event):
        print("on_mouse_move")


if __name__ == "__main__":
    args = sys.argv
#    if len(args) < 2:
#        print("Usage: python main.py [DATA_SET]")
#        sys.exit(1)
    dataset = args[-1]
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    ata = ATApp(phase=dataset)
    ata.show()
    ata.init()
    sys.exit(app.exec_())
