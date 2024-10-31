"""main.py: Main entry point for Evaluator GUI"""
__author__      = "Minyoung Kim"
__license__ = "MIT"
__maintainer__ = "Minyoung Kim"
__email__ = "minykim@mit.edu"
__date__ = "02/22/2019"


import sys
import shutil
import os.path
import math
import time
from tqdm import tqdm
import pickle
from glob import glob
import numpy as np
from sklearn.metrics import confusion_matrix
from functools import partial
import pandas as pd
from tabulate import tabulate


# UI
from PyQt5 import QtGui, QtWidgets
import qdarkstyle
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
from matplotlib.figure import Figure
if is_pyqt5():
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)

import vispy.app
import evaluator_ui as EUI
from itertools import cycle

from vispy import app, scene
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score as aps
from sklearn.metrics import precision_recall_curve as prc

# torch
import torch
import torch.nn.functional as F

# Internal
from utils.data.dataset import MicrogliaDataset, TrapDataset
from utils.train.preprocessor import TrainPreprocessor
import utils.util as bmUtil
from utils.util import PRT
from utils.const import Phase, Dataset, ModelType, EvalCol, NormalizationType
from utils.viz.bmCanvas import BMCanvas
from utils.params import TrainParams
from train.trainer import Trainer
from train.trapTrainer import TRAPTrainer
from train.bfmcTrainer import BFMCTrainer
from train.mTrainer import MTrainer
from utils.data.dbLabel import MDBLabel, TDBLabel



class EvaluatorApp(QtGui.QMainWindow, EUI.Ui_MainWindow):
    class UI_MODE(object):
        EVAL = 'EVAL'
        CORRECTING = 'CORRECTING'

    def __init__(self, **args):
        """init"""
        self.phase = args.pop('phase')

        super(EvaluatorApp, self).__init__(**args)

        # setup UI
        self.setupUi(self)

        self.ui_mode = self.UI_MODE.CORRECTING
        self.prep = None
        self.params = None
        self.MLabel = MDBLabel()
        #self.MLabel = TDBLabel()
        self.classes = self.MLabel.get_ordered_labels_by_id()
        self.lblButtons = [ self.rbtnRamified, self.rbtnAmoeboid, self.rbtnGarbage, self.rbtnUncertain ]
        self.modeButtons = [ self.rbtnModeEval, self.rbtnModeCorrecting ]
        self.lblAccs = None
        self.current_gt_label = None

        self.dl_return_fname = True
        self.dpi = 96
        self.old_gidx = None

        self.screenshot_dir = "screenshot"
        self.eval_csv = "evaluation.csv"
        self.eval_logfile = "" # to be set in build_params()

        # initialize data
        self.fnames = None
        self.inputs = None
        self.outputs = None
        self.enc_preds = None
        self.cluster_preds = None
        self.gts = None
        self.gts_corrected = None
        self.log_df = None

        self.win_g = self.geometry()
        self.gl_g = self.gridGboxLog.geometry()
        self.gg_g = self.gridGboxGraphs.geometry()
        self.gp_g = self.gridGboxProbDist.geometry()


    def setup(self):
        # initialize default paths
        self.config_paths()

        # link gui item connections
        self.link_actions()

        # init canvas and replace placeholder
        cw = self.canvasPlaceholder.frameGeometry().width()
        ch = self.canvasPlaceholder.frameGeometry().height()
        self.canvas = BMCanvas(keys='interactive', size=(cw, ch), show=True, logWindow=self.logwin)
        self.caption = self.canvas.create_text((10, ch-20), "", 8, (1.0, 1.0, 1.0))
        self.caption2 = self.canvas.create_text((10, ch-40), "", 8, (1.0, 1.0, 1.0), bold=True)
        self.caption3 = self.canvas.create_text((100, 30), "", 8, (1.0, 1.0, 1.0))

        self.gridLayout_2.removeWidget(self.canvasPlaceholder)
        self.gridLayout_2.addWidget(self.canvas.native, 0, 0, 1, 1) # raw, colume, rowspan, colspan

        # init plots
        self.reset_plots(init=True)

        self.log("Press [Build] to build parameters and initialize!", PRT.WARNING)


    def config_paths(self):
        self.train_model_root = str(self.txt_trainRoot.toPlainText())
        self.log("Train model root: [%s]"%self.train_model_root, PRT.LOG)
        self.expr = str(self.txt_expr.toPlainText())
        self.expr_root = os.path.join(self.train_model_root, self.expr)
        print("model: ", self.txt_model.toPlainText())
#        model_base = sorted(glob("%s/*.pth"%self.expr_root))[-1]

        model_base = str(self.txt_model.toPlainText().split('\'')[0])
        print("model_base: ", model_base)
        self.model = os.path.join(self.train_model_root, self.expr, model_base)
        self.model_id = "%s_%d"%(self.expr, int(model_base.split('.')[0].split('_')[-1]))
        self.log("model: %s, model_id: %s"%(self.model, self.model_id))

        #TODO: get text from GUI
        self.save_dir="/media/ssdshare2/general/MYK/data/analysis/morphet/eval"


    def reset_plots(self, init=False):
        if init:
            # initialize with/height for each plot section
            self.cw_grv1 = int(self.grView1.frameGeometry().width() * 0.95)
            self.ch_grv1 = int(self.grView1.frameGeometry().height() * 0.9)
            self.cw_grv2 = int(self.grView2.frameGeometry().width() * 0.94)
            self.ch_grv2 = int(self.grView2.frameGeometry().height() * 0.94)
            self.cw_grv3 = int(self.grView3.frameGeometry().width())
            self.ch_grv3 = int(self.grView3.frameGeometry().height())
            self.cw_grv4 = int(self.grView3.frameGeometry().width())
            self.ch_grv4 = int(self.grView3.frameGeometry().height())
            self.log("self.ch_grv4: {}".format(self.ch_grv4), PRT.ERROR)

            self.plot1 = QtGui.QGraphicsScene(0, 0, self.cw_grv1, self.ch_grv1)
            self.plot2 = QtGui.QGraphicsScene(0, 0, self.cw_grv1, self.ch_grv1)
            self.plot3 = QtGui.QGraphicsScene(0, 0, self.cw_grv3, self.ch_grv3)
            self.plot4 = QtGui.QGraphicsScene(0, 0, self.cw_grv4, self.ch_grv4)
        else:
            self.plot1.clear()
            self.plot2.clear()
            self.plot3.clear()
            self.plot4.clear()

        # now, hide!!
        self.toggle_hide()


    def toggle_hide(self, hide=True):
        if hide:
            # move logs up
            self.gridGboxLog.move(self.gg_g.x(), self.gg_g.y())
        else:
            # move logs down
            self.gridGboxLog.move(self.gl_g.x(), self.gl_g.y())

        # hide graph box
        self.gridGboxGraphs.setHidden(hide)

        # hide stat box
        self.gridGboxStat.setHidden(hide)

        # hide prob box
        self.gridGboxProbDist.setHidden(hide)

        # finally, resize MainWindow
        if hide:
            r_w = self.win_g.width() - self.gp_g.width() - 9
            r_h = self.win_g.height() - self.gg_g.height() - 4
            self.setMinimumSize(QtCore.QSize(r_w, r_h))
            self.setMaximumSize(QtCore.QSize(r_w, r_h))
            self.resize(r_w, r_h)
        else:
            self.setMinimumSize(QtCore.QSize(self.win_g.width(), self.win_g.height()))
            self.setMaximumSize(QtCore.QSize(self.win_g.width(), self.win_g.height()))
            self.resize(self.win_g.width(), self.win_g.height())


    def link_actions(self):
        """link action functions to target UI items"""

        self.btnBuild.clicked.connect(self.build_params)
        self.btnEval.clicked.connect(self.evaluate)
        self.btnSkip.clicked.connect(self.show_next_fp)

        self.btnPrev.clicked.connect(partial(self.move_on, False))
        self.btnNext.clicked.connect(partial(self.move_on, True))

        for lb in self.lblButtons:
            lb.clicked.connect(self.on_label_corrected)

        for rb in self.modeButtons:
            rb.clicked.connect(self.on_mode_change)

        # connects labels' mousePressEvent
        self.lbl_datasetPath.mousePressEvent = lambda e:self.text_clicked(e,
                                                        self.lbl_datasetPath, self.txt_datasetPath)
        self.lbl_trainRoot.mousePressEvent = lambda e:self.text_clicked(e,
                                                        self.lbl_trainRoot, self.txt_trainRoot)
        self.lbl_model.mousePressEvent = lambda e:self.text_clicked(e,
                                                        self.lbl_model, self.txt_model, self.expr_root)
        self.lbl_expr.mousePressEvent = lambda e:self.text_clicked(e,
                                                        self.lbl_expr, self.txt_expr, self.train_model_root)

        # connects Combobox changes
        self.cboxDataset.currentIndexChanged.connect(self.on_cbox_changed)
        self.cboxNumClass.currentIndexChanged.connect(self.on_cbox_changed)
        self.cboxDataSubset.currentIndexChanged.connect(self.on_cbox_changed)
        self.cboxDataSampling.currentIndexChanged.connect(self.on_cbox_changed)

        # menu actions
        self.actionScreenshot.triggered.connect(self.save_screenshot)
        self.actionQuit.triggered.connect(self.quit)
        self.actionSaveGT.triggered.connect(self.save_groundtruth)


    def on_mode_change(self):
        sender = self.sender()
        mode = str(sender.text())

        if mode == self.ui_mode:
            return

        if mode == self.UI_MODE.EVAL:
            msg = "Switching to Eval mode!\n"
            msg += "You shouldn't go into this mode if you're doing label correction!\n"
            msg += "Seeing model's behavior might make you biased on the correction.\n"
            msg += "If you go into this mode, stopping correction is highly recommended.\n"
            msg += "Do you still want to switch the mode?"
            choice = QtGui.QMessageBox.question(self, 'Warning!', msg,
                                                QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)

            if choice == QtGui.QMessageBox.Yes:
                self.ui_mode = mode
                # show viz
                self.toggle_hide(False)
                self.update_evaluation(corrected=True)
                self.show_sample()
            elif choice == QtGui.QMessageBox.No:
                # reset rbtn
                for rb in self.modeButtons:
                    if rb == sender:
                        rb.setChecked(False)
                    else:
                        rb.setChecked(True)

        elif mode == self.UI_MODE.CORRECTING:
            msg = "You're trying to switch to Correcting mode, however, "
            msg += "it's not recommended to correct labels after seeing "
            msg += "the model evaluation report. Do you still want to switch?\n"
            msg += "If you press 'No', the application will be closed. (Don't worry! "
            msg += "All of your works would be saved!"

            choice = QtGui.QMessageBox.question(self, 'Warning!', msg,
                                                QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)

            if choice == QtGui.QMessageBox.Yes:
                self.ui_mode = mode
                self.reset_plots()

            elif choice == QtGui.QMessageBox.No:
                if self.log_df is not None:
                    self.save_groundtruth()
                self.quit()

    def save_logs(self):
        if self.log_df is None:
            self.log("[GT Saving] No logs yet for saving!", PRT.WARNING)
            return

        # reset index!
        self.log_df = self.log_df.reset_index(drop=True)

        if os.path.isfile(self.eval_logfile):
            # save backup
            backupf = "{}_backup_{}.csv".format(self.eval_logfile.split('.')[0], bmUtil.get_current_time())
            os.rename(self.eval_logfile, backupf)

        if self.log_df is not None:
            self.log("[GT Saving] Saving logs at [%s]"%self.eval_logfile, PRT.STATUS)
            self.log_df.to_csv(self.eval_logfile, sep=',', index=False)


    def save_groundtruth(self):
        if self.gts_corrected is None:
            self.log("[GT Saving] Nonthing to save!", PRT.ERROR)
            return

        # save logs for later
        self.save_logs()

        gts = np.array(self.gts)
        gts_c = np.array(self.gts_corrected)

        self.log("len(gts): %d"%len(gts), PRT.LOG)
        self.log("len(gts_corrected): %d"%len(gts_c), PRT.LOG)

        agreement = gts == gts_c
        if all(agreement):
            # unanimous
            self.log("[GT Saving] Unanimous!!", PRT.ERROR)
            return

        # create dir to store originals if needed
        orig_dir = os.path.join(self.lbl_dir, "original")
        bmUtil.CHECK_DIR(orig_dir)

        corrected_pfs = []
        for i in range(len(agreement)):
            if not agreement[i]:
                #fname = self.fnames[i].split('/')[-1]
                fname = self.get_curr_fid(i)
                cmd = "grep %s %s/*.p"%(fname, self.lbl_dir)
                self.log("shell cmd: {}".format(cmd), PRT.LOG)
                r = os.popen(cmd).read()
                pfile_str = r.split(':')[0]
                pfile_split = pfile_str.split(' ')
                pfile = ""
                for item in pfile_split:
                    if '.p' in item :
                        pfile = item

                print("pfile: ", pfile)
                assert pfile != ""

                key = fname
                #key = r.split("'")[1]

                #if not os.path.isfile(pfile):
                #    # there could be only one file
                #    pfile = glob("%s/*.p"%self.lbl_dir)[0]

                corrected_pfs.append(pfile)

                labels = pickle.load(open(pfile, "rb"))
                try:
                    assert labels[key] == gts[i]
                except AssertionError:
                    print("i: ", i, "key: ", key)
                    print("labels[key]: ", labels[key])
                    print("gts[i]: ", gts[i])

                labels[key] = gts_c[i]

                # move original, and save
                try:
                    shutil.move(pfile, orig_dir)
                except shutil.Error as err:
                    f_exist = err.args[0].split("'")[1]
                    # file exists
                    self.log("File [%s] exists, thus removing before backup"%f_exist, PRT.WARNING)
                    os.remove(f_exist)
                    # try moving again
                    shutil.move(pfile, orig_dir)

                with open(pfile, 'wb') as fp:
                    pickle.dump(labels, fp)

        self.log("[GT Saving] Files corrected: {}".format(list(set(corrected_pfs))), PRT.LOG)

        # Now, copy gts_corrected to gts! (to get up-to-date labels)
        self.gts = self.gts_corrected.copy()


    def save_screenshot(self):
        bmUtil.CHECK_DIR(self.screenshot_dir)
        filename = "screenshot_{}.png".format(bmUtil.get_current_time())
        fpath = os.path.join(self.screenshot_dir, filename)
        self.log("Saving a screenshot to %s"%fpath, PRT.LOG)
        p = QtGui.QScreen.grabWindow(QtWidgets.QApplication.primaryScreen(), self.winId())
        p.save(fpath)
        self.log("Saving a screenshot to %s (Done)"%fpath, PRT.STATUS)


    def on_label_corrected(self):
        if self.current_gt_label is None:
            # nothing to do
            return

        old_gt = str(self.current_gt_label.text())

        sender = self.sender()
        new_gt = str(sender.text())

        if old_gt == new_gt:
            # nothing changed!
            return

        self.log("GT changed from [{}] to [{}]".format(old_gt, new_gt), PRT.WARNING)
        gidx = self.get_curr_gid()

        self.gts_corrected[gidx] = self.MLabel.get_id(new_gt)

        self.log("gidx: {}, gt: {}, gt_corrected: {}, fn: {}".format(gidx, self.gts[gidx], self.gts_corrected[gidx],
                                                                     self.fnames[gidx]))

        self.enable(self.current_gt_label)
        self.current_gt_label = sender
        self.disable(self.current_gt_label)

        self.update_evaluation(corrected=True)



    def update_evaluation(self, corrected):
        if self.gts is None:
            return

        if corrected:
            gts_all = self.gts_corrected
        else:
            gts_all = self.gts

        # remove unnecessary items (containing not-being-considered labels)
        indices = np.where(gts_all < self.params.num_clusters)
        gts = gts_all[indices]
        preds = self.enc_preds[indices]
        outs = self.outputs[indices]

#        if self.model_type == ModelType.BMTR:
#            cluster_preds = self.cluster_preds[indices]
#        else:
        cluster_preds = None

        if self.ui_mode == self.UI_MODE.EVAL:
            # flush figures
            plt.clf()
            plt.close()

            # correct plots
            self.show_confusion_matrix(preds, gts, self.params.num_clusters,
                                        "Confusion matrix (enc_pred)", save_dir=self.save_dir)
            self.show_prcurves(outs, gts, self.params.num_clusters,
                               save_dir=self.save_dir)

            # update numbers
            self.update_stats(gts, preds, cluster_preds)


    def update_stats(self, gts, preds, cluster_preds=None):
        # global accuracy
        acc_all = bmUtil.calc_acc(gts, preds) * 100.
        self.lbl_acc_all_val.setText("%.2f%%"%acc_all)
        self.lbl_ap_all_val.setText("%.2f"%self.avg_precision_all)

        # accuracy per class
        for i in range(self.params.num_clusters):
            ind = [ x for x, e in enumerate(gts) if e == i]
            acc = bmUtil.calc_acc(gts[ind], preds[ind]) * 100.
            if self.lblAccs[i] is not None:
                self.lblAccs[i].setText("%.2f%%"%acc)

        # logging
        self.log("\nExpr: %s"%(self.expr), PRT.LOG)
        self.log("Phase: {} (Sampling={})".format(self.subset, self.get_if_sample()), PRT.LOG)
        self.log("Accuracy\n   ----------------------", PRT.LOG)
        self.log("   enc_acc: %.2f%%"%acc_all, PRT.LOG)
        if cluster_preds is not None:
            self.cluster_acc = bmUtil.calc_acc(gts, cluster_preds)
            self.log("   cluster_acc: %.2f%%"%(self.cluster_acc * 100.), PRT.LOG)
        self.log("   ----------------------", PRT.LOG)


    def on_cbox_changed(self):
        # enable Build button
        self.reset_build()


    def reset_build(self):
        self.config_paths()     # update paths
        self.disable(self.btnNext)
        self.disable(self.btnPrev)
        self.disable(self.btnSkip)
        self.disable(self.btnEval)
        #self.rbtnModeCorrecting.setChecked(True)
        self.enable(self.btnBuild)


    def build_params(self):
        self.disable(self.btnBuild)
        self.log("Building parameters...", PRT.STATUS)
        print("Building parameters...")

        # get normlization details
        ntype =  self.cbBox_NormType.currentText()
        norm_clip = True if self.chkBox_Clip.isChecked() else False
        no2d = True if self.chkBox_No2D.isChecked() else False
        self.log("Normalization: ntype: {}, clip? {}".format(ntype, norm_clip), PRT.STATUS)
        print("Normalization: ntype: {}, clip? {}".format(ntype, norm_clip))

        self.config_paths()

        if self.rbtnBFMC.isChecked():
            self.model_type = ModelType.BFMC
        elif self.rbtnBMTR.isChecked():
            self.model_type = ModelType.BMTR
        elif self.rbtnTRAP.isChecked():
            self.model_type = ModelType.TRAP
        elif self.rbtnALTR.isChecked():
            self.model_type = ModelType.ALTR
        else:
            self.log("Unrecognized model type!", PRT.ERROR)

        self.log("model_type: [%s]"%self.model_type, PRT.LOG)

        dataset = str(self.cboxDataset.currentText())
        dataset_path = os.path.join(str(self.txt_datasetPath.toPlainText()), dataset)
        print("dataset_path: ", dataset_path)
        dataset_name = str(self.txt_datasetName.toPlainText())

        nclass = str(self.cboxNumClass.currentText())
        bsize = str(self.sbBatchSize.value())

        self.log("dataset: [%s] [%s]"%(dataset, dataset_path), PRT.LOG)
        self.log("# class: [%s], batchsize: [%s]"%(nclass, bsize), PRT.LOG)
        self.subset = str(self.cboxDataSubset.currentText())
        sampling = str(self.cboxDataSampling.currentText())

        self.log("sample? ({}): {}".format(self.subset, sampling), PRT.LOG)

        if self.model_type == ModelType.TRAP:
            args = ['GUI', '-ph', self.phase, '-bs', '8', '-e', '2', '-ts', 'GUI', '-nc', '2',
                    '-ie', 'True', '-mt', self.model_type,
                    '-ds', 'trap', '-us', sampling, '-usv', sampling,
                    '-dw', '16', '-dh', '16', '-dd', '8',
                    '-aw', self.model, '-dp', dataset_path,
                    '-nt', ntype, '-cl', str(norm_clip),
                    '--debug']
        else:
            args = ['GUI', '-ph', self.phase, '-bs', bsize, '-e', '2', '-ts', 'GUI', '-nc', nclass,
                    '-ie', 'True', '-mt', self.model_type,
                    '-ds', dataset_name, '-us', sampling, '-usv', sampling,
                    '-aw', self.model, '-dp', dataset_path,
                    '-nt', ntype, '-cl', str(norm_clip),
                    '--debug']
            if no2d:
                args += ['-n2d']

        self.dataset_path = dataset_path
        print("dataseT_path: ", dataset_path)
        self.params = TrainParams()
        self.params.build(args, "Evaluator_TRParser")

        # set save directory
        self.cluster_save_dir = os.path.join(self.train_model_root, self.expr, "clustered")
        self.log("cluster_save_dir: %s"%self.cluster_save_dir)
        bmUtil.CHECK_DIR(self.cluster_save_dir)
        for i in range(self.params.num_clusters):
            bmUtil.CHECK_DIR("%s/%d"%(self.cluster_save_dir, i))

        # set labels directory, load if log exists
        self.lbl_dir = os.path.join(dataset_path, self.subset, "labels")
        self.eval_logfile = os.path.join(dataset_path, self.subset, self.eval_csv)
        if os.path.isfile(self.eval_logfile):
            self.log("Reading exisisting Log DataFrame from [%s]!"%self.eval_logfile, PRT.STATUS)
            self.log_df = pd.read_csv(self.eval_logfile)
        else:
            self.log_df = None

        # build dataset
        self.log("building dataset...", PRT.STATUS)
        db = self.build_dataset(self.params, ntype, norm_clip,
                                self.log, self.dl_return_fname)

        # build trainer
        self.log("building trainer...", PRT.STATUS)
        self.trainer = self.build_trainer(self.params, db, self.model_type)

        # set GUI labels
        if self.params.num_clusters == 3:
            self.lblAccs = [ self.lbl_acc_r_val, self.lbl_acc_a_val, self.lbl_acc_g_val ]
        else:
            self.lblAccs = [ self.lbl_acc_r_val, None, self.lbl_acc_a_val, self.lbl_acc_g_val ]

        # ready for evaluation, reset progress bar
        self.reset_progressbar(self.pBarEval)
        self.enable(self.pBarEval)
        self.enable(self.btnEval)
        self.log("Press [Evaluate]!", PRT.WARNING)


    @staticmethod
    def build_dataset(params, ntype, norm_clip, log=None, return_fname=False):
        assert params is not None

        prep = TrainPreprocessor()
        print("params.dataset: ", params.dataset)
        isLabeled = True if params.dataset in [Dataset.MICROGLIA_LABELED, Dataset.TRAP] else False
        phases = params.phase.split('_')
        mdb = {}
        mdb['name'] = params.dataset

        if True:
            for p in phases:
                d = MicrogliaDataset(params.data_path, params.file_ext, p,
                                     params.num_clusters, prep,
                                     with_labels=isLabeled,
                                     return_fname=return_fname,
                                     ntype=ntype, clip=norm_clip)
                if log:
                    log("Total [ {} ] data found in [ {} ] phase (with_label={}).".format(len(d), d.phase, isLabeled), PRT.STATUS2)
                mdb[p] = d
        else:
            print("return_fname: ", return_fname)
            print("islabeled: ", isLabeled)
            for p in phases:
                d = TrapDataset(data_path=params.data_path, ext=params.file_ext, phase=p,
                                data_size=[params.data_d, params.data_h, params.data_w],
                                num_clusters=params.num_clusters, preprocessor=prep,
                                with_labels=isLabeled, return_fname=return_fname,
                                ntype=ntype, clip=norm_clip)
                if log:
                    log("Total [ {} ] data found in [ {} ] phase (with_label={}).".format(len(d), d.phase, isLabeled), PRT.STATUS2)
                mdb[p] = d

        return mdb


    @staticmethod
    def build_trainer(params, db, mt):

        if mt == ModelType.BMTR:
            trainer = Trainer("BMTR", params, db)
        elif mt == ModelType.BFMC:
            # BFMC
            trainer = BFMCTrainer(name="BMTR", params=params, dataset=db)
        elif mt == ModelType.TRAP:
            print("TRAP Trainer")
            # TRAP
            trainer = TRAPTrainer(name="TRAP", params=params, dataset=db)
        elif mt == ModelType.ALTR:
            trainer = MTrainer(name="ALTR", params=params, dataset=db)
        else:
            raise 'UnknownModelType'

        return trainer


    def show_prcurves(self, predictions, gts, num_clusters, save_dir=None, debug=False):
        gts_onehot = np.zeros((gts.shape[0], num_clusters), dtype=np.int8)
        for i in range(gts.shape[0]):
            gts_onehot[i][gts[i]] = 1

        precision = dict()
        recall = dict()
        avg_precision = dict()
        for i in range(num_clusters):
            if debug:
                print("i: ", i)
                print("gts_onehot: ", gts_onehot)
                print("predictions: ", predictions)
            precision[i], recall[i], _ = prc(gts_onehot[:, i], predictions[:, i])
            avg_precision[i] = aps(gts_onehot[:, i], predictions[:, i])
            self.log('Average precision score for class %d (%s): %.3f'%(i, self.MLabel.get_name(i), avg_precision[i]), PRT.STATUS2)

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = prc(gts_onehot.ravel(), predictions.ravel())
        avg_precision["micro"] = aps(gts_onehot, predictions, average="micro")
        self.avg_precision_all = avg_precision["micro"]
        self.log('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(avg_precision["micro"]), PRT.STATUS2)

        # visualize
        plt.figure(figsize=(self.cw_grv3/self.dpi, self.ch_grv3/self.dpi), dpi=self.dpi)
        plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2, where='post')
        plt.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('AP(micro, all)={0:0.2f}'.format(avg_precision["micro"]))

        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.15)
        fcanvas = FigureCanvas(fig)

        fcanvas.setGeometry(10, 3, self.cw_grv3, self.ch_grv3)
        self.plot3 = QtGui.QGraphicsScene(0, 0, self.cw_grv3, self.ch_grv3)
        self.plot3.addWidget(fcanvas)
        self.grView3.setScene(self.plot3)

        if save_dir is not None:
            dataset = str(self.cboxDataset.currentText())
            fname = os.path.join(save_dir,
                                 "prcurve_mid-%s_DS-%s-%s.pdf"%(self.model_id, dataset, self.subset))
            fig.savefig(fname, format='pdf', dpi=800)
            self.log("PR curve is saved to [ %s ]"%(fname), PRT.STATUS)


        # visualize PR Curves per class
        # setup plot details
        colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

        #plt.figure(figsize=(self.cw_grv4/self.dpi, self.ch_grv4/self.dpi), dpi=self.dpi)
        plt.figure(figsize=(4, 7), dpi=self.dpi)
        f_scores = np.linspace(0.2, 0.8, num=4)
        lines = []
        labels = []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

        lines.append(l)
        labels.append('iso-f1 curves')
        l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
        lines.append(l)
        labels.append('m-avg PR (area={0:0.2f})'
                      ''.format(avg_precision["micro"]))

        for i, color in zip(range(num_clusters), colors):
            l, = plt.plot(recall[i], precision[i], color=color, lw=2)
            lines.append(l)
            labels.append('PR [class %d-%s] (area=%.2f)'%(i, self.MLabel.get_name(i), avg_precision[i]))
        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.15)
        fig.tight_layout()
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR-Curve per class')
        plt.legend(lines, labels, loc=(.03, .03), prop=dict(size=5))

        #fcanvas = FigureCanvas(plt.gcf())
        fcanvas = FigureCanvas(fig)
        fcanvas.setGeometry(10, 3, self.cw_grv4, self.ch_grv4)
        self.plot4 = QtGui.QGraphicsScene(0, 0, self.cw_grv4, self.ch_grv4)
        self.plot4.addWidget(fcanvas)
        self.grView4.setScene(self.plot4)

        if save_dir is not None:
            dataset = str(self.cboxDataset.currentText())
            fname = os.path.join(save_dir,
                                 "prcurve_perClass_mid-%s_DS-%s-%s.pdf"%(self.model_id, dataset, self.subset))
            fig.savefig(fname, format='pdf', dpi=800)
            self.log("PR curve is saved to [ %s ]"%(fname), PRT.STATUS)



    def show_confusion_matrix(self, predictions, gts, num_clusters,
                              title="Confusion matrix", save_dir=None):
        CM = bmUtil.ConfusionMatrix()
        fig = CM.show_confusion_matrix(predictions, gts, num_clusters, "Confusion matrix (enc_pred)", True,
                                       figsize=(5,12))
        fcanvas = FigureCanvas(fig)
        fcanvas.setGeometry(10, 3, self.cw_grv1, self.ch_grv1)

        self.plot1 = QtGui.QGraphicsScene(0, 0, self.cw_grv1, self.ch_grv1)
        self.plot1.addWidget(fcanvas)
        self.grView1.setScene(self.plot1)
        if save_dir is not None:
            dataset = str(self.cboxDataset.currentText())
            fname = os.path.join(save_dir,
                                 "confusion_matrix_mid-%s_DS-%s-%s.pdf"%(self.model_id, dataset, self.subset))
            fig.savefig(fname, format='pdf', dpi=800)
            self.log("ConfusionMatrix is saved to [ %s ]"%(fname), PRT.STATUS)


    def get_if_sample(self):
        if self.subset == Phase.TRAIN:
            sampler = bool(self.params.use_sampler)
        elif self.subset == Phase.VAL:
            sampler = bool(self.params.use_sampler_in_val)
        else:
            sampler = False

        return sampler


    def evaluate(self):
        self.disable(self.btnEval)
        np.set_printoptions(suppress=True)
        torch.set_printoptions(precision=1)

        sampler = self.get_if_sample()
        self.log("phase: {}, use_sampler: {}".format(self.subset, sampler), PRT.STATUS)
        print("phase: {}, use_sampler: {}".format(self.subset, sampler))

        # inference
        max_iter = None
        self.outputs, \
        self.enc_preds, \
        self.cluster_preds, \
        self.gts, \
        self.inputs, \
        self.fnames = self.infer(phase=self.subset, sampler=sampler, max_iter=max_iter)
        self.gts_corrected = self.gts.copy()

        # update plots and statistics
        self.update_evaluation(corrected=False)

        # visualize
        self.start_sample_viewer()


    def update_idx(self, is_forward):
        if is_forward:
            self.s_idx += 1
            if self.s_idx == self.params.batch_size:
                self.s_idx = 0
                self.s_bidx += 1
            if self.get_curr_gid() >= len(self.gts):
                # reached the end, reset to 0
                self.s_idx = 0
                self.s_bidx = 0
        else:
            self.s_idx -= 1
            if self.s_idx < 0:
                self.s_idx = self.params.batch_size - 1
                self.s_bidx -= 1
            if self.s_bidx < 0:
                # reached the front, reset to 0
                self.s_idx = 0
                self.s_bidx = 0


    def get_curr_gid(self):
        gidx = self.s_bidx * self.params.batch_size + self.s_idx
        if gidx >= len(self.gts):
            # reached the end, reset to 0
            self.s_idx = 0
            self.s_bidx = 0
            return 0

        return gidx


    def get_sub_idx_by_gid(self, gid):
        s_idx = gid % self.params.batch_size
        s_bidx = int(float(gid) / self.params.batch_size)

        return s_idx, s_bidx


    def start_sample_viewer(self):
        if self.log_df is None:
            # start from stratch
            self.s_idx = 0
            self.s_bidx = 0
        else:
            gmax = self.log_df[EvalCol.GID].max()
            self.s_idx, self.s_bidx = self.get_sub_idx_by_gid(gmax)

        self.show_sample(cmap="viridis")


    def get_curr_fid(self, idx):
        # extract id part of file name
        return '/'.join(self.fnames[idx].split('/')[-2:])


    def log_curr_sample(self):
        gidx = self.get_curr_gid()
        fid = self.get_curr_fid(gidx)
        gt = self.gts[gidx]
        gt_c = self.gts_corrected[gidx]
        pred_id = self.enc_preds[gidx]
        correct_gt = gt == pred_id
        correct_gt_c = gt_c == pred_id

        log = {}
        log[EvalCol.GID] = gidx
        log[EvalCol.FID] = fid
        log[EvalCol.GT] = gt
        log[EvalCol.GT_CORRECTED] = gt_c
        log[self.model_id] = pred_id
        log[EvalCol.VISITED] = 1
        l_df = pd.DataFrame([log])

        if self.log_df is None:
            self.log_df = pd.DataFrame(l_df)
            self.log("[NEW] DF head: {}".format(self.log_df.head()), PRT.LOG)
        else:
            row = self.log_df.loc[self.log_df[EvalCol.GID] == gidx]
            if row.empty:
                self.log("Appending DF: {}".format(l_df), PRT.LOG)
                self.log_df = self.log_df.append(l_df)
            else:
                self.log_df.loc[self.log_df[EvalCol.GID] == gidx, EvalCol.GT] = gt
                self.log_df.loc[self.log_df[EvalCol.GID] == gidx, EvalCol.GT_CORRECTED] = gt_c
                self.log_df.loc[self.log_df[EvalCol.GID] == gidx, self.model_id] = pred_id
                self.log_df.loc[self.log_df[EvalCol.GID] == gidx, EvalCol.VISITED] += 1
                check = self.log_df.loc[self.log_df[EvalCol.GID] == gidx]

            #print(tabulate(self.log_df, headers='keys', tablefmt='psql'))


    def show_next_fp(self):
        # save current sample's log
        self.log_curr_sample()

        found = False
        cnt = 0
        while not found:
            self.update_idx(is_forward=True)

            gidx = self.get_curr_gid()
            gt = self.gts_corrected[gidx]
            pred_id = self.enc_preds[gidx]
            correct = gt == pred_id
            found = not correct
            cnt += 1
            if cnt > 10000:
                break

        if found:
            self.show_sample()
        else:
            self.log("Couldn't find incorrectly predicted samples within 10000 samples!", PRT.WARNING)


    def move_on(self, is_forward):
        # save current sample's log
        self.log_curr_sample()

        self.update_idx(is_forward=is_forward)
        self.show_sample()


    def show_sample(self, cmap=None):
        if self.gts is None:
            return

        gidx = self.get_curr_gid()
        vol = self.inputs[self.s_bidx][self.s_idx]
        gt = self.gts_corrected[gidx]
        pred = self.outputs[gidx]
        pred_id = self.enc_preds[gidx]
        probs = F.softmax(torch.from_numpy(pred), dim=0).data.numpy()

        if gidx != 0:
            self.enable(self.btnPrev)
        else:
            self.disable(self.btnPrev)

        if gidx < len(self.gts_corrected) - 1:
            self.enable(self.btnNext)
        else:
            self.disable(self.btnNext)

        self.enable(self.btnSkip)

        # update plot
        self.update_prediction(probs, gt)

        correct = gt == pred_id

        # update canvas
        self.canvas.set_volume(vol)
        if cmap:
            self.canvas.volume.cmap = cmap
        fn = self.get_curr_fid(gidx)
        self.caption.text = "gidx: %d, f: %s"%(gidx, fn)
        self.caption2.text = "correct" if correct else "wrong"

        tot_n_viewed = 0
        n_incorrects = 0
        if self.log_df is not None:
            tot_n_viewed = len(self.log_df.index)
            if self.model_id in self.log_df.columns:
                df_inc = self.log_df.loc[self.log_df[EvalCol.GT_CORRECTED] != self.log_df[self.model_id]]
                n_incorrects = df_inc[self.model_id].count()

        percentage = float(tot_n_viewed) / float(len(self.gts)) * 100.
        n_tot_incorrects = self.calc_num_incorrects()
        inc_percentage = float(n_incorrects) / float(n_tot_incorrects) * 100.

        self.caption3.text = "%d/%d (all) viewed (%.2f%%), %d/%d (FP) viewed (%.2f%%)"%(tot_n_viewed,
                                                                                        len(self.gts), percentage,
                                                                                        n_incorrects, n_tot_incorrects,
                                                                                        inc_percentage)


    def calc_num_incorrects(self):
        return sum(np.array(self.gts_corrected) != np.array(self.enc_preds))


    def update_prediction(self, probs, gt):
        # update prediction
        for idx, lb in enumerate(self.lblButtons):
            # reset selection
            self.enable(lb)
            self.btnGroup_label.setExclusive(False)
            lb.setChecked(False)
            self.btnGroup_label.setExclusive(True)

            if idx == gt:
                if self.ui_mode == self.UI_MODE.EVAL:
                    lb.setChecked(True)
                    self.disable(lb)
                self.current_gt_label = lb


        if self.ui_mode == self.UI_MODE.EVAL:
            # update distribution plot
            figure = Figure(figsize=(self.cw_grv2/self.dpi, self.ch_grv2/self.dpi), dpi=self.dpi)
            axes = figure.gca()
            axes.set_title("Prob Dist. (gt: {})".format(gt))

            print("Probs: ", probs)
            # correct or incorrect
            edgecolor = "green" if gt == np.argmax(probs) else "red"
            axes.bar(self.classes[:len(probs)], probs, color=(0.1, 0.1, 0.1, 0.1), edgecolor=edgecolor)
            fcanvas = FigureCanvas(figure)
            fcanvas.setGeometry(8, 4, self.cw_grv2, self.ch_grv2)
            self.plot2 = QtGui.QGraphicsScene(0, 0, self.cw_grv2, self.ch_grv2)
            self.plot2.addWidget(fcanvas)
            self.grView2.setScene(self.plot2)



    @staticmethod
    def reset_progressbar(bar):
        bar.setValue(0)


    @staticmethod
    def update_progressbar(bar, at, total):
        val = math.ceil(float(at)/float(total) * 100.)
        bar.setValue(val)
        QtWidgets.QApplication.processEvents()


    def infer(self, phase=Phase.TRAIN, sampler=False, max_iter=None, save=False):
        max_iter = None

        self.trainer.curr_phase = phase
        n_batches = self.trainer.n_batches[phase]
        data_iter = iter(self.trainer.data_loader[phase])
        total_iter = min(max_iter, n_batches) if max_iter is not None else n_batches

        fnames_all = None
        gts_all = []
        inputs_all = []
        cluster_preds = None
        enc_preds = None
        net_outputs = None
        with torch.no_grad():
            for di in tqdm(range(total_iter), desc=phase):
                if sampler:
                    inputs_orig = next(iter(self.trainer.data_loader[phase]))
                else:
                    inputs_orig = next(data_iter)

                if self.dl_return_fname:
                    fnames = np.array(inputs_orig[0])
                    fnames_all = fnames if di == 0 else np.append(fnames_all, fnames)
                    inputs = inputs_orig[1:]
                else:
                    inputs = inputs_orig

                if self.model_type == ModelType.TRAP:
                    _, inputs, labels = self.trainer.form_net_inputs(inputs_orig)
                elif self.model_type == ModelType.ALTR:
                    inputs, labels = self.trainer.form_net_inputs(inputs, rescale=True)
                else:
                    inputs, labels = self.trainer.form_net_inputs(inputs)

                if inputs.size(0) == 1:
                    # discard. Pytorch forward() requires more than 1.
                    continue

                if self.model_type in [ModelType.ALTR, ModelType.BMTR]:
                    encoded_feat, decoded_output, _ = self.trainer.net(inputs, deconv=True)
                    encoded_feat_npy = encoded_feat.data.cpu().numpy()
                    net_outputs = encoded_feat_npy if di == 0 else np.vstack((net_outputs, encoded_feat_npy))

                    y_pred_z_cuda = torch.argmax(encoded_feat, dim=1)
                    if False:
                        ef_softmax = F.softmax(encoded_feat)
                        print("ef_softmax: {}".format(ef_softmax.data.cpu().numpy()))
                        print("y_pred_z_cuda: ", y_pred_z_cuda)
                    y_pred_z = y_pred_z_cuda.data.cpu().numpy()

                elif self.model_type == ModelType.TRAP:
                    encoded_feat, _,yhat = self.trainer.net(inputs, cluster=True)
                    yz_cu = torch.argmax(encoded_feat, dim=1)
                    yhat_npy = bmUtil.t2npy(encoded_feat)
                    net_outputs = yhat_npy if di == 0 else np.vstack((net_outputs, yhat_npy))
                    y_pred_z = bmUtil.t2npy(yz_cu)

                else:
                    # original BFMC
                    _, _, yhat = self.trainer.net(inputs)
                    yz_cu = torch.argmax(yhat, dim=1)

                    yhat_npy = bmUtil.t2npy(yhat)
                    net_outputs = yhat_npy if di == 0 else np.vstack((net_outputs, yhat_npy))
                    y_pred_z = bmUtil.t2npy(yz_cu)

                gts_npy = labels.data.cpu().numpy()
                inputs_npy = inputs.squeeze().data.cpu().numpy()

                # accumulate results
                gts_all = gts_npy if di == 0 else np.append(gts_all, gts_npy)
                inputs_all.append(inputs_npy)

#                if self.model_type == ModelType.BMTR:
#                    cluster_preds = y_pred if di == 0 else np.append(cluster_preds, y_pred)
                enc_preds = y_pred_z if di == 0 else np.append(enc_preds, y_pred_z)

                # save clustered data
                if save:
                    for idx in range(len(y_pred_z)):
                        g_idx = di * inputs.size(0) + idx
                        cluster_id = y_pred_z[idx]
                        correct_id = gts_npy[idx]
                        fname = "%s_%d_sa_%d_gt_%d_pd_%d.npy"%(phase, g_idx, sampler, correct_id, cluster_id)
                        save_f = "%s/%d/%s"%(self.cluster_save_dir, cluster_id, fname)
                        np.save(save_f, inputs[idx].data.cpu().numpy())

                # update progressbar
                self.update_progressbar(self.pBarEval, di + 1, total_iter)

        self.log("Done Clustering...", PRT.STATUS)
        if save:
            self.log("Saved %d numpy arrays into files under %s"%(len(enc_preds), self.cluster_save_dir), PRT.LOG)

        return net_outputs, enc_preds, cluster_preds, gts_all, np.array(inputs_all), fnames_all


    def text_clicked(self, event, caller, obj, prefix=None):
        self.log("caller text: {}".format(caller.text()), PRT.ERROR)
        caller_name = caller.text()
        currVal = str(obj.toPlainText())
        is_dir = False if 'Model' in caller_name else True
        updated = self.load_directory(lastVal=currVal, prefix=prefix, is_dir=is_dir)
        if updated is not None:
            obj.setPlainText(updated)

            if caller_name == 'datapath':
                # update dataset selection (subdirs)
                subdirs = next(os.walk(updated))[1]
                self.log("subdirs: {}".format(subdirs))
                self.cboxDataset.clear()
                self.cboxDataset.addItems(subdirs)

            self.reset_build()






    def quit(self):
        """quit the app"""

        for i in range(2, -1, -1):
            if i > 1:
                self.log("Closing in %d seconds..."%i, PRT.WARNING)
            elif i == 1:
                self.log("Goodbye!", PRT.ERROR)

            time.sleep(1)

        sys.exit(0) # why it doesn't fully kill the application??


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


    def load_directory(self, lastVal="/", prefix=None, is_dir=True):
        """connect with menubar's Load option"""

        if prefix is not None:
            preset = os.path.join(prefix, lastVal)
        else:
            preset = lastVal

        if is_dir:
            # directory
            newVal = str(QtGui.QFileDialog.getExistingDirectory(None, 'Select folder', preset,
                                                                   QtGui.QFileDialog.ShowDirsOnly))
        else:
            # file
            newVal = str(QtGui.QFileDialog.getOpenFileName(None, "Select file", preset))

        if lastVal == newVal:
            self.log("selected the same path as current, do nothing!", PRT.WARNING)
            return

        if newVal == "":
            self.log("Selection cancelled.")
            return None

        self.log("got new val: {}".format(newVal), PRT.LOG)

        return newVal if prefix is None else newVal.split('/')[-1]


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


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
#    app.use('glfw')
#    app.setStyle("fusion")
#    if len(sys.argv) != 2:
#        print("\nUsage: python main.py [PHASE]\n")
#        sys.exit(1)

    #app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    phase = sys.argv[-1]
    eapp = EvaluatorApp(phase=phase)
    eapp.setup()
    eapp.show()
    sys.exit(app.exec_())
