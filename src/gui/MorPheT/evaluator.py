"""main.py: Main entry point for Evaluator GUI"""
"""taken from the standalone Evaluator GUI"""
__author__      = "Minyoung Kim"
__license__ = "MIT"
__maintainer__ = "Minyoung Kim"
__email__ = "minykim@mit.edu"
__date__ = "09/29/2021"


import sys
import os.path
import math
import numpy as np
from sklearn.metrics import confusion_matrix
from functools import partial
import pandas as pd

# UI
from PyQt5 import QtGui
#from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
from matplotlib.backends.qt_compat import is_pyqt5
from matplotlib.figure import Figure
if is_pyqt5():
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
else:
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)

from itertools import cycle

import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score as aps
from sklearn.metrics import precision_recall_curve as prc

# torch
import torch
import torch.nn.functional as F

# Internal
import utils.util as bmUtil
from utils.util import PRT
from utils.const import Phase, Dataset, ModelType, EvalCol
from utils.params import TrainParams
from train.main import build_dataset, build_trainer
from utils.data.dbLabel import MDBLabel, TDBLabel


class EvaluatorApp():
    class UI_MODE(object):
        EVAL = 'EVAL'
        CORRECTING = 'CORRECTING'

    def __init__(self, qtApp, phase='val'):
        """init"""
        self.app = qtApp
        self.phase = phase

        self.ui_mode = self.UI_MODE.CORRECTING
        self.prep = None
        self.params = None
        self.MLabel = MDBLabel()
        #self.MLabel = TDBLabel()
        self.classes = self.MLabel.get_ordered_labels_by_id()
        self.lblAccs = [ self.app.lbl_acc_r_val, self.app.lbl_acc_a_val, self.app.lbl_acc_g_val ]
        self.current_gt_label = None

        self.dl_return_fname = True
        self.dpi = 96
        self.old_gidx = None

        self.screenshot_dir = "screenshot"
        self.eval_csv = "evaluation.csv"
        self.eval_logfile = "" # to be set in build_params()

        # initialize data
        self.outputs = None
        self.enc_preds = None
        self.cluster_preds = None
        self.gts = None
        self.log_df = None

        self.link_actions()
        self.reset_plots(init=True)


    def reset_plots(self, init=False):
        if init:
            # initialize with/height for each plot section
            self.cw_grv1 = int(self.app.grView1.frameGeometry().width() * 0.95)
            self.ch_grv1 = int(self.app.grView1.frameGeometry().height() * 0.9)
            self.cw_grv3 = int(self.app.grView3.frameGeometry().width() * 0.87)
            self.ch_grv3 = int(self.app.grView3.frameGeometry().height() * 0.92)
            self.cw_grv4 = int(self.app.grView3.frameGeometry().width() * 0.87)
            self.ch_grv4 = int(self.app.grView3.frameGeometry().height() * 0.92)

            self.plot1 = QtGui.QGraphicsScene(0, 0, self.cw_grv1, self.ch_grv1)
            self.plot2 = QtGui.QGraphicsScene(0, 0, self.cw_grv1, self.ch_grv1)
            self.plot3 = QtGui.QGraphicsScene(0, 0, self.cw_grv1, self.ch_grv1)
            self.plot4 = QtGui.QGraphicsScene(0, 0, self.cw_grv1, self.ch_grv1)
        else:
            self.plot1.clear()
            self.plot2.clear()
            self.plot3.clear()
            self.plot4.clear()


    def link_actions(self):
        """link action functions to target UI items"""
        self.app.btnEval.clicked.connect(self.evaluate)


    def update_evaluation(self, corrected):
        if self.gts is None:
            return

        gts_all = self.gts

        # remove unnecessary items (containing not-being-considered labels)
        indices = np.where(gts_all < self.params.num_clusters)
        gts = gts_all[indices]
        preds = self.enc_preds[indices]
        outs = self.outputs[indices]

        if self.model_type == ModelType.BMTR:
            cluster_preds = self.cluster_preds[indices]
        else:
            cluster_preds = None

        # correct plots
        self.show_confusion_matrix(preds, gts, self.params.num_clusters, "Confusion matrix (enc_pred)")
        self.show_prcurves(outs, gts, self.params.num_clusters)

        # update numbers
        self.update_stats(gts, preds, cluster_preds)


    def update_stats(self, gts, preds, cluster_preds=None):
        # global accuracy
        acc_all = bmUtil.calc_acc(gts, preds) * 100.
        self.app.lbl_acc_all_val.setText("%.2f%%"%acc_all)
        self.app.lbl_ap_all_val.setText("%.2f"%self.avg_precision_all)

        # accuracy per class
        for i in range(self.params.num_clusters):
            ind = [ x for x, e in enumerate(gts) if e == i]
            acc = bmUtil.calc_acc(gts[ind], preds[ind]) * 100.
            self.lblAccs[i].setText("%.2f%%"%acc)

        # logging
        self.app.log("\nExpr: %s"%(self.expr), PRT.LOG)
        self.app.log("Phase: {} (Sampling={})".format(self.subset, self.get_if_sample()), PRT.LOG)
        self.app.log("Accuracy\n   ----------------------", PRT.LOG)
        self.app.log("   enc_acc: %.2f%%"%acc_all, PRT.LOG)
        if cluster_preds is not None:
            self.cluster_acc = bmUtil.calc_acc(gts, cluster_preds)
            self.app.log("   cluster_acc: %.2f%%"%(self.cluster_acc * 100.), PRT.LOG)
        self.app.log("   ----------------------", PRT.LOG)


    def build_params(self):
        self.app.log("Building parameters...", PRT.STATUS)
        self.expr = self.app.model_id
        self.mparams = self.app.model_params[self.expr]

        self.model_type = self.mparams['net_type']
        dataset_name = self.mparams['train_data']
        dataset_path = self.mparams['train_data_path']
        nclass = str(self.mparams['num_class'])

        self.model = self.mparams['model_file']
        mf_splits = self.mparams['model_file'].split('/')
        self.train_model_root = '/'.join(self.model.split('/')[:-1])

        # TODO: get from GUI
#        bsize = str(self.sbBatchSize.value())
#        sampling = str(self.cboxDataSampling.currentText())
#        self.subset = str(self.cboxDataSubset.currentText())
        bsize = '48'
        self.subset = 'val'
        sampling = 'No'

        self.app.log("model_type: [%s]"%self.model_type, PRT.LOG)
        self.app.log("model: [%s]"%self.model, PRT.LOG)
        self.app.log("train_model_root: [%s]"%self.train_model_root, PRT.LOG)
        self.app.log("dataset: [%s]"%(dataset_path), PRT.LOG)
        self.app.log("# class: [%s], batchsize: [%s]"%(nclass, bsize), PRT.LOG)
        self.app.log("sample? ({}): {}".format(self.subset, sampling), PRT.LOG)

        if self.model_type == ModelType.TRAP:
            args = ['GUI', '-ph', self.phase, '-bs', '8', '-e', '2', '-ts', 'GUI', '-nc', '2',
                    '-ie', 'True',
                    '-ds', 'trap', '-us', sampling, '-usv', sampling,
                    '-dw', '16', '-dh', '16', '-dd', '8',
                    '-mt', self.model_type,
                    '-aw', self.model, '-dp', dataset_path, '--debug']
        else:
            args = ['GUI', '-ph', self.phase, '-bs', bsize, '-e', '2', '-ts', 'GUI', '-nc', nclass,
                    '-ie', 'True',
                    '-ds', dataset_name, '-us', sampling, '-usv', sampling,
                    '-mt', self.model_type,
                    '-aw', self.model, '-dp', dataset_path, '--debug']

        self.dataset_path = dataset_path
        self.params = TrainParams()
        self.params.build(args, "Evaluator_TRParser")

        # set save directory
        self.cluster_save_dir = os.path.join(self.train_model_root, self.expr, "clustered")
        self.app.log("cluster_save_dir: %s"%self.cluster_save_dir)
        bmUtil.CHECK_DIR(self.cluster_save_dir)
        for i in range(self.params.num_clusters):
            bmUtil.CHECK_DIR("%s/%d"%(self.cluster_save_dir, i))

        # set labels directory, load if log exists
        self.lbl_dir = os.path.join(dataset_path, self.subset, "labels")
        self.eval_logfile = os.path.join(dataset_path, self.subset, self.eval_csv)
        if os.path.isfile(self.eval_logfile):
            self.app.log("Reading exisisting Log DataFrame from [%s]!"%self.eval_logfile, PRT.STATUS)
            self.log_df = pd.read_csv(self.eval_logfile)
        else:
            self.log_df = None

        # build dataset
        self.app.log("building dataset...", PRT.STATUS)
        db = build_dataset(self.params, self.app.log)

        # build trainer
        self.app.log("building trainer...", PRT.STATUS)
        self.trainer = build_trainer(self.params, db)


    def show_prcurves(self, predictions, gts, num_clusters):
        gts_onehot = np.zeros((gts.shape[0], num_clusters), dtype=np.int8)
        for i in range(gts.shape[0]):
            gts_onehot[i][gts[i]] = 1

        precision = dict()
        recall = dict()
        avg_precision = dict()
        for i in range(num_clusters):
            precision[i], recall[i], _ = prc(gts_onehot[:, i], predictions[:, i])
            avg_precision[i] = aps(gts_onehot[:, i], predictions[:, i])
            self.app.log('Average precision score for class %d (%s): %.3f'%(i, self.MLabel.get_name(i), avg_precision[i]), PRT.STATUS2)

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = prc(gts_onehot.ravel(), predictions.ravel())
        avg_precision["micro"] = aps(gts_onehot, predictions, average="micro")
        self.avg_precision_all = avg_precision["micro"]
        self.app.log('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(avg_precision["micro"]), PRT.STATUS2)

        # visualize
        plt.figure(figsize=(int(self.cw_grv3/self.dpi),
                            int(self.ch_grv3/self.dpi)), dpi=self.dpi)
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

        fcanvas.setGeometry(2, 2, self.cw_grv3, self.ch_grv3)
        self.plot3 = QtGui.QGraphicsScene(0, 0, self.cw_grv3, self.ch_grv3)
        self.plot3.addWidget(fcanvas)
        self.app.grView3.setScene(self.plot3)

        # visualize PR Curves per class
        # setup plot details
        colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

        plt.figure(figsize=(self.cw_grv4/self.dpi, self.ch_grv4/self.dpi), dpi=self.dpi)
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
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR-Curve per class')
        plt.legend(lines, labels, loc=(.03, .03), prop=dict(size=5))

        #fcanvas = FigureCanvas(plt.gcf())
        fcanvas = FigureCanvas(fig)
        fcanvas.setGeometry(2, 2, self.cw_grv4, self.ch_grv4)
        self.plot4 = QtGui.QGraphicsScene(0, 0, self.cw_grv4, self.ch_grv4)
        self.plot4.addWidget(fcanvas)
        self.app.grView4.setScene(self.plot4)


    def show_confusion_matrix(self, predictions, gts, num_clusters, title="Confusion matrix"):
        CM = bmUtil.ConfusionMatrix()
        fig = CM.show_confusion_matrix(predictions, gts, num_clusters,
                                       "Confusion matrix (enc_pred)",
                                       return_fig=True, classes=self.classes,
                                       figsize=(5,5))
        fcanvas = FigureCanvas(fig)
        fcanvas.setGeometry(2, 2, self.cw_grv1, self.ch_grv1)

        self.plot1 = QtGui.QGraphicsScene(0, 0, self.cw_grv1, self.ch_grv1)
        self.plot1.addWidget(fcanvas)
        self.app.grView1.setScene(self.plot1)


    def get_if_sample(self):
        if self.subset == Phase.TRAIN:
            sampler = bool(self.params.use_sampler)
        elif self.subset == Phase.VAL:
            sampler = bool(self.params.use_sampler_in_val)
        else:
            sampler = False

        return sampler


    def evaluate(self):
        self.app.disable(self.app.btnEval)

        # build params
        self.build_params()
        np.set_printoptions(suppress=True)
        torch.set_printoptions(precision=1)
        sampler = self.get_if_sample()
        self.app.log("phase: {}, use_sampler: {}".format(self.subset, sampler), PRT.STATUS)

        # inference
        max_iter = None
        self.outputs, \
        self.enc_preds, \
        self.cluster_preds, \
        self.gts = self.infer(phase=self.subset, sampler=sampler, max_iter=max_iter)

        # update plots and statistics
        self.update_evaluation(corrected=False)

        self.app.enable(self.app.btnEval)


    def infer(self, phase=Phase.TRAIN, sampler=False, max_iter=None, save=False):
        net_outputs, \
        enc_preds, \
        cluster_preds, \
        gts, \
        enc_acc, \
        cluster_acc = self.trainer.validate(return_result=True,
                                            pBar=self.app.pBar,
                                            pBarFunc=self.app.update_progressbar)

        return net_outputs, enc_preds, cluster_preds, gts
