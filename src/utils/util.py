"""util.py: util functions"""
__author__      = "Minyoung Kim"
__license__ = "MIT"
__maintainer__ = "Minyoung Kim"
__email__ = "minykim@mit.edu"
__date__ = "07/23/2018"


#----------
# imports
#----------
import os
from os import system
from os.path import *
import inspect
import numpy as np
from colored import fg, bg, attr
from random import shuffle
import matplotlib.pyplot as plt
import scipy.stats as stats
import itertools
import ast
from sklearn.metrics import confusion_matrix, accuracy_score

import pandas as pd
from datetime import datetime

import matplotlib
import matplotlib.cm
from matplotlib.ticker import MultipleLocator


def visualize(array):
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, -z, zdir='z', c='red')
    plt.show()


def itk_visualize(array):
    """visualize numpy data with itkwidget"""
    itk_image = itk.GetImageViewFromArray(array)
    itkw.view(itk_image)


def print_info(name, array):
    print("{}: min({}), max({}), mean({}), shape({})".format(name, np.min(array), np.max(array),
                                                             np.mean(array), array.shape))


##----------------------------------------------------------
# CLASS PRT()
# - print message to terminal with colors different by level
#-----------------------------------------------------------
class PRT(object):
    STATUS = "status-progress"
    STATUS2 = "status-progress-less-important"
    ERROR = "error"
    WARNING = "warning"
    LOG = "log"
    LOGW = "logw"
    FLAGS = {
             STATUS:'#00af00',       # Green3
             STATUS2: '#0087ff',     # DodgerBlue1
             ERROR:'#d70000',        # Red3
             WARNING:'#ff5f00',      # OrangeRed1
             LOG:'#4e4e4e',          # Grey30
             LOGW:'#cecece'          # Brighter Grey
            }

    def __init__(self):
        pass


    @staticmethod
    def p(_msg, _flag):
        if _flag not in PRT.FLAGS.keys():
            #print(colored("INVALID FLAG for PRT_MSG()!", 'red'))
            print("%sINVALID FLAG for PRT_MSG()!%s"%(fg('red'), attr('reset')))
            exit(1)

        #print(colored(_msg, PRT.FLAGS[_flag]))
        print("%s%s%s"%(fg(PRT.FLAGS[_flag]), _msg, attr('reset')))


    @staticmethod
    def html(_caller, _msg, _flag):
        #redText = "<span style=\" color:#ff0000;\" >"
        text = "[{}] <span style=\" color:{};\" >".format(_caller, PRT.FLAGS[_flag])
        text += _msg
        text += "</span>"

        return text



def print_class_params(class_name, class_vars, only=None, exclude=None, returnOnly=False,
                       in_dict=False):
    """print class parameters in a nice format

    Parameters
    ----------
    class_name: string
        name of the class
    class_var: Class Variable Object
        list of tuples of variable name and value
    only: list
        return only variables specified
    exclude: list
        return variables except for specified
    """

    p = PRT()
    items = sorted(class_vars.items())
    filtered = {} if in_dict else []

    for item in items:
        if only is not None:
            if item[0] not in only:
                continue

        if exclude is not None:
            if item[0] in exclude:
                continue

        if in_dict:
            filtered[item[0]] = item[1]
        else:
            filtered.append(item)

    if in_dict:
        return filtered
    else:
        entry = '\n'.join("[ %s ]\t: %s" % item for item in items)
        title_b = "[---------- %s() Variables and their values (BEGIN) ----------]"%class_name
        title_e = "[---------- %s() Variables and their values (END) ----------]"%class_name

        if not returnOnly:
            p.p(title_b, p.LOG)
            p.p("{}".format(entry), p.LOG)
            p.p(title_e, p.LOG)

        return title_b, entry, title_e





##----------------------------------------------------------
# DEF WAIT
# - stop for input
#-----------------------------------------------------------
def WAIT(_str=""):
    st = inspect.stack()
    try:
        wait = input("WAIT(%s) %s"%(str(':'.join(np.array(st[1][1:4]))), _str))
    except:
        pass



##----------------------------------------------------------
# DEF CHECK_DIR
# - check if path exists, and create if not
#-----------------------------------------------------------
def CHECK_DIR(path, _check=True, _verbose=True):
    if _check:
        if not isdir(path):
            if _verbose:
                print("%s doesn't exist, thus create one..."%path)
            system("mkdir -p " + path)



#-----------------------------------------------------------
# DEF PLOT_HIST
# - plot histograms with provided data
#-----------------------------------------------------------
def PLOT_HIST(_data, _titles, _visf=None):
    plt.ion()
    fig = plt.figure(figsize=(20, 30))
    fig.suptitle("Histogram of Data", fontsize=16, weight='bold')

    dim = len(_data)

    for i in range(dim):
        subdata = _data[i]
        subdata.sort()
        max_val, min_val = np.max(subdata), np.min(subdata)
        mean = np.mean(subdata)
        std = np.std(subdata)
        pdf = stats.norm.pdf(subdata, mean, std)

        # plot
        ax = plt.subplot(dim, 1, i+1)
        title = _titles[i] + " histogram (mean: %f, std: %f, max: %f, min: %f)"%(mean, std, max_val, min_val)
        ax.set_title(title, fontsize=10, weight='bold')
        plt.plot(subdata, pdf, '-o', color='red')
        plt.hist(subdata, normed=True, color='yellow', alpha=0.6)

    plt.show()
    plt.ioff()

    if _visf:
        plt.savefig(_visf, dpi=5)
        print("plotHistograms(): histogram plot is saved in %s!"%_visf)

    return



#-----------------------------------------------------------
# DEF read_examples
#-----------------------------------------------------------
def read_examples(_files):
    examples = None
    for f in _files:
        data = np.load(f)
        if data.size:
            if examples is None:
                examples = data
            else:
                examples = np.append(examples, data, axis=0)
    return examples


#-----------------------------------------------------------
# DEF unique_rows
#-----------------------------------------------------------
def unique_rows(_a):
    a = np.ascontiguousarray(_a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


def calc_acc(gt, pred, clustering=False):
    if clustering: # no specific label for class (random-order)
        return cluster_acc(gt, pred)
    else:
        return classification_acc(gt, pred)


def cluster_acc(gt, pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        gt: true labels, numpy.array with shape `(n_samples,)`
        pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    gt = gt.astype(np.int64)
    assert pred.size == gt.size
    D = max(pred.max(), gt.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    for i in range(pred.size):
        w[pred[i], gt[i]] += 1

    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)

    return sum([w[i, j] for i, j in ind]) * 1.0 / pred.size

def classification_acc(gt, pred):

    return accuracy_score(gt, pred)



class ConfusionMatrix(object):
    def __init__(self):
        pass

    def plot_confusion_matrix(self, cm, classes, title='Confusion matrix',
                              cmap=plt.cm.Blues, return_fig=None, debug=False,
                              figsize=(5, 10),
                              xlabel='Predicted label', ylabel='True label',
                              shownorm=True):
        """
        (brought from scikit-learn example)
        This function prints and plots the confusion matrix.
        Two plots are drawn, one for raw results, the other for normalized
        """
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        if debug:
            print('Confusion matrix, without normalization: {}'.format(cm))
            print('Confusion matrix, with normalization: {}'.format(cm_norm))

        if shownorm:
            fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
            ftitle = title + ": raw(left), normalized(right)"
        else:
            fig, axes = plt.subplots(1, 1, figsize=figsize, sharey=True)
            ftitle = title + "(raw)"

        plt.subplots_adjust(wspace=0.4, top=0.99, bottom=0.01)
        if not return_fig:
            fig.tight_layout()
        fig.suptitle(ftitle, verticalalignment='top', fontsize = 7)

        tick_marks = [x for x in range(len(classes))]
        plt.setp(axes, xticks=tick_marks, xticklabels=classes,
                       yticks=tick_marks, yticklabels=classes,
                       ylabel=ylabel, xlabel=xlabel)
        fmt = 'd'
        thresh = cm.max() / 2.
        fmt_norm = '.2f'
        thresh_norm = cm_norm.max() / 2.

        if shownorm:
            ax1, ax2 = axes
            pos1 = ax1.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.colorbar(pos1, ax=ax1, fraction=0.046, pad=0.04)
            pos2 = ax2.imshow(cm_norm, interpolation='nearest', cmap=cmap)
            plt.colorbar(pos2, ax=ax2, fraction=0.046, pad=0.04)
            plt.sca(ax2)
            ax2.set_ylabel('')
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                ax1.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
                ax2.text(j, i, format(cm_norm[i, j], fmt_norm),
                         horizontalalignment="center",
                         color="white" if cm_norm[i, j] > thresh_norm else "black")

        else:
            pos1 = axes.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.colorbar(pos1, ax=axes, fraction=0.046, pad=0.04)
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                axes.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        if return_fig:
            return plt.gcf()
        else:
            plt.show()


    def show_confusion_matrix(self, predictions, gts, num_clusters, title="Confusion matrix",
                              return_fig=None, classes=None, figsize=(5, 10)):
        if classes is None:
            classes = ['0', '1', '2', '3']
        classes = classes[:num_clusters]

        pred_nonzero = np.count_nonzero(predictions)
        gt_nonzero = np.count_nonzero(gts)
#        print("GT: # of Ramified: %d, # of Amoeboid: %d"%(len(gts) - gt_nonzero, gt_nonzero))
#        print("Prediction: # of Ramified: %d, # of Amoeboid: %d"%(len(predictions) - pred_nonzero, pred_nonzero))

        gts_lbl = ['']*len(gts)
        predictions_lbl = ['']*len(gts)
        for i in range(len(gts)):
            gts_lbl[i] = classes[gts[i]]
            predictions_lbl[i] = classes[predictions[i]]
        #print("classes: ", classes)
        #print("gts_lbl: ", gts_lbl[:10])
        #print("predictions_lbl: ", predictions_lbl[:10])

        #cnf_matrix = confusion_matrix(gts, predictions, labels=classes)
        cnf_matrix = confusion_matrix(gts_lbl, predictions_lbl, labels=classes)
        np.set_printoptions(precision=2)

        return self.plot_confusion_matrix(cnf_matrix, classes=classes, title=title,
                                          return_fig=return_fig, figsize=figsize)


def t2npy(t):
    """convert torch tensor to numpy array

    Parameters
    ----------
    t: Torch Tensor
        data to convert to numpy array
    """

    return t.data.cpu().numpy()


def bmVstack(stack, data):
    """vstack new data to a stack, return data if stack is None"""
    if stack is None:
        return data
    else:
        return np.vstack((stack, data))


def bmAppend(stack, data):
    """append new data to a stack, return data if stack is None"""
    if stack is None:
        return data
    else:
        return np.append(stack, data)


def resample_coords(npy_list, ratio=(1.0, 1.0, 1.0)):
    new_list = []
    zrat, yrat, xrat = ratio
    for item in npy_list:
        z, y, x = item
        nz = int(z*zrat)
        ny = int(y*yrat)
        nx = int(x*xrat)
        new_list.append([nz, ny, nx])

    return new_list


def convert_cc_npy_ratio(npy_file, ratio=(1.0, 1.0, 1.0)):
    data = np.load(npy_file)
    print("len(data): ", len(data))

    new_data = resample_coords(data, ratio)

    new_fname = npy_file[:-4] + "_new" + npy_file[-4:]
    np.save(new_fname, new_data)
    print("Done!")


def convert_cc_csv_ratio(csv_file, ratio=(1.0, 1.0, 1.0)):
    zrat, yrat, xrat = ratio

    df = pd.read_csv(csv_file)
    print("df.head(): ", df.head())
    for index, row in df.iterrows():
        row['z'] = int(row['z'] * zrat)
        row['y'] = int(row['y'] * yrat)
        row['x'] = int(row['x'] * xrat)
    print("df.head(): ", df.head())
    new_csvf = csv_file[:-4] + "_new" + csv_file[-4:]
    df.to_csv(new_csvf, sep=',', index=False)


def colorize(value, vmin=None, vmax=None, cmap='viridis'):
    """
    A utility function for Torch/Numpy that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap.
    Arguments:
      - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
        [height, width, 1].
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: Matplotlib default colormap)

    Returns a 4D uint8 tensor of shape [height, width, 4].
    """

    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin!=vmax:
        value = (value - vmin) / (vmax - vmin) # vmin..vmax
    else:
        # Avoid 0-division
        value = value*0.
    # squeeze last dim if it exists
    value = value.squeeze()
    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(t2npy(value), bytes=True) # (nxmx4)

    return value


def get_current_time():
    return datetime.now().strftime('%Y-%m-%d_%H%M%S')

def BM_lEval(arg):
    if arg in ['None', 'none']:
        return None
    else:
        try:
            arg_le = ast.literal_eval(arg)
            return arg_le
        except ValueError:
            return arg
