"""params.py: Parameter Classes"""
__author__      = "Minyoung Kim"
__license__ = "MIT"
__maintainer__ = "Minyoung Kim"
__email__ = "minykim@mit.edu"
__date__ = "08/15/2018"

import sys
sys.path.append("../")
import os
import argparse
import socket
from distutils.util import strtobool
import ast
import json

# internal
import utils.util as bmUtil
from utils.const import Phase, Dataset, ModelType, StainChannel, NormalizationType


class BaseParams(object):
    """BaseParams Class"""
    prt = bmUtil.PRT()

    def __init__(self):
        self.debug = False


    def _parser(self, desc=None):
        """add list of arguments to ArgumentParser
           : This is a placeholder for subclasses

        Params
        ---------
        desc: description of Params set
        """
        parser = argparse.ArgumentParser(description=desc)
        parser.add_argument('-dbg', '--debug', action='store_true', default=False)

        return parser


    def build(self, argv, desc, returnOnly=False):
        """parse arguments

        Params
        ---------
        argv: parameter arguments from command line
        desc: description of the Params set
        """

        # parse
        parser = self._parser(desc)
        args = parser.parse_args(argv[1:])
        args_dict = vars(args)

        # update class variables with args passed
        vars(self).update(args_dict)

        # placeholder function for arguments post-processing
        self.postproc_args()

        # print if needed
        if self.debug:
            self.print_params(returnOnly)


    def postproc_args(self):
        # Nothing to do for base class
        pass


    def print_params(self, returnOnly=False, in_dict=False):
        """print class parameters in a nice format"""
        return bmUtil.print_class_params(self.__class__.__name__, vars(self),
                                         returnOnly=returnOnly, in_dict=in_dict)


class MorphPredictionParams(BaseParams):
    """Parameter Class for Morphology Prediction"""

    def __init__(self):
        super(MorphPredictionParams, self).__init__()


    def _parser(self, desc=None):
        parser = super(MorphPredictionParams, self)._parser(desc)
        parser.add_argument('-pf', '--param_file', required=True, help='parameter file path')
        parser.add_argument('-ck', '--checkpoint', default=None, help='for data reuse')
        parser.add_argument('-nvz', '--do_not_visualize', action='store_true', default=False)
        parser.add_argument('-mt', '--model_type', default=ModelType.BMTR)
        parser.add_argument('-chn', '--channel', default=['ch1'],
                            help='channel types, [ch1, ch2, ch3, ch4]')

        return parser



class CellCenterDetectionParams(BaseParams):
    """CellCenterDetectionParams Class - holds parameters for CCD using Phathom"""
    def __init__(self):
        super(CellCenterDetectionParams, self).__init__()


    def _parser(self, desc=None):
        parser = super(CellCenterDetectionParams, self)._parser(desc)
        parser.add_argument('-pf', '--param_file', required=True, help='parameter file path')
        parser.add_argument('-ch', '--channel', required=True,
                            help='stain channel type, [GFP, AF, ToPro3, IBA1, ...]')
        parser.add_argument('-sn', '--slice_no', default=2,
                            help='z-slice idx for sample visualization')
        parser.add_argument('-dm', '--damp', default=2,
                            help='damping thr for center coordinate filtering')

        return parser


class DataConversionParams(BaseParams):
    """DataConversionParams Class - holds parameters for data conversion"""
    def __init__(self):
        super(DataConversionParams, self).__init__()


    def _parser(self, desc=None):
        parser = super(DataConversionParams, self)._parser(desc)
        parser.add_argument('-dr', '--d_root', help='root directory of data', required=True)
        parser.add_argument('-ch', '--channel', default=StainChannel.GFP,
                            help='stain channel type, [GFP, AF, ToPro3]')
        parser.add_argument('-cs', '--chunk_size', default='(100, 100, 100)',
                            help='zarr chunk size')
        parser.add_argument('-bw', '--batchwise', action='store_true', default=False,
                            help='run in batch or not')
        parser.add_argument('-ext', '--file_ext', default='tif', help='file extension of raw data')

        return parser


    def postproc_args(self):
        self.chunk_size = ast.literal_eval(self.chunk_size)


class DataGenParams(BaseParams):
    """DataGenParams Class - holds parameters for data generation phase"""
    def __init__(self):
        super(DataGenParams, self).__init__()


    def _parser(self, desc=None):
        parser = super(DataGenParams, self)._parser(desc)
        parser.add_argument('-dr', '--d_root', default='/data_ssd/brain_mapping/', help='root directory of raw data')
        parser.add_argument('-dp', '--data_rel_path', default='', help='rel data path')
        parser.add_argument('-dt', '--data_type', default='cc', help="type of data to generate (e.g. 'cc', '16x480x480')")
        parser.add_argument('-dw', '--data_width', type=int, default=4506, help="raw data width (x-axis)")
        parser.add_argument('-dh', '--data_height', type=int, default=7118, help="raw data height (y-axis)")
        parser.add_argument('-dd', '--data_depth', type=int, default=2000, help="raw data depth (z-axis)")

        parser.add_argument('-ext', '--file_ext', default='tiff', help='file extension of raw data')
        parser.add_argument('-ac', '--ann_cell_center', default='cell_centers.npy', help='numpy file of cell center coordinates')
        parser.add_argument('-ap', '--ann_pos', default=None, help='json file of positive coordinates')
        parser.add_argument('-an', '--ann_neg', default=None, help='json file of negative coordinates')
        parser.add_argument('-ts', '--timestmp', required=True,
                                                 help='timestamp that would automatically set by caller')
        parser.add_argument('-nc', '--num_cpu', type=int, default=10, help='number of CPUs to use')
        parser.add_argument('-ns', '--num_samples', type=int, default=10, help='number of samples to generate per CPU')
        parser.add_argument('-sp', '--save_path', default=None, help='directory where data being saved')
        parser.add_argument('-dns', '--do_not_save', type=strtobool, default=False)

        parser.add_argument('-ch', '--csz_half', type=int, default=16, help="center-crop volume half (xy-axis)")
        parser.add_argument('-cq', '--csz_quart', type=int, default=8, help="center-crop volume quart (z-axis)")

        # for multiple channels
        parser.add_argument('-mc', '--multi_channel', default=None, help='format:[ch1, ch2, ch3, ch4]')

        return parser


    def postproc_args(self):
        if self.ann_cell_center:
            self.ann_cell_center = os.path.join(self.d_root, self.ann_cell_center)
            if not os.path.isfile(self.ann_cell_center):
                self.prt.p("Cell Center Annotation file (%s) doesn't exist! resetting to None."%self.ann_cell_center,
                         self.prt.WARNING)
                self.ann_cell_center = None

        self.ann_pos = self.d_root + self.ann_pos if self.ann_pos else None
        self.ann_neg = self.d_root + self.ann_neg if self.ann_neg else None
        self.data_path = os.path.join(self.d_root, self.data_rel_path)

        if not os.path.isdir(self.data_path):
            self.prt.p("Please set correct [d_root](%s) and [data_rel_path](%s)!"%(self.d_root, self.data_rel_path), self.prt.ERROR)
            sys.exit(1)

        if self.multi_channel is not None:
            self.multi_channel = ast.literal_eval(self.multi_channel)

        if not self.do_not_save:
            if self.save_path is None:
                self.save_path = os.path.join(self.d_root, "training_data", self.data_type, self.timestmp)
            else:
                self.save_path = os.path.join(self.save_path,
                                              "%s_%s"%(self.timestmp, self.data_type))
            bmUtil.CHECK_DIR(self.save_path)


    def add_phase_to_savepath(self, phase):
        self.save_path = os.path.join(self.save_path, phase)



class CellDataParams(DataGenParams):
    """Basic Parameter Class for 3-D Volumetric Cell dataset"""

    def __init__(self):
        super(CellDataParams, self).__init__()

    def _parser(self, desc=None):
        parser = super(CellDataParams, self)._parser(desc)
        parser.add_argument('-pj', '--param_json', default='params.json', help='Parameter JSON filename')
        parser.add_argument('-ar', '--ann_roi', default='/data/raw/taeyun/labels/all', help='ROI annotation file directory')
        return parser

    def postproc_args(self):
        super(CellDataParams, self).postproc_args()

        # read json
        pf = os.path.join(self.d_root, self.param_json)
        with open(pf) as fp:
            params = json.load(fp)

        self.voxel_size = params['voxel_size']
        self.data_depth = params['dd']
        self.data_height = params['dh']
        self.data_width = params['dw']
        self.ranges = [params['zr'], params['yr'], params['xr']]
        try:
            self.btype = params['btype']
            self.bid = params['bid']
        except KeyError:
            # no information available
            pass

        self.clims = []
        for ch in sorted(self.multi_channel):
            try:
                self.clims.append(params[ch]['clim'])
            except KeyError:
                self.clims.append(None)



class TrainParams(BaseParams):
    """TrainParams Class - holds parameters for training phase"""

    def __init__(self):
        super(TrainParams, self).__init__()


    def _parser(self, desc=None):
        parser = super(TrainParams, self)._parser(desc)
        parser.add_argument('-dp', '--data_path', default='/data_ssd/brain_mapping/training_data/cc/073118',
                                                  help='training data location')
        parser.add_argument('-ext', '--file_ext', default='npy',
                                                  help='file extension of training data')
        parser.add_argument('-dw', '--data_w', type=int, default=32)
        parser.add_argument('-dh', '--data_h', type=int, default=32)
        parser.add_argument('-dd', '--data_d', type=int, default=16)
        parser.add_argument('-rw', '--resnet_weights', default=None)
        parser.add_argument('-aw', '--ae_weights', default=None,
                                                help='pretrained autoencoder weights')
        parser.add_argument('-awf', '--freeze_aw', action='store_true', default=False,
                                                help='freeze pretrained autoencoder weights')
        parser.add_argument('-pw', '--pretrained_weights_all', default=None,
                                                help='pretrained weights (autoencoder -> cluster)')
        parser.add_argument('-ph', '--phase', default=Phase.TRAIN, required=True)
        parser.add_argument('-mt', '--model_type', default=ModelType.BMTR)
        parser.add_argument('-bs', '--batch_size', type=int, default=1)
        parser.add_argument('-e', '--epoch', type=int, default=800)
        parser.add_argument('-ce', '--clustering_epoch', type=int, default=8000)
        parser.add_argument('-nc', '--num_clusters', type=int, default=2)
        parser.add_argument('-lg', '--log_root', default='./src/train/logs')
        parser.add_argument('-sr', '--save_root', default='./src/train/weights/models')
        parser.add_argument('-fr', '--full_resolution', action='store_true', default=False)
        parser.add_argument('-fi', '--force_inference', action='store_true', default=False)
        parser.add_argument('-sh', '--shuffle', action='store_true', default=False)
        parser.add_argument('-us', '--use_sampler', type=strtobool, default=True,
                                                help='use Sampler in DataLoader in Train phase')
        parser.add_argument('-usv', '--use_sampler_in_val', type=strtobool, default=False,
                                                help='use Sampler in DataLoader in Validate phase')
        parser.add_argument('-de', '--description', help='description of training')
        parser.add_argument('-ds', '--dataset', default=Dataset.MICROGLIA, help='Dataset name, e.g. [mnist, microglia]')
        parser.add_argument('-ts', '--timestmp', required=True,
                                                  help='timestamp that would automatically set by caller')
        parser.add_argument('-ie', '--is_eval', type=strtobool, default=False, help='network on eval mode or not')
        parser.add_argument('-n2d', '--no2dUnet', action='store_true', default=False)
        parser.add_argument('-nt', '--norm_type', required=True, default=NormalizationType.ZERO_AND_ONE,
                                                  help='options: [zero_and_one,  minus_one_and_one, zero_mean]')
        parser.add_argument('-cl', '--clip_data', required=True, type=strtobool, default=False)
        parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
        parser.add_argument('-al', '--alpha', default=0.5, type=float,
                                                help='alpha for recon and class loss weighting')
        parser.add_argument('-na', '--no_alpha', action='store_true', default=False,
                                                help='use alpha or not')

        return parser


    def postproc_args(self):
        self.hostname = socket.gethostname()
        self.log_dir = '{}/{}-{}/{}'.format(self.log_root, self.timestmp,
                                            self.hostname, self.phase)

        if Phase.TRAIN in self.phase:
            self.save_dir = '{}/{}-{}'.format(self.save_root, self.timestmp, self.hostname)
            bmUtil.CHECK_DIR(self.save_dir)
        else:
            self.save_dir = None

        self.set_preprocessing_params()


    def set_preprocessing_params(self):
        """build parameter map for preprocessing"""

        self.preproc_map = {
                                #"flip_rate": 0.5,
                                #
                           }
