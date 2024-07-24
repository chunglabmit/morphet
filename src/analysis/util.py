# -*- coding: utf-8 -*-
"""util.py: utility for analysis"""
__author__      = "Minyoung Kim"
__license__ = "MIT"
__maintainer__ = "Minyoung Kim"
__email__ = "minykim@mit.edu"
__date__ = "10/15/2018"

import sys
import os
import yaml
from datetime import datetime
from argparse import Namespace

# internal
from cellCentersAnalyzer import CellCenterAnalyzer as CCA
from utils.const import StainChannel


def get_sample_volume(data_root,
                      checkpoint=None,
                      param_file="params.json",
                      channel1=StainChannel.GFP,
                      channel1_clim=None,
                      channel2=StainChannel.AUTOFLUORESCENCE,
                      channel2_clim=None,
                      dist_thr=150.0,
                      vx=None, vy=None, vz=None):
    """more generic function for retrieving a sample volume to visualize

    Parameters
    ----------
    data_root: string
        path to working directory
    checkpoint: string
        (optional) checkpoint time
    param_file: string
        .json parameter file name
    channel1: string
        First StainChannel
    channel2: string
        Second StainChannel
    dist_thr: float
        (optional) distance threshold for center filtering using the second channel
    """

    dist_thr = 150.0
    param_f = os.path.join(data_root, param_file)
    print("reading parameter file [%s]"%param_f)
    with open(param_f) as fp:
        params_dict = yaml.safe_load(fp)

    params = Namespace(**params_dict)
    paramsByChn1 = Namespace(**params_dict[channel1])

    ranges = (params.zr, params.yr, params.xr)

    chn1_rel_path = os.path.join(paramsByChn1.rel_path, params.tif_rel_path)
    chn1_cc_npy = os.path.join(paramsByChn1.rel_path, paramsByChn1.cc_npy) \
                  if hasattr(paramsByChn1, 'cc_npy') else ""


    if channel2:
        paramsByChn2 = Namespace(**params_dict[channel2])
        chn2_rel_path = os.path.join(paramsByChn2.rel_path, params.tif_rel_path)
        chn2_cc_npy = os.path.join(paramsByChn2.rel_path, paramsByChn2.cc_npy) \
                      if hasattr(paramsByChn2, 'cc_npy') else ""

    now = datetime.now()
    current_time = now.strftime("%Y%m%d-%H%M") if checkpoint is None else checkpoint

    gargs = [ sys._getframe().f_code.co_name,
              '-dr', params.d_root,
              '-dt', '%s_%s'%(params.name, channel1),
              '-dw', str(params.dw),
              '-dh', str(params.dh),
              '-dd', str(params.dd),
              '-ts', current_time ]

    title = "Sample Volume (%s)"%params.name

    args_chn1 = [ '-dp', chn1_rel_path, '-ac', chn1_cc_npy ]
    args_chn2 = [ '-dp', chn2_rel_path, '-ac', chn2_cc_npy ] if channel2 else None

    cca = CCA(title=title, keys='interactive', size=(800, 600), show=True, logWindow=None,
              gparams=gargs, mparams=args_chn1, nparams=args_chn2,
              ranges=ranges, dist_threshold=dist_thr,
              clims=[channel1_clim, channel2_clim], voxel_size=[vz, vy, vx])

    return cca



def get_a_sample_volume(args, fromPhathom=False):
    zr = (18, 50)
    yr = None
    xr = None
    dist_thr = 150.0

    cc_npy = "cell_centers_Phathom.npy" if fromPhathom else "cell_centers.npy"
    title = "Microglia-Nuclei_Phathom" if fromPhathom else "Microglia-Nuclei"

    #GFP_REL_PATH = 'E10.5_small_substack/100_GFP_Hysteresis'
    GFP_REL_PATH = 'E10.5_small_substack/100_GFP'
    TOPRO3_REL_PATH = 'E10.5_small_substack/100_ToPro'
    params_GFP = ['-dp', GFP_REL_PATH, '-ac', '%s/%s'%(GFP_REL_PATH, cc_npy)]
    params_ToPro3 = ['-dp', TOPRO3_REL_PATH, '-ac', '%s/%s'%(TOPRO3_REL_PATH, cc_npy)]

    cca = CCA(title=title, keys='interactive', size=(800, 600), show=True, logWindow=None,
              gparams=args, mparams=params_GFP, nparams=params_ToPro3,
              ranges=[zr, yr, xr], dist_threshold=dist_thr)
    return cca
