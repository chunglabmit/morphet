"""tif2zarr.py: Class to convert tif zstack into zarr format
    - The conversion from tif zstack to zarr is needed in such following cases:
       1) run cell centroid detection algorithm with phathom
          - see codes under src/ccd/
       2) visualize big data on neuroglancer
          - it helps reducing buffering in loading
    - Corresponding notebook: src/notebooks/ConvertTiff2Zarr.ipynb
"""
__author__      = "Minyoung Kim"
__license__ = "MIT"
__maintainer__ = "Minyoung Kim"
__email__ = "minykim@mit.edu"
__date__ = "12/14/2018"


import os
import sys
try:
    assert sys.version_info >= (3, 5)
except AssertionError:
    print("Python 3.5 or higher is required to run!")
    sys.exit(1)

import json
import numpy as np
from argparse import Namespace
from tqdm import tqdm
import math

if sys.version_info >= (3,0):
    from PyQt5 import QtGui
else:
    from PyQt4 import QtGui

# phathom
from phathom import io

# internal
from utils.params import DataConversionParams
from utils.data import data_generator, preprocessing
from utils.const import NormalizationType, StainChannel
import utils.util as bmUtil
from utils.util import PRT


class Tif2Zarr(object):
    def __init__(self, p, logWindow=None, qtProgressBar=None,
                 paramf='params.json', z_rel_path = 'zarr'):
        """init function

        Parameters
        ----------
        p: DataConversionParams()
        """

        self.data_root = p.data_root
        self.batchwise = p.batchwise
        self.channel = p.channel
        self.param_file = os.path.join(self.data_root, paramf)
        try:
            with open(self.param_file) as fp:
                self.params_dict = json.load(fp)
                self.params_dictByChn = self.params_dict[self.channel]
                self.p = Namespace(**self.params_dict)
                self.pByChn = Namespace(**self.params_dictByChn)
        except IOError:
            print("parameter file (%s) not found"%self.param_file)

        self.ZARR_REL_PATH = z_rel_path
        self.tiffpath = os.path.join(self.data_root, self.pByChn.tif_rel_path)
        self.zarrpath = os.path.join(self.data_root, self.pByChn.zarr_rel_path)
        self.vol_size = (self.p.dd, self.p.dh, self.p.dw)
        self.chunk_size = p.chunk_size

        # GUI
        self.logwin = logWindow
        self.pbar = qtProgressBar

        bmUtil.print_class_params(self.__class__.__name__, vars(self))


    def convert(self):
        # check if Zarr is already been created!
        if os.path.exists(self.zarrpath):
            # TODO: check size with the raw data
            self.log("Zarr already exists! check the directory: %s"%self.zarrpath)
            return

        else:
            # create zarr directory
            try:
                os.mkdir(self.zarrpath)
                self.log("Successfully created Zarr directory [ %s ]"%self.zarrpath)
            except OSError:
                self.log("Failed to create Zarr directory [ %s ]. Check permission and retry!"%self.zarrpath, PRT.ERROR)
                return

        self.zarr = io.zarr.new_zarr(self.zarrpath, self.vol_size, self.chunk_size, np.float32)
        # TODO: move read_tiff3d() to IO class instead of DG
        dg = data_generator.DataGenerator(self.p)

        if self.batchwise:
            batch_size = self.chunk_size[0]
            smean = 0
            svar = 0
            gmin = 2**16
            gmax = 0
            cnt = 0
            for s in tqdm(range(0, self.vol_size[0], batch_size), desc="Converting Tif2Zarr batchwise.."):
                cnt += 1
                # check if it's last batch
                if s + batch_size > self.vol_size[0]:
                    print("last batch!")
                    stack_size = self.vol_size[0] - s
                else:
                    stack_size = batch_size

                subvol = dg._read_tiff3d(self.tiffpath, self.vol_size, s, stack_size, ext=self.p.file_ext)

                # write to Zarr
                io.zarr.write_subarray(subvol, self.zarr, (s, 0, 0))

                # calculate stats
                lmean, lvar, lstd, lmin, lmax, lshape = dg._print_arr_info(subvol)
                smean += lmean
                svar += lvar
                gmin = min(gmin, lmin)
                gmax = max(gmax, lmax)

                # update progressbar if exist
                self.update_progressbar(s + batch_size, self.vol_size[0])

            gmean = smean / cnt
            gvar = svar / cnt
            gstd = math.sqrt(gvar)

            # update param file
            self.params_dictByChn['mean'] = gmean
            self.params_dictByChn['std'] = gstd
            self.params_dictByChn['min'] = gmin
            self.params_dictByChn['max'] = gmax
            self.params_dictByChn['shape'] = self.vol_size

        else:
            tiffs = dg._read_tiff3d(self.tiffpath, self.vol_size)
            # write to Zarr
            io.zarr.write_subarray(subvol, self.zarr, (s, 0, 0))

            # update param file
            info = dg._print_arr_info(tiffs)
            self.params_dictByChn['mean'] = info[0]
            self.params_dictByChn['std'] = info[2]
            self.params_dictByChn['min'] = info[3]
            self.params_dictByChn['max'] = info[4]
            self.params_dictByChn['shape'] = info[5]

        # update param file
        print("new params: ", self.params_dictByChn)
        #self.params_dict[self.channel].update(self.params_dictByChn)

        self.log("new params: {}".format(self.params_dictByChn))
        with open(self.param_file, 'w') as fp:
            json.dump(self.params_dict, fp, indent=4, separators=(',', ': '), sort_keys=True)
            fp.write('\n')


    def update_progressbar(self, at, total):
        if self.pbar is None:
            return

        val = math.ceil(float(at)/float(total) * 100.)
        self.pbar.setValue(val)
        QtGui.QApplication.processEvents()


    def log(self, msg, flag=PRT.LOG):
        """log wrapper"""
        if self.logwin is None:
            print(msg)
            return

        self.logwin.append(PRT.html(self.__class__.__name__, msg, flag))
        self.logwin.moveCursor(QtGui.QTextCursor.End)
        QtGui.QApplication.processEvents()



if __name__ == "__main__":
    dcp = DataConversionParams()
    dcp.build(sys.argv, "tif2zarr")

    t2z = Tif2Zarr(dcp)
    t2z.convert()
