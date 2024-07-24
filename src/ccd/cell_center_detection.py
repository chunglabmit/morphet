#!/usr/bin/env python
# coding: utf-8

# ## Cell Center Detection using Phathom
# - This notebook is to analyze cell center detection algorithm in Phathom for Microglia/Nuclei dataset
#
"""cell_center_detection.py: run cell-center-detection with Phathom"""
__author__      = "Minyoung Kim"
__license__ = "MIT"
__maintainer__ = "Minyoung Kim"
__email__ = "minykim@mit.edu"
__date__ = "11/01/2018"

import sys
# Python version check - due to Phathom
try:
    assert sys.version_info >= (3,5)
except:
    sys.exit("Python >= 3.5 is required!")

from PyQt5 import QtGui

import os
import matplotlib.pyplot as plt
import numpy as np
import json
from functools import partial
from argparse import Namespace
import multiprocessing
from tqdm import tqdm

# PHATHOM
from phathom import io
from phathom import utils as phaUtil
from phathom.segmentation.segmentation import find_centroids
from phathom.phenotype.celltype import nucleus_probability, nuclei_centers_probability2, calculate_eigvals

# BM
from utils.params import DataGenParams, CellCenterDetectionParams
from utils.data.data_generator import DataGenerator
from analysis.cellCentersAnalyzer import CellCenterAnalyzer
from utils.train.preprocessor import TrainPreprocessor
from utils.util import PRT


class BM_CCD(object):
    def __init__(self, json_params, channel, slice_no=118, damp=2,
                       qtLogWindow=None, qtProgressBarSetup=None,
                       qtProgressBarRun=None):

        self.channel = channel
        self.slice_no = slice_no
        self.damp = damp

        # GUI
        self.logwin = qtLogWindow
        self.pbarSetup = qtProgressBarSetup
        self.pbarRun = qtProgressBarRun

        self.tPrep = TrainPreprocessor()

#        with open(jsonfile) as fp:
#            self.pDict = json.load(fp)
        self.pDict = json_params
        self.pDictByChn = self.pDict[self.channel]

        self.init_sub_params()
        self.dg = None
        self.load_data()


    def _pop(self, key):
        """pop value with key from dictionary if exists

        Params
        ------
        key: name of key
        """

        val = None
        try:
            val = self.pDict.pop(key)
        except KeyError:
            pass

        return val


    def update_params(self, channel=None, slice_no=None, damp=None, xr=None, yr=None, zr=None):
        if channel:
            self.channel = channel
        if slice_no:
            self.slice_no = slice_no
        if damp:
            self.damp = damp
        if ((xr is not None) or (yr is not None) or (zr is not None)):
            self.p.zr = zr
            self.p.yr = yr
            self.p.xr = xr
            self.load_data()


    @staticmethod
    def update_progressbar(pbar, at, total):
        if pbar is None:
            return

        val = np.ceil(float(at)/float(total)) * 100.
        pbar.setValue(val)
        QtGui.QApplication.processEvents()


    def log(self, msg, flag=PRT.LOG):
        """log wrapper"""
        if self.logwin is None:
            print(msg)
            return

        self.logwin.append(PRT.html(self.__class__.__name__, msg, flag))
        self.logwin.moveCursor(QtGui.QTextCursor.End)
        QtGui.QApplication.processEvents()


    def update_cell_centers_file(self, cc_npy):
        """update cell_centers.npy file within DataGenerator Class it owns

        Params
        ------
        cc_npy: new ccd Numpy file to update
        """
        self.dg.p.ann_cell_center = cc_npy
        self.log("Updated cell centers file to [ %s ]"%cc_npy, PRT.STATUS)


    def init_sub_params(self):
        self.p = Namespace(**self.pDict)
        self.pByType = Namespace(**self.pDictByChn)
        #rel_path = self.pByType.rel_path

        # build DataGenParams input
        self.dgParamsByType = ['-dp', self.pByType.tif_rel_path,
                                '-ac', '%s/cell_centers.npy'%self.p.d_root,
                                '-ext', self.p.file_ext]
        save_path = os.path.join(self.p.d_root, self.p.inf_rel_path)
        self.params = ['BM_CCD()', '-dr', self.p.d_root, '-dt', 'test',
                       '-sp', save_path,
                       '-dw', '%d'%self.p.dw, '-dh', '%d'%self.p.dh, '-dd', '%d'%self.p.dd,
                       '-ts', 'PIPELINE', '--debug']


    def load_data(self):
        dgParams = DataGenParams()
        dgParams.build(self.params + self.dgParamsByType, "DGP-CCD")
        # comment due to Pickle error when called from GUI
        #self.dg = DataGenerator(dgParams, self.logwin, self.pbarSetup)
        self.dg = DataGenerator(dgParams, None, None)
        self.dg.load_all()

        self.centers_shield = []
        self.vol_raw, centers_shield = self.dg.grab_sub_zyxplane_and_coords(self.p.zr,
                                                                            self.p.yr,
                                                                            self.p.xr)
        if centers_shield:
            centers_shield = CellCenterAnalyzer.adjust_coordinates(centers_shield,
                                                                   self.p.zr,
                                                                   self.p.yr,
                                                                   self.p.xr)
            self.centers_shield = np.array(centers_shield)


    def get_a_slice(self, z=None, raw=False):
        if z is None:
            z = self.slice_no

        try:
            if z > len(self.vol_raw):
                print("slice number exceeds the total depth!")
                return None
            else:
                if raw:
                    return self.vol_raw[z]
                else:
                    return self.vol[z]
        except AttributeError:
            return self.vol_raw[z]


    def read_full_tif_at(self, z):
        return self.dg._read_tiff_image(self.dg.files[z])


    def preprocess(self, normByPercentile=None, clip=False, norm_min=None, norm_max=None):
        """preprocess volume and save to separate variable to keep the raw
        """
        if self.vol_raw is not None:
            self.vol = self.tPrep.preprocess_all(self.vol_raw.copy(),
                                                 normByPercentile=normByPercentile,
                                                 norm_min=norm_min,
                                                 norm_max=norm_max,
                                                 clip=clip)


    def show_a_slice(self):
        if self.vol is not None:
            a_slice = self.vol[self.slice_no]
            print(a_slice.min(), a_slice.max())

            plt.imshow(a_slice)
            plt.show()


    @staticmethod
    def subtract_and_process(vol1, vol2, tPrep):
        diff = vol1 - vol2
        # re-process
        volume = tPrep.preprocess_all(diff.copy())

        return volume


    def detect_centers(self, sigma=4.0, steepness=200, offset=0.5, threshold=0.9, min_dist=3,
                       mean=1.0, stdev=None, arr_to_subtract=None, raw=False, viz=True,
                       axes=[None, None, None]):
        # NOTE: for raw Microglia
        #   use defaults
        # NOTE: if normalized:
        #   sigma = 1.0
        #   steepness = 0.4
        # NOTE: for HysteresisThresholded Images
        #   sigma = 4.0
        #   steepness = 100
        #   offset = 0.03
        #   threshold=0.5
        #   h=0.01
        print("raw? ", raw)
        if raw:
            vol = self.vol_raw.copy()
        else:
            vol = self.vol.copy()

        if arr_to_subtract is not None:
            volume = BM_CCD.subtract_and_process(vol, arr_to_subtract, self.tPrep)
        else:
            volume = vol

        # TODO: pass by dictionary
        p = {'sigma': sigma, 'steepness': steepness, 'offset': offset, 'threshold': threshold,
             'min_dist': min_dist, 'mean': mean, 'stdev': stdev,
             'arr_to_subtract': arr_to_subtract.shape if arr_to_subtract is not None else None,
             'raw': raw, 'viz': viz}
        self.log("detect_centers(): Params: {}".format(p))

        self.m_probs_c, self.m_probs_i, self.m_probs, \
        self.centroids, self.cs = self.calc_centroids(volume.copy(),
                                                      sigma, steepness, offset,
                                                      threshold, min_dist,
                                                      slice_no=self.slice_no,
                                                      damp=self.damp,
                                                      mean=mean, stdev=stdev,
                                                      viz=viz)
        if viz:
            # visualize
            # - Centers from Phathom
            a_slice = self.get_a_slice()
            self.viz_all(a_slice, self.cs, None, self.cs, reverse=True, s=5)
            # - Centers from Shield-2018
            cs_s = self.get_centroids_in_range(self.centers_shield, self.slice_no, self.damp)
            if len(cs_s):
                self.viz_all(a_slice, cs_s, None, cs_s, s=5, reverse=True)


    @staticmethod
    def viz_all(a_slice, m_cs, n_slice, n_cs, s=50, reverse=False, dpi=90, cmap='viridis', clim=None):
        xidx, yidx = (2, 1) if reverse else (1, 2)

        if clim is None:
            clim = (np.min(a_slice), np.max(a_slice))

        nc = 2 if n_slice is None else 3
        fig, axs = plt.subplots(figsize=(10, 6), nrows=1, ncols=nc, sharex=True, dpi=dpi)
        axs[0].set_title('Microglia')
        axs[0].imshow(a_slice, cmap=cmap, clim=clim)
        # Microglia
        axs[1].set_title('Microglia w/ CCD')
        axs[1].imshow(a_slice, cmap=cmap, clim=clim)
        if len(m_cs):
            axs[1].scatter(m_cs[:, xidx], m_cs[:, yidx], alpha=0.6, s=s, color='red')

        if n_slice is not None:
            # Nuclei
            axs[2].set_title('Nuclei')
            #axs[1].imshow(n_slice, clim=[0, 4500], cmap='Greys')
            axs[2].imshow(n_slice, cmap='Greys')
            if len(n_cs):
                axs[2].scatter(n_cs[:, xidx], n_cs[:, yidx], alpha=0.6, s=s, color='red')
        plt.show()


    @staticmethod
    def get_centroids_in_range(centroids, slice_no, damp):
        """filter centroids with slice_no and damp range

        Parameters
        ----------
        centroids: numpy array
            array of center coordinates
        slice_no: integer
            slice index (pivot)
        damp: integer
            +- range
        """

        crr = np.array([])
        if len(centroids):
            cr = centroids[np.where(centroids[:, 0] >= max(0, slice_no-damp))]
            crr = cr[np.where(cr[:, 0] <= slice_no+damp)]
            print("%d/%d centers sitting on the slice [%d]"%(len(crr), len(centroids),
                                                             slice_no))
        return crr


    @staticmethod
    def viz_hist(slice_):
        plt.figure()
        plt.imshow(slice_)
        plt.show()
        plt.hist(slice_.ravel(), bins=256)
        plt.show()


    @staticmethod
    def viz_probs(probs_slice, name, bins=10, clim=None):
        data = probs_slice.copy()
        fig, axs = plt.subplots(figsize=(8, 4), nrows=1, ncols=2)

        if clim:
            pos1 = axs[0].imshow(data, cmap='viridis', clim=clim)
        else:
            pos1 = axs[0].imshow(data, cmap='viridis')

        plt.colorbar(pos1, ax=axs[0], fraction=0.046, pad=0.04)
        axs[1].hist(data.flatten(), bins=bins)

        fig.suptitle(name)
        fig.tight_layout()
        plt.show()


    @staticmethod
    def _calc_centroids_partial(input_tuple, overlap, sigma, min_intensity, steepness, offset,
                                I0=None, stdev=None, prob_thresh=0.5, min_dist=1,
                                normalize=True, norm_min=None, norm_max=None,
                                normByPercentile=None, clip=False,
                                zarr_subtract=None, prob_output=None):

        subtract = True if zarr_subtract is not None else False

        arr, start_coord, chunks = input_tuple
        ghosted_chunk, start_ghosted, _ = phaUtil.extract_ghosted_chunk(arr, start_coord, chunks, overlap)
        if ghosted_chunk.max() < min_intensity:
            return None


        if subtract:
            ghosted_chunk_subtract, _, _ = phaUtil.extract_ghosted_chunk(zarr_subtract, start_coord, chunks, overlap)


        if normalize:
            tPrep = TrainPreprocessor()
            gc_npy = np.asarray(ghosted_chunk)
            gc_npy_norm = tPrep.preprocess_all(gc_npy,
                                               normByPercentile=normByPercentile,
                                               norm_min=norm_min,
                                               norm_max=norm_max,
                                               clip=clip)
            if subtract:
                sb_npy = np.asarray(ghosted_chunk_subtract)
                sb_npy_norm = tPrep.preprocess_all(sb_npy,
                                                   normByPercentile=normByPercentile,
                                                   norm_min=norm_min,
                                                   norm_max=norm_max,
                                                   clip=clip)
                ghosted_chunk = BM_CCD.subtract_and_process(gc_npy_norm.copy(), sb_npy_norm.copy(), tPrep)
            else:
                ghosted_chunk = gc_npy_norm


        sum_limit = 0.05
        if np.sum(ghosted_chunk) < sum_limit:
            # most likely backgrounds
            return None

        try:
            prob_c, prob_i, prob = nucleus_probability(ghosted_chunk,
                                                       sigma=sigma,
                                                       steepness=steepness,
                                                       offset=offset,
                                                       I0=I0,
                                                       stdev=stdev)
        except np.linalg.LinAlgError:
            # eigenvalue didn't converge
            return None


        if prob_output is not None:
            start_local = start_coord - start_ghosted
            stop_local = np.minimum(start_local + np.asarray(chunks), ghosted_chunk.shape)
            prob_valid = phaUtil.extract_box(prob, start_local, stop_local)
            stop_coord = start_coord + np.asarray(prob_valid.shape)
            phaUtil.insert_box(prob_output, start_coord, stop_coord, prob_valid)

        centers_local = nuclei_centers_probability2(prob, threshold=prob_thresh, min_dist=min_dist)
        print("len(centers_local): ", len(centers_local))

        if centers_local.size == 0:
            return None

        # Filter out any centers detected in ghosted area
        centers_interior = phaUtil.filter_ghosted_points(start_ghosted, start_coord, centers_local, chunks, overlap)

        # change to global coordinates
        centers = centers_interior + start_ghosted
        print("len(centers): ", len(centers))
        return centers


    @staticmethod
    def calc_centroids_parallel(zarr,
                                sigma, steepness, offset, I0, stdev,  # for nucleus_probability
                                prob_thresh, min_dist,          # for nuclei_centers_probability
                                min_intensity,
                                chunk_size, overlap, nb_workers=None,
                                normalize=True, norm_min=None, norm_max=None,
                                normByPercentile=None, clip=False,
                                zarr_subtract=None, prob_output=None,
                                pbar=None):

        """calculate center coordinates of cell in volume
        """
        np.warnings.filterwarnings('ignore')
        f = partial(BM_CCD._calc_centroids_partial,
                    overlap=overlap,
                    sigma=sigma,
                    min_intensity=min_intensity,
                    steepness=steepness,
                    offset=offset,
                    I0=I0,
                    stdev=stdev,
                    prob_thresh=prob_thresh,
                    min_dist=min_dist,
                    normalize=normalize,
                    norm_min=norm_min,
                    norm_max=norm_max,
                    normByPercentile=normByPercentile,
                    clip=clip,
                    zarr_subtract=zarr_subtract,
                    prob_output=prob_output)
        results = BM_CCD._pmap_chunks(f, zarr, chunk_size, nb_workers, use_imap=True, pbar=pbar)

        return results


    @staticmethod
    def _pmap_chunks(f, arr, chunks=None, nb_workers=None, use_imap=False, pbar=None):
        """Maps a function over an array in parallel using chunks
           (grabbed from Phathom.utils)

        The function `f` should take a reference to the array, a starting index, and the chunk size.
        Since each subprocess is handling it's own indexing, any overlapping should be baked into `f`.
        Caution: `arr` may get copied if not using memmap. Use with SharedMemory or Zarr array to avoid copies.

        Parameters
        ----------
        f : callable
            function with signature f(arr, start_coord, chunks). May need to use partial to define other args.
        arr : array-like
            an N-dimensional input array
        chunks : tuple, optional
            the shape of chunks to use. Default tries to access arr.chunks and falls back to arr.shape
        nb_workers : int, optional
            number of parallel processes to apply f with. Default, cpu_count
        use_imap : bool, optional
            whether or not to use imap instead os starmap in order to get an iterator for tqdm.
            Note that this requires input tuple unpacking manually inside of `f`.

        Returns
        -------
        result : list
            list of results for each chunk

        """
        ctx = multiprocessing.get_context("spawn")

        if chunks is None:
            try:
                chunks = arr.chunks
            except AttributeError:
                chunks = arr.shape

        if nb_workers is None:
            nb_workers = int( multiprocessing.cpu_count() / 2 )

        start_coords = phaUtil.chunk_coordinates(arr.shape, chunks)

        args_list = []
        for i, start_coord in enumerate(start_coords):
            args = (arr, start_coord, chunks)
            args_list.append(args)

        t_iter = len(args_list)
        if nb_workers > 1:
            with ctx.Pool(processes=nb_workers) as pool:
                if use_imap:
#                   results = list(tqdm(pool.imap(f, args_list), total=len(args_list)))
                    results = []
                    for idx, s in enumerate(tqdm(pool.imap(f, args_list), total=t_iter, desc="pmap_chunks")):
                        results.append(s)
                        BM_CCD.update_progressbar(pbar, idx+1, t_iter)
                else:
                    results = list(pool.starmap(f, args_list))
        else:
            if use_imap:
                results = list(tqdm(map(f, args_list), total=len(args_list)))
            else:
                results = list(starmap(f, args_list))
        return results



    @staticmethod
    def calc_centroids(volume, sigma, steepness, offset, threshold, min_dist, slice_no,
                       damp=0, mean=1.0, stdev=None, viz=True):
        """calculate center coordinates of cell in volume
        """
        print("sigma: ", sigma, "steepness: ", steepness, "offset: ", offset, "I0: ", mean, "stdev: ", stdev)
        probs_c, probs_i, probs = nucleus_probability(volume, sigma=sigma, steepness=steepness, offset=offset, I0=mean, stdev=stdev)

        if viz:
            BM_CCD.viz_probs(probs_c[slice_no], "curvature")
            BM_CCD.viz_probs(probs_i[slice_no], "intensity")
            BM_CCD.viz_probs(probs[slice_no], "all")

        centers = nuclei_centers_probability2(probs, threshold=threshold, min_dist=min_dist)
        print("len(centers): ", len(centers))
        centers_at_slice = BM_CCD.get_centroids_in_range(np.array(centers), slice_no, damp)

        #return probs, centers, centers_at_slice
        return probs_c, probs_i, probs, centers, centers_at_slice



if __name__ == '__main__':

    ccdParams = CellCenterDetectionParams()
    ccdParams.build(sys.argv, "CCD-Params")
    bccd = BM_CCD(ccdParams.param_file, ccdParams.channel,
                  ccdParams.slice_no, ccdParams.damp)
    bccd.detect_centers()
