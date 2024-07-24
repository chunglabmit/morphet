"""alignment.py: alignent related functionalities"""
__author__      = "Minyoung Kim"
__license__ = "MIT"
__maintainer__ = "Minyoung Kim"
__email__ = "minykim@mit.edu"
__date__ = "10/05/2019"

import os
#import matplotlib
#import matplotlib.pyplot as plt
#from skimage import img_as_float
#from skimage import exposure
#from scipy.ndimage import zoom
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
from glob import glob
from PyQt5 import QtGui
from dask.array.image import imread as dimread
import tifffile
import imagesize
import multiprocessing
from multiprocessing import Pool
import neuroglancer
from argparse import Namespace

# Nuggt
from nuggt.utils.warp import Warper
from nuggt.align import ViewerPair

# internal
from utils.util import PRT
from utils.data.data_generator import DataGenerator
from utils.const import NormalizationType, AtlasConst
from utils.data.preprocessing import BMPreprocessing
#from analysis.alignmentAnalyzer import AlignmentAnalyzer


def process_a_file(fname, scale, xr=None, yr=None):
    img = DataGenerator._read_tiff_image_roi(fname, yr, xr)
    return BMPreprocessing._resize(img, scale)



class MouseAtlas(object):
    def __init__(self, alignment_json, rl_file, ann_tif):
        self.ajson = alignment_json
        self.rl_file = rl_file
        self.ann_tif = ann_tif
        print("MouseAtlas(): ajson: ", self.ajson)
        #print("rl_file: ", self.rl_file)
        #print("ann_tif: ", self.ann_tif)
        self.load()


    def load(self):
        self.df = pd.read_csv(self.rl_file)
        self.anns_mask = tifffile.imread(self.ann_tif)    # (228, 391, 320) zyx
        with open(self.ajson, "r") as fp:
            self.alignment = json.load(fp)


        # SRC: moving, DST: reference
        self.warper = Warper(self.alignment['moving'], self.alignment['reference'])
        self.warper_r = Warper(self.alignment['reference'], self.alignment['moving'])


    def get_af_tif(self):
        ann_tif_file = os.path.basename(self.ann_tif)
        af_tif = ann_tif_file.replace("annotation", "autofluorescence")
        return os.path.join(os.path.dirname(self.ann_tif), af_tif)


    def get_label(self, pt, debug=True):
        w = self.warper(pt) # (z, y, x)
        #w = [int(x)-1 for x in w[0]]
        w = [int(x) for x in w[0]]
        try:
            label_no = int(self.anns_mask[w[0]][w[1]][w[2]])
            row = self.df.loc[self.df['id'] == label_no]
            label_name = row['acronym'].values[0]
            label_full_name = row['name'].values[0]
            if debug and label_no:
                print("pt: ", pt, " --> ", "w: ", w, "label_no: ", label_no, "name: ", label_name)
        except IndexError:
            if debug:
                print("IndexError! pt: {}, w: {}, mask.shape: {}".format(pt, w, self.anns_mask.shape))
            #label_no = -1
            label_no = 0
            label_name = ""
            label_full_name = ""

        if label_no != 0 and debug:
            print("Found mapping, pt: {}, w: {}, mask.shape: {}".format(pt, w, self.anns_mask.shape))

        return label_no, label_name, label_full_name


    def get_parent(self, acronym, depth):
        assert depth <= 10 and depth >= 0

        cur_row = self.df[self.df['acronym'] == acronym]
        cur_depth = int(cur_row['depth'])

        while cur_depth > depth:
            cur_row = self.df[self.df['id'] == int(cur_row['parent_structure_id'])]
            cur_depth = int(cur_row['depth'])

        return str(cur_row['acronym'].values[0]), cur_depth


    def get_parent_labels(self, label_id, return_acronym=False, return_rid=False):
        curr_id = label_id

        labels = []
        while curr_id != -1:
            row = self.df.loc[self.df['id'] == curr_id]
            if row.empty:
                break
            if return_acronym:
                labels.append(row['acronym'].values[0])
            elif return_rid:
                labels.append(row['id'].values[0])
            else:
                labels.append(row['name'].values[0])
            curr_id = row['parent_structure_id'].values[0]

        return labels


class BMAligner(object):
    """Class for handling alignment"""
    def __init__(self, droot, subdir, atlas_root, marker,
                       xidx=2, yidx=1, zidx=0,
                       clip_x=None, clip_y=None, clip_z=None,
                       flip_x=False, flip_y=False, flip_z=False,
                       num_threads=6,
                       qtLogWindow=None, qtProgressBar_r=None, qtProgressBar_a=None):

        self.prep = BMPreprocessing()
        self.warper = None

        self.droot = droot
        self.subdir = subdir
        self.atlas_root = atlas_root
        self.marker = marker
        self.xidx = xidx
        self.yidx = yidx
        self.zidx = zidx
        self.clip_x = clip_x
        self.clip_y = clip_y
        self.clip_z = clip_z
        self.flip_x = flip_x
        self.flip_y = flip_y
        self.flip_z = flip_z
        self.num_threads = num_threads
        self.logwin = qtLogWindow
        self.pbar_r = qtProgressBar_r
        self.pbar_a = qtProgressBar_a


    def print_flip_info(self):
        self.log("self.flip_x: {}".format(self.flip_x))
        self.log("self.flip_y: {}".format(self.flip_y))
        self.log("self.flip_z: {}".format(self.flip_z))
        self.log("self.x_idx: {}".format(self.xidx))
        self.log("self.y_idx: {}".format(self.yidx))
        self.log("self.z_idx: {}".format(self.zidx))


    def get_rescaled_output_name(self):
        output = "%s/%s_downsampled"%(self.droot, self.marker)
        if self.flip_x:
            output += "_flipX"
        if self.flip_y:
            output += "_flipY"
        if self.flip_z:
            output += "_flipZ"
        output += ".tif"

        return output


    def rescale_image(self, atlas_file):
        fpath = os.path.join(self.droot, self.subdir)
        self.log("Rescaling the image stack [%s]"%fpath)
        output = self.get_rescaled_output_name()

        if os.path.isfile(output):
            self.log("Rescale is already done on current settings! (TIF: %s)"%output, PRT.WARNING)
            self.update_progressbar(self.pbar_r, 1, 2)   # half-complete!
            return output

        files = sorted(glob("%s/*.tif*"%fpath))
        self.log("len(files): %d"%len(files))

        if len(files) == 0:
            self.log("No files in the path [ %s ]!"%fpath, PRT.ERROR)
            return False

        apath = os.path.join(self.atlas_root, atlas_file)
        try:
            atlas_file = dimread(apath)[0]
        except FileNotFoundError:
            self.log("Could not find the atlas file [ %s ]!"%apath, PRT.ERROR)
            return False

        iw, ih = imagesize.get(files[0])
        if self.clip_x:
            iw = self.clip_x[1] - self.clip_x[0]
        if self.clip_y:
            ih = self.clip_y[1] - self.clip_y[0]
        z0, z1 = (0, len(files)) if self.clip_z is None else self.clip_z

        atlas_shape = atlas_file.shape
        scale_x = atlas_shape[self.xidx] / iw
        scale_y = atlas_shape[self.yidx] / ih
        self.log("scale_x, scale_y: {}, {}".format(scale_x, scale_y))

        stack = []
        with Pool(self.num_threads) as pool:
            futures = []
            for z in np.linspace(z0, z1 - 1, atlas_shape[self.zidx]):
                if np.floor(z) == np.ceil(z):
                    future = pool.apply_async(
                                    process_a_file,
                                (files[int(z)], (scale_y, scale_x), self.clip_x, self.clip_y)
                             )
                    futures.append((future, future, .5))
                else:
                    future1 = pool.apply_async(
                                process_a_file,
                                (files[int(np.floor(z))], (scale_y, scale_x), self.clip_x, self.clip_y)
                              )
                    future2 = pool.apply_async(
                                process_a_file,
                                (files[int(np.ceil(z))], (scale_y, scale_x), self.clip_x, self.clip_y)
                              )
                    futures.append((future1, future2, 1 - (z - np.floor(z))))

            cnt = 0
            for future1, future2, frac in tqdm(futures):
                p1 = future1.get()
                p2 = future2.get()
                plane = p1.astype(np.float32) * frac +\
                        p2.astype(np.float32) * (1 - frac)
                stack.append(plane.astype(p1.dtype))

                self.update_progressbar(self.pbar_r, cnt, len(futures))
                cnt += 1

        img = np.array(stack).transpose(self.zidx, self.yidx, self.xidx)
        if self.flip_x:
            img = img[..., ::-1]
        if self.flip_y:
            img = img[:, ::-1]
        if self.flip_z:
            img = img[::-1]
        tifffile.imsave(output, img)
        self.log("Rescaled TIF is saved to %s"%output, PRT.STATUS)

        return output


    def rescale_coords_to_original(self, aligned_json, moving_tif):
        self.update_progressbar(self.pbar_a, 0, 1)
        path_s = aligned_json.split("/")
        rescaled_json = os.path.join("%s"%'/'.join(path_s[:-1]), "rescaled_%s"%path_s[-1])
        if os.path.isfile(rescaled_json):
            self.log("Scaling back to original is already done!", PRT.WARNING)
            self.update_progressbar(self.pbar_a, 1, 1)

        else:
            with open(aligned_json) as fp:
                aj = json.load(fp)

            fpath = os.path.join(self.droot, self.subdir)
            files = sorted(glob("%s/*"%fpath))
            self.log("len(files): %d"%len(files))
            if len(files) == 0:
                self.log("No files in the path [ %s ]!"%fpath, PRT.ERROR)
                return None

            # get moving_tif shape
            alignment_shape = tifffile.imread(moving_tif).shape

            # get z-stack shape (original TIF files)
            iw, ih = imagesize.get(files[0])
            x0, x1 = (0, iw) if self.clip_x is None else self.clip_x
            y0, y1 = (0, ih) if self.clip_y is None else self.clip_y
            z0, z1 = (0, len(files)) if self.clip_z is None else self.clip_z
            stack_shape = [z1 - z0, y1 - y0, x1 - x0]

            # get points, transform, and dump to new json
            points = np.array(aj["moving"])
            xform = self.transform(points, alignment_shape, stack_shape)
            xform[:, 0] += z0
            xform[:, 1] += y0
            xform[:, 2] += x0
            aj["moving"] = xform.tolist()
            with open(rescaled_json, "w") as fp:
                json.dump(aj, fp)
            self.log("Dumped JSON [ %s ]"%rescaled_json, PRT.STATUS2)

            # done!
            self.update_progressbar(self.pbar_a, 1, 1)

        return rescaled_json


    def transform(self, points, alignment_shape, stack_shape):
        """Transform the points from the alignment frame to the stack frame"""
        ssz, ssy, ssx = stack_shape
        points = np.atleast_2d(points)

        zc = points[:, self.zidx] * ssz / alignment_shape[self.zidx]
        if self.flip_z:
            zc = ssz - 1 - zc
        yc = points[:, self.yidx] * ssy / alignment_shape[self.yidx]
        if self.flip_y:
            yc = ssy - 1 - yc
        xc = points[:, self.xidx] * ssx / alignment_shape[self.xidx]
        if self.flip_x:
            xc = ssx - 1 - xc

        return np.column_stack((zc, yc, xc))


    def launch_viewer(self, moving, reference, annotation, points,
                      mv_vox=[1.0, 1.0, 1.0],
                      ref_vox=[1.0, 1.0, 1.0],
                      scs=None):
        """launch neuroglancer with alignment results
        :param moving: moving tif file
        :param reference: reference tif file
        :param annotation: annotation ATLAS tif file
        :param points: rescaled points (JSON file)
        :mv_vox: moving voxel size
        :ref_vox: reference voxel size
        :scs: static-content-source address
        """
        args_dict = {}
        args_dict['reference-image'] = reference
        args_dict['segmentation'] = annotation
        args_dict['points'] = points
        if scs is not None:
            neuroglancer.set_static_content_source(url=scs)
            args_dict['static-content-source'] = scs
        args_dict['reference-voxel-size'] = ref_vox
        args = Namespace(**args_dict)

        ref_img = tifffile.imread(reference).astype(np.float32)
        moving_img = tifffile.imread(moving).astype(np.float32)
        ann_img = tifffile.imread(annotation).astype(np.uint32)

        mvoxel = [x*1000. for x in mv_vox]
        rvoxel = [x*1000. for x in ref_vox]
        vp = ViewerPair(ref_img, moving_img, ann_img, points, rvoxel, mvoxel)
#                        n_workers=multiprocessing.cpu_count())

        ref_viewer = repr(vp.reference_viewer)
        mv_viewer = repr(vp.moving_viewer)
        self.log("Reference Viewer: %s"%ref_viewer)
        self.log("Moving Viewer: %s"%mv_viewer)

        return ref_viewer, mv_viewer


    def log(self, msg, flag=PRT.LOG):
        """log wrapper"""
        if self.logwin is None:
            print(msg)
            return

        self.logwin.append(PRT.html(self.__class__.__name__, msg, flag))
        self.logwin.moveCursor(QtGui.QTextCursor.End)
        QtGui.QApplication.processEvents()


    def update_progressbar(self, pbar, at, total):
        if pbar is None:
            return

        val = np.ceil(float(at)/float(total) * 100.)
        pbar.setValue(val)
        QtGui.QApplication.processEvents()
