"""data_generator.py: Class to generate training data from raw tiff images
    - moved/modified from InitialDataAnalysisAndGeneration.ipynb
"""
__author__      = "Minyoung Kim"
__license__ = "MIT"
__maintainer__ = "Minyoung Kim"
__email__ = "minykim@mit.edu"
__date__ = "09/20/2018"


import os
import math
import multiprocessing
from multiprocessing import Pool
import cv2
import numpy as np
import glob
import json
from PIL import Image
import tifffile
import pylab
from matplotlib import pyplot as plt
import skimage
from skimage import io
from random import randint
from tqdm import tqdm
from dask.array.image import imread as dimread
from argparse import Namespace

from PyQt5 import QtGui, QtWidgets
from phathom import io

# internal
import utils.util as bmUtil
from utils.util import PRT


class DataGenerator(object):
    prt = bmUtil.PRT()

    def __init__(self, params, qtLogWindow=None, qtProgressBar=None):
        self.p = params
        self.pdict = None
        self.pByType = self.load_params_from_json()

        # GUI
        self.logwin = qtLogWindow
        self.pbar = qtProgressBar

        self.cc_all = None
        self.full_images = {}
        self.max_num_full_images = 40

        self.zarrs = None


    def load_params_from_json(self, paramf='params.json'):
        paramf = glob.glob("%s/%s"%(self.p.d_root, paramf))[0]
        with open(paramf, "r") as fp:
            self.pdict = json.load(fp)
        pm = self.pdict[self.pdict['marker']]
        print("pm: ", pm)
        p_byType = Namespace(**pm)


        # get dimension
        self.p.data_width = self.pdict['dw']
        self.p.data_height = self.pdict['dh']
        self.p.data_depth =  self.pdict['dd']
        self.p.file_ext = self.pdict['file_ext']

        # load cell center file from json
        if 'cc_npy' in pm.keys():
            self.p.ann_cell_center = os.path.join(self.p.d_root, pm['cc_npy'])

        return p_byType


    def retrieve_files(self):
        """load data from files"""

        print("multi channel?", self.p.multi_channel)
        # retrieve files from data path
        if self.p.multi_channel is None:
            dp = self.p.data_path
            assert(dp is not None)
            self.files = sorted(glob.glob('{}/*.{}'.format(dp, self.p.file_ext)))
            if not len(self.files):
                self.files = sorted(glob.glob('{}/*/*.{}'.format(dp, self.p.file_ext)))

            self.num_files = len(self.files)
#            assert self.num_files > 0
            if self.p.debug:
                print("self.files[0]: ", self.files[0])

            # load ZARR
            if self.pByType.zarr_rel_path:
                #dp_zarr = os.path.join(self.p.data_path, "%s_zarr"%self.p.file_ext)
                dp_zarr = os.path.join(self.p.d_root, self.pByType.zarr_rel_path)
                zar = io.zarr.open(dp_zarr)
                self.zarrs = zar

        else:
            dp = []
            self.files = {}
            self.zarrs = {}
            for item in self.p.multi_channel:
                pByType = self.pdict[item]
                dp_item = os.path.join(self.p.data_path, pByType['tif_rel_path'])
                print("dp_item: ", dp_item)
                #dp_item = os.path.join(self.p.data_path, item)
                dp.append(dp_item)

                files = sorted(glob.glob('{}/*.{}'.format(dp_item, self.p.file_ext)))
                self.files[item] = files

                # load ZARR
                dp_item_zarr = dp_item + "_zarr"
                zar = io.zarr.open(dp_item_zarr)
                self.zarrs[item] = zar

                if self.p.debug:
                    print("self.files[{}][0]: ", self.files[item][0])

            self.num_files = len(self.files[self.p.multi_channel[0]])
            assert self.num_files > 0

        self.log("{} files are retrieved from {}".format(self.num_files, dp), PRT.STATUS)

    @staticmethod
    def _zero_pad_3d(vol, center, target_size):
        """zero pad to make the volume to target_size
           coordinate: zyx-order
        """
        z, y, x = center
        td, th, tw = target_size
        vd, vh, vw = vol.shape

        pad_z = float(td - vd) / 2.
        pad_y = float(th - vh) / 2.
        pad_x = float(tw - vw) / 2.

        padded = np.pad(vol,
                        ((int(pad_z), int(np.ceil(pad_z))),
                         (int(pad_y), int(np.ceil(pad_y))),
                         (int(pad_x), int(np.ceil(pad_x)))
                        ),
                        mode='constant')
        assert padded.shape == target_size

        return padded

    @staticmethod
    def _zero_pad_2d(img, target_size, align='center'):
        """zero pad to make the image to target_size (image will be centered)
           coordinate: zyx-order

           align: 'center', 'left', or 'right'

        """
        th, tw = target_size
        ih, iw = img.shape

        if align == 'center':
            pad_y = float(th - ih) / 2.
            pad_x = float(tw - iw) / 2.
            padded = np.pad(img,
                            ((int(pad_y), int(np.ceil(pad_y))),
                             (int(pad_x), int(np.ceil(pad_x)))
                            ),
                            mode='constant')
        else:
            pad_y = float(th - ih) / 2.
            pad_x = tw - iw
            if align == 'left':
                padded = np.pad(img,
                                ((int(pad_y), int(np.ceil(pad_y))),
                                 (0, pad_x)
                                ),
                                mode='constant')
            else: # right
                padded = np.pad(img,
                                ((int(pad_y), int(np.ceil(pad_y))),
                                 (pad_x, 0)
                                ),
                                mode='constant')

        assert padded.shape == target_size

        return padded


    @staticmethod
    def _center_crop_image(img, y, hh, x, hw):
        """center crop image with center coordinates and half-sizes"""

        res_h, res_w = img.shape
        ypad = None
        if (y - hh) < 0:
            #print("out of y-axis border(top), y: {}, hh: {}".format(y, hh))
            ypad = hh - y

        if (y + hh) > res_h:
            #print("out of y-axis border(bottom), y: {}, hh: {}".format(y, hh))
            if ypad is None:
                ypad = y + hh - res_h
            else:
                ypad += y + hh - res_h

        xpad = None
        if (x - hw) < 0:
            #print("out of x-axis border(left), x: {}, hw: {}".format(x, hw))
            xpad = hw - x

        if (x + hw) > res_w:
            #print("out of x-axis border(right), x: {}, hw: {}".format(x, hw))
            if xpad is None:
               xpad = x + hw - res_w
            else:
               xpad += x + hw - res_w

        img_crop = img[max(0, y - hh):min(res_h, y + hh), max(0, x - hw):min(res_w, x + hw)].copy()

        # fill with black
        if ypad is not None:
            ypad_half = float(ypad) / 2.0
            img_crop = cv2.copyMakeBorder(img_crop, int(ypad_half), int(math.ceil(ypad_half)),
                                                    0, 0,
                                                    cv2.BORDER_CONSTANT, value=[0, 0, 0])
        if xpad is not None:
            xpad_half = float(xpad) / 2.0
            img_crop = cv2.copyMakeBorder(img_crop, 0, 0,
                                                    int(xpad_half), int(math.ceil(xpad_half)),
                                                    cv2.BORDER_CONSTANT, value=[0, 0, 0])
        assert img_crop.shape == (2 * hh, 2 * hw)

        return img_crop


    def grab_patchcubes(self, yxs, z, hw, hh, hd):
        """return cropped patch cubes at z, yxs, with size of (2 * hd, 2 * hh, 2 * hw)

        Parameters
        ----------
        yxs: list
            list of tuples of (y, x) coordinates
        z: integer
            depth
        hw: integer
            half-width
        hh: integer
            half-height
        hd: integer
            half-depth
        """

        patchcubes = []

        # read tif image if necessary
        for i in range(z - hd, z + hd, 1):
            if i < 0 or i >= len(self.files):
                continue
            if i not in self.full_images.keys():
                # not loaded yet, load
                if self.p.debug:
                    print("%s not loaded yet, so load..."%self.files[i])
                self.full_images[i] = self._read_tiff_image(self.files[i])

        # grab patchcubes

        for yx in yxs:
            y, x = yx
            slices = []
            for i in range(z - hd, z + hd, 1):
                if i < 0 or i >= len(self.files):
                    slices.append(np.zeros((2 * hh, 2 * hw)))
                    continue
                a_slice = self._center_crop_image(self.full_images[i], y, hh, x, hw)
                slices.append(a_slice)

            assert len(slices) == 2*hd
            patchcubes.append(np.array(slices, dtype=np.float32))

        # cleanup
        self.cleanup_full_images()

        return patchcubes


    def grab_a_patchcube(self, x, y, z, hw, hh, hd, use_zarr=True):
        """return cropped patch cube at (z, y, x) with size of (2 * hd, 2 * hh, 2 * hw)"""

        z_center = int(np.round(z))

        MULTI_CHN = isinstance(self.files, dict)
        if MULTI_CHN:
            keys = sorted(self.files.keys())
            images = {}
            for key in keys:
                images[key] = []
        else:
            num_files = len(self.files)
            images = []

        if use_zarr:
            zr = [max(0, z - hd), min(self.p.data_depth, z + hd)]
            yr = [max(0, y - hh), min(self.p.data_height, y + hh)]
            xr = [max(0, x - hw), min(self.p.data_width, x + hw)]
            if MULTI_CHN:
                for key in keys:
                    zvol = self.zarrs[key][zr[0]:zr[1], yr[0]:yr[1], xr[0]:xr[1]]
                    zvol = self._zero_pad_3d(zvol, (z, y, x), (hd*2, hh*2, hw*2)) # zyx-order
                    images[key].append(zvol)
            else:
                zvol = self.zarrs[zr[0]:zr[1], yr[0]:yr[1], xr[0]:xr[1]]
                zvol = self._zero_pad_3d(zvol, (z, y, x), (hd*2, hh*2, hw*2)) # zyx-order
                images.append(zvol)

        else:       # use tiff
            for i in range(z - hd, z + hd, 1):
                if i < 0 or i >= self.num_files:
                    # add an empty XY-plane
                    if MULTI_CHN:
                        for key in keys:
                            images[key].append(np.zeros((2 * hh, 2 * hw)))
                    else:
                        images.append(np.zeros((2 * hh, 2 * hw)))
                    continue

                if MULTI_CHN:
                    for key in keys:
                        img = self._read_tiff_image(self.files[key][i])
                        img_crop = self._center_crop_image(img, y, hh, x, hw)
                        images[key].append(img_crop)
                else:
                    img = self._read_tiff_image(self.files[i])
                    img_crop = self._center_crop_image(img, y, hh, x, hw)
                    images.append(img_crop)

        # format as numpy
        if MULTI_CHN:
            images_all = []
            for key in keys:
                images_all.append(images[key])

            images_all = np.array(images_all, dtype=np.float32)
            return images_all

        else:
            images = np.array(images, dtype=np.float32)
            return images


    def cleanup_full_images(self):
        fi_size = len(self.full_images.keys())

        if fi_size > self.max_num_full_images:
            num_to_del = fi_size - self.max_num_full_images
            keys_to_del = sorted(self.full_images.keys())[:num_to_del]
            for k in keys_to_del:
                if self.p.debug:
                    print("removing %d-th image"%k)
                del self.full_images[k]


    def grab_yxplane_with_depth(self, zrange, returnFiles=False):
        """return z-stack of XY plane with zrange"""

        zs, ze = zrange
        images = []
        files = []
        for i in range(zs, ze, 1):
            files.append(self.files[i])
            img = self._read_tiff_image(self.files[i])
            images.append(img)

        if returnFiles:
            return np.array(images, dtype=np.float32), files

        return np.array(images, dtype=np.float32)


    def grab_sub_zyxplane_new(self, zr=None, yr=None, xr=None):
        """return sub XYZ plane with ranges

        Parameter
        ------------
        zr: a tuple of zmin and zmax
        yr: a tuple of ymin and ymax
        xr: a tuple of xmin and xmax
        """

        print("zr, yr, xr: ", zr, yr, xr)
        zs, ze = zr
        ys, ye = yr
        xs, xe = xr
        print("self.files[0]: ", self.files[0])
        fr = '/'.join(self.files[0].split('/')[:-1])
        img = dimread('%s/*.%s'%(fr, self.p.file_ext))

        print("img.shape: ", img.shape)
        roi = img[zs:ze, ys:ye, xs:xe]  # crop if out of bound in x, y-axis
#        roi = img[zs:ze, :, :]  # crop if out of bound in x, y-axis
        print("roi.shape: ", roi.shape)
        return roi.compute()




    def grab_sub_zyxplane(self, zr=None, yr=None, xr=None):
        """return sub XYZ plane with ranges

        Parameter
        ------------
        zr: a tuple of zmin and zmax
        yr: a tuple of ymin and ymax
        xr: a tuple of xmin and xmax
        """
        MULTI_CHN = isinstance(self.files, dict)
        if MULTI_CHN:
            keys = sorted(self.files.keys())

        if zr:
            zs, ze = zr
        else:
            if MULTI_CHN:
                zs, ze = [0, len(self.files[keys[0]])]
            else:
                zs, ze = [0, len(self.files)]
        assert zs >= 0

        height = self.p.data_height if yr is None else yr[1] - yr[0]
        width = self.p.data_width if xr is None else xr[1] - xr[0]

        if MULTI_CHN:
            images = {}
            for i in sorted(keys):
                if "RAW" in i:
                    images[i] = np.zeros((ze - zs, height*2, width*2), dtype=np.float32)
                else:
                    images[i] = np.zeros((ze - zs, height, width), dtype=np.float32)
        else:
            if "RAW" in self.files[0]:
                images = np.zeros((ze - zs, height*2, width*2), dtype=np.float32)
            else:
                images = np.zeros((ze - zs, height, width), dtype=np.float32)

        for i in tqdm(range(zs, ze, 1), desc="reading file"):
            if MULTI_CHN:
                for key in keys:
                    if "RAW" in key:
                        # TODO: make an argument
                        jump = 1250
                        img = self._read_tiff_image_roi(self.files[key][i-jump], (yr[0]*2, yr[1]*2), (xr[0]*2, xr[1]*2))
                        images[key][i-zs] = img
                    else:
                        img = self._read_tiff_image_roi(self.files[key][i], yr, xr)
                        images[key][i-zs] = img
            else:
                if "RAW" in self.files[0]:
                    # TODO: make an argument
                    jump = 1250
                    img = self._read_tiff_image_roi(self.files[i-jump], (yr[0]*2, yr[1]*2), (xr[0]*2, xr[1]*2))
                else:
                    img = self._read_tiff_image_roi(self.files[i], yr, xr)
                images[i-zs] = img


            # GUI if exist
            self.update_progressbar(i - zs + 1, ze - zs + 1)

        return images



    @staticmethod
    def _show_image(img):
        pylab.rcParams['figure.figsize'] = (12, 12)
        pylab.imshow(img, cmap='gray')
        pylab.show()


    @staticmethod
    def _read_tiff_image_roi(f, yr, xr):
        """read a tiff image from file f, then return only ROI"""

        if yr:
            ys, ye = yr
            assert ys >= 0
        if xr:
            xs, xe = xr
            assert xs >= 0

        img = DataGenerator._read_tiff_image(f)

        if yr:
            img = img[ys:ye, :]
        if xr:
            img = img[:, xs:xe]

        return img


    @staticmethod
    def _read_tiff_image(f):
        """read a tiff image from file f"""
        image = cv2.imread(f, -1)
        #image = dimread(f)

        return image


    @staticmethod
    def _read_tiff3d(path, volume_size, z_start=0, stack_size=None, ext='tif'):
        """read multiple tiffs (z-stack) and return volumetric numpy array"""

        assert(path is not None)

        files = sorted(glob.glob('%s/*.%s'%(path, ext)))
        d, h, w = volume_size

        zs = z_start
        ze = d if stack_size is None else zs + stack_size

        images = []
        for i in range(zs, ze):
            try:
                img = cv2.imread(files[i], -1)[:h, :w]  # crop if out of bound in x, y-axis
                #img = dimread(files[i], -1)[:h, :w]  # crop if out of bound in x, y-axis
                images.append(img)
            except IndexError e:
                print("zs: %d, ze: %d, i: %d"%(zs, ze, i))

        images = np.array(images, dtype=np.float32)
        return images


    @staticmethod
    def _print_arr_info(arr):
        mean_ = np.mean(arr)
        min_ = np.min(arr)
        max_ = np.max(arr)
        #std_ = np.std(arr, dtype=np.float64)
        var_ = np.var(arr)
        std_ = np.std(arr)
        shape_ = arr.shape

        print("mean: {}, var: {}, std: {}, min: {}, max: {}, shape: {}".format(mean_, var_, std_, min_, max_, shape_))
        return (float(mean_), float(var_), float(std_), int(min_), int(max_), shape_)


    @staticmethod
    def scale_linear_bycolumn(rawpoints, high=255.0, low=0.0):
        mins = np.min(rawpoints, axis=0)
        maxs = np.max(rawpoints, axis=0)
        rng = maxs - mins
        return high - (((high - low) * (maxs - rawpoints)) / rng)


    def load_annotations(self, shuffle=True):
        """load annotations from .npy, obtaind by cell detector"""
        self.pos = np.array(json.load(open(self.p.ann_pos))) if self.p.ann_pos else []
        self.neg = np.array(json.load(open(self.p.ann_neg))) if self.p.ann_neg else []
        if self.p.ann_cell_center:
            self.cc_all = np.load(self.p.ann_cell_center)    # (z, y, x) order
            if shuffle:
                np.random.shuffle(self.cc_all)
            self.prt.p("Total {} positive / {} negative cell centers available from {} ({} samples)".format(len(self.pos), \
                                                                                    len(self.neg), \
                                                                                    self.p.ann_cell_center, \
                                                                                    len(self.cc_all)),
                                                                                    self.prt.STATUS)
            self.prt.p("Availablity: %.3f%%"%(float(len(self.pos)+len(self.neg)) / len(self.cc_all)), self.prt.LOG)
        else:
            self.prt.p("No annotation file is provided...do nothing.", self.prt.LOG)


    def get_all_centroids(self):
        return self.cc_all


    def load_all(self):
        """load list of files under the directory (set by Params()),
           and annotations from annotation file (set by Params())
        """

        self.retrieve_files()
        self.load_annotations()


    def get_centerCrop_coordinate(self, coord, zxyorder=False):
        coord_int = self._get_int_coordinate(coord)
        if zxyorder:
            z, x, y = coord_int
        else:
            z, y, x = coord_int

        z_start, z_end = (z - self.p.csz_quart, z + self.p.csz_quart)
        y_start, y_end = (y - self.p.csz_half, y + self.p.csz_half)
        x_start, x_end = (x - self.p.csz_half, x + self.p.csz_half)

        return [(z_start, z_end), (y_start, y_end), (x_start, x_end)]


    def get_a_sample(self, samples, show=False, zxyorder=False, idx=None):
        """TODO: remove while loop"""

        a_sample = None
        while a_sample is None:
            if idx is None:
                idx = randint(0, len(samples))
            a_sample = samples[idx]

            zse, yse, xse = self.get_centerCrop_coordinate(a_sample, zxyorder)
            z_start, z_end = zse
            y_start, y_end = yse
            x_start, x_end = xse

            if idx is None:
                #TODO: REMOVE the condition above
                if (z_start <= 0) or (x_start <= 0) or (y_start <= 0):
                    a_sample = None
                    continue
                if (z_end >= self.p.data_depth) or (x_end >= self.p.data_width) or (y_end >= self.p.data_height):
                    a_sample = None
                    continue

        a_sample_int = self._get_int_coordinate(a_sample)
        if zxyorder:
            z, x, y = a_sample_int
        else:
            z, y, x = a_sample_int

        cube = self.grab_a_patchcube(x, y, z, self.p.csz_half, self.p.csz_half, self.p.csz_quart)
        if show and cube is not None:
            print("min: ", np.min(cube), "max: : ", np.max(cube))
            plt.imshow(cube[self.p.csz_quart], cmap='gray')
            plt.show()

        return (z, y, x), cube


    def get_samples_from_cc(self, num_samples, cc_list=None, start_idx=None):
        """get [num_samples] samples from all cell_centers randomly"""
        sample_map = {}
        for i in range(num_samples):
            if cc_list is None:
                if start_idx is None:
                    crd, a_sample = self.get_a_sample(self.cc_all)
                else:
                    crd, a_sample = self.get_a_sample(self.cc_all, idx=start_idx+i)
            else:
                assert start_idx is not None
                crd, a_sample = self.get_a_sample(cc_list, idx=start_idx+i)

            if a_sample is None:
                print("Sample is NONE, Coordinate must be wrong -> ", crd)
                continue

            if crd in sample_map.keys():
                # already there
                continue
            sample_map[crd] = a_sample

        return sample_map


    def save_data_to_file(self, fname, data, gidx=None):
        num_files_per_dir = 10000
        if gidx is not None:
            # create subdirectory
            subdir="%06d"%(gidx / num_files_per_dir)
            bmUtil.CHECK_DIR(os.path.join(self.p.save_path, subdir))
            np.save(os.path.join(self.p.save_path, subdir, fname), data)
        else:
            np.save(os.path.join(self.p.save_path, fname), data)


    def generate_data_parallel(self, num_samples, ncpu=None, batch_size=10, cc_list=None):
        nworkers = ncpu if ncpu is not None else multiprocessing.cpu_count()
        print("nworkers: ", nworkers)

        # TODO


    def build_zcrd_map(self, cc_list):
        zcrd_map = {}
        for cc in cc_list:
            z = cc[0]
            if z not in zcrd_map.keys():
                zcrd_map[z] = [[cc[1], cc[2]]]
            else:
                zcrd_map[z].append([cc[1], cc[2]])

        return zcrd_map


    def generate_data_sequential(self, cc_list=None):
        """generate_data in z-stack order, to minimize file IO on tif images
        This function can be used where `cc_list` is passed, and can be loaded sequentially.
        Note that generate_data() should be used if random sampling is required, such as when
        generating training data

        Parameters
        ----------
        cc_list: list
            list of centroids (in zyx order)
        """

        zcrd_map = self.build_zcrd_map(cc_list)
        keys = sorted(zcrd_map.keys())
        gidx = 0
        for key in tqdm(keys, desc="[{}] Data Generation".format(os.getpid())):
            crds = zcrd_map[key]
            patchcubes = self.grab_patchcubes(crds, key, self.p.csz_half, self.p.csz_half, self.p.csz_quart)

            for pi, pc in enumerate(patchcubes):
                fname = "cell_%06d_zyx-%d-%d-%d.npy"%(gidx, key, crds[pi][0], crds[pi][1])
                if self.p.debug:
                    self.prt.p("Saving data to %s.."%fname, self.prt.STATUS)
                self.save_data_to_file(fname, pc, gidx)
                gidx += 1



    def generate_data(self, num_samples, batch_size=10, cc_list=None, jump_to=0):
        n_batch = int(np.ceil(float(num_samples) / float(batch_size)))
        print("batch_size: {}, n_batch: {}, total: {}".format(batch_size, n_batch, num_samples))

        gidx = 0
        for idx in tqdm(range(n_batch), desc="[{}] Data Generation".format(os.getpid())):
            start_idx = idx * batch_size
            if idx == n_batch - 1:
                print("last batch!")
                batch_size = num_samples - start_idx
                print("idx: ", idx, "batch_size: ", batch_size, "start_idx: ", start_idx)

            sample_map = self.get_samples_from_cc(batch_size, cc_list, start_idx + jump_to)
            for key in sample_map.keys():
                fname = "cell_%06d_zyx-%d-%d-%d.npy"%(gidx, key[0], key[1], key[2])
                self.prt.p("Saving data to %s.."%fname, self.prt.STATUS)
                self.save_data_to_file(fname, sample_map[key], gidx)
                gidx += 1


    def retrieve_zyx_from_filename(self, fname):
        """file format sample: /xxx/xxx/xxx/cell_000168_zyx-1077-5565-3398.npy
        """
        s = fname.split('/')[-1].split('.')[0].split('-')
        z, y, x = [int(x) for x in s[1:]]

        return z, y, x


    def grab_sub_zyxplane_and_coords(self, zr=None, yr=None, xr=None, z_damp=None):
        """
        grab sub-volume along with center coordinates in ranges (zr, yr, xr)

        Parameters
        ----------
        zr: (zmin, zmax); if None, return whole range
        yr: (ymin, ymax); if None, return whole range
        xr: (xmin, ymin); if None, return whole range
        """

        vol = self.grab_sub_zyxplane(zr, yr, xr)
        centers = self.get_coords_in_range(zr, yr, xr, z_damp=z_damp)

        return vol, centers


    def get_num_centers(self, cc_list=None):
        all = self.cc_all if cc_list is None else cc_list
        return len(all)


    def get_coords_in_range(self, zr=None, yr=None, xr=None, cc_list=None, z_damp=None):
        """return set of coordinates in ranges in zyx plane

        Parameter
        ------------
        zr: a tuple of zmin and zmax
        yr: a tuple of ymin and ymax
        xr: a tuple of xmin and xmax
        """

        filtered = self.cc_all if cc_list is None else cc_list
        if filtered is None:
            return filtered # return None if cell centers are not provided

        total_num = len(filtered)
        if zr:
            if z_damp is not None:
                z0, z1 = zr
                z0 = max(0, z0 - z_damp)
                z1 = min(z1 + z_damp, self.pByType.shape[0])
                zr = (z0, z1)
            filtered = self._return_coords_filtered_by_range(filtered, zr, axis=0)
        if yr:
            filtered = self._return_coords_filtered_by_range(filtered, yr, axis=1)
        if xr:
            filtered = self._return_coords_filtered_by_range(filtered, xr, axis=2)

        print("%d / %d selected"%(len(filtered), total_num))
        for i in range(len(filtered)):
            coord = filtered[i]
            filtered[i] = DataGenerator._get_int_coordinate(coord)

        return filtered


    @staticmethod
    def _get_int_coordinate(coord):
        return [int(np.round(x)) for x in coord]


    @staticmethod
    def _return_coords_filtered_by_range(coords, r, axis=0):
        """filter (z, y, x) coordinates with range in axis provided (default=0, z-axis)

        Parameter
        ------------
        coords: original set of coords (z, y, x)
        r: [start, end] range
        axis: axis to filter
        """
        rs, re = r
        assert rs >= 0

        coords_sorted = sorted(coords, key=lambda x: x[axis])
        coords_filtered = [ x for x in coords_sorted if x[axis] >= rs and x[axis] < re]

        return coords_filtered


    def update_progressbar(self, at, total):
        if self.pbar is None:
            return

        val = math.ceil(float(at)/float(total) * 100.)
        self.pbar.setValue(val)
        QtWidgets.QApplication.processEvents()


    def log(self, msg, flag=PRT.LOG):
        """log wrapper"""
        if self.logwin is None:
            print(msg)
            return

        self.logwin.append(PRT.html(self.__class__.__name__, msg, flag))
        self.logwin.moveCursor(QtGui.QTextCursor.End)
        QtWidgets.QApplication.processEvents()
