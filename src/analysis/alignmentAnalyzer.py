"""alignmentAnalyzer.py"""
__author__      = "Minyoung Kim"
__license__ = "MIT"
__maintainer__ = "Minyoung Kim"
__email__ = "minykim@mit.edu"
__date__ = "04/15/2024"


import os
import cv2
import numpy as np
import json
import tifffile
from tqdm import tqdm
from itertools import combinations
from scipy.ndimage import map_coordinates
import pandas as pd
import shutil

# multiprocessing
from functools import partial
from itertools import repeat
from multiprocessing import Pool

import matplotlib.pyplot as plt
from matplotlib import transforms
# Generate Colormaps
from matplotlib import cm
from matplotlib.colors import ListedColormap,LinearSegmentedColormap

# internal
from align.alignment import MouseAtlas
import utils.util as bmUtil
from utils.const import AtlasConst as AC
from utils.const import MetaData as MD
from utils.const import ParamConst


# define top and bottom colormaps
data = np.random.random([100, 100]) * 10
# create colormaps
N = 256
magenta = np.ones((N, 4))
magenta[:, 0] = np.linspace(255/N, 0, N) # R = 255
magenta[:, 1] = np.linspace(0, 0, N) # G = 0
magenta[:, 2] = np.linspace(255/N, 0, N)  # B = 255
magenta_cmp = ListedColormap(magenta)
magenta_r_cmp = magenta_cmp.reversed()

#cyan = np.ones((N, 4))
#cyan[:, 0] = np.linspace(0, 0, N) # R = 255
#cyan[:, 1] = np.linspace(255/N, 0, N) # G = 0
#cyan[:, 2] = np.linspace(255/N, 0, N)  # B = 255
#cyan_cmp = ListedColormap(cyan)
#cyan_r_cmp = cyan_cmp.reversed()

color = np.ones((N, 4))
color[:, 0] = np.linspace(0, 0, N) # R = 255
color[:, 1] = np.linspace(0, 255/256, N) # G = 232
color[:, 2] = np.linspace(0, 255/256, N)  # B = 11
cyan = ListedColormap(color)
cyan_r = cyan.reversed()


class AlignmentAnalyzer(object):
    """AlignmentAnalyzer Class
        - contains functions for alignment analysis
    """
    metafile = "params.json"
    color = [ 'r', 'g', 'b', 'm', 'c' ]
    def __init__(self, data_loc, marker, atlas_dir, rl_file):
        """init
           data_loc: map (dictionary) that contains data name and its location.
           marker: channel to analyze (cell type)
        """
        self.data_loc = data_loc
        self.marker = marker
        self.atlas_dir = atlas_dir
        self.rl_file = os.path.join(self.atlas_dir, rl_file)
        self.ann_df = pd.read_csv(self.rl_file)

        self.data = None
        self.rs_align_json = {}
        self.align_json = {}
        self.ann_tif = {}
        self.cc_npy = {}
        self.cc_csv = {}
        self.dId = {} # data id
        self.al_info = {}
        self.raw_dim = {}
        self.ds_dim = {}

        # grab metadata from "params.json"
        self.load_metadata()


    def load_metadata(self):
        """load_metadata
           : load meta data from metafile (params.json)
        """

        for k in list(self.data_loc.keys()):
            self.rs_align_json[k] = []
            self.align_json[k] = []
            self.ann_tif[k] = []
            self.cc_npy[k] = []
            self.cc_csv[k] = []
            self.dId[k] = []
            self.al_info[k] = []
            self.raw_dim[k] = []
            self.ds_dim[k] = []
            for item in self.data_loc[k]:
                print("dataset: {}".format(item))
                jsonf = os.path.join(item, self.metafile)
                with open(jsonf, 'r') as fp:
                    params = json.load(fp)

                align_info = params[AC.ALGN_INFO]
                self.al_info[k].append(align_info)

                self.rs_align_json[k].append(align_info[AC.RS_ALGND_JSON])
                # align_json is more accurate than rescaled as it's aligned with downsampled version
                self.align_json[k].append(align_info[AC.ALGND_JSON])
                self.ann_tif[k].append(align_info[AC.ANN_TIF])
                self.cc_npy[k].append(params[self.marker][MD.CC_NPY])
                self.cc_csv[k].append(params[self.marker][MD.CC_CSV])
                self.dId[k].append(params[ParamConst.NAME])

                self.raw_dim[k].append(params[self.marker]['shape'])
                ds_vol = tifffile.imread(params[AC.ALGN_INFO]['downsampled_tif'])
                self.ds_dim[k].append(ds_vol.shape)


    def load_data(self, use_rescaled=False, update_csv=True):
        print("use_rescaled_json? ", use_rescaled)
        self.data = {}
        for k in list(self.data_loc.keys()):
            self.data[k] = []
            for idx in range(len(self.data_loc[k])):
                item = self.data_loc[k][idx]
                print("idx, item: [ {}, {} ]".format(idx, item))

                # create MouseAtlas
                #js_file = os.path.join(item, self.rs_align_json)
                if use_rescaled:
                    js_file = self.rs_align_json[k][idx]
                else:
                    js_file = self.align_json[k][idx]
                ann_tif = os.path.join(self.atlas_dir, self.ann_tif[k][idx])
                ma = MouseAtlas(js_file, self.rl_file, ann_tif)

                # load cell coordinates
                cc_csv_file = os.path.join(item, self.cc_csv[k][idx])
                print("cc_csv: ", cc_csv_file)
                cc_df = pd.read_csv(cc_csv_file)
                #cc_npy_file = os.path.join(item, self.cc_npy[k][idx])
                #cc_npy = np.load(cc_npy_file)
                #print("cc_npy: ", cc_npy_file)

                al_info = self.al_info[k][idx]
                raw_dim = self.raw_dim[k][idx]
                ds_dim = self.ds_dim[k][idx]
                print("al_info: {}".format(al_info))

                # get label map

                if AC.REGION in cc_df:
                    print("Region extraction is already done! using ones from CC_CSV.")
                else:
                    cc_df = self.get_region_info(cc_df, ma, al_info,
                                                 raw_dim, ds_dim, use_rescaled)
                    if update_csv:
                        # backup current CSV
                        cc_csv_file_backup = cc_csv_file + "_backup_%s"%bmUtil.get_current_time()
                        shutil.copyfile(cc_csv_file, cc_csv_file_backup)

                        # update CSV with the new df
                        cc_df.to_csv(cc_csv_file, sep=',', index=False)
                        print("Backed up csv file (%s) and updated with Region column."%cc_csv_file_backup)

                self.data[k].append([ma, cc_df])


    @staticmethod
    def _raw_to_ds(raw_point, raw_dim, ds_dim, al_info):
        zidx = al_info['zidx']
        yidx = al_info['yidx']
        xidx = al_info['xidx']

        rzz, ryy, rxx = raw_dim
        dzz = ds_dim[zidx]
        dyy = ds_dim[yidx]
        dxx = ds_dim[xidx]

        # get corresponding coordinate on RAW data
        transformed = np.zeros((3, ), dtype=int)
        transformed[zidx]= int(raw_point[0] / rzz * dzz)
        transformed[yidx] = int(raw_point[1] / ryy * dyy)
        transformed[xidx] = int(raw_point[2] / rxx * dxx)
        if al_info['flip_z']:
            transformed[0] = ds_dim[0] - transformed[0]
        if al_info['flip_y']:
            transformed[1] = ds_dim[1] - transformed[1]
        if al_info['flip_x']:
            transformed[2] = ds_dim[2] - transformed[2]

        return transformed


    def get_region_info(self, cc_df, ma, al_info, raw_dim, ds_dim, use_rescaled=False):
        """get dictionary of region labels from MouseAtlas class"""
        crds = cc_df[['z', 'y', 'x']].values
        lnames = []
        for c in tqdm(crds, desc="getting labels"):
            if use_rescaled:
                # direct warping with raw coordinates
                pt = c
            else:
                # transform raw coordinate to downsampled space
                pt = self._raw_to_ds(c, raw_dim, ds_dim, al_info)

            _, lname, _ = ma.get_label(pt, debug=False)
            lnames.append(lname)

        # add to df
        cc_df[AC.REGION] = lnames

        return cc_df


    def generate_label_map(self, df, conditions):
        """
        example: conditions = { "col1": [1, 2, 3],
                                "col2": [4, 5] }
                    keys: AND operation
                    list values: OR operation
                 : retreive where col1 = 1 or 2 or 3, and
                                  col2 = 4 or 5
        """
        df_new = df
        for key in conditions.keys(): # AND
            cond = conditions[key] # list of OR

            #df_new = df_new[df_new[key]==cond]

            sub_dfs = []
            for cc in cond:
                sub_df = df_new[df_new[key] == cc]
                sub_dfs.append(sub_df)
            # concat
            df_new = pd.concat(sub_dfs)


        lmap = {}
        cells_no_reg = df_new[df_new[AC.REGION].isnull()][AC.REGION].values
        print("# of cells not assigned to any region: ", len(cells_no_reg))
        df_new = df_new.dropna()
        regions = df_new[AC.REGION].unique().tolist()
        for reg in tqdm(regions, "generate lmap"):
            cells = df_new[df_new[AC.REGION] == reg][['z', 'y', 'x']].values
            lmap[reg] = cells

        return df_new, lmap


    def compute_population(self, df_colname, conditions):
        df_all = {}
        for j, key in enumerate(list(self.data.keys())):
            print("key: ", key)
            if key not in conditions.keys():
                # skip
                continue
            dset = self.data[key]
            dset_conds = conditions[key]

            # loop through dataset
            df = self.ann_df[['id', 'name', 'acronym', 'parent_structure_id', 'depth']].copy()
            for i in range(len(dset)):
                m_atlas, df_cc_csv = dset[i]

                dname = self.dId[key][i]
                print("dname: ", dname)
                df[dname] = 0
                vol_coln = 'volume_%s'%df_colname[key][i]
                print("vol_coln: ", vol_coln)
                vol_coln_norm = vol_coln + "_norm"
                if vol_coln not in df:
                    df[vol_coln] = self.ann_df[vol_coln]
                if vol_coln_norm not in df:
                    df[vol_coln_norm] = self.ann_df[vol_coln_norm]

                # get label map
                df_cc_csv_new, lblmap = self.generate_label_map(df_cc_csv, dset_conds)

                # add to dataframe
                regions = list(lblmap.keys())
                counts = [len(lblmap[key]) for key in regions]
                for r, c in zip(regions, counts):
                    if r == '':
                        print("empty region (not assigned): %d"%c)
                        continue

                    df.at[df[df['acronym'] == r].index[0], dname] = c

                    # update all parent regions
                    curr_id = df[df['acronym'] == r]['id'].iloc[0]
                    while True:
                        pid = int(df[df['id'] == curr_id]['parent_structure_id'])
                        if pid == -1:
                            break
                        p_row = df[df['id'] == pid]
                        df.at[p_row.index[0], dname] += c
                        curr_id = pid

            df_all[key] = df

        return df_all


    def compute_normalized_density(self, df_all, data_type, df_colname,
                                   depth_at=2, roi_list=None,
                                   compute_type='density', rel_vol_ratio=None):
        df_t = df_all[data_type]
        data_t = self.dId[data_type]

        # get data at specific DEPTH
        final_df = df_t[['id', 'name', 'acronym', 'depth']]
        if roi_list is not None:
            final_df = final_df[final_df['acronym'].isin(roi_list)]
        else:
            final_df = final_df[final_df['depth']==depth_at]

        for i in range(len(data_t)):
            dname = data_t[i]

            if compute_type == 'count':
                # get cell count
                compute_coln = '%s_count'%dname
                df_t[compute_coln] = df_t[dname].copy()
            else: # density
                # compute density
                compute_coln = '%s_density'%dname
                #vn_coln = 'volume_%s_norm'%df_colname[data_type][i]
                vn_coln = 'volume_%s'%df_colname[data_type][i]
                # divide by region volume
                df_t[compute_coln] = df_t[dname] / df_t[vn_coln]

                if rel_vol_ratio is not None:
                    # multiply with relative volume factor
                    df_t[compute_coln] *= rel_vol_ratio

            # filter by condition (either roi, or depht_at)
            if roi_list is not None:
                #print("got roi list, use instead depth_at")
                da_df = df_t[df_t['acronym'].isin(roi_list)]
            else:
                # compute density at specified depth
                da_df = df_t[df_t['depth']==depth_at]

            #final_df[vn_coln] = da_df[vn_coln]
            final_df[compute_coln] = da_df[compute_coln]

            # shouldn't be needed
            #final_df[density_coln] = da_df[cnt_coln].copy() / da_df[cnt_coln].sum()
            #final_df[density_coln] = da_df[cnt_coln].copy()
            #final_df = final_df[final_df['density'] > 0]

        return final_df


    def compute_counts(self, df_all, data_type, df_colname, depth_at=2):
        df_t = df_all[data_type]
        data_t = self.dId[data_type]

        # get data at specific DEPTH
        final_df = df_t[['id', 'name', 'acronym', 'depth']]
        final_df = final_df[final_df['depth']==depth_at]

        for i in range(len(data_t)):
            dname = data_t[i]
            # get Normalized Density
            cnt_coln = dname
            print("cnt_coln: ", cnt_coln)
            vn_coln = 'volume_%s'%df_colname[data_type][i]
            print("vn_coln: ", vn_coln)

            # compute density at specified depth
            da_df = df_t[df_t['depth']==depth_at]

            final_df[cnt_coln] = da_df[cnt_coln]
            final_df[vn_coln] = da_df[vn_coln]
            density_coln = '%s_density'%dname
            final_df[density_coln] = da_df[dname].copy()

            #final_df = final_df[final_df['density'] > 0]
        return final_df


    def analyze_within_class(self):
        print("Analyzing within class...")


    def plot_between_class(self, fx, fys, labels, step=20):
        if len(fx) < step:
            step = len(fx)

        ite = int(np.ceil(len(fx)/step))
        print("ite: ", ite)

        fig, axes = plt.subplots(nrows=ite, ncols=1, figsize=(30, 4*ite), dpi=80)
        if ite == 1:
            axes = [axes]
        plt.title(">10% discrepency in # of cells")

        for i in range(ite):
            ss = step * i
            ee = ss + step

            ind = np.arange(step)
            width = 0.3

            max_y = 0
            for j in range(len(fys)):
                axes[i].bar(fx[ss:ee], fys[j][ss:ee], color=self.color[j], align='center', alpha=0.4, label=labels[j])
                if max(fys[j][ss:ee]) > max_y:
                    max_y = max(fys[j][ss:ee])

            axes[i].set_ylabel("# of cells")
            axes[i].set_ylim(0, max_y)
            axes[i].legend(loc='best')
        plt.show()


    def analyze_between_class(self):
        print("Analyzing between class...")

        fx, fys, labels = self.region_diff(min_cells=50)
        self.plot_between_class(fx, fys, labels)

    def analyze(self):
        self.analyze_within_class()
        self.analyze_between_class()


    def rat_diff_enough(self, counts, thr):
        indices = range(len(counts))
        combo = list(combinations(indices, 2))

        for c in combo:
            c1 = counts[c[0]]
            c2 = counts[c[1]]
            rat = float(abs(c1 - c2)) / max(c1, c2)
            if rat > thr:    # if there's any
                return True

        return False


    def filter_by_ratio(self, r_is, lms, thr=0.1, min_cells=200, max_cells=8000):
        """filter regions by ratio of two datasets"""

        fx = []
        fys = []
        for i in range(len(lms)):
            fys.append([])

        for r in r_is:
            if r == 'root':
                continue

            counts = [len(lm[r]) for lm in lms]

            if max(counts) < min_cells:
                continue
            if max(counts) > max_cells:
                continue

            if not self.rat_diff_enough(counts, thr):
                continue

            fx.append(r)

            for i in range(len(lms)):
                fys[i].append(counts[i])

        return fx, fys


    def region_diff(self, min_cells=200, max_cells=8000):
        """analyze difference between two dataset by region"""
        assert self.data is not None

        classes = list(self.data.keys())
        print("classes: ", classes)

        lms = []
        for i in range(len(classes)):
            lms.append(self.data[classes[i]][0][2])    # [ma, cc, lm]

        rgs = []
        r_df = None
        r_is = None
        for i in range(len(lms)):
            rg = list(lms[i].keys())
            rgs.append(rg)
            if i == 0:
                r_df = set(rg)
                r_is = set(rg)

            else:
                r_df -= set(rg)
                r_is = r_is.intersection(rg)

        r_df = list(r_df)
        r_is = list(r_is)

        # retrieve ys
        ys = []

        for i in range(len(lms)):
            y = [ len(lms[i][x]) for x in r_is ]
            ys.append(y)

        fx, fys = self.filter_by_ratio(r_is, lms, min_cells=min_cells, max_cells=max_cells)

        return fx, fys, classes


    @staticmethod
    def draw_figure(moving_XY,  af_vol, ams, pt, w,
                    lbl_map=None, clim=[100, 1000], ann_clim=[15000, 18000],
                    cmap_n='gist_ncar_r', sharex=True, rot_deg=0):
        nc = 3 if lbl_map is None else 4
        # DRAW
        fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=nc, sharex=sharex)
        cmp = plt.get_cmap(cmap_n)
        cmp.set_under(color='black')

        # rotate if needed
        tr = transforms.Affine2D().rotate_deg(rot_deg)

        # plot
        msz = 9
        pt_marker = '*' # 'v' # "X"
        ax[0].imshow(moving_XY, transform=tr+ax[0].transData, clim=clim, cmap=cyan)
        ax[0].set_title('Ours')
        ax[0].plot(pt[2], pt[1], marker=pt_marker, markersize=msz, color="white")

        ax[1].imshow(af_vol[w[0]], cmap='gray')
        ax[1].plot(w[2], w[1], marker=pt_marker, markersize=msz, color="white")
        ax[1].set_title('Reference')

        ax[2].imshow(ams, clim=ann_clim, cmap=cmp)
        ax[2].plot(w[2], w[1], marker=pt_marker, markersize=msz, color="white")
        ax[2].set_title('Annotation')

        if lbl_map is not None:
            ax[3].imshow(moving_XY, clim=clim, cmap='gray')
            ax[3].imshow(lbl_map, clim=ann_clim, alpha=0.4, cmap=cmp)
            ax[3].plot(pt[2], pt[1], marker=pt_marker, markersize=msz, color="white")
            ax[3].set_title('Overlay')

        # remove ticks
        for aax in ax:
            aax.set_yticks([])
            aax.set_xticks([])
            aax.axis('off')

        fig.tight_layout()
        plt.show()

        return fig


    @staticmethod
    def _align_partial(ma, img, lblmap, x_start, x_end, z, thr):
        point_list = []
        yy, xx = img.shape
        for x in range(x_start, x_end):
            for y in range(yy):
                if img[y][x] > thr:
                    point_list.append([z, y, x])
                    a_pt = [z, y, x]
                    w = ma.warper(a_pt)
                    w = [int(x) for x in w[0]]
                    try:
                        lno = int(ma.anns_mask[w[0]][w[1]][w[2]])
                    except:
                        #print("EXCEPTION: a_pt: ", a_pt, ", w: ", w, ", anns_mask.shape: ", ma.anns_mask.shape)
                        continue
                    if(all(i >= 0 for i in w)) and (lno > 0):
                        lblmap[y][x] = lno

        return point_list, lblmap


    @staticmethod
    def align_point(ma, point, moving_tif, af_tif, params,
                    raw_tiff_dir=None, raw_dim=None,
                    ext='tif', flip_x=False, flip_y=False, flip_z=False,
                    is_raw=False, clim=[0, 1500], ann_clim=[15000, 30000],
                    map_slice=False, intensity_thr=110, ncpu=10,
                    cmap_n='gist_ncar_r', figsize=(15,8)):
        af_vol = tifffile.imread(af_tif)
        moving_vol = tifffile.imread(moving_tif) # downsampled TIF
        assert(af_vol.shape == moving_vol.shape)

        if is_raw:
            # Results on RAW Data
            md, mh, mw = moving_vol.shape
            # get corresponding coordinate on RAW data
            rzz, ryy, rxx = raw_dim
            raw_z = int(point[0] / md * rzz)
            raw_y = int(point[1] / mh * ryy)
            raw_x = int(point[2] / mw * rxx)
            if flip_x:
                raw_x = rxx - raw_x
            if flip_y:
                raw_y = ryy - raw_y
            if flip_z:
                raw_z = rzz - raw_z

            tifpath = os.path.join(raw_tiff_dir, "img_%04d.%s"%(raw_z, ext))
            moving_XY = tifffile.imread(tifpath)
            pt = [raw_z, raw_y, raw_x]

        else:
            # get a slice
            moving_XY = np.asarray(moving_vol[point[0]])
            pt = point

            # rescale moving_XY with the original aspect ratio
            mXY_h, mXY_w = moving_XY.shape
            cparams = params[params['marker']]
            if params['align_info']['xidx'] == 2:
                raw_h, raw_w = cparams['shape'][1:]
            else:
                raw_w, raw_h = cparams['shape'][1:]
            ar_w = int(raw_w * mXY_h / raw_h)
            rescale_to = (ar_w, mXY_h)
            rescaled_ptx = pt[2] * (ar_w / mXY_w)

        print("moving slice shape: ", moving_XY.shape, ", pt: ", pt, ", is_raw?: ", is_raw)

        nc = 3
        lbl_map = None

        if map_slice:
            nc = 4
            yy, xx = moving_XY.shape
            point_list = []
            lbl_map = np.zeros((yy, xx)) #, dtype=np.uint32)

            pt_size = int(xx/ncpu)
            rr = np.arange(0, xx, pt_size)
            rr_end = rr + pt_size
            rr_end[-1] = xx
            rrlen = len(rr)
            print("ncpu: ", ncpu, ", pt_size: ", pt_size, ", rrlen: ", rrlen)
            args_partial = list(zip(repeat(ma), repeat(moving_XY), repeat(lbl_map), rr, rr_end, repeat(pt[0]), repeat(intensity_thr)))
        #    args_partial = list(zip([ma]*rrlen, [moving_XY]*rrlen, [lbl_map]*rrlen, rr, rr_end, [pt[0]]*rrlen, [intensity_thr]*rrlen))
            with Pool() as pool:
                L = pool.starmap(_align_partial, args_partial)
                for item in L:
                    ptl, lmap = item
                    point_list += ptl
                    lbl_map += lmap

            print("[ %d / %d ] are nonzero. (%.02f %%)"%(len(point_list), xx*yy, len(point_list)*100./(xx*yy)))

        # rescale w/ origina aspect_ratio
        if not is_raw:
            moving_XY = cv2.resize(moving_XY, dsize=rescale_to, interpolation=cv2.INTER_CUBIC)
            if map_slice:
                lbl_map = cv2.resize(lbl_map, dsize=rescale_to, interpolation=cv2.INTER_CUBIC)


        # WARP POINT
        w = ma.warper(pt)
        w = [int(x) for x in w[0]]
        ams = ma.anns_mask[w[0]]
        lno = int(ma.anns_mask[w[0]][w[1]][w[2]])
        print("pt(z, y, x): ", pt, ", w(z, y, x): ", w, ", label_no: ", lno)

        # DRAW
        sharex = False if is_raw else True
        fig = draw_figure(moving_XY,  af_vol, ams, [pt[0], pt[1], rescaled_ptx], w, lbl_map, clim, ann_clim, cmap_n, sharex=sharex)

        if is_raw:
            return moving_XY, af_vol, ams, lbl_map, w
        else:
            return moving_XY, af_vol, ams, lbl_map, w, rescaled_ptx

    @staticmethod
    def warp_image(z0, z1, warper, src_image, shape, is_ann):
        z, y, x = np.mgrid[z0:z1, 0:shape[1], 0:shape[2]]
        aligned = np.zeros((shape[1], shape[2]))
        src_coords = np.column_stack([z.flatten(),
                                      y.flatten(), x.flatten()])
        if is_ann:
            warped = warper(src_coords)
            warp_map = {}
            for scrd, wcrd in zip(src_coords, warped):
                sc_z, sc_y, sc_x = scrd
                key = "%d,%d,%d"%(sc_z, sc_y, sc_x)
                warp_map[key] = wcrd

            for xx in range(shape[2]):
                for yy in range(shape[1]):
                    key = "%d,%d,%d"%(z0, yy, xx)
                    wz, wy, wx = [int(i) for i in warp_map[key]]
                    try:
                        lbl = src_image[wz][wy][wx]
                        aligned[yy][xx] = lbl
                    except:  # out-of-bound
                        #print("wz, wy, wx: ", wz, wy, wx, "src_image.shape: ", src_image.shape)
                        continue

            return (z0, aligned)
        else:
            ii, jj, kk = [_.reshape(z.shape) for _ in warper(src_coords).transpose()]
            return (z0, map_coordinates(src_image, [ii, jj, kk]))


    @staticmethod
    def align_image(warper, src_image, dst_image, ncpu=20, is_ann=False):
        """Warp the source image into the destination image's space"""
        decimation = max(1, np.min(dst_image.shape) // 5)
        inputs = [
            np.arange(0,
                      dst_image.shape[_]+ decimation - 1,
                      decimation)
            for _ in range(3)]

        alignment_image = np.zeros(src_image.shape)
        warper = warper.approximate(*inputs)

        with Pool(ncpu) as pool:
            futures = []
            for z0 in range(0, dst_image.shape[0]):
                z1 = z0 + 1
                futures.append(
                    pool.apply_async(warp_image,
                                     (z0, z1, warper, src_image, dst_image.shape, is_ann)))
            for future in tqdm(futures, desc="Warping image"):
                myz, aligned = future.get()
                alignment_image[myz:myz+1] = aligned

        return alignment_image


    @staticmethod
    def compute_density_on_slice(wd, params, ma, cc_list, ref_slices, damp=1, region_ids=None):
        is_absolute_density = False if region_ids is None else True
        print("is_absolute_density? ", is_absolute_density)

        al_info = params['align_info']
        marker = al_info['aligned_on']
        raw_dim = params[marker]['shape']
        ds_vol = tifffile.imread(al_info['downsampled_tif'])
        ds_dim = ds_vol.shape

        # create cell_dicts w/ ref_slices as keys
        cell_dicts = {}
        for rs in ref_slices:
            cell_dicts[rs] = {}

        # create list of slices again with ref_slices and damp
        ref_slices_with_damp = {}
        for rs in ref_slices:
            candidate_slices = np.arange(rs, rs + damp, 1)
            for ii in candidate_slices:
                ref_slices_with_damp[ii] = rs

        for cc in tqdm(cc_list, desc="Warping CCs"):
            # Transform the raw point to downsampled space
            pt = AlignmentAnalyzer._raw_to_ds(cc, raw_dim, ds_dim, al_info)

            # Warp point
            w = ma.warper(pt)
            w = [int(x) for x in w[0]]
            slice_no = w[0]

            if slice_no in ref_slices_with_damp.keys():
                try:
                    lno = int(ma.anns_mask[w[0]][w[1]][w[2]])
                except IndexError:
                    # Out of Bound
                    lno = -1

                cell_dict = cell_dicts[ref_slices_with_damp[slice_no]]

                if is_absolute_density:
                    if lno not in region_ids:
                        continue
                if lno not in cell_dict.keys():
                    cell_dict[lno] = [w]
                else:
                    cell_dict[lno].append(w)

        return cell_dicts


    @staticmethod
    def get_density_map(wd, params, ma, cc_list, s_indices, damp=1, region_ids=None):
        dd, hh, ww = ma.anns_mask.shape
        dmaps = np.zeros((len(s_indices), hh, ww))

        def _get_cell_population(ma, slice_no, cell_dict, bg_val=-1):
            ann_mask = ma.anns_mask[slice_no]
            pop_map = np.zeros(ann_mask.shape)

            h, w = ann_mask.shape
            for yy in range(h):
                for xx in range(w):
                    lbl_no = ann_mask[yy][xx]
                    if lbl_no == 0:
                        pop_map[yy][xx] = bg_val # background

                    elif lbl_no in cell_dict.keys():
                        pop_map[yy][xx] = len(cell_dict[lbl_no])

            return pop_map

        # get cell_dicts
        cell_dicts = AlignmentAnalyzer.compute_density_on_slice(wd, params, ma, cc_list,
                                                                s_indices, damp, region_ids)

        # fill dmaps
        for idx, si in enumerate(tqdm(s_indices, "slices")):
            cell_dict = cell_dicts[si]
            # compute cell population by region
            dmaps[idx] = _get_cell_population(ma, si, cell_dict)

        return cell_dicts, dmaps


    @staticmethod
    def get_area_of_region(cell_dicts, ma, damp):
        area_dict = {}

        zs = cell_dicts.keys()
        area_dicts = {}
        area_masks = {}
        for zno in zs:
            ann_zno = ma.anns_mask[zno]
            area_mask = np.ones(ann_zno.shape)

            area_dicts[zno] = {}

            labels = list(cell_dicts[zno].keys())
            #print("zno: ", zno, "labels: ", labels)
            new_zs = np.arange(zno, zno + damp, 1)

            for ll in labels:
                locs_xx, locs_yy = np.where(ann_zno==ll)

                # compute area for a region w/ label ll
                area = 0
                for nz in new_zs:
                    ann_mask =ma.anns_mask[nz]
                    area += len(np.where(ann_mask==ll)[0])
                    #print("\tnz: ", nz, "area: ", area)
                #print("== label: ", ll, "cum area: ", area)
                area_dicts[zno][ll] = area

                # create area mask
                for xx, yy in zip(locs_xx, locs_yy):
                    area_mask[xx][yy] = area

            area_masks[zno] = area_mask


        return area_dicts, area_masks
