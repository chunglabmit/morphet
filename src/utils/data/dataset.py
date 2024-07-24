"""dataset.py: dataset classes"""
__author__      = "Minyoung Kim"
__license__ = "MIT"
__maintainer__ = "Minyoung Kim"
__email__ = "minykim@mit.edu"
__date__ = "08/15/2018"


#----------
# imports
#----------
import numpy as np
import pickle
import glob
import os
from torch.utils.data import Dataset
import torch
from utils.const import Phase
import utils.util as bmUtil
from utils.const import Phase, NormalizationType

class MicrogliaDataset(Dataset):
    """Microglia dataset"""

    def __init__(self, data_path, ext, phase, num_clusters, preprocessor, with_labels=False, return_fname=False,
                 data_size=None, mdg=None, centroids=None, ntype=NormalizationType.ZERO_AND_ONE, clip=False):
        """
        Params
        ----------
        data_path: path of dataset location
        ext: extension of files
        phase: phase of training data to retrieve
        num_clusters: number of clusters / classes
        preprocessor: Preprocessor() object for the dataset
        return_fname: return filename in __getitem__ if set True
        """

        self.prep = preprocessor
        self.data_path = data_path
        self.file_ext = ext
        self.phase = phase
        self.num_clusters = num_clusters
        self.return_fname = return_fname
        self.with_labels = with_labels
        self.data_size = data_size
        self.mdg = mdg
        self.centroids = centroids
        self.norm_percentile = (0.5, 99.5)
        self.ntype = ntype
        self.clip = clip

        if self.with_labels:
            self.labels, self.weights = self.build_labels(self.data_path, self.phase, self.num_clusters)
        else:
            self.labels, self.weights = (None, None)

        if self.mdg is None:
            # load list of numpy files
            self.files = self._load_files(self.data_path, self.phase, self.file_ext, self.labels)
        else:
            assert self.centroids is not None
            print("len(centroids): ", len(self.centroids))


    @staticmethod
    def build_labels(path, phase, num_clusters):
        label_files = glob.glob(os.path.join(path, phase) + '/labels/*.p')
        print("loaded %d label files from [%s] set"%(len(label_files), phase))

        if len(label_files) == 0:
            return None

        label_map = {}
        cluster_cnt = [0] * num_clusters
        for f in label_files:
            with open(f, 'rb') as fp:
                data = pickle.load(fp)

            new_data = dict(("%s/%s/%s"%(path, phase, key), value) for (key, value) in data.items())
            label_map.update(new_data)

        for k in list(label_map):
            # filter labels not going to be considered, depending on # of clusters defined
            if label_map[k] >= num_clusters:
                del label_map[k]
                continue
            cluster_cnt[label_map[k]] += 1

        print("cluster_cnt: {}".format(cluster_cnt))
        weights = max(cluster_cnt) / torch.Tensor(cluster_cnt)
        weights = weights / torch.max(weights)

        weights_all = []
        for k in label_map.keys():
            weights_all.append(weights[label_map[k]])

        assert len(weights_all) == len(label_map.keys())

        return label_map, weights_all


    def __len__(self):
        """return number of data samples in current phase"""

        if self.mdg is None:
            return len(self.files)
        else:
            return len(self.centroids)


    def __getitem__(self, idx):
        """return data at [idx] from current phase"""

        fname = self.files[idx]
        data_raw = np.load(fname).squeeze()
        zz, xx, yy = data_raw.shape
        #rescale_z = 16. / zz
        #rescale_x = 32. / yy
        #rescale_y = 32. / xx

        # preprocess
        if self.prep is not None:
            #resize = (rescale_z, rescale_y, rescale_x)
            data = self.prep.preprocess_all(data_raw, normByPercentile=self.norm_percentile, # [0.5, 99.5] by default
                                            ntype=self.ntype, clip=self.clip)#, resize=resize)
            #print("min: {}, max: {}, mean: {}".format(data.min(), data.max(), data.mean()))
        else:
            data = data_raw.copy()

        if self.return_fname:
            if self.with_labels:
                return fname, data, self.labels[fname]
            else:
                return fname, data, data_raw
        else:
            if self.with_labels:
                return data, self.labels[fname]
            else:
                return data


    @staticmethod
    def _load_files(path, phase, ext, labels=None):
        """load files in current phase only"""

        if labels is None:
            if phase == Phase.REALTIME:
                files = sorted(glob.glob(path + '/*.%s'%ext))
                if not len(files):  # subdirectories!
                    files = sorted(glob.glob(path + '/*/*.%s'%ext))
            else:
                files = glob.glob(os.path.join(path, phase) + '/*/*.%s'%ext)
        else:
            files = list(labels.keys())

        print("loaded %d sample files from [%s] set"%(len(files), phase))
        return files


class MPDataset(MicrogliaDataset):
    def __init__(self, **args):
        super(MPDataset, self).__init__(**args)


class TrapDataset(MicrogliaDataset):
    def __init__(self, **args):
        super(TrapDataset, self).__init__(**args)
        assert self.data_size is not None

    def __getitem__(self, idx):
        """return data at [idx] from current phase"""

        hd, hh, hw = self.data_size # (z, y, x) order

        if self.mdg is None:
            fname = self.files[idx]
            #print("IDX[ {} ] loading data from {}".format(idx, fname))
            data_raw = np.load(fname)
        else:
            # format fname as cell_000168_zyx-1077-5565-3398.npy
            z, y, x = self.centroids[idx]
            fname = "cell_%6d_zyx-%d-%d-%d.realtime"%(idx, z, y, x)
            patch = self.mdg.grab_a_patchcube(x, y, z, hw, hh, hd, use_zarr=True)
            data_raw = patch[0]

        if data_raw is not None:
            # preprocess
            if self.prep is not None:
                data = self.prep.preprocess_all(data_raw, normByPercentile=self.norm_percentile,
                                                ntype=self.ntype, clip=self.clip)

                #print("min: {}, max: {}, mean: {}".format(data.min(), data.max(), data.mean()))
            else:
                data = data_raw.copy()

            if self.mdg is None:
                if len(data.shape) == 5: # (1, 1, z, y, x)
                    data = np.squeeze(data, 1)
                elif len(data.shape) == 3: # (z, y, x)
                    data = data[np.newaxis, :, :, :]

                od, oh, ow = data.shape[-3:]
                if od > 2*hd:
                    midpoint = int(od / 2)
                    data = data[:, midpoint-hd:midpoint+hd, :, :]
                if oh > 2*hh:
                    midpoint = int(oh / 2)
                    data = data[:, :, midpoint-hh:midpoint+hh, :]
                if ow > 2*hw:
                    midpoint = int(ow / 2)
                    data = data[:, :, :, midpoint-hw:midpoint+hw]

            data = data[:, np.newaxis, :, :]
        else:
            data = None

        if self.return_fname:
            if self.with_labels:
                return fname, data, self.labels[fname]
            else:
                return fname, data, data_raw
        else:
            if self.with_labels:
                return data, self.labels[fname]
            else:
                return data


class RealTimeDataset(TrapDataset):
    def __init__(self, **args):
        super(RealTimeDataset, self).__init__(**args)
