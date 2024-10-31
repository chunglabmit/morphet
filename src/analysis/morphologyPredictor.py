# -*- coding: utf-8 -*-
"""morphologyPredictor.py: Class for predicting morphology of microglia and visualizing"""
__author__      = "Minyoung Kim"
__license__ = "MIT"
__maintainer__ = "Minyoung Kim"
__email__ = "minykim@mit.edu"
__date__ = "10/17/2018"
import sys
sys.path.append("../../")
import os
import warnings
warnings.filterwarnings("always")


from multiprocessing import Pool, freeze_support
from functools import partial

from pathlib import Path
import datetime
import numpy as np
from glob import glob
import torch
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy
from vispy import app
import json
import yaml
from argparse import Namespace
import pandas as pd

# internal
from utils.const import Dataset, Phase, VizMarkers, StainChannel, ModelType
from utils.data.microgliaDB import MDBLabel
from utils.data.dataset import MicrogliaDataset, TrapDataset, RealTimeDataset
from utils.params import TrainParams, MorphPredictionParams, DataGenParams
from utils.data.data_generator import DataGenerator
import utils.util as bmUtil
from utils.train.preprocessor import TrainPreprocessor
from train.trainer import Trainer
from train.mTrainer import MTrainer
from train.bfmcTrainer import BFMCTrainer
from analysis.baseAnalyzer import BaseAnalyzer
from analysis.unsupervisedLearner import BMClustering

MLABEL = MDBLabel()

def io_worker(fnames, data):
    for fn, d in zip(fnames, data):
        np.save(fn, d)

class MorphologyPredictor(BaseAnalyzer):
    prt = bmUtil.PRT()

    def __init__(self, **args):
        self.gargs = args.pop('gargs')
        self.margs = args.pop('margs')
        self.ranges = args.pop('ranges')
        self.in_memory = args.pop('in_memory')
        print("in_memory? ", self.in_memory)
        try:
            self.cc_csv = args.pop('cc_csv')
        except KeyError:
            pass

        self.curr_vol = VizMarkers.GFP_MICROGLIA
        self.mdb = None
        self.predictor = None
        self.is_prediction_run = False
        self.marker_size = 10  # for figure, pt 15
        self.marker_alpha= 0.9 # for figure, alpha 0.7

        super(MorphologyPredictor, self).__init__(**args)


    def set_range(self, axis, ranges):
        self.ranges[axis] = ranges


    def setup_dg(self, name="MP", ranges=[None, None, None], ccd=None):
        if ccd is None:
            if self.gargs is None or self.margs is None:
                all_args = None
            else:
                all_args = self.gargs + self.margs
            dgParams = DataGenParams()
            dgParams.build(all_args, "DataGenParser-%s"%name)
            self.dg = DataGenerator(dgParams)
        else:
            self.dg = ccd.dg


    def load_data(self, clim, overlayCenters=False, ccd=None,
                        ranges=None, z_damp=0,
                        loadonly=False, marker_color=None,
                        shuffle=True):

        if self.gargs is None or self.margs is None:
            all_args = None
        else:
            all_args = self.gargs + self.margs

        if ranges is None:
            ranges = self.ranges

        # setup data
        self.mVol, \
        self.mRawPoints, \
        self.mPoints = self.form_data(all_args, VizMarkers.GFP_MICROGLIA,
                                      ranges=ranges, ccd=ccd, shuffle=shuffle,
                                      z_damp=z_damp)
        self.clim_min, self.clim_max = clim
        if not loadonly:
            self.set_volume(self.mVol.copy(), clim=[self.clim_min, self.clim_max], scalebar=False)
            self.volume.cmap = 'grays'  # viridis, hot, hsl, grays

            # show
            if overlayCenters:
                self.overlay_marker(self.mPoints, VizMarkers.GFP_MICROGLIA, "m", color=marker_color)


    def get_data_type(self):
        return self.dg.p.data_type


    def get_save_path(self):
        return self.dg.p.save_path


    def setup_model(self, train_params, full_inference):
        """setup trainer and preprocessor class for prediction"""

        print("train_params: {}".format(train_params))

        # configure train params
        self.trParams = TrainParams()
        r = self.trParams.build(train_params, "BM_TRParser_MP", True)  # BM Train Parser for Morphology Predictor

        if r is not None:
            tb, entry, te = r
            entry = entry.replace("\n", "<br/>")
            self.log(tb, self.prt.LOG)
            self.log("{}".format(entry), self.prt.STATUS2)
            self.log(te, self.prt.LOG)

        # get preprocessor
        self.prep = TrainPreprocessor()


    def run(self, full_inference, savePredictionArray=False, savePredictionLabel=False, postfix="",
                  markers=[ VizMarkers.M_RAMIFIED, VizMarkers.M_AMOEBOID, VizMarkers.M_GARBAGE ],
                  isRaw=False, z_damp=0, usv_model_id=None, usv_model=None, progSignal=None,
                  rescale_input=False, in_data_size=None):

        self.predict(full_inference,
                     savePredictionArray=savePredictionArray, savePredictionLabel=savePredictionLabel,
                     postfix=postfix, z_damp=z_damp,
                     usv_model_id=usv_model_id, usv_model=usv_model, progSignal=progSignal,
                     rescale_input=rescale_input, in_data_size=in_data_size)

        self.mark_predictions(markers, isRaw)
        self.is_prediction_run = True


    def build_dataset(self, full_inference, custom_input_size=None):
        """build dataLoader for inference, from generated data by Cell Center Detector"""
        self.prt.p("full inference? {}".format(full_inference), self.prt.STATUS)

        mdb = {}
        phases = self.trParams.phase.split('_')
        centroids = self.dg.get_all_centroids() if full_inference else self.mRawPoints

        if not self.in_memory:
            # only generate data if not in in_memory mode
            # generate input data for inference, center-cropped by given cell center coordinates"""
            num_samples = len(centroids)
            print("save_path: ", self.get_save_path())
            files = glob(os.path.join(self.get_save_path(), "*", "*.npy"))
            if len(files) > 0:
                self.prt.p("MP:build_dataset():Data has already been generated...(%d/%d)"%(len(files), num_samples),
                           self.prt.WARNING)
            else:
                self.prt.p("MP:build_dataset():Start generating (%d) samples: "%num_samples, self.prt.STATUS)
                self.dg.generate_data_sequential(cc_list=centroids)

        if custom_input_size is not None:
            data_d, data_h, data_w = custom_input_size
        else:
            data_d, data_h, data_w = [self.trParams.data_d, self.trParams.data_w, self.trParams.data_w]

        # createt dataset
        mdb['name'] = self.trParams.dataset
        print("self.trParams.data_path: ", self.trParams.data_path)
        for p in phases:
            if self.trParams.dataset == Dataset.TRAP:
                d = TrapDataset(data_path=self.trParams.data_path,
                                ext=self.trParams.file_ext,
                                phase=p,
                                num_clusters=self.trParams.num_clusters,
                                preprocessor=self.prep,
                                with_labels=False,
                                return_fname=True,
                                #data_size=[8, 16, 16], # half-the size
                                data_size=[int(data_d/2),
                                           int(data_h/2),
                                           int(data_w/2)],
                                mdg=self.dg,
                                centroids=centroids)
            elif self.trParams.dataset == Dataset.MICROGLIA:
                d = MicrogliaDataset(self.trParams.data_path, self.trParams.file_ext, p,
                                     self.trParams.num_clusters, self.prep, with_labels=False, return_fname=True,
                                     ntype=trParams.norm_type, clip=trParams.clip_data)

            else: # Real-Time Dataset
                print("self.trParams.data(d, h, w): %d, %d, %d"%(data_d, data_h, data_w))
                d = RealTimeDataset(data_path=self.trParams.data_path,
                                    ext=self.trParams.file_ext,
                                    phase=p,
                                    num_clusters=self.trParams.num_clusters,
                                    preprocessor=self.prep,
                                    with_labels=False,
                                    return_fname=True,
#                                    data_size=[8, 16, 16], # half-the size
                                    data_size=[int(data_d/2),
                                               int(data_h/2),
                                               int(data_w/2)],
                                    mdg=self.dg,
                                    centroids=centroids,
                                    ntype=self.trParams.norm_type, clip=self.trParams.clip_data)

            print("Total [ {} ] data found in [ {} ] phase (with_label={}).".format(len(d), d.phase, False))
            mdb[p] = d
            print("mdb: ", mdb)

        return mdb


    def build_infer_module(self, mdb=None, model_type=None, full_inference=None,
                            skip_mdb_for_now=False, data_size=None):
        """build neural network and trainer module"""

        assert model_type is not None

#        if self.mdb is None:
#            self.mdb = self.build_dataset(full_inference)
        if not skip_mdb_for_now:
            if mdb is None:
                mdb = self.build_dataset(full_inference, data_size)

        self.mdb = mdb
        # get trainer type
        if model_type == ModelType.BMTR:
            self.predictor = Trainer("BMTR-MP", self.trParams, self.mdb)
        elif model_type == ModelType.BFMC:
            self.predictor = BFMCTrainer(name="BFMC-MP", params=self.trParams, dataset=self.mdb)
        elif model_type == ModelType.ALTR:
            self.predictor = MTrainer(name="ALTR", params=self.trParams, dataset=self.mdb)
        elif model_type == ModelType.KMNS:
            self.predictor = BMClustering()
        else:
            self.log("Unknown Model Type [ %s ]!"%model_type, self.prt.ERROR)
            raise UnknownModelTypeError


    def predict_a_batch(self, full_inference, outlayer=None,
                        phase=Phase.REALTIME, nb=1, rescale_input=False,
                        in_data_size=None):
        activation = {}
        hooks = {}

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        #ae_file_split = self.trParams.ae_weights.split('/')
        #model_name = ae_file_split[-1].split('_')[0]
        model_type = self.trParams.model_type
        print("model_type: ", model_type)

        # build dataset
        if self.mdb is None:
            self.mdb = self.build_dataset(full_inference)

        print("self.mdb: ", self.mdb)
        # build network and trainer
        if self.predictor is None:
            self.build_infer_module(mdb=self.mdb, model_type=model_type, data_size=in_data_size)

        # set eval mode
        self.predictor.net.eval()
        self.predictor.curr_phase = phase
        print("self.predictor.nbatches.keys: ", self.predictor.n_batches)
        n_batches = self.predictor.n_batches[phase]
        print("n_batches: ", n_batches)
        data_iter = iter(self.predictor.data_loader[phase])

        if outlayer is not None:
            ols, olnames = outlayer
            for ol, olname in zip(ols, olnames):
                hook = ol.register_forward_hook(get_activation(olname))
                hooks[olname] = hook

        enc_preds = None
        encoded_feat = None
        resnet_feat_npy = None
        decoded_output = None
        fnames = None

        niter = 0
        with torch.no_grad():
            for di in tqdm(range(n_batches), desc=phase):
                while niter < nb:
                    # Get Input
                    if model_type in [ModelType.BMTR, ModelType.BFMC, ModelType.ALTR]:
                        fnames, inputs, _ = next(data_iter)
                        inputs, labels = self.predictor.form_net_inputs(inputs, rescale=rescale_input)
                    else:
                        # TRAP
                        inputs_cpu = next(data_iter)
                        fnames, inputs, labels = self.predictor.form_net_inputs(inputs_cpu, rescale=rescale_input)
                    # get numpy for saving
                    inputs_npy = bmUtil.t2npy(inputs).squeeze()
                    print("inputs_npy.shape: ", inputs_npy.shape)

                    zproj_feat = None
                    
                    # Inference
                    if model_type in [ModelType.BMTR, ModelType.ALTR]:
                        encoded_feat, decoded_output, zproj_feat = self.predictor.net(inputs, deconv=True)

                        resnet_feat_npy = bmUtil.t2npy(decoded_output[0])
                        print("resnet_feat_npy.shape: ", resnet_feat_npy.shape)
                        yz_cu = torch.argmax(encoded_feat, dim=1)
                        yz = bmUtil.t2npy(yz_cu)
                    elif model_type == ModelType.BFMC:
                        _, _, yhat = self.predictor.net(inputs)
                        yz_cu = torch.argmax(yhat, dim=1)
                        yz = bmUtil.t2npy(yz_cu)
                    elif model_type == ModelType.TRAP:
                        encoded_feat, decoded_output, yhat = self.predictor.net(inputs, cluster=True)
                        yz_cu = torch.argmax(encoded_feat, dim=1)
                        yz = bmUtil.t2npy(yz_cu)
                    else:
                        raise UnknownModelTypeError

                    #enc_preds = bmUtil.bmAppend(enc_preds, yz)
                    enc_preds = yz
                    niter += 1

            self.prt.p("Done Prediction...", self.prt.LOG)

            for k in hooks.keys():
                hooks[k].remove()
        return inputs_npy, encoded_feat, enc_preds, resnet_feat_npy, decoded_output, activation, fnames, zproj_feat


    def predict(self, full_inference, savePredictionArray=False, savePredictionLabel=False, postfix="",
                z_damp=0, usv_model_id=None, usv_model=None, progSignal=None,
                rescale_input=False, in_data_size=None):
        """run inference model using data provided

        :progSignal: for MorPheT GUI - progress bar update signal

        """
        print("save array? ", savePredictionArray)
        print("save label? ", savePredictionLabel)
        print("Unsupervised model: ", usv_model)
        print("in_data_size: ", in_data_size)

        ae_file_split = self.trParams.ae_weights.split('/')
        print("ae_file_split: ", ae_file_split)
        #model_name = ae_file_split[-1].split('_')[0]
        model_type = self.trParams.model_type
        print("model_type: ", model_type)

        model_subcls = None
        preds_subcls = None
        if usv_model is not None:
            model_subcls = BMClustering()
            model_subcls.load_model(usv_model)

        df = None

        run_inference = True
        phase = Phase.REALTIME
        if savePredictionArray:
            pred_save_dir = "%s/prediction/%s"%(self.get_save_path(), postfix)
            print("PRED_SAVE_DIR: ", pred_save_dir)
            bmUtil.CHECK_DIR(pred_save_dir)
            pred_cnts = []
            for i in range(self.trParams.num_clusters):
                bmUtil.CHECK_DIR("%s/%d"%(pred_save_dir, i))
                pred_cnts.append(0)
            num_files_per_dir = 10000

        if savePredictionLabel:
            self.csv_pred_name = '-'.join([model_type] + ae_file_split[-2].split('-')[:2])
            print("self.csv_pred_name: ", self.csv_pred_name)
            if usv_model is not None:
                self.csv_pred_subcls_name = usv_model_id
                print("self.csv_pred_subcls_name: ", self.csv_pred_subcls_name)

            else:
                self.csv_pred_subcls_name = None

            df = pd.read_csv(self.cc_csv)

            if self.csv_pred_name in df.columns:
                df_not_labeled = df.loc[(df[self.csv_pred_name] == -1)]
                print("len(df_not_labeled): ", len(df_not_labeled))
                if len(df_not_labeled) < 10 and not self.trParams.force_inference:
                    self.prt.p("Every centroid is already labeled! Directly visualizing stuff..", self.prt.STATUS2)
                    run_inference = False
#            else:
#                df[self.csv_pred_name] = -1
#                if self.csv_pred_subcls_name is not None:
#                    df[self.csv_pred_subcls_name] = -1

        if run_inference:
            # build dataset
            mdb = self.build_dataset(full_inference, in_data_size)

            # build network and trainer
            self.build_infer_module(mdb, model_type, True, data_size=in_data_size)

            # set eval mode
            self.predictor.net.eval()
            self.predictor.curr_phase = phase
            n_batches = self.predictor.n_batches[phase]
            data_iter = iter(self.predictor.data_loader[phase])

            if savePredictionLabel:
                pred_dict = {'z': [], 'y': [], 'x': [], self.csv_pred_name: []}
                if self.csv_pred_subcls_name is not None:
                    pred_dict[self.csv_pred_subcls_name] = []

            enc_preds = None
            with torch.no_grad():
                self.log("Running inference...# batches: [ %d ]"%n_batches, self.prt.STATUS2)
                for di in tqdm(range(n_batches), desc=phase):
                    if progSignal:
                        progSignal.emit(di+1, n_batches)
                    # Get Input
                    if model_type in [ModelType.BMTR, ModelType.BFMC, ModelType.ALTR]:
                        fnames, inputs, _ = next(data_iter)
                        inputs, labels = self.predictor.form_net_inputs(inputs, rescale=rescale_input)
                    else:
                        # TRAP
                        inputs_cpu = next(data_iter)
                        fnames, inputs, labels = self.predictor.form_net_inputs(inputs_cpu, rescale=rescale_input)
                    # get numpy for saving
                    inputs_npy = bmUtil.t2npy(inputs).squeeze()

                    # Inference
                    if model_type in [ModelType.BMTR, ModelType.ALTR]:
                        encoded_feat, decoded_output, middle_feat = self.predictor.net(inputs, deconv=True)
                        resnet_feat_npy = bmUtil.t2npy(decoded_output[0])
                        dz_feat_npy = bmUtil.t2npy(decoded_output[1])
                        yz_cu = torch.argmax(encoded_feat, dim=1)
                        yz = bmUtil.t2npy(yz_cu)
                        if model_subcls is not None:
                            preds_subcls = model_subcls.predict(dz_feat_npy)

                    elif model_type == ModelType.BFMC:
                        _, _, yhat = self.predictor.net(inputs)
                        yz_cu = torch.argmax(yhat, dim=1)
                        yz = bmUtil.t2npy(yz_cu)
                    elif model_type == ModelType.TRAP:
                        encoded_feat, decoded_output, yhat = self.predictor.net(inputs, cluster=True)
                        yz_cu = torch.argmax(encoded_feat, dim=1)
                        yz = bmUtil.t2npy(yz_cu)
                    else:
                        raise UnknownModelTypeError

                    enc_preds = bmUtil.bmAppend(enc_preds, yz)

                    # save
                    if savePredictionArray or savePredictionLabel:
                        fns = []
                        fns_r = []
                        fns_dz = []

                        for idx in range(len(yz)):
                            g_idx = di * inputs.size(0) + idx
                            cluster_id = yz[idx]
                            if model_subcls is not None:
                                subcls_id = preds_subcls[idx]
                            z, y, x = self.dg.retrieve_zyx_from_filename(fnames[idx])
                            if savePredictionLabel:
                                pred_dict['z'].append(z)
                                pred_dict['y'].append(y)
                                pred_dict['x'].append(x)
                                pred_dict[self.csv_pred_name].append(cluster_id)
                                #df.loc[(df['z'] == z) & (df['y'] == y) & (df['x'] == x), self.csv_pred_name] = cluster_id
                                if self.csv_pred_subcls_name is not None:
                                    pred_dict[self.csv_pred_subcls_name].append(subcls_id)
                                    #df.loc[(df['z'] == z) & (df['y'] == y) & (df['x'] == x), self.csv_pred_subcls_name] = subcls_id

                            if savePredictionArray:
                                pred_cnts[cluster_id] += 1
                                subdir = "%06d"%(pred_cnts[cluster_id] / num_files_per_dir)
                                tgtdir = os.path.join(pred_save_dir, str(cluster_id), subdir)
                                bmUtil.CHECK_DIR(tgtdir)

                                fname = os.path.join(tgtdir, "%s_%06d_pd-%d_feat-zyx-%d-%d-%d.npy"%(phase, g_idx, cluster_id, z, y, x))
                                fns.append(fname)

                                fname = os.path.join(tgtdir, "%s_%06d_pd-%d_feat-zyx-%d-%d-%d_rnfeat.npy"%(phase, g_idx, cluster_id, z, y, x))
                                fns_r.append(fname)

                                fname = os.path.join(tgtdir, "%s_%06d_pd-%d_feat-zyx-%d-%d-%d_dzfeat.npy"%(phase, g_idx, cluster_id, z, y, x))
                                fns_dz.append(fname)

                        if savePredictionArray:
                            # now save everything
                            resnet_feat_npy = resnet_feat_npy.squeeze()
                            all_fns = fns + fns_r + fns_dz
                            self.save_data_to_disk_mp(all_fns, [inputs_npy, resnet_feat_npy, dz_feat_npy])

            self.prt.p("Done Prediction...", self.prt.LOG)

            # formulate coordinates per class, and update csv if needed
            if savePredictionLabel:
                drdr = os.path.dirname(self.cc_csv)
                df_pred = pd.DataFrame.from_dict(pred_dict)
                new_csvf = os.path.join(drdr, "cc_csv_df_pred.csv")
                df_pred.to_csv(new_csvf, sep=',', index=False)
                self.prt.p("Saved new prediction into csv [%s]"%(new_csvf), self.prt.LOG)

                df[self.csv_pred_name] = df_pred[self.csv_pred_name]
                if self.csv_pred_subcls_name is not None:
                    df[self.csv_pred_subcls_name] = df_pred[self.csv_pred_subcls_name]

                df.to_csv(self.cc_csv, sep=',', index=False)
                self.prt.p("Saved labels into csv [%s]"%(self.cc_csv), self.prt.LOG)

                # merge pred_dict with df
                #self.prt.p("Merging pred_dict with df...", self.prt.LOG)
                #df = df.merge(df_pred, on=['z', 'y', 'x'])
                #self.prt.p("Merging pred_dict with df...(Done)", self.prt.LOG)
                #df.to_csv(os.path.join(drdr, "Merged.csv"), sep=',', index=False)

        if df is not None:
            self.prediction_points, self.subclasses = self.get_points_per_class(df, z_damp)

    def save_data_to_disk_mp(self, fns, data_list, ncpu=32):
        freeze_support()
        data = []
        for item in data_list:
            for npydata in item:
                data.append(npydata)

        ns = len(fns)
        try:
            assert ns == len(data)
        except:
            print("ERROR! ns: ", ns, "len(data): ", len(data))
            raise AttributeError

        bsz = int(np.ceil(ns / ncpu))

        fns_chunks = [fns[i*bsz:(i+1)*bsz] for i in range(ncpu)]
        data_chunks = [data[i*bsz:(i+1)*bsz] for i in range(ncpu)]
        full_args = zip(fns_chunks, data_chunks)
        with Pool(processes=ncpu) as pool:
            pool.starmap(io_worker, full_args)



    def get_points_per_class(self, dataframe=None, z_damp=0):
        """this is needed becuase cropped data can be generated earlier and
            doesn't necessarilly follow the order in cell_coordinates list
            by the time of prediction/save"""

        zr, yr, xr = self.ranges
        prediction_points = {}
        subclasses = {}
        for i in range(self.trParams.num_clusters):
            prediction_points[i] = []
            subclasses[i] = []

        if dataframe is not None:
            for i in range(self.trParams.num_clusters):
                self.prt.p("[Morphology Predictor] retrieving predictions for class %d"%i, self.prt.STATUS2)
                df = dataframe.loc[(dataframe[self.csv_pred_name] == i)]
                dfc = df[['z', 'y', 'x']]
                coords = dfc.values.tolist()

                # check sub-phenotype
                handle_subcol = True
                subcol = '%s_sub'%self.csv_pred_name
                if subcol in df.columns:
                    dfs = df[subcol]
                    num_nan = dfs.isnull().sum().sum()
                    if num_nan > 0:
                        handle_subcol = False

                handle_subcol = False
                if handle_subcol:
                    subcl_list = np.array(dfs.values.tolist(), dtype=int)

                if zr is None and yr is None and xr is None:
                    prediction_points[i] = np.array(coords)
                else:
                    coords_in_range = self.dg.get_coords_in_range(zr, yr, xr, np.array(coords))
                    self.log("[ class %d ]: (%d / %d) selected"%(i, len(coords_in_range),
                                                               self.dg.get_num_centers()),
                                                               self.prt.LOG)
                    for cir in coords_in_range:
                        if handle_subcol:
                            index = [cir_idx for cir_idx, crd in enumerate(coords) if crd[0]==cir[0] and crd[1]==cir[1] and crd[2]==cir[2]]
                            assert len(index) == 1
                            subclasses[i].append(subcl_list[index[0]])

                        rel_coord = BaseAnalyzer.adjust_coordinates(deepcopy([cir]), zr, yr, xr, z_damp)[0]
                        prediction_points[i].append(rel_coord)

        else:
            ppath = self.get_save_path()
            files = sorted(glob(ppath + "/*.npy"))

            for i in range(self.trParams.num_clusters):
                self.prt.p("retrieving predictions for class %d"%i, self.prt.STATUS2)
                subpath = ppath + "/prediction/%d"%i
                subfiles = sorted(glob(subpath +"/*.npy"))
                for sf in subfiles:
                    sample_id = int(sf.split('/')[-1].split('_')[1])
                    coord = [int(x) for x in files[sample_id].split('/')[-1].split('.')[0].split('-')[1:]]
                    rel_coord = BaseAnalyzer.adjust_coordinates(deepcopy([coord]), zr, yr, xr, z_damp)[0]
                    prediction_points[i].append(rel_coord)

        return prediction_points, subclasses


    def mark_predictions(self, markers, isRaw=False):
        print("num_clusters: ", self.trParams.num_clusters)
        """overlay centers with predictions"""
        main_colors = [
            [0.92, 0.098, 0.15],
            [0.098, 0.94, 0.15],
            [0.0, 1.0, 1.0],
            [0.3, 0.5, 1.0],
            [0.9, 0.5, 0.7],
            [0.3, 0.3, 0.9],
        ]
        bindings = [
            "a","b","c", "d", "e", "f", "g", "h", "i", "j", "k",
        ]


        if True:
            labels = {}
            for i in range(len(markers)):
                print("Labels i: ", i)
                c = main_colors[i]
                c.append(self.marker_alpha)
                labels[i] = {"marker":markers[i], "binding":bindings[i], "color": c, "symbol": 'disc', 'ssz': self.marker_size}
        else:
            labels = {
    #                    0: {"marker":markers[0], "binding":"r", "color": [0.92, 0.098, 0.29, 0.9], "symbol": 'disk', 'ssz': 15},
    #                    1: {"marker":markers[1], "binding":"a", "color": [0.0, 1.0, 1.0, 0.9], "symbol": 'disc', 'ssz': 8},
    #                    2: {"marker":markers[2], "binding":"e", "color": [0.0, 1.0, 0.0, 0.7], "symbol": 'x', 'ssz': 11 }
                        0: {"marker":markers[0], "binding":"r", "color": [0.92, 0.098, 0.15, self.marker_alpha], "symbol": 'disc', 'ssz': self.marker_size},
                        1: {"marker":markers[1], "binding":"a", "color": [0.098, 0.94, 0.15, self.marker_alpha], "symbol": 'disc', 'ssz': self.marker_size},
                        2: {"marker":markers[2], "binding":"e", "color": [0.0, 1.0, 1.0, self.marker_alpha], "symbol": 'x', 'ssz': self.marker_size}
                      }

        for i in range(self.trParams.num_clusters):
            points = np.array(self.prediction_points[i])
            if isRaw:
                points[:, 1] *= 2
                points[:, 2] *= 2

            if len(points):
                l = labels[i]
                if False:
                #if i == 2:
                    subclasses = np.array(self.subclasses[i])
                    print("len(points): ", len(points), "len(subclasses): ", len(subclasses))

                    subbindings = ["h", "j", "k", "l"]
                    subcolors= [[1.0, 0.0, 1.0, 0.7], [1.0, 1.0, 0.0, 0.7], [1.0, 0.0, 0.3, 0.7], [0.3, 0.7, 1.0, 0.7]]
                    if len(subclasses):
                        for si in range(int(np.max(subclasses)) + 1):
                            sub_points = points[np.where(subclasses==si)]
                            self.overlay_marker(sub_points, "subclass %d of %s"%(si, l["marker"]),
                                                subbindings[si], subcolors[si], size=8)

                self.overlay_marker(points, l["marker"], l["binding"], color=l["color"], alpha=0.9, size=l['ssz'], symbol=l["symbol"])


    def switch_volume(self):
        if self.curr_vol == VizMarkers.GFP_MICROGLIA:
            self.set_volume(self.clusterVol[0].copy())
            self.volume.cmap = 'hot'
            self.curr_vol = VizMarkers.M_RAMIFIED
        elif self.curr_vol == VizMarkers.M_RAMIFIED:
            self.set_volume(self.clusterVol[1].copy())
            self.curr_vol = VizMarkers.M_AMOEBOID
            self.volume.cmap = 'hot'
        elif self.curr_vol == VizMarkers.M_AMOEBOID:
            self.set_volume(self.clusterVol[2].copy())
            self.curr_vol = VizMarkers.M_GARBAGE
            self.volume.cmap = 'hot'
        elif self.curr_vol == VizMarkers.M_GARBAGE:
            self.set_volume(self.mVol.copy())
            self.curr_vol = VizMarkers.GFP_MICROGLIA
            self.volume.cmap = 'grays'


    def update_clim_max(self, decrease=False):
        if decrease:
            self.clim_max -= 150
        else:
            self.clim_max += 150


    def update_clim_min(self, decrease=False):
        if decrease:
            self.clim_min -= 150
        else:
            self.clim_min += 150


    def on_key_press(self, event):
        for key in self.key_bindings:
            if event.text == key:
                self.toggle_marker(self.key_bindings[key])

        if event.text in ["5", "6"]:
            if event.text == "5":
                self.marker_size -= 1
            else:
                self.marker_size += 1

            print("new marker_size: ", self.marker_size)

        if event.text == 's':
            self.set_volume_style(cmapToggle=True)

        if event.text in ["1", "2", "3", "4"]:
            if event.text == "1":
                self.update_clim_max()
            elif event.text == "2":
                self.update_clim_max(decrease=True)
            elif event.text == "3":
                self.update_clim_min()
            elif event.text == "4":
                self.update_clim_min(decrease=True)
            print("setting clim_min: {}, clim_max: {}".format(self.clim_min, self.clim_max))
            self.set_volume(self.mVol.copy(), clim=(self.clim_min, self.clim_max), scalebar=True)
            print("setting clim_min: {}, clim_max: {}(Done)".format(self.clim_min, self.clim_max))

        if event.text == 'x':
            self.cAxis.visible = not self.cAxis.visible



    @staticmethod
    def _grab_cube_by_point(p, vol, new_vol, cubsize=(16, 32, 32)):
        nv = new_vol.copy()
        vz, vy, vx = vol.shape
        #ddh, dhh, dwh = [x/2 for x in cubsize]
        ddh, dhh, dwh = cubsize
        zs, ze = (max(0, p[0] - ddh), min(p[0] + ddh, vz))
        ys, ye = (max(0, p[1] - ddh), min(p[1] + ddh, vy))
        xs, xe = (max(0, p[2] - ddh), min(p[2] + ddh, vx))

        for i in range(zs, ze):
            for j in range(ys, ye):
                for k in range(xs, xe):
                    nv[i][j][k] = vol[i][j][k]

        return nv


def build_args(channel, checkpoint, params_f=None, params_dict=None):
    try:
        if params_dict is None:
            assert params_f is not None
            with open(params_f) as fp:
                params_dict = yaml.safe_load(fp)
        else:
            assert params_dict is not None

        print("channel: ", channel)
        params = Namespace(**params_dict)
        paramsByChn = Namespace(**params_dict[channel])
    except IOError:
        print("Check parameter file again and re-run please.")
        sys.exit(1)
    except KeyError:
        print("Check channel (%s) and re-run please."%channel)
        sys.exit(1)

    now = datetime.datetime.now()
    current_time = now.strftime("%Y%m%d-%H%M") if checkpoint is None else checkpoint

    #rel_path = os.path.join(paramsByChn.rel_path, params.tif_rel_path)
    #save_path = os.path.join(params.d_root, paramsByChn.rel_path, params.inf_rel_path)
    #argsByChn = [ '-dp', rel_path,
    #             '-ac', os.path.join(paramsByChn.rel_path, paramsByChn.cc_npy) ]
    #rel_path = os.path.join(paramsByChn.tif_rel_path)
    datatype = '%s_%s'%(params.name, channel)
    save_path = os.path.join(params.d_root, params.inf_rel_path)
    try:
        cc_npy = paramsByChn.cc_npy
        cc_csv = os.path.join(params.d_root, paramsByChn.cc_csv)
    except AttributeError:
        # no cc available yet
        cc_npy = None
        cc_csv = None

    argsByChn = [ '-dp', paramsByChn.tif_rel_path, '-ac', cc_npy ]

    gargs = [ sys._getframe().f_code.co_name,
              '-dr', params.d_root, '-dt', datatype, '-sp', save_path,
              '-ts', current_time, '-dw', str(params.dw), '-dh', str(params.dh),
              #'-ch', str(params.ch),
              #'-cq', str(params.cq),
              '-ext', str(params.file_ext)]
#              '--debug' ]

    ranges = [params.zr, params.yr, params.xr]
    full_inference = params.full_inference
    clim = paramsByChn.clim

    return params_dict, argsByChn, gargs, ranges, cc_csv, full_inference, clim


def main(params_f, model_type=ModelType.BMTR, channel=StainChannel.GFP,
         checkpoint=None, in_memory=False, no_viz=False):

    # load parameters from .json file
    params, \
    argsByChn, \
    gargs, \
    ranges, \
    cc_csv, \
    full_inference, \
    clim = build_args(channel, checkpoint, params_f=params_f)

    # create predictor
    mp = MorphologyPredictor(title="MorphologyPredictor", keys='interactive', size=(800, 600),
                             show=True, logWindow=None,
                             gargs=gargs, margs=argsByChn, ranges=ranges,
                             in_memory=in_memory,
                             cc_csv=cc_csv)

    if not no_viz:
        # STEP 1. Load data as volume and retrieve cell centers from files
        mp.load_data(clim)

    # STEP 2. Setup Trainer for inference
    # TODO: create another type of Params()
    if model_type == ModelType.BFMC:
        dset = Dataset.MICROGLIA
        num_clusters = '3'
        if True:
            EXPR="20190221-124316-bumblebee"
            pretrained = '/data_ssd2/weights/models/%s/BFMC_Phase2_ALL_00098.pth'%(EXPR)
        elif True:
            EXPR="20190123-114107-bumblebee"
            pretrained = '/data_ssd/weights/models/%s/BFMC_Phase2_ALL_00021.pth'%(EXPR)
        elif True:
            EXPR="20190118-152411-bumblebee"
            pretrained = '/data_ssd/weights/models/%s/BFMC_Phase2_ALL_00279.pth'%(EXPR)

    elif model_type == ModelType.BMTR:
        dset = Dataset.REALTIME
        if True:
            EXPR="20190312-170708-bumblebee"
            pretrained = '/media/share12/MYK/models/microglia/%s/BMTR_AE_00200.pth'%(EXPR)
            num_clusters = '3'
        elif True:
            EXPR="20190313-125936-bumblebee"
            pretrained = '/data_ssd2/weights/models/%s/BMTR_ClusterTraining_00000.pth'%(EXPR)
            num_clusters = '3'
        elif True:
            #EXPR="20180919-154658-bumblebee"
            #EXPR="20180830-140211-bumblebee"
            EXPR="20181022-185608-bumblebee"
            pretrained = '/data_ssd/weights/models/%s/BMTR_AE_00099.pth'%(EXPR)
            num_clusters = '3'
        else:
            EXPR="20180827-172754-bumblebee"
            #EXPR="20180906-142038-bumblebee"
            pretrained = '/home/mykim/Bumblebee/data/weights/models/%s/BMTR_AE_00095.pth'%(EXPR)
            num_clusters = '2'

        norm_type=NormalizationType.ZERO_AND_ONE
        clip_data=True
    elif model_type == ModelType.TRAP:
        # TRAP
        from pathlib import Path
        homedir = str(Path.home())
        dset = Dataset.TRAP
        TRAIN_ROOT = os.path.join(homedir, "cbm/src/train")
        num_clusters = '2'
        #EXPR="20200526-132807-bumblebee"
        EXPR="20200705-130651-bumblebee"
        pretrained = os.path.join(TRAIN_ROOT, "weights", "models", EXPR, "TRAP_AE_00099.pth")

    elif model_type == ModelType.ALTR:
        dset = Dataset.REALTIME
        EXPR="20210914-234831-bumblebee"
        pretrained = '/media/share12/MYK/models/microglia/%s/ALTR_AE_00063.pth'%(EXPR)
        num_clusters = '3'
        norm_type=NormalizationType.ZERO_AND_ONE
        clip_data=True
    else:
        raise UnknownModelTypeError

    print("model_type: ", model_type)
    if dset == Dataset.MICROGLIA:
        args_infer = ['morphologyPredictor', '-ph', Phase.REALTIME, '-bs', '10', '-e', '2',
                        '-ts', Phase.REALTIME, '-nc', num_clusters, '-ds', dset, '-us', 'False',
                        '-aw', pretrained, '-mt', model_type, '-dp', mp.get_save_path(), '-ie', 'True', '--debug']
    else:
        # TRAP or REALTIME
        patch_sz = params['inf_patch_size']
        print("patch size: ", patch_sz)
        args_infer = ['morphologyPredictor', '-ph', Phase.REALTIME, '-bs', '32', '-e', '2',
                        '-ts', Phase.REALTIME, '-nc', num_clusters, '-ds', dset, '-us', 'False', '-usv', 'False',
                        '-dw', str(patch_sz[2]), '-dh', str(patch_sz[1]), '-dd', str(patch_sz[0]),
                        '-aw', pretrained, '-mt', model_type, '-dp', mp.get_save_path(), '-ie', 'True',
                        '-cl', clip_data, '-nt', norm_type, '--debug']


    print("args_infer: ", args_infer)
    mp.setup_model(args_infer, full_inference)

    # STEP 3. Run Prediction and visualize results
    mp.run(full_inference, savePredictionArray=False, savePredictionLabel=True)


if __name__ == '__main__':

    p = MorphPredictionParams()
    p.build(sys.argv, "MPParser")
    main(p.param_file, p.model_type, p.channel, p.checkpoint,
         in_memory=True, no_viz=p.do_not_visualize)
    app.run()
