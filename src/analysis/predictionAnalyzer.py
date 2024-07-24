"""predictionAnalyzer.py: contains PredictionAnalyzer Class"""
__author__      = "Minyoung Kim"
__license__ = "MIT"
__maintainer__ = "Minyoung Kim"
__email__ = "minykim@mit.edu"
__date__ = "01/30/2019"

import os
import pandas as pd
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import json
from copy import deepcopy

import torch
import torchvision.utils as vutils

from utils.data.microgliaDB import MDBLabel
from .baseAnalyzer import BaseAnalyzer


class PredictionAnalyzer(object):
    """PredictionAnalyzer Class
        - contains statistical functions for prediction analysis
    """
    def __init__(self, droot, channel, num_class=3,
                       labels={0: "Ramified", 1: "Amoeboid", 2: "Garbage"}):
        self.dr = droot
        self.chn = channel
        self.nc = num_class
        self.labels = labels
        self.params = None
        self.params_f = None
        self.MLABEL = MDBLabel()


    def update_data_root(self, data_path):
        self.dr = data_path


    def config_params(self, dpath=None):
        try:
            params_f = glob("%s/params.json"%self.dr)[0]
            if params_f == self.params_f:
                print("same file!, do nothing!")
                return

            with open(params_f) as fp:
                self.params = json.load(fp)
            self.params_f = params_f
        except:
            print("Error loading json file!")

        if dpath is None:
#            self.data_path = os.path.join(self.dr,
#                                          self.params[self.chn]['tif_rel_path'],
#                                          self.params['inf_rel_path'],
#                                          '%s_%s'%(self.params['name'], self.chn),
#                                          self.params[self.chn]['inf_dpath'])
            self.data_path = os.path.join(self.dr,
                                          self.params['inf_rel_path'],
                                          '%s_%s'%(self.params['name'], self.chn))

        else:
            self.data_path = os.path.join(self.dr, dpath)

        print("[PredictionAnalyzer]: data_path: ", self.data_path)
        self.csvf = os.path.join(self.dr, self.params[self.chn]['cc_csv'])

        # load data frame from .csv
        self.df = pd.read_csv(self.csvf)


    def get_param_dict(self):
        return self.params


    def update_dataframe(self):
        # in case csv file has updated outside
        self.df = pd.read_csv(self.csvf)


    def load_predictions(self, models):
        self.predictions = {}
        try:
            for m in models:
                ps = [self.df.loc[(self.df[m] == i)] for i in range(self.nc)]
                self.predictions[m] = ps
                print("[{}] class distribution: {}".format(m, [len(ps[i]) for i in range(self.nc)]))

        except KeyError:
            # TEMPORARY
            self.predictions[m] = [ self.df, [], [] ]
            print("[{}] class distribution: {}".format(m, [len(self.predictions[m][i]) for i in range(self.nc)]))


    def load_agreement(self, m1, m2):
        self.df_agreed = self.df.loc[self.df[m1] == self.df[m2]]
        self.df_disagreed = self.df.loc[self.df[m1] != self.df[m2]]
        assert(len(self.df) == len(self.df_agreed) + len(self.df_disagreed))


    def draw_pie(self, m, nc=None, ax=None):
        #explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
        sizes = [len(self.predictions[m][i]) for i in range(self.nc)]
        up_to = len(sizes) if nc is None else nc

        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(aspect="equal"))
        p, tx, autotexts = ax.pie(sizes[:up_to],
                                  labels=list(self.labels.values())[:up_to],
                                  colors = ['lightcoral', 'gold', 'yellowgreen', 'lightskyblue'],
                                  autopct='%1.1f%%',
                                  shadow=True,
                                  wedgeprops={"edgecolor":"k",'linewidth': 1, 'antialiased': True},
                                  startangle=90)

        for i, a in enumerate(autotexts):
            t = a.get_text()
            a.set_text("{}\n({:,})".format(t, sizes[i]))

        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        #ax.set_title("[{} | {}]".format(self.params['age'], m))
        ax.set_title("[ {} ]".format(self.params['age']))

        if ax is None:
            plt.tight_layout()
            plt.show()


    @staticmethod
    def get_img(f):
        data_tensor = torch.from_numpy(np.load(f))
        data_extend = data_tensor.unsqueeze(1)
        x = vutils.make_grid(data_extend, normalize=True, scale_each=True, nrow=8)
        return np.transpose(x.numpy(), (1, 2, 0))


    def get_points_per_class(self, dg, model_name):
        prediction_points = {}
        for i in range(self.nc):
            prediction_points[i] = []
        zr = self.params['zr']
        yr = self.params['yr']
        xr = self.params['xr']

        for i in range(self.nc):
            print("retrieving predictions for class %d"%i)
            df = self.df.loc[(self.df[model_name] == i)]
            df = df[['z', 'y', 'x']]
            coords = df.values.tolist()

            if zr is None and yr is None and xr is None:
                prediction_points[i] = np.array(coords)
            else:
                coords_in_range = dg.get_coords_in_range(zr, yr, xr, np.array(coords))
                print("[ %s ]: (%d / %d) selected"%(self.MLABEL.get_name(i), len(coords_in_range),
                                                    dg.get_num_centers()))
                for cir in coords_in_range:
                    rel_coord = BaseAnalyzer.adjust_coordinates(deepcopy([cir]), zr, yr, xr)[0]
                    prediction_points[i].append(rel_coord)

        return prediction_points
