"""phenotypeHeatmap.py

 * Move functions form analysis Jupyter noteobook: 3.PhenotypeHeatmap.ipynb
"""
__author__      = "Minyoung Kim"
__license__ = "MIT"
__maintainer__ = "Minyoung Kim"
__email__ = "minykim@mit.edu"
__date__ = "04/21/2024"


from pathlib import Path
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt


class PhenotypeHeatmap(object):
    """PhenotypeHeatmap Class
        - contains functions for cell phenotype heatmap visualizaiton
    """
    def __init__(self):
        """init
        """
        return

    @staticmethod
    def create_heatmap(cells, vol_size, heatmap_size):
        d, h, w = vol_size
        bh, bw = heatmap_size
        heatmap = np.zeros((int(np.ceil(h/bh))+1, int(np.ceil(w/bw))+1))

        for c in cells:
            by, bx = [int(np.ceil(c[1]/bh)), int(np.ceil(c[2]/bw))]
            heatmap[by][bx] += 1

        return heatmap


    @staticmethod
    def compute_global_coordinate(zyx, params):
        z, y, x = zyx
        return (z + int(params['zr'][0]), y + int(params['yr'][0]), x + int(params['xr'][0]))


    @staticmethod
    def get_subcls_heatmaps(params, df, model_id, sv_crds,
                            model_dict, usv_model_id, vol_size, grid_size):

        usv_model_dict = model_dict[usv_model_id]
        subcls_ids = list(usv_model_dict['class'].keys())

        # initialize dictionary
        subcls_dict = {}
        for sid in subcls_ids:
            subcls_dict[int(sid)] = []

        # count cells per sub-class
        for crd in tqdm(sv_crds, "centroids"):
            zzg, yyg, xxg = PhenotypeHeatmap.compute_global_coordinate(crd, params)
            row = df[(df['z'] == zzg) & (df['y'] == yyg) & (df['x'] == xxg)]
            #assert row[model_id].values[0] == sv_cls_id # others = 2
            subcls = row[usv_model_id].values[0]
            subcls_dict[subcls].append(crd)

        heatmaps = {}
        # get heatmaps
        for key in subcls_dict.keys():
            subcls_crds = subcls_dict[key]
            print("sub-cls [ %d ] len: %d"%(key, len(subcls_crds)))
            heatmaps[key] = PhenotypeHeatmap.create_heatmap(subcls_crds, vol_size, grid_size)

        return heatmaps, subcls_dict


    @staticmethod
    def viz_subcls_heatmaps(heatmaps, vminmax, cms=None, figsize=(14,4), dpi=300):
        vmin, vmax = vminmax
        nc = len(heatmaps)
        if cms is None:
            cms = ['Reds', 'Greens', 'Blues', 'Greys', 'Reds', 'Greens', 'Blues', 'Greys', 'Reds', 'Greens', 'Blues', 'Greys', 'Reds', 'Greens', 'Blues', 'Greys']

        fig, axes = plt.subplots(figsize=figsize, ncols=nc, nrows=1, dpi=dpi)
        for idx, key in enumerate(heatmaps.keys()):
            hmap = heatmaps[key]
            r = sns.heatmap(hmap, cmap=cms[idx], vmin=vmin, vmax=vmax, ax=axes[idx])
            r.set_title("Heatmap of Sub-Class(%d)"%idx)
            axes[idx].axis('off')
            axes[idx].set_aspect('equal')

        fig.tight_layout()
        if False:
            figf = os.path.join(savepath,  "%s_phenotype_heatmap_Others_Subcls.png"%dname)
            print("figf: ", figf)
            fig.savefig(figf)
            print("fig saved")
        plt.show()
        return fig
