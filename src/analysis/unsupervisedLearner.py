"""unsupervisedLearner.py: Unsupervised Learning Class"""
__author__      = "Minyoung Kim"
__license__ = "MIT"
__maintainer__ = "Minyoung Kim"
__email__ = "minykim@mit.edu"
__date__ = "03/01/2024"

import os
import sys
import numpy as np
import torchvision.utils as vutils
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle


# internal
import utils.util as bmUtil
from utils.data.preprocessing import BMPreprocessing as bmPrep


class BMClustering(object):
    prt = bmUtil.PRT()
    CM = 'cividis'

    def __init__(self):

        self.feats = None
        self.cids = None
        self.feat1_idx = None
        self.feat2_idx = None
        self.kmeans = None
        mpl.rc('image', cmap=self.CM)


    def set_data(self, feats, cids):
        self.feats = feats
        self.cids = cids


    def print_params(self):
        # TODO: make a base class and inherit
        return bmUtil.print_class_params(self.__class__.__name__, vars(self))


    def set_feat_pair(self, feat1_idx, feat2_idx):
        self.feat1_idx = feat1_idx
        self.feat2_idx = feat2_idx


    @staticmethod
    def viz_subclusters(cids, inputs_npy, y_pred, sub_cls_id, ncr=(7, 2)):
        indices = np.where(y_pred==sub_cls_id)[0]
        num_sub_cls = len(indices)
        nc, nr = ncr
        is_oneRow = True if nr == 1 else False


        num_viz = nc * nr
        fig, axes = plt.subplots(figsize=(10, 2), ncols=nc, nrows=nr)

        for i in range(num_viz):
            if i >= len(indices):
                break
            # determine plot position

            if not is_oneRow:
                ri = int(i  / nc)
            ci = i % nc

            # get prediction idx w/ sub_cls_id specified
            idx = indices[i]

            # pull up cell_id
            cid = cids[idx]

            # pull up image
            cell_npy = inputs_npy[cid]
            cell_zproj = bmPrep._max_proj(cell_npy)

            if is_oneRow:
                axes[ci].imshow(cell_zproj)
                axes[ci].set_title("C#%d"%(cid))
            else:
                axes[ri][ci].imshow(cell_zproj)
                axes[ri][ci].set_title("cid: %d"%(cid))

        fig.suptitle("Sub Class: [ %d ] "%sub_cls_id, y=1.08)
        for ax in axes:
            if is_oneRow:
                ax.axis("off")
            else:
                for axx in ax:
                    axx.axis("off")
        fig.tight_layout()
        plt.show()


    @staticmethod
    def viz_cell(data, cids, ftitle="", figsize=(10,2), clim=[0, 1.0], cm='viridis'):
        n_cell = len(cids)
        fig, axes = plt.subplots(figsize=figsize, ncols=n_cell, nrows=1)

        for idx, cid in enumerate(cids):
            cimg = data[cid]
            zproj = bmPrep._max_proj(cimg)

            if n_cell == 1:
                axes.imshow(zproj, clim=clim, cmap=cm)
                axes.set_title("Cell #%d"%cid)
                axes.axis('off')

            else:
                axes[idx].imshow(zproj, clim=clim, cmap=cm)
                axes[idx].set_title("Cell #%d"%cid)
                axes[idx].axis('off')


        fig.suptitle(ftitle, y=1.04)

        plt.show()
        return fig


    # grabbed from internet
    @staticmethod
    def find_elbow(data):

        def _get_data_radiant(data):
            return np.arctan2(data[:, 1].max() - data[:, 1].min(),
                              data[:, 0].max() - data[:, 0].min())

        theta = _get_data_radiant(data)

        # make rotation matrix
        co = np.cos(theta)
        si = np.sin(theta)
        rotation_matrix = np.array(((co, -si), (si, co)))

        # rotate data vector
        rotated_vector = data.dot(rotation_matrix)

        # return index of elbow
        return np.where(rotated_vector == rotated_vector[:, 1].min())[0][0]


    def determine_K(self, max_num_clusters=10, use_elbow=False, viz=False, save_path=None):
        SQD = []
        xs = range(2, max_num_clusters+1)

        s_scores = []
        for k in xs:
            kmeans = KMeans(n_clusters=k, random_state=0)
            y_pred = kmeans.fit_predict(self.feats)
            # compute mean silhouette coefficient
            s_score = silhouette_score(self.feats, y_pred)
            s_scores.append(s_score)
            SQD.append(kmeans.inertia_/len(self.feats))

        k_det = np.argmin(s_scores) + 1
        ys = SQD

        if viz:
            plt.plot(s_scores)
            plt.title("s_score")
            plt.show()

            plt.figure(figsize=(16, 8))
            plt.plot(xs, ys, 'bx-')
            plt.xlabel('# clusters')
            plt.ylabel('Mean(SQ. dist from points to cluster centroid')
            plt.title('Elbow Method')
            plt.show()

        data = np.array([xs, ys])
        data = np.swapaxes(data, 0, 1)
        elbow_idx = BMClustering.find_elbow(data)
        print("k_det: ", k_det, ", elbow index: ", elbow_idx)
        if use_elbow:
            k_det = elbow_idx

        if save_path is not None:
            now = bmUtil.get_current_time()
            silhouette_score_file = os.path.join(save_path,
                                        "silhouette_scores_%s.csv"%now)
            np.savetxt(silhouette_score_file, np.array(s_scores), delimiter=",")
            print("determine_K(): saved silhouette_scores to %s"%silhouette_score_file)

            elbow_idx_file = os.path.join(save_path,
                                        "elbow_indices_%s.csv"%now)
            np.savetxt(elbow_idx_file, data, delimiter=",")
            print("determine_K(): saved elbow indices to %s"%elbow_idx_file)

        return k_det


    def run_kmeans(self, max_nc=10, use_elbow=False, force_nc=None,
                   cm=None, viz=False, figsize=(8,8),
                   save_path=None, pretrained_model=None):
        if cm is None:
            cm = self.CM

        # determine K
        self.k_det = self.determine_K(max_num_clusters=max_nc, use_elbow=use_elbow,
                                 viz=viz, save_path=save_path) if force_nc is None else force_nc

        if pretrained_model is not None:
            self.load_model(pretrained_model)
            y_pred = self.predict(self.feats)

        else:
            self.kmeans = KMeans(n_clusters=self.k_det, random_state=0)
            y_pred = self.kmeans.fit_predict(self.feats)

        cluster_centers = self.kmeans.cluster_centers_

        xs = self.feats[:, self.feat1_idx]
        ys = self.feats[:, self.feat2_idx]
        cluster_xs = cluster_centers[:,self.feat1_idx]
        cluster_ys = cluster_centers[:,self.feat2_idx]
        ccenters = np.swapaxes(np.array([cluster_xs, cluster_ys]), 0, 1)

        # top4 closest samples from each cluster
        cells_picked = []
        for i in range(self.k_det):
            cctr = ccenters[i]
            indices = np.where(y_pred==i)[0]
            ccids = np.array(self.cids)[indices]
            xs_c = xs[indices]
            ys_c = ys[indices]

            # compute distance from cluster_center
            cdists = []
            for idx, (xsc, ysc) in enumerate(zip(xs_c, ys_c)):
                cdist = np.linalg.norm([xsc, ysc] - cctr)
                cdists.append(cdist)

            # get 4 closest cells
            ncc = 12
            res = sorted(range(len(cdists)), key=lambda sub: cdists[sub])[:ncc]
            if False:
                # Printing result
                print("\t\tIndices list of min K elements is : ", res)
                #print("\t\tiidx: ", iidx, "final indices: ", indices[iidx])
                #print("\t\tselected cell ids: ", ccids[iidx[:ncc]])
                print("\t\tselected cell ids: ", ccids[res])
            cells_picked.append(ccids[res])

        return y_pred, cells_picked


    def viz_kmeans_result(self, y_pred, figsize=(8,8), dpi=160, cm=None,
                          ss=250, s=20, fs=10, cids=None):
        if cm is None:
            cm = self.CM

        xs = self.feats[:, self.feat1_idx]
        ys = self.feats[:, self.feat2_idx]

        cluster_centers = self.get_cluster_centers()
        cluster_xs = cluster_centers[:,self.feat1_idx]
        cluster_ys = cluster_centers[:,self.feat2_idx]

        # Plot
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=figsize, dpi=dpi)
        scatter = axs.scatter(xs, ys, c=y_pred, cmap=cm, alpha=0.4, s=s)
        if False:
            for i, cid in enumerate(self.cids):
                axs.annotate(cid, (xs[i], ys[i]))

        if cids is not None:
            xs_c = []
            ys_c = []
            for cc in cids:
                ii = list(self.cids).index(cc)
                xs_c.append(xs[ii])
                ys_c.append(ys[ii])

            for i, cid in enumerate(cids):
                axs.annotate(cid, (xs_c[i], ys_c[i]), fontsize=fs)

        # draw cluster centers
        axs.scatter(cluster_xs, cluster_ys, c='r', edgecolors='black', marker='s', s=ss)
        legend1 = axs.legend(*scatter.legend_elements(), title="sub classes")
        axs.add_artist(legend1)
        axs.set_title("K-Means clustering results")
        axs.set_xlabel("Feat # %d"%self.feat1_idx)
        axs.set_ylabel("Feat # %d"%self.feat2_idx)
        axs.grid(False)
        axs.spines['top'].set_visible(False)
        axs.spines['right'].set_visible(False)

        plt.show()
        return fig


    def save_model(self, savepath, model_name):
        assert self.kmeans is not None
        fullpath = os.path.join(savepath, model_name + ".pkl")
        with open(fullpath, "wb") as fp:
            pickle.dump(self.kmeans, fp)

        print("BMClustering(): Saved model to [ %s ]"%fullpath)
    def save_model(self, savepath, model_name):
        assert self.kmeans is not None
        fullpath = os.path.join(savepath, model_name + ".pkl")
        with open(fullpath, "wb") as fp:
            pickle.dump(self.kmeans, fp)

        print("BMClustering(): Saved model to [ %s ]"%fullpath)


    def load_model(self, mpath, model_name, ext=".pkl"):
        model_path = os.path.join(mpath, model_name + ext)
        with open(model_path, "rb") as fp:
            self.kmeans = pickle.load(fp)
        print("BMClustering(): Loaded model from [ %s ]."%model_path)

    def load_model(self, model_path):
        with open(model_path, "rb") as fp:
            self.kmeans = pickle.load(fp)
        print("BMClustering(): Loaded model from [ %s ]."%model_path)


    def predict(self, data):
        return self.kmeans.predict(data)


    def get_cluster_centers(self):
        return self.kmeans.cluster_centers_


    @staticmethod
    def viz_vol2d(vol, clim=None):
        fig, ax = plt.subplots(figsize=(30, 3), ncols=len(vol), nrows=1)
        for idx, img in enumerate(vol):
            ax[idx].imshow(img, clim=clim)
            ax[idx].axis("off")
        plt.show()
