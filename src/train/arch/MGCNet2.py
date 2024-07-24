"""MGCNet2.py: Microglia Clustering Model"""
__author__      = "Minyoung Kim"
__license__ = "MIT"
__maintainer__ = "Minyoung Kim"
__email__ = "minykim@mit.edu"
__date__ = "08/15/2018"


import sys
sys.path.append("../../")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter

# internal
import utils.util as bmUtil
from utils.const import Dataset
from train.arch import resnet3D


class MGCNet3DResNet(nn.Module):
    """Microglia Clustering Network"""
    prt = bmUtil.PRT()

    def __init__(self, params):
        super(MGCNet3DResNet, self).__init__()

        self.p = params
        self.alpha = 1.0
        self.lstm_last_hdim = 64
        self.zdepth = 16
        self.cfeat_dim = self.p.num_clusters                     # cluster feature dim
        self.resnet_out_dim = 512           # 2048
#        self.in_w = self.p.data_w
#        self.in_h = self.p.data_h
#        self.in_d = self.p.data_d
        self.in_w = 32
        self.in_h = 32
        self.in_d = 16
        self.in_chn = 1
        """
        if self.p.dataset == Dataset.MNIST:
            self.in_w, self.in_h = [28, 28]
            self.in_w, self.in_h = [28, 28]
        elif Dataset.MICROGLIA in self.p.dataset:
            self.in_w, self.in_h = [32, 32]
        else:
            raise DatasetNotRecognizedError
        """

        self.build_model()
        if (self.p.pretrained_weights_all) is None and (self.p.ae_weights is None):
            self.init_hidden()


    def build_model(self):
        print("full resolution? {}".format(self.p.full_resolution))
        self.module = resnet3D.resnet34(num_classes=self.p.num_clusters, sample_size=self.in_w,
                                        in_chn=self.in_chn, sample_duration=self.in_d).cuda()

        # load pretrained weights if provided
        if self.p.resnet_weights:
            weights = torch.load(self.p.resnet_weights)['state_dict']
            my_dict = self.module.state_dict()
            for k, v in my_dict.items():
                if k in weights:
                    my_dict[k] = weights[k]
            self.module.load_state_dict(my_dict)
            self.prt.p("loaded pretrained weights from %s"%(self.p.resnet_weights), self.prt.LOG)

        self.fc = nn.Linear(self.resnet_out_dim, self.p.num_clusters).cuda()
        if self.p.full_resolution:
            self.fcconv = nn.Conv3d(self.resnet_out_dim, self.p.num_clusters, kernel_size=1, stride=(1, 1, 1), bias=True)

        self.dfc = nn.Linear(self.p.num_clusters, self.resnet_out_dim).cuda()

        self.cluster_layer = Parameter(torch.Tensor(self.p.num_clusters, self.cfeat_dim).cuda(), requires_grad=True)


    def forward(self, x, deconv=True, cluster=False):
        """forward propagate all"""

        dz = [None]
        # handle case when batch_size == 1
        bsz = x.size(0)
        x = torch.transpose(x, 1, 2).squeeze()  # swap channel and depth
        if bsz == 1:
            x = x.unsqueeze(0)

        # stack
        if self.in_chn == 3:
            x = torch.stack([x]*3, dim=1)       # make it 3 channels for ResNet3D 3-chn input Network
        else:
            x = x.unsqueeze(1)

        x, middle_feat = self.module(x, self.p.full_resolution)

        if self.p.full_resolution:
            xl = self.fcconv(x)
            return xl, None, None

        xl = self.fc(x)

        if deconv:
            dz = self.dfc(xl)

        # cluster
        #soft_label = self.cluster(x) if cluster else None
        soft_label = self.cluster(xl.view(bsz, -1)) if cluster else None

        return xl, (middle_feat, dz), soft_label


    def config_encoder(self, learnable=True):
        """change encoder paramters' state whether requires back propagation

        :param learnable: make it learnable or not
        """
        for param in self.module.parameters():
            param.requires_grad = learnable


    def target_distribution(self, q):
        """compute target distribution by first raising q to the second power,
           followed by normalizing by frequency per cluster.

           The target distribution would aim to:
                i. Strengthen predictions, i.e., improve cluster purity
                ii. Put more emphasis on data points assigned with high confidence
                iii. Prevent large clusters from distorting the hidden feature space

        :param q: encoded features
        """

        weight = q**2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()


    def cluster(self, z):
        """soft labeling by clustering
           - student t-distribution as same as used in t-SNE algorithm
                q_ij = 1 / (1 + dist(x_i, u_j)^2), then normalize

        :param z: feature data

        """
        # unsqueeze -> (batch_size, 1, 3)
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = torch.pow(q, (self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return q


    def init_hidden(self):
        # init cluster_layer
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.fc.reset_parameters()
        self.dfc.reset_parameters()


    def print_params(self):
        return bmUtil.print_class_params(self.__class__.__name__, vars(self))
