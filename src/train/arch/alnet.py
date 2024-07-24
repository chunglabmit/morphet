"""alnet.py: Active Learning Network"""
__author__      = "Minyoung Kim"
__license__ = "MIT"
__maintainer__ = "Minyoung Kim"
__email__ = "minykim@mit.edu"
__date__ = "09/13/2021"


import sys
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
from train.arch import resnet as resnet2D
from train.arch import unet


class ALNet(nn.Module):
    """Microglia Clustering Network"""
    prt = bmUtil.PRT()

    def __init__(self, params):
        super(ALNet, self).__init__()

        self.p = params
        self.cfeat_dim = self.p.num_clusters                     # cluster feature dim
        print("self.p.no2dUnet: ", self.p.no2dUnet)
        if self.p.no2dUnet:
            self.resnet_out_dim = 512 # 2048
        else:
            self.resnet_out_dim = 1024           # 2048

#        self.in_w = 32
#        self.in_h = 32
#        self.in_d = 16
        self.in_w = self.p.data_w
        self.in_h = self.p.data_h
        self.in_d = self.p.data_d
        self.zdepth = int(self.in_d / 2.)
        self.in_chn = 1

        self.build_model()
        if (self.p.pretrained_weights_all) is None and (self.p.ae_weights is None):
            self.init_hidden()


    def build_model(self):
        print("full resolution? {}".format(self.p.full_resolution))
        self.module = resnet3D.resnet34(num_classes=self.p.num_clusters, sample_size=self.in_w,
                                        in_chn=self.in_chn, sample_duration=self.in_d).cuda()

#        self.module_zp = resnet2D.resnet10(num_classes=self.p.num_clusters, nchn=1,
#                                           c1_ks=5, mp_ks=5, ap_ks=4, stride=1,
#                                           headless=True).cuda()

        print("NO 2d Unet? ", self.p.no2dUnet)
        if self.p.no2dUnet:
            self.module_zp = None
        else:
            pool_kernel_size = int(self.in_w / 8.)
            print("POOL_KERNAL_SIZE: ", pool_kernel_size)
            self.module_zp = unet.UNet(1, self.p.num_clusters, bilinear=False,
                                        pool_kernel_size=pool_kernel_size).cuda()

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


    def forward(self, x, deconv=True):
        """forward propagate all"""
        if self.p.no2dUnet:
            x_2d = None
        else:
            # compute maxprojection
            x_2d = torch.max(x, 1)[0]

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
        if self.p.no2dUnet:
            stacked = x
            unet_dec = None
        else:
            feat_mproj, unet_dec = self.module_zp(x_2d)
            feat_mproj = feat_mproj.squeeze().unsqueeze(0)
            #stacked = torch.cat((x, feat_mproj), 1)
            if x.size()[0] != 1:
                # Handle corner case where the last batch only has one sample resulting in dim
                # (e.g., [1,512]) which shouldn't be squeezed
                stacked = torch.cat((x, feat_mproj.squeeze()), 1)
            else:
                stacked = torch.cat((x, feat_mproj), 1)

        if self.p.full_resolution:
            xl = self.fcconv(x)
            return xl, None, None

        xl = self.fc(stacked)

        if deconv:
            dz = self.dfc(xl)

        return xl, (middle_feat, dz), (x_2d, unet_dec)


    def config_encoder(self, learnable=True):
        """change encoder paramters' state whether requires back propagation

        :param learnable: make it learnable or not
        """
        for param in self.module.parameters():
            param.requires_grad = learnable


    def init_hidden(self):
        self.fc.reset_parameters()
        self.dfc.reset_parameters()


    def print_params(self):
        return bmUtil.print_class_params(self.__class__.__name__, vars(self))
