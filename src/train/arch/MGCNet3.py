"""MGCNet3.py: Microglia Classification Model"""
__author__      = "Minyoung Kim"
__license__ = "MIT"
__maintainer__ = "Minyoung Kim"
__email__ = "minykim@mit.edu"
__date__ = "01/16/2019"

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
from train.arch import autoencoder_resnet3D as aeResNet3D

class BGFilteringMorphClassifier3D(nn.Module):
    """Background Filtering & Morphology Classification 3-D Network - ResNet backbone"""
    prt = bmUtil.PRT()

    def __init__(self, params):
        super(BGFilteringMorphClassifier3D, self).__init__()

        self.p = params
        self.alpha = 1.0
        self.zdepth = 16
        self.cfeat_dim = self.p.num_clusters                     # cluster feature dim
        #self.resnet_out_dim = 2048           # 2048
        self.resnet_out_dim = 512 # 2048
        self.in_w = self.p.data_w
        self.in_h = self.p.data_h
        self.in_d = self.p.data_d
        self.in_chn = 1

        self.build_model()
        if (self.p.pretrained_weights_all) is None and (self.p.ae_weights is None):
            self.init_hidden()

    def build_model(self):
#        self.autoencoder = aeResNet3D.MCAutoencoder(sample_size=self.in_w,
        self.autoencoder = aeResNet3D.MCAutoencoder_s(sample_size=self.in_w,
                                                      in_chn=self.in_chn,
                                                      sample_duration=self.in_d).cuda()

        # load pretrained weights if provided
        if self.p.resnet_weights:
            self.autoencoder.load_resnet_weights(self.p.resnet_weights)
            self.prt.p("loaded pretrained weights from %s"%(self.p.resnet_weights), self.prt.LOG)

#        self.fc1 = nn.Linear(self.resnet_out_dim, 64).cuda()
#        self.fc2 = nn.Linear(64, self.p.num_clusters).cuda()
        self.fc = nn.Linear(self.resnet_out_dim, self.p.num_clusters)


    def lock_encoder(self):
        for param in self.autoencoder.parameters():
            param.requires_grad = False


    def unlock_encoder(self):
        for param in self.autoencoder.parameters():
            param.requires_grad = True


    def forward(self, x, deconv=True, ae_only=False):
        """forward propagate all
            [ Conv3d ]
                input: (N, C_in, D, H, W)
                output: (N, C_out, D_out, H_out, W_out)
        """
        # handle case when batch_size == 1
        bsz = x.size(0)
        if bsz == 1:
            x = x.unsqueeze(0)

        # stack or unsqueeze
        if self.in_chn == 3:
            x = torch.stack([x]*3, dim=1)       # make it 3 channels for ResNet3D 3-chn input Network

        encoded, decoded = self.autoencoder(x, deconv)

        if ae_only:
            return encoded, decoded

#        x = F.relu(self.fc1(encoded[0]))
#        x = self.fc2(x)
        x = self.fc(encoded[0])
        return encoded, decoded, x


    def init_hidden(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()


    def print_params(self):
        return bmUtil.print_class_params(self.__class__.__name__, vars(self))
