"""MGCNet.py: Microglia Clustering Model"""
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
from .resnet import BasicBlock
from .resnet_feat import ResNetFeat
from .convlstm import ConvLSTM
from .deconvlstm import DeConvLSTM
from utils.const import Dataset


class MGCNet3D(nn.Module):
    """Microglia Clustering Network"""
    prt = bmUtil.PRT()

    def __init__(self, params):
        super(MGCNet3D, self).__init__()

        self.p = params
        self.alpha = 1.0
        self.lstm_last_hdim = 64
        self.zdepth = 16
        self.cfeat_dim = self.p.num_clusters                     # cluster feature dim
        if self.p.dataset == Dataset.MNIST:
            self.in_w, self.in_h = [28, 28]
        elif self.p.dataset == Dataset.MICROGLIA:
            self.in_w, self.in_h = [32, 32]
        else:
            raise DatasetNotRecognizedError

        self.build_model()
        if self.p.pretrained_weights_all is None:
            self.init_hidden()

        self.print_params()


    def build_model(self):

        # 3D conv / deconv
        """
        conv3d: input (N, C_in, D, H, W) and output (N, C_out, D_out, H_out, W_out)
        """
        self.conv3d_1 = nn.Conv3d(1, 64, (3, 3, 3), stride=2, padding=0, dilation=2).cuda()
        self.deconv3d_1 = nn.ConvTranspose3d(64, 1, (3, 3, 3), stride=2, padding=0, output_padding=1, dilation=2).cuda()

        self.conv3d_2 = nn.Conv3d(64, 256, (3, 3, 3), stride=1, padding=0).cuda()
        self.deconv3d_2 = nn.ConvTranspose3d(256, 64, (3, 3, 3), stride=1, padding=0).cuda()

        self.conv3d_3 = nn.Conv3d(256, 16, (1, 7, 7), stride=1).cuda()
        self.deconv3d_3 = nn.ConvTranspose3d(16, 256, (1, 7, 7), stride=1).cuda()

        self.conv3d_4 = nn.Conv3d(16, 1, (1, 1, 1), stride=1).cuda()
        self.deconv3d_4 = nn.ConvTranspose3d(1, 16, (1, 1, 1), stride=1).cuda()

        # pool / unpool
        self.pool = nn.MaxPool3d((4, 4, 4), stride=1, return_indices=True, ceil_mode=True).cuda()
        self.dpool = nn.MaxUnpool3d((4, 4, 4), stride=1).cuda()

        # fully connected layer
        self.fc = nn.Linear(1296, self.cfeat_dim).cuda()
        self.dfc = nn.Linear(self.cfeat_dim, 1296).cuda()

        self.dropout3d = nn.Dropout3d(p=0.2)

        self.cluster_layer = Parameter(torch.Tensor(self.p.num_clusters, self.cfeat_dim).cuda(), requires_grad=True)
        print("self.cluster_layer: ", self.cluster_layer)


    def forward(self, x, deconv=True, cluster=False):
        """forward propagate all"""

        dz = [None]

        # encoder
        x = torch.transpose(x, 1, 2)  # swap channel and depth
        x = F.leaky_relu(self.conv3d_1(x))
        #x = self.dropout3d(x)
        x = F.leaky_relu(self.conv3d_2(x))
        x, indices = self.pool(x)
        x = torch.tanh(self.conv3d_3(x))
        x = self.conv3d_4(x)

        bsz, dd, dchn, dw, dh = x.size()

        # decoder
        if deconv:
            dz = self.deconv3d_4(x)
            dz = torch.tanh(self.deconv3d_3(dz))
            dz = self.dpool(dz, indices)
            dz = F.leaky_relu(self.deconv3d_2(dz))
            dz = F.leaky_relu(self.deconv3d_1(dz))
            dz = torch.transpose(dz, 1, 2)

        # cluster
        #soft_label = self.cluster(x) if cluster else None
        soft_label = self.cluster(x.view(bsz, -1)) if cluster else None

        return x, dz, soft_label


    def config_encoder(self, learnable=True):
        """change encoder paramters' state whether requires back propagation

        :param learnable: make it learnable or not
        """
        pass


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
        #q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return q


    def init_hidden(self):
        # init cluster_layer
        torch.nn.init.xavier_normal_(self.cluster_layer.data)



    def print_params(self):
        return bmUtil.print_class_params(self.__class__.__name__, vars(self))




class MGCNetLSTM(nn.Module):
    """Microglia Clustering Network"""
    prt = bmUtil.PRT()

    def __init__(self, params):
        super(MGCNetLSTM, self).__init__()

        self.p = params
        self.alpha = 1.0
        self.lstm_last_hdim = 8
        #self.lstm_last_hdim = 64
        self.zdepth = 16
        self.cfeat_dim = self.p.num_clusters                     # cluster feature dim
        self.lstm_num_layers = 3
        self.lstm_hidden_dims = {1: [self.lstm_last_hdim], 2: [256, self.lstm_last_hdim], 3: [64, 64, self.lstm_last_hdim]}
        self.dlstm_hidden_dims = {1: [1], 2: [256, 1], 3: [64, 64, 1]}
        self.use_softmax_for_distribution = False
        if self.p.dataset == Dataset.MNIST:
            self.in_w, self.in_h = [28, 28]
        elif self.p.dataset == Dataset.MICROGLIA:
            self.in_w, self.in_h = [32, 32]
        else:
            raise DatasetNotRecognizedError

        self.build_model()
        if self.p.pretrained_weights_all is None:
            self.init_hidden()

        self.print_params()


    def build_model(self):
#        # build resnet18
#        self.resnet = ResNetFeat(BasicBlock, [2, 2, 2, 2])
#        if False: #self.p.resnet_weights is not None:
#            weights = torch.load(self.p.resnet_weights)
#            my_dict = self.resnet.state_dict()
#            # only load necessary layers
#            pretrained_dict = {k: v for k, v in weights.items() if k in my_dict}
#            my_dict.update(pretrained_dict)
#            self.resnet.load_state_dict(pretrained_dict)
#            self.prt.p("loaded pretrained resnet model from %s"%self.p.resnet_weights, self.prt.LOG)

        # lstm
        self.lstm = ConvLSTM(input_size=(self.in_w, self.in_h),
                            input_dim=1,
                            hidden_dim = self.lstm_hidden_dims[self.lstm_num_layers],
                            kernel_size=(3, 3),
                            num_layers = self.lstm_num_layers,
                            batch_first=True,
                            bias=True,
                            return_all_layers=False).cuda()

        # fully-connected
        """
        self.fc1 = nn.Linear(self.lstm_last_hdim * self.in_w * self.in_h, 2000).cuda()
        self.fc2 = nn.Linear(2000, self.cfeat_dim).cuda()
        self.dfc2 = nn.Linear(self.cfeat_dim, 2000).cuda()
        self.dfc1 = nn.Linear(2000, self.lstm_last_hdim * self.in_w * self.in_h).cuda()
        """
        self.fc = nn.Linear(self.lstm_last_hdim * self.in_w * self.in_h, self.cfeat_dim).cuda()
        self.dfc = nn.Linear(self.cfeat_dim, self.lstm_last_hdim * self.in_w * self.in_h).cuda()


        self.dlstm = DeConvLSTM(input_size=(self.in_w, self.in_h),
                            input_dim=self.lstm_last_hdim,
                            hidden_dim = self.dlstm_hidden_dims[self.lstm_num_layers],
                            kernel_size=(3, 3),
                            num_layers = self.lstm_num_layers,
                            batch_first=True,
                            bias=True,
                            return_all_layers=False).cuda()

        self.cluster_layer = Parameter(torch.Tensor(self.p.num_clusters, self.cfeat_dim).cuda(), requires_grad=True)
        print("self.cluster_layer: ", self.cluster_layer)


    def forward(self, _in, deconv=True, cluster=False):
        """forward propagate all"""

        decoded_output = [None]
        # 1. forward encoder
        output, states = self.lstm(_in)         # output: (num_layer, batch_size, 16, last_hdim, self.in_w, self.in_h)
        output = output[-1]

        # 2. get encoder output
        depth = output.size(1)
        last_output = output[:, -1, :, :, :]     # (batch_size, last_hdim, self.in_w, self.in_h)

        b, hdim, h, w = last_output.size()
        encoded_feat  = self.fc(last_output.view(-1, hdim*h*w))

        """
        fc = self.fc1(last_output.view(-1, hdim*h*w))
        fc = F.relu(fc)
        encoded_feat = self.fc2(fc)
        """

        # decoder
        if deconv:
            """
            dz = self.dfc2(encoded_feat)
            dz = F.relu(dz)
            dz = self.dfc1(dz)
            dz = dz.view(b, hdim, h, w)
            """
            dz = self.dfc(encoded_feat)
            dz = dz.view(b, hdim, h, w)

            # option 1
            dz = dz.unsqueeze(1)
            dz = dz.expand(b, depth, hdim, h, w)
            # option 2
            #dz = torch.stack([dz]*depth, dim=1)        # stack depth-times for total timestep

            decoded_output, decoded_states = self.dlstm(dz, states)
            decoded_output = decoded_output[-1]

        # cluster
        soft_label = self.cluster(encoded_feat) if cluster else None

        return encoded_feat, decoded_output, soft_label


    def config_encoder(self, learnable=True):
        """change encoder paramters' state whether requires back propagation

        :param learnable: make it learnable or not
        """

        for param in self.lstm.parameters():
            param.requires_grad = learnable

        for param in self.fc.parameters():
            param.requires_grad = learnable

        """
        for param in self.fc1.parameters():
            param.requires_grad = learnable
        for param in self.fc2.parameters():
            param.requires_grad = learnable
        """


    def target_distribution(self, q):
        """compute target distribution by first raising q to the second power,
           followed by normalizing by frequency per cluster.
           
           The target distribution would aim to:
                i. Strengthen predictions, i.e., improve cluster purity
                ii. Put more emphasis on data points assigned with high confidence
                iii. Prevent large clusters from distorting the hidden feature space

        :param q: encoded features
        """

        if self.use_softmax_for_distribution:
            return torch.log(q)
        else:
            weight = q**2 / q.sum(0)
            return (weight.t() / weight.sum(1)).t()


    def cluster(self, z):
        """soft labeling by clustering
           - student t-distribution as same as used in t-SNE algorithm
                q_ij = 1 / (1 + dist(x_i, u_j)^2), then normalize

        :param z: feature data
        
        """
        if self.use_softmax_for_distribution:
            #q = torch.argmax(z, dim=1)
            q = F.softmax(z)
        else:
            # unsqueeze -> (batch_size, 1, 3)
            q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
            q = torch.pow(q, (self.alpha + 1.0) / 2.0)
            #q **= (self.alpha + 1.0) / 2.0
            #q = q.pow((self.alpha + 1.0) / 2.0)
            q = (q.t() / torch.sum(q, 1)).t()

        return q


    def init_hidden(self):
        # init cluster_layer
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.xavier_uniform_(self.dfc.weight)
        """
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.dfc1.weight)
        torch.nn.init.xavier_uniform_(self.dfc2.weight)
        """



    def print_params(self):
        return bmUtil.print_class_params(self.__class__.__name__, vars(self))

