"""autoencoder_resnet3D.py: Autoencoder network with resnet3D backbone"""
__author__      = "Minyoung Kim"
__license__ = "MIT"
__maintainer__ = "Minyoung Kim"
__email__ = "minykim@mit.edu"
__date__ = "01/16/2019"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from .resnet3D import downsample_basic_block, BasicBlock, Bottleneck, TransBasicBlock
from torch.autograd import Function

class MCAutoencoder_s(nn.Module):
    # ResNet3D-34 similar
    # use BasicBlock

    def __init__(self, **kwargs):
        super(MCAutoencoder_s, self).__init__()
        self.encoder = ResNetEncoder(BasicBlock, [3, 4, 6, 3], **kwargs)
        self.decoder = ResNetDecoder_s(TransBasicBlock, [6, 4, 3, 3], **kwargs)

    def load_resnet_weights(self, f):
        weights = torch.load(f)['state_dict']
        my_dict = self.encoder.state_dict()
        for k, v in my_dict.items():
            if k in weights:
                my_dict[k] = weights[k]
        self.encoder.load_state_dict(my_dict)

    def forward(self, x, decode=True, debug=False):
        x, ls = self.encoder(x)
        # x: Final features
        # ls: intermittent features starting just before skip module
        if debug:
            for idx, item in enumerate(ls):
                print("Encoder[l{}] size: {}".format(idx, item.size()))
        encoded = [x, ls[0], ls[3]]
        decoded = self.decoder(*ls) if decode else None
        return encoded, decoded


class MCAutoencoder(nn.Module):
    def __init__(self, **kwargs):
        super(MCAutoencoder, self).__init__()
        self.encoder = ResNetEncoder(Bottleneck, [3, 4, 6, 3], **kwargs)
        self.decoder = ResNetDecoder(TransBasicBlock, [6, 4, 3, 3], **kwargs)

    def load_resnet_weights(self, f):
        weights = torch.load(f)['state_dict']
        my_dict = self.encoder.state_dict()
        for k, v in my_dict.items():
            if k in weights:
                my_dict[k] = weights[k]
        self.encoder.load_state_dict(my_dict)

    def forward(self, x, decode=True, debug=False):
        x, ls = self.encoder(x)
        # x: Final features
        # ls: intermittent features starting just before skip module
        if debug:
            for idx, item in enumerate(ls):
                print("Encoder[l{}] size: {}".format(idx, item.size()))
        encoded = [x, ls[0], ls[3]]
        decoded = self.decoder(*ls) if decode else None
        return encoded, decoded


class Binary(Function):
    @staticmethod
    def forward(ctx, input):
        return F.relu(Variable(input.sign())).data

    @staticmethod
    def backward(ctx, grad_output):
        return gard_output


class ResNetEncoder(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 in_chn=3):

        self.inplanes = 64
        super(ResNetEncoder, self).__init__()

        self.conv1 = nn.Conv3d(in_chn,
                               64,
                               kernel_size=7,
                               stride=(1, 2, 2),
                               padding=(3, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))

        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                                nn.Conv3d(self.inplanes,
                                          planes * block.expansion,
                                          kernel_size=1,
                                          stride=stride,
                                          bias=False),
                                 nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x, full_resolution=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        l0 = x
        x = self.maxpool(x)
        x = self.layer1(x)
        l1 = x
        x = self.layer2(x)
        l2 = x
        x = self.layer3(x)
        l3 = x
        x = self.layer4(x)
        l4 = x
#        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x, [l0, l1, l2, l3, l4]


class ResNetDecoder_s(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 in_chn=3):

        self.inplanes = 256
        super(ResNetDecoder_s, self).__init__()

        self.dlayer1 = self._make_transpose(block, 256, layers[0], stride=2)
        self.dlayer2 = self._make_transpose(block, 128, layers[1], stride=2)
        self.dlayer3 = self._make_transpose(block, 64, layers[2], stride=2)
        self.dlayer4 = self._make_transpose(block, 64, layers[3], stride=2)

        self.ag0 = self._make_agant_layer(64, 64)
        self.ag1 = self._make_agant_layer(64, 64)
        self.ag2 = self._make_agant_layer(128, 128)
        self.ag3 = self._make_agant_layer(256, 256)
        self.ag4 = self._make_agant_layer(512, 256)

        self.inplanes = 64

        # final
        #self.conv6 = self._make_transpose(block, 64, in_chn)
        self.deconv6 = nn.ConvTranspose3d(self.inplanes,
                                          in_chn,
                                          kernel_size=7,
                                          stride=(1, 2, 2),
                                          padding=(3, 3, 3),
                                          output_padding=(0, 1, 1),
                                          bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_transpose(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                            nn.ConvTranspose3d(self.inplanes,
                                               planes,
                                               kernel_size=2,
                                               stride=stride,
                                               padding=0,
                                               bias=False),
                            nn.BatchNorm3d(planes))

        layers = []
        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)


    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
                    nn.Conv3d(inplanes,
                              planes,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              bias=False),
                    nn.BatchNorm3d(planes),
                    nn.ReLU(inplace=True))
        return layers


    def forward(self, l0, l1, l2, l3, l4, debug=True):
        ag4 = self.ag4(l4)

        x = self.dlayer1(ag4)
        decoded0 = x
        x = x + self.ag3(l3)

        x = self.dlayer2(x)
        x = x + self.ag2(l2)

        x = self.dlayer3(x)
        x = x + self.ag1(l1)

        x = self.dlayer4(x)
        x = x + self.ag0(l0)
        decoded = x

        x = self.deconv6(x)

        return decoded0, decoded, x


class ResNetDecoder(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 in_chn=3):

        self.inplanes = 512
        #self.inplanes = 1024
        super(ResNetDecoder, self).__init__()

        self.dlayer1 = self._make_transpose(block, 1024, layers[0], stride=2)
        self.dlayer2 = self._make_transpose(block, 512, layers[1], stride=2)
        self.dlayer3 = self._make_transpose(block, 256, layers[2], stride=2)
        self.dlayer4 = self._make_transpose(block, 64, layers[3], stride=2)

        self.ag0 = self._make_agant_layer(64, 64)
        self.ag1 = self._make_agant_layer(64 * 4, 256)
        self.ag2 = self._make_agant_layer(128 * 4, 512)
        self.ag3 = self._make_agant_layer(256 * 4, 1024)
        self.ag4 = self._make_agant_layer(512 * 4, 512)

        self.inplanes = 64

        # final
        #self.conv6 = self._make_transpose(block, 64, in_chn)
        self.deconv6 = nn.ConvTranspose3d(self.inplanes,
                                          in_chn,
                                          kernel_size=7,
                                          stride=(1, 2, 2),
                                          padding=(3, 3, 3),
                                          output_padding=(0, 1, 1),
                                          bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_transpose(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                            nn.ConvTranspose3d(self.inplanes,
                                               planes,
                                               kernel_size=2,
                                               stride=stride,
                                               padding=0,
                                               bias=False),
                            nn.BatchNorm3d(planes))

        layers = []
        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)


    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
                    nn.Conv3d(inplanes,
                              planes,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              bias=False),
                    nn.BatchNorm3d(planes),
                    nn.ReLU(inplace=True))
        return layers


    def forward(self, l0, l1, l2, l3, l4, debug=False):
#        if debug: print("Decoder[l4]: {}".format(l4.size()))
        ag4 = self.ag4(l4)
#        if debug: print("Decoder[ag4]: {}".format(ag4.size()))
        x = self.dlayer1(ag4)
        if debug: print("Decoder[dlayer1]: {}".format(x.size()))

        decoded0 = x
#        if debug: print("Decoder[ag3]: {}".format(ag3.size()))
        x = x + self.ag3(l3)
#        if debug: print("Decoder[x+ag3]: {}".format(x.size()))

        x = self.dlayer2(x)
        if debug: print("Decoder[dlayer2]: {}".format(x.size()))

        x = x + self.ag2(l2)
#        if debug: print("Decoder[x+ag2]: {}".format(x.size()))

        x = self.dlayer3(x)
        if debug: print("Decoder[dlayer3]: {}".format(x.size()))

        x = x + self.ag1(l1)
#        if debug: print("Decoder[ag1]: {}".format(x.size()))

        x = self.dlayer4(x)
        if debug: print("Decoder[dlayer4]: {}".format(x.size()))

        x = x + self.ag0(l0)
#        if debug: print("Decoder[ag0]: {}".format(x.size()))

        decoded = x
#        x = self.conv6(x)
        x = self.deconv6(x)
        if debug: print("Decoder[dconv6]: {}".format(x.size()))
  #      x = F.sigmoid(x)

        return decoded0, decoded, x
