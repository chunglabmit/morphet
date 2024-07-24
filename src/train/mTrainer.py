"""mTrainer.py: Trainer for microglia cell phenotyping"""
__author__      = "Minyoung Kim"
__license__ = "MIT"
__maintainer__ = "Minyoung Kim"
__email__ = "minykim@mit.edu"
__date__ = "09/13/2021"

import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torchvision import transforms as T
import itertools
from tensorboardX import SummaryWriter
import math
from scipy.ndimage import zoom

# internal
from train.trainer import Trainer
from train.arch.alnet import ALNet
from utils.const import Phase, Dataset, ModelType
import utils.util as bmUtil

class MTrainer(Trainer):
    def __init__(self, **args):
        super(MTrainer, self).__init__(**args)

    def init_network(self, is_eval=False):
        if is_eval:
            return ALNet(self.p).cuda().eval()
        else:
            return ALNet(self.p).cuda()


    def init_data_loader(self, updated_dataset=None):
        kwargs = {'num_workers': 16, 'pin_memory': True}        # only for cuda!
        self.data_loader = {}
        self.n_batches = {}

        if updated_dataset is not None:
            self.dataset = updated_dataset

        for key in self.dataset.keys():
            if key == 'name':
                continue

            if key == Phase.TRAIN and self.p.use_sampler:
            #if self.p.use_sampler:
                weights = self.dataset[key].weights
                #lbl_cnt = self.dataset[key].label_cnt
                #weights = 1 / torch.Tensor(lbl_cnt)
                #weights = max(lbl_cnt) / torch.Tensor(lbl_cnt)
                #weights = weights / torch.max(weights)
                sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, self.p.batch_size)
                # shuffle must be False if sampler is passed
                shuffle = False
            elif key == Phase.VAL and self.p.use_sampler_in_val:
                weights = self.dataset[key].weights
                sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, self.p.batch_size)
                shuffle = False

            elif key == Phase.REALTIME:
                sampler = None
                shuffle = self.p.shuffle
            else:
                sampler = None
                shuffle = self.p.shuffle

            self.data_loader[key] = DataLoader(self.dataset[key],
                                          batch_size=self.p.batch_size,
                                          shuffle=shuffle,
                                          sampler=sampler,
                                          **kwargs)
            self.n_batches[key] = int(math.ceil(float(len(self.dataset[key])) / float(self.p.batch_size)))



    def validate(self, return_result=False,
                 pBar=None, pBarFunc=None):
        """validate Autoencoder part of model"""

        self.curr_phase = Phase.VAL
        zs, gts, _ = self.forward_all(self.n_batches[self.curr_phase],
                                             desc="VAL_AE", return_labels=True,
                                             pBar=pBar, pBarFunc=pBarFunc)
        results = self.report_accuracy(zs, gts)

        if return_result:
            return results


    def report_accuracy(self, zs, gts, debug=True):
        """print accuracy and report to tensorboard

        Params
        ----------
        zs: last layer's output (prob distribution)
        gts: ground truth
        debug: print messages or not
        """

        prefix=ModelType.ALTR
        log_iter = self.w_iter if self.curr_phase == Phase.TRAIN else self.w_iter_val

        y_pred_z_cuda = torch.argmax(zs, dim=1)
        y_pred_z = y_pred_z_cuda.data.cpu().numpy()
        gts_npy = gts.data.cpu().numpy()
        accuracy = bmUtil.calc_acc(gts_npy, y_pred_z) # from zs

        if debug:
            self.prt.p("[{}] y_pred_z[:20]: {}".format(self.curr_phase, y_pred_z[:20]), self.prt.LOG)
            self.prt.p("[{}] gts[:20]: {}".format(self.curr_phase, gts_npy[:20]), self.prt.LOG)
            self.prt.p("[%s] Accuracy: %.2f%%"%(self.curr_phase, accuracy * 100.), self.prt.LOG)

        if self.writer:
            self.writer.add_scalar('%s/%s/Accuracy'%(self.curr_phase, prefix), accuracy, log_iter)

        return zs.data.cpu().numpy(), y_pred_z, gts_npy, accuracy * 100


    def train(self, validate=False):
        """train model"""

        # log description to tensorboard
        self.log_desc_to_tfboard()

#        loss_alpha = 2.0
#        alpha_factor = 0.995
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.p.learning_rate, momentum=0.9)
        #optimizer = torch.optim.Adam(self.net.parameters(), lr=self.p.learning_rate)
#        optimizer = torch.optim.NAdam(self.net.parameters(), lr=self.p.learning_rate, betas=(0.9, 0.999),
#                                      eps=1e-08, weight_decay=0, momentum_decay=0.004)
        criterion_class = nn.CrossEntropyLoss()
        criterion_recon = nn.MSELoss(reduction='mean')      # option: size_average=True

        class_loss = None

        # freeze if set
        if self.p.freeze_aw:
            print("FREEZING PRETRAINED WEIGHT PART!")
            self.net.config_encoder(learnable=False)

        for ei in range(self.p.epoch):
            recon_loss_all = torch.cuda.FloatTensor([]).unsqueeze(0)[0]
            class_loss_all = torch.cuda.FloatTensor([]).unsqueeze(0)[0]
            total_loss_all = torch.cuda.FloatTensor([]).unsqueeze(0)[0]

            self.curr_phase = Phase.TRAIN
            acc_all = []
            n_batches = int(math.ceil(self.n_batches[self.curr_phase]))
            labels = None
            for di in tqdm(range(n_batches), desc="TRAIN"):
                # form input data
                inputs_cpu = next(iter(self.data_loader[self.curr_phase]))
                inputs, labels = self.form_net_inputs(inputs_cpu, rescale=True)

                # reset gradient
                optimizer.zero_grad()

                # forward
                middle_feat = None
                enc, decoded_output, zproj_feat  = self.net(inputs, deconv=True)

                # compute loss
                #recon_loss = criterion_recon(decoded_output[0], decoded_output[1])
                if self.p.no2dUnet:
                    recon_loss = torch.cuda.FloatTensor([0])[0]
                else:
                    recon_loss = criterion_recon(zproj_feat[0], zproj_feat[1])

                class_loss = criterion_class(enc, labels)
                if self.p.no_alpha:
                    total_loss = recon_loss + class_loss
                else:
                    total_loss = self.p.alpha * recon_loss + (1 - self.p.alpha) * class_loss
                #print("total_loss: {}, recon_loss: {}, class_loss: {}".format(total_loss, recon_loss, class_loss))

                total_loss_all = torch.cat([total_loss_all, total_loss.unsqueeze(0)], dim=0)
                recon_loss_all = torch.cat([recon_loss_all, recon_loss.unsqueeze(0)], dim=0)
                if class_loss:
                    class_loss_all = torch.cat([class_loss_all, class_loss.unsqueeze(0)], dim=0)

                # backward prop, and update weights
                total_loss.backward()
                optimizer.step()

                # report accuracy
                if labels is not None:
                    _, _, _, accuracy = self.report_accuracy(enc, labels, debug=False)
                    acc_all.append(accuracy)

                # visualize
                self.visualize_AE_Training(ei, di, zproj_feat[0], zproj_feat[1], {'total_loss':total_loss, 'recon_loss':recon_loss,
                                                      'class_loss':class_loss}, prefix=ModelType.ALTR)
                resnet_x4_feat = decoded_output[0]
                self.visualize_feat(ei, di, resnet_x4_feat, 'ResNet-feat-x4', labels=labels)

            self.prt.p("AVG recon_loss: %.4f, class_loss: %.4f, total_loss: %.4f"%(torch.mean(recon_loss_all),
                                                                                   torch.mean(class_loss_all),
                                                                                   torch.mean(total_loss_all)), self.prt.STATUS)
            self.prt.p("AVG Accuracy: %.2f%%"%(np.mean(np.array(acc_all))), self.prt.STATUS)

            # validate only if the dataset has lables
            if validate and (self.dataset['name'] in [ Dataset.MNIST, Dataset.MICROGLIA_LABELED ]):
                self.validate()

            # save model after an epoch
            self.save_model(ei, "AE")


    def forward_all(self, n_batches, num_sample_batches=0, desc='FORWARD ALL',
                    is_validate=False, return_labels=False, prefix=ModelType.ALTR,
                    pBar=None, pBarFunc=None):
        """forward prop on all available training data and return yhat tensors

        Params
        ----------
        n_batches: number of batches (usually, n_batches = n_iter / batch_size)
        """

        isVal = self.curr_phase == Phase.VAL
        sampled = (isVal and self.p.use_sampler_in_val) or (not isVal and self.p.use_sampler)

        if isVal:
            criterion_recon = nn.MSELoss(reduction='mean')      # option: size_average=True
            criterion_class = nn.CrossEntropyLoss()

        f_iter = iter(self.data_loader[self.curr_phase])
        zs = []
        sample_batches = []
        labels =[]
        #n_batches = int(math.ceil(n_batches))
        n_batches = int(n_batches)

        with torch.no_grad():
            for di in tqdm(range(n_batches), desc=desc):
                if sampled:
                    inputs_ = self.form_net_inputs(next(iter(self.data_loader[self.curr_phase])), rescale=True)
                else:
                    try:
                        inputs_ = self.form_net_inputs(next(iter(f_iter)), rescale=True)
                    except StopIteration:
                        print("EXCEPTION [ StopIteration ] di: ", di, "n_batches: ", n_batches)
                        continue


                if return_labels:
                    inputs, label = inputs_
                else:
                    inputs, _ = inputs_

                sample = inputs.squeeze()[:, int(self.net.zdepth/2), :, :]
                middle_feat = None
                z, decoded_output, zproj_feat = self.net(inputs, deconv=True)

                if isVal:
                    if self.p.no2dUnet:
                        recon_loss = torch.cuda.FloatTensor([0])[0]
                    else:
                        recon_loss = criterion_recon(zproj_feat[0], zproj_feat[1])

                    class_loss = criterion_class(z, label)

                    if self.p.no_alpha:
                        total_loss = recon_loss + class_loss
                    else:
                        total_loss = self.p.alpha * recon_loss + (1 - self.p.alpha) * class_loss

                    if self.writer:
                        self.writer.add_scalar('%s/%s/recon_loss'%(self.curr_phase, prefix), recon_loss, self.w_iter_val)
                        self.writer.add_scalar('%s/%s/class_loss'%(self.curr_phase, prefix), class_loss, self.w_iter_val)
                        self.writer.add_scalar('%s/%s/total_loss'%(self.curr_phase, prefix), total_loss, self.w_iter_val)
                        self.w_iter_val += 1

                if di == 0:
                    if return_labels:
                        labels = label
                    zs = z
                    if num_sample_batches > 0:
                        sample_batches = sample
                else:
                    if return_labels:
                        labels = torch.cat((labels, label), dim=0)
                    zs = torch.cat((zs, z), dim=0)
                    if di < num_sample_batches:
                        sample_batches = torch.cat((sample_batches, sample), dim=0)

                # if ProgressBar is passed
                if pBar:
                    pBarFunc(pBar, di + 1, n_batches)

        if return_labels:
            return zs, labels, sample_batches
        else:
            return zs, sample_batches


    def form_net_inputs(self, _in, rescale=False):
        """form final inputs for the network to be fed, handling differently by dataset"""
        label = None
        dname = self.dataset['name']
        print("_in.size: ", _in.shape)
        print("rescale?: ", rescale)

        if dname in [Dataset.MICROGLIA, Dataset.REALTIME]:
            if rescale:
                _, in_dd, _, in_dh, in_dw = _in.shape
                dh_factor = self.net.in_h / float(in_dh)
                dw_factor = self.net.in_w / float(in_dw)
                dd_factor = self.net.in_d / float(in_dd)
                if dh_factor * dw_factor * dd_factor == 1.0:
                    # no rescale needed
                    inputs = _in.cuda()
                else:
                    in_resized = []
                    for item in _in:
                        arr = np.squeeze(item)
                        new_arr = zoom(arr, (dd_factor, dh_factor, dw_factor))
                        in_resized.append(new_arr)
                    inputs = torch.from_numpy(np.array(in_resized)[:, :, np.newaxis, :, :]).cuda()
            else:
                inputs = _in.cuda()
        elif dname == Dataset.MICROGLIA_LABELED:
            inputs, label = _in
            if rescale:
                _, in_dd, in_dh, in_dw = inputs.shape
                dh_factor = self.net.in_h / float(in_dh)
                dw_factor = self.net.in_w / float(in_dw)
                dd_factor = self.net.in_d / float(in_dd)
                if dh_factor * dw_factor * dd_factor == 1.0:
                    # no rescale needed
                    inputs = inputs.squeeze().unsqueeze(2).cuda()
                else:
                    in_resized = []
                    for item in inputs:
                        arr = np.squeeze(item)
                        new_arr = zoom(arr, (dd_factor, dh_factor, dw_factor))
                        in_resized.append(new_arr)

                    inputs = torch.from_numpy(np.array(in_resized)[:, :, np.newaxis, :, :]).cuda()

                    #inputs.shape:  torch.Size([64, 16, 64, 64])
                    #inputs.size():  torch.Size([64, 16, 1, 64, 64])
            else:
                inputs = inputs.squeeze().unsqueeze(2).cuda()

            label = label.cuda()
        else:
            raise NotImplementedError

        if label is not None:
            return inputs, label.cuda()
        else:
            return inputs, label
