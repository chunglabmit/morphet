"""trainer_BFMC.py: Model Training Class for BFMC networks"""
__author__      = "Minyoung Kim"
__license__ = "MIT"
__maintainer__ = "Minyoung Kim"
__email__ = "minykim@mit.edu"
__date__ = "01/16/2019"

import numpy as np
from tqdm import tqdm
from random import randint
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import itertools
from sklearn.cluster import KMeans
from tensorboardX import SummaryWriter

# internal
from train.trainer import Trainer
from train.arch.MGCNet3 import BGFilteringMorphClassifier3D
from utils.const import Phase, Dataset
import utils.util as bmUtil

pdist = nn.PairwiseDistance(p=2)

class BFMCTrainer(Trainer):
    def __init__(self, **args):
        super(BFMCTrainer, self).__init__(**args)
        self.rl0_weight = 1.       #1e-4
        self.rl1_weight = 1.       #1e-6
        self.rl2_weight = 1.       #1e-6

    def init_network(self, is_eval=False):
        print("BGFilteringMorphClassifier3D Net Initialization!")
        # override
        print("is_eval? ", is_eval)
        if is_eval:
            return BGFilteringMorphClassifier3D(self.p).cuda().eval()
        else:
            return BGFilteringMorphClassifier3D(self.p).cuda()


    def load_weights(self, ae_only=False):
        """load pretrained weights

        Params
        ----------
        ae_only: load only pretrained autoencoder or not
        """

        w_type = "AE" if ae_only else "ALL"
        weights_f = self.p.ae_weights if ae_only else self.p.pretrained_weights_all
        weights = torch.load(weights_f)
        try:
            self.net.load_state_dict(weights)
        except Exception as e:
            self.prt.p("Error: " + str(e), self.prt.ERROR)
            self.prt.p("Layers don't match between pretraiend weights & network...loading only availables!", self.prt.WARNING)
            my_dict = self.net.state_dict()
            for k, v in my_dict.items():
                if k == 'fcconv.weight':
                    fcw = weights['fc.weight']
                    o, d = fcw.size()
                    fcw = fcw.view(o, d, 1, 1, 1)
                    my_dict[k] = fcw
                elif k == 'fcconv.bias':
                    my_dict[k] = weights['fc.bias']
                elif "fc2" in k:    # due to possibility of change in number of classes
                    continue
                elif k in weights:
                    my_dict[k] = weights[k]

            self.net.load_state_dict(my_dict)

        self.prt.p("loaded pretrained %s weights from %s"%(w_type, weights_f), self.prt.LOG)


    def validate(self, stage):
        """validate Autoencoder part of model"""

        self.curr_phase = Phase.VAL
        yhats, gts, _ = self.forward_all(self.n_batches[self.curr_phase], desc="VAL_AE", return_labels=True)
        acc, probs = self.report_accuracy(yhats, gts, stage=stage)


    def report_accuracy(self, zs, gts, stage="Phase1_AE", debug=True):
        """print accuracy and report to tensorboard

        Params
        ----------
        zs: last layer's output (prob distribution)
        gts: ground truth
        debug: print messages or not
        """

        log_iter = self.w_iter if self.curr_phase == Phase.TRAIN else self.w_iter_val

        sm = F.softmax(zs, dim=1)
        predicted = torch.argmax(sm, dim=1)
        correct = (predicted == gts).sum().item()
        total = gts.size(0)
        enc_acc = float(correct) / float(total)

        class_correct = list(0. for i in range(self.p.num_clusters))
        class_total = list(0. for i in range(self.p.num_clusters))
        c = (predicted == gts).squeeze()
        for i in range(len(c)):
            label = gts[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

        if debug:
            self.prt.p("[{}] predicted[:20]: {}".format(self.curr_phase, predicted[:20]), self.prt.LOG)
            self.prt.p("[{}] gts[:20]: {}".format(self.curr_phase, gts[:20]), self.prt.LOG)
            self.prt.p("[%s] enc_acc: %.2f%%"%(self.curr_phase, enc_acc * 100.), self.prt.LOG)
            for i in range(self.p.num_clusters):
                print("[%s] Accuracy of %d : %2d%%"%(self.curr_phase, i, 100 * class_correct[i] / class_total[i]))

        self.writer.add_scalar('%s/%s/enc_acc'%(self.curr_phase, stage), enc_acc, log_iter)

        return enc_acc * 100., sm


    def train(self, validate=False):
        # log description to tensorboard
        self.log_desc_to_tfboard()

        # 1. train autoencoder
        learning_rate = 1e-4

        if not self.ae_preloaded:
            self.train_ae(learning_rate, validate)
        else:
            # lock encoder
            self.prt.p("locking encoder...", self.prt.STATUS)
            self.net.lock_encoder()
            self.prt.p("locking encoder...(Done)", self.prt.STATUS)
            # 2. train background filter and classifie
            self.train_bfmc(learning_rate, validate)



    def train_bfmc(self, learning_rate=1e-4, validate=False):
        stage = "Phase2_ALL"

        optimizer = torch.optim.SGD(self.net.parameters(), lr=learning_rate, momentum=0.9)
        criterion_class = nn.CrossEntropyLoss()
        bg_label = self.p.num_clusters - 1

        for ei in range(self.p.epoch):
            self.curr_phase = Phase.TRAIN
            class_loss_all = []
            acc_all = []
            n_batches = self.n_batches[self.curr_phase]
            labels = None
            for di in tqdm(range(n_batches), desc="TRAIN"):
                inputs_cpu = next(iter(self.data_loader[self.curr_phase]))
                inputs, labels = self.form_net_inputs(inputs_cpu)
                nbi = (labels != bg_label).nonzero()    # non-bg indices

                optimizer.zero_grad()
                enc, dec, yhat = self.net(inputs)

                assert labels is not None
                class_loss = criterion_class(yhat, labels)

                if di == 0:
                    class_loss_all = class_loss.unsqueeze(0)
                else:
                    class_loss_all = torch.cat([class_loss_all, class_loss.unsqueeze(0)], dim=0)

                class_loss.backward()
                optimizer.step()

                if labels is not None:
                    acc, _ = self.report_accuracy(yhat, labels, stage=stage, debug=False)
                    acc_all.append(acc)

                # visualize
                self.visualize_training(ei, di, inputs, dec[2], {'class_loss':class_loss}, "Phase2_ALL",
                                        enc[1], dec[1])

            self.prt.p("AVG class_loss: %.4f"%torch.mean(class_loss_all), self.prt.STATUS)
            self.prt.p("AVG acc: %.2f%%"%(np.mean(np.array(acc_all))), self.prt.STATUS)

            # validate only if the dataset has lables
            if validate and (self.dataset['name'] in [ Dataset.MNIST, Dataset.MICROGLIA_LABELED ]):
                self.validate(stage)

            # save model after an epoch
            self.save_model(ei, stage)


    def train_bfmc_include_reconloss(self, learning_rate=1e-4, validate=False):
        stage = "Phase2_ALL"

        optimizer = torch.optim.SGD(self.net.parameters(), lr=learning_rate, momentum=0.9)
        criterion_recon0, criterion_recon1, criterion_recon2 = self.get_recon_criterions()
        criterion_class = nn.CrossEntropyLoss()
        bg_label = self.p.num_clusters - 1

        for ei in range(self.p.epoch):
            self.curr_phase = Phase.TRAIN
            recon_loss_all = []
            class_loss_all = []
            total_loss_all = []
            enc_acc_all = []
            n_batches = self.n_batches[self.curr_phase]
            labels = None
            for di in tqdm(range(n_batches), desc="TRAIN"):
                inputs_cpu = next(iter(self.data_loader[self.curr_phase]))
                inputs, labels = self.form_net_inputs(inputs_cpu)
                nbi = (labels != bg_label).nonzero()    # non-bg indices

                optimizer.zero_grad()
                enc, dec, yhat = self.net(inputs)
                recon_l0 = criterion_recon0(inputs, dec[2])     # raw vs final decoded
                recon_l1 = criterion_recon1(enc[1], dec[1])     # second biggest
                recon_l2 = criterion_recon2(enc[2], dec[0])     # smallest kernels
                recon_loss = self.rl0_weight * recon_l0 + self.rl1_weight * recon_l1 + self.rl2_weight * recon_l2

                if labels is not None:
                    class_loss = criterion_class(yhat, labels)
                else:
                    class_loss = 0.0
                total_loss = recon_loss + class_loss
 #               print("total_loss: %.4f, recon_loss: %.4f (%.4f, %.4f, %.4f), class_loss: %.4f"%(total_loss, recon_loss, recon_l0,
 #                                                                                                recon_l1, recon_l2, class_loss))
                if di == 0:
                    total_loss_all = total_loss.unsqueeze(0)
                    recon_loss_all = recon_loss.unsqueeze(0)
                    class_loss_all = class_loss.unsqueeze(0)
                else:
                    total_loss_all = torch.cat([total_loss_all, total_loss.unsqueeze(0)], dim=0)
                    recon_loss_all = torch.cat([recon_loss_all, recon_loss.unsqueeze(0)], dim=0)
                    class_loss_all = torch.cat([class_loss_all, class_loss.unsqueeze(0)], dim=0)

                # backward prop, and update weights
                total_loss.backward()
                optimizer.step()

                # report accuracy
                if labels is not None:
                    enc_acc, _ = self.report_accuracy(yhat, labels, stage=stage, debug=False)
                    enc_acc_all.append(enc_acc)

                # visualize
#                self.net.autoencoder.register_forward_hook(get_)
                self.visualize_training(ei, di, inputs, dec[2], {'total_loss':total_loss, 'recon_loss':recon_loss,
                                                      'class_loss':class_loss}, "Phase2_ALL",
                                        enc[1], dec[1])

            self.prt.p("AVG recon_loss: %.4f, class_loss: %.4f, total_loss: %.4f"%(torch.mean(recon_loss_all),
                                                                                   torch.mean(class_loss_all),
                                                                                   torch.mean(total_loss_all)), self.prt.STATUS)
            self.prt.p("AVG enc_acc: %.2f%%"%(np.mean(np.array(enc_acc_all))), self.prt.STATUS)

            # validate only if the dataset has lables
            if validate and (self.dataset['name'] in [ Dataset.MNIST, Dataset.MICROGLIA_LABELED ]):
                self.validate(stage)

            # save model after an epoch
            self.save_model(ei, stage)


    def get_recon_criterions(self):
        #criterion_recon0 = nn.SmoothL1Loss()
        #criterion_recon1 = nn.SmoothL1Loss()
        #criterion_recon2 = nn.SmoothL1Loss()
        return nn.MSELoss(), nn.MSELoss(), nn.MSELoss()


    def train_ae(self, learning_rate=1e-4, validate=False):
        """train autoencoder"""
        stage = "Phase1_AE"
        optimizer = torch.optim.SGD(self.net.parameters(), lr=learning_rate, momentum=0.9)
        #optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        criterion_recon0, criterion_recon1, criterion_recon2 = self.get_recon_criterions()

        for ei in range(self.p.epoch):
            self.curr_phase = Phase.TRAIN
            recon_loss_all = []
            n_batches = self.n_batches[self.curr_phase]
            for di in tqdm(range(n_batches), desc="TRAIN"):
                inputs_cpu = next(iter(self.data_loader[self.curr_phase]))
                inputs, _ = self.form_net_inputs(inputs_cpu)

                optimizer.zero_grad()
                enc, dec = self.net(inputs, ae_only=True)
                if False:
                    print("enc[0].size: ", enc[0].size())
                    print("enc[1].size: ", enc[1].size())
                    print("enc[2].size: ", enc[2].size())
                    print("dec[0].size: ", dec[0].size())
                    print("dec[1].size: ", dec[1].size())
                    print("dec[2].size: ", dec[2].size())

                recon_l0 = criterion_recon0(inputs, dec[2])     # raw vs final decoded
                recon_l1 = criterion_recon1(enc[1], dec[1])     # second biggest
                recon_l2 = criterion_recon2(enc[2], dec[0])     # smallest kernels
                print("recon l0: {}, l1: {}, l2: {}".format(recon_l0, recon_l1, recon_l2))
                recon_loss = self.rl0_weight * recon_l0 + self.rl1_weight * recon_l1 + self.rl2_weight * recon_l2
                print("recon_loss all: {}".format(recon_loss))

                if di == 0:
                    recon_loss_all = recon_loss.unsqueeze(0)
                else:
                    recon_loss_all = torch.cat([recon_loss_all, recon_loss.unsqueeze(0)], dim=0)

                # backward prop, and update weights
                recon_loss.backward()
                optimizer.step()

                # visualize
                self.visualize_training(ei, di, inputs, dec[2], {'recon_loss':recon_loss}, stage,
                                        enc[1], dec[1])

            self.prt.p("AVG recon_loss: %.4f"%(torch.mean(recon_loss_all)), self.prt.STATUS)

            # validate only if the dataset has lables
            if validate and (self.dataset['name'] in [ Dataset.MNIST, Dataset.MICROGLIA_LABELED ]):
                self.validate(stage)

            # save model after an epoch
            self.save_model(ei, stage)


    def forward_all(self, n_batches, num_sample_batches=0, desc='FORWARD ALL', is_validate=False, return_labels=False):
        """forward prop on all available training data and return yhat tensors

        Params
        ----------
        n_batches: number of batches (usually, n_batches = n_iter / batch_size)
        """

        isVal = self.curr_phase == Phase.VAL
        sampled = (isVal and self.p.use_sampler_in_val) or (not isVal and self.p.use_sampler)

        if isVal:
            criterion_recon0, criterion_recon1, criterion_recon2 = self.get_recon_criterions()
            criterion_class = nn.CrossEntropyLoss()

        f_iter = iter(self.data_loader[self.curr_phase])
        yhats = []
        sample_batches = []
        labels =[]
        print("n_batches: ", n_batches)

        with torch.no_grad():
            for di in tqdm(range(n_batches), desc=desc):
                if sampled:
                    inputs_ = self.form_net_inputs(next(iter(self.data_loader[self.curr_phase])))
                else:
                    inputs_ = self.form_net_inputs(next(f_iter))


                if return_labels:
                    inputs, label = inputs_
                else:
                    inputs, _ = inputs_

                sample = inputs.squeeze()[:, int(self.net.zdepth/2), :, :]
                #z, decoded_output, yhat = self.net(inputs)
                #enc, dec, yhat = self.net(inputs)
                enc, dec, yhat = self.net(inputs)

                if isVal:
                    recon_l0 = criterion_recon0(inputs, dec[2])     # raw vs final decoded
                    recon_l1 = criterion_recon1(enc[1], dec[1])     # second biggest
                    recon_l2 = criterion_recon2(enc[2], dec[0])     # smallest kernels
                    recon_loss = self.rl0_weight * recon_l0 + self.rl1_weight * recon_l1 + self.rl2_weight * recon_l2
                    #recon_loss0 = criterion_recon0(inputs, dec[1])
                    #recon_loss1 = criterion_recon1(enc, dec[0])
                    #recon_loss = rl0_weight * recon_loss0 + rl1_weight * recon_loss1

                    class_loss = criterion_class(yhat, label)
                    total_loss = recon_loss + class_loss

                    if self.writer:
                        self.writer.add_scalar('%s/recon_loss'%self.curr_phase, recon_loss, self.w_iter_val)
                        self.writer.add_scalar('%s/class_loss'%self.curr_phase, class_loss, self.w_iter_val)
                        self.writer.add_scalar('%s/total_loss'%self.curr_phase, total_loss, self.w_iter_val)
                        self.w_iter_val += 1

                if di == 0:
                    if return_labels:
                        labels = label
                    yhats = yhat
                    if num_sample_batches > 0:
                        sample_batches = sample
                else:
                    if return_labels:
                        labels = torch.cat((labels, label), dim=0)
                    yhats = torch.cat((yhats, yhat), dim=0)
                    if di < num_sample_batches:
                        sample_batches = torch.cat((sample_batches, sample), dim=0)

        if return_labels:
            return yhats, labels, sample_batches
        else:
            return yhats, sample_batches


    def form_net_inputs(self, _in):
        """form final inputs for the network to be fed, handling differently by dataset"""

        label = None
        dname = self.dataset['name']

        if dname == Dataset.MNIST:
            data, label = _in
            data = data.cuda()
            # as MNIST has only one channel in depth, stack up to lstm depth
            inputs = torch.stack([data]*16, dim=1)

        elif dname == Dataset.MICROGLIA:
            # unsqueeze and load to gpu
            inputs = _in.cuda().unsqueeze(1)        # expand for channel

        elif dname == Dataset.MICROGLIA_LABELED:
            inputs, label = _in
            inputs = inputs.cuda().unsqueeze(1)     # expand for channel
            label = label.cuda()
        else:
            raise NotImplementedError

        if label is not None:
            return inputs, label.cuda()
        else:
            return inputs, label


    def visualize_training(self, epoch, batch_idx, data_in, data_out, loss, stage,
                                 enc_feat=None, dec_feat=None):
        """visualize results on Tensorboard

        Params
        ----------
        epoch: current epoch
        batch_idx: current batch idx
        data_in: input data Tensor
        data_out: decoded net output data Tensor
        loss: loss dictionary
        """

        log_idx = self.w_iter
        dname = self.dataset['name']
        sidx = randint(0, data_in.size(0)-1)      # pick a sample from a batch

        for key in loss.keys():
            self.writer.add_scalar('%s/%s/%s'%(self.curr_phase, stage, key), loss[key], log_idx)

        # visualize images

        # 1. Raw Input vs Decoded Input
        # data_in : (batch_size, chn, z, y, x)
        if data_out is not None:
            grid_data = torch.cat((data_in[sidx], data_out[sidx]), dim=1)
        else:
            grid_data =data_in[sidx]
        # transpose for make_grid() taking 4D mini-batch Tensor of shape (BxCxHxW)
        grid_data_t = torch.transpose(grid_data, 0, 1)

        normalize = True if dname == Dataset.MNIST else False
        x = vutils.make_grid(grid_data_t, normalize=normalize, nrow=16, padding=2)
        self.writer.add_image('%s/Input_Top_Decoded_Bottom'%self.curr_phase, x, log_idx)

        # 2. First Enc feat vs Second-Last Dec feat
        # enc_feat: (batch_size, output, D, H, w) (8, 64, 16, 16, 16)
        if enc_feat is not None:
            assert dec_feat is not None
            oidx = randint(0, enc_feat.size(1)-1) # sample output idx
            grid_enc_data = torch.cat((enc_feat[sidx][oidx],dec_feat[sidx][oidx]), dim=0).unsqueeze(1)
            x2 = vutils.make_grid(grid_enc_data, normalize=normalize, nrow=16, padding=2)
            self.writer.add_image('%s/INTERIM_Enc_Top_Dec_Bottom'%self.curr_phase, x2, log_idx)

        self.w_iter += 1
