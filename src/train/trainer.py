"""trainer.py: Model Training Class"""
__author__      = "Minyoung Kim"
__license__ = "MIT"
__maintainer__ = "Minyoung Kim"
__email__ = "minykim@mit.edu"
__date__ = "08/15/2018"

import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import itertools
from sklearn.cluster import KMeans
from tensorboardX import SummaryWriter
import math
import json

import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt


# internal
#from arch.MGCNet import MGCNetLSTM, MGCNet3D
from train.arch.MGCNet2 import MGCNet3DResNet
from utils.const import Phase, Dataset, ModelType
import utils.util as bmUtil

pdist = nn.PairwiseDistance(p=2)

class Trainer(object):
    prt = bmUtil.PRT()

    def __init__(self, name, params, dataset):
        self.expr_name = name
        self.p = params
        self.dataset = dataset
        self.update_interval = 300
        self.gamma = 0.1        # coefficient of clustering loss
        self.writer = SummaryWriter(self.p.log_dir) if self.p.timestmp != Phase.REALTIME else None


        self.w_iter = 0
        self.w_iter_val = 0
        self.ae_preloaded = False
        self.curr_phase = None

        # initialize network
        self.net = self.init_network(self.p.is_eval)

#        if self.p.phase == Phase.VAL:
#            self.load_weights()
        if self.p.ae_weights is not None:
            self.ae_preloaded = True
            self.load_weights(ae_only=True)

        elif self.p.pretrained_weights_all is not None:
            self.ae_preloaded = True
            self.load_weights()

        try:
            self.prt.p("dataset: {}".format(self.dataset), self.prt.WARNING)
            self.init_data_loader()
        except Exception as e:
            self.prt.p("Error: " + str(e), self.prt.ERROR)
            self.prt.p("init_data_loader() failed. skip loading data", self.prt.WARNING)
            pass

        if self.p.save_dir is not None:
            self.save_params()


    def init_network(self, is_eval=False):
        # create a default network
        # NOTE: For special network, override this function
        # get network
        #net = MGCNet3D(trParams)
        #net = MGCNetLSTM(trParams)
        #net = MGCNet3DResNet(trParams)
        #net = BFMC3DResNet(trParams)

        if is_eval:
            return MGCNet3DResNet(self.p).cuda().eval()
        else:
            return MGCNet3DResNet(self.p).cuda()


    def print_params(self):
        # TODO: make a base class and inherit
        return bmUtil.print_class_params(self.__class__.__name__, vars(self))


    @staticmethod
    def _save_params(msg, fname, save_dir, is_json=False):
        """save parameters to a file

        Params
        ----------
        msg: lines of messages as a list
        fname: base filename
        save_dir: save location
        """

        fullpath = os.path.join(save_dir, fname)
        if is_json:
            with open(fullpath, 'w') as fp:
                json.dump(msg, fp, indent=4, separators=(',', ':'), sort_keys=True)
                fp.write('\n')
        else:
            tb, info, te = msg
            with open(fullpath, 'a') as fp:
                fp.write(tb + '\n')
                fp.write(info + '\n')
                fp.write(te + '\n')


    def save_params(self):
        """save all params information to files"""

        self._save_params(self.net.print_params(), 'NetParams.txt', self.p.save_dir)
        # save TrainParams() both in txt and json
        self._save_params(self.p.print_params(), 'Params.txt', self.p.save_dir)
        self._save_params(self.p.print_params(in_dict=True), 'Params.json',
                          self.p.save_dir, is_json=True)
        self._save_params(self.print_params(), 'TrainerParams.txt', self.p.save_dir)


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
            else:
                sampler = None
                shuffle = self.p.shuffle

            self.data_loader[key] = DataLoader(self.dataset[key],
                                          batch_size=self.p.batch_size,
                                          shuffle=shuffle,
                                          sampler=sampler,
                                          **kwargs)
            self.n_batches[key] = int(math.ceil(float(len(self.dataset[key])) / float(self.p.batch_size)))


    def load_weights(self, ae_only=False):
        """load pretrained weights

        Params
        ----------
        ae_only: load only pretrained autoencoder or not
        """

        w_type = "AE" if ae_only else "ALL"
        weights_f = self.p.ae_weights if ae_only else self.p.pretrained_weights_all

        #weights = torch.load(weights_f)['state_dict']
        weights = torch.load(weights_f)
        """
        print("weights.keys: ", weights.keys())
        for key in weights:
            print("key: ", key)
            if key == "state_dict":
                print("subkeys: ", weights[key].keys())
            else:
                print("val: ", weights[key])
            sys.exit(0)
        """
        try:
            self.net.load_state_dict(weights)
            print("self.net.cluster_layer: {}".format(self.net.cluster_layer))
        except Exception as e:
            self.prt.p("Error: " + str(e), self.prt.ERROR)
            self.prt.p("Layers don't match between pretraiend weights & network...loading only availables!", self.prt.WARNING)
            my_dict = self.net.state_dict()
            for k, v in my_dict.items():
                # TODO: handle this by excluding non-existing layers from 'weights',
                #       rather than hard-code specific layers which might not compatible for all models
                if 'fc' in k or 'dfc' in k or 'cluster_layer' in k:
                    continue
                #print("k: ", k, "v: ", v)
                #if 'module' in k:
                #    continue
                try:
                    my_dict[k] = weights[k]
                except:
                    self.prt.p("Exception occured during weight loading, trying to resolve..", self.prt.ERROR)
                    if k == 'fcconv.weight':
                        fcw = weights['fc.weight']
                        o, d = fcw.size()
                        fcw = fcw.view(o, d, 1, 1, 1)
                        my_dict[k] = fcw
                    elif k == 'fcconv.bias':
                        my_dict[k] = weights['fc.bias']
            self.net.load_state_dict(my_dict)

        self.prt.p("loaded pretrained %s weights from %s"%(w_type, weights_f), self.prt.LOG)
#        self.prt.p("Cluster Centers: {}".format(self.net.cluster_layer.data), self.prt.STATUS)


    def run(self):
        """end-to-end run of Trainer"""

        # run by phase
        if self.p.phase == Phase.TRAIN:
            self.prt.p("Start Training...", self.prt.STATUS)
            self.train()

        elif self.p.phase == Phase.TRAIN_VAL:
            self.prt.p("Start Training w/ Validation...", self.prt.STATUS)
            self.train(validate=True)

        elif self.p.phase == Phase.VAL:
            self.prt.p("Start Validating...", self.prt.STATUS)
            self.validate()

        elif self.p.phase == Phase.TEST:
            self.prt.p("Start Testing...", self.prt.STATUS)
            self.test()

        else:
            raise NotImplementedError


    def validate(self):
        """validate Autoencoder part of model"""

        self.curr_phase = Phase.VAL
        yhats, zs, gts, _ = self.forward_all(self.n_batches[self.curr_phase], desc="VAL_AE", return_labels=True)

        self.report_accuracy(yhats, zs, gts, prefix=ModelType.BMTR)


    def report_accuracy(self, yhats, zs, gts, prefix="Unknown", debug=True):
        """print accuracy and report to tensorboard

        Params
        ----------
        yhats: soft-labels predicted by model's cluster_layer
        zs: last layer's output (prob distribution)
        gts: ground truth
        debug: print messages or not
        """

        log_iter = self.w_iter if self.curr_phase == Phase.TRAIN else self.w_iter_val

        y_pred_cuda = torch.argmax(yhats, dim=1)
        y_pred_z_cuda = torch.argmax(zs, dim=1)

        y_pred = y_pred_cuda.data.cpu().numpy()
        y_pred_z = y_pred_z_cuda.data.cpu().numpy()

        gts_npy = gts.data.cpu().numpy()

        enc_acc = bmUtil.calc_acc(gts_npy, y_pred_z)
        cluster_acc = bmUtil.calc_acc(gts_npy, y_pred)

        if debug:
            self.prt.p("[{}] y_pred_z[:20]: {}".format(self.curr_phase, y_pred_z[:20]), self.prt.LOG)
            self.prt.p("[{}] y_pred[:20]: {}".format(self.curr_phase, y_pred[:20]), self.prt.LOG)
            self.prt.p("[{}] gts[:20]: {}".format(self.curr_phase, gts_npy[:20]), self.prt.LOG)
            self.prt.p("[%s] enc_acc: %.2f%%"%(self.curr_phase, enc_acc * 100.), self.prt.LOG)
            self.prt.p("[%s] cluster_acc: %.2f%%"%(self.curr_phase, cluster_acc * 100.), self.prt.LOG)

        if self.writer:
            self.writer.add_scalar('%s/%s/enc_acc'%(self.curr_phase, prefix), enc_acc, log_iter)
            self.writer.add_scalar('%s/%s/cluster_acc'%(self.curr_phase, prefix), cluster_acc, log_iter)

        return enc_acc * 100., cluster_acc * 100.


    def train(self, validate=False):
        """train model by 1) training LSTM Autoencoder, 2) training cluster_layer jointly"""

        # log description to tensorboard
        self.log_desc_to_tfboard()

        # train
        self.train(validate=validate)

        # freeze feature extractor?
        #self.net.config_encoder(learnable=False)

        # train cluster layer
        #self.train_cluster(validate=validate)


    def train(self, validate=False):
        """train"""

        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.p.learning_rate, momentum=0.9)
        #optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        criterion_recon = nn.MSELoss(size_average=True)      # option: size_average=True
        criterion_class = nn.CrossEntropyLoss()

        if self.p.freeze_aw:
            print("FREEZING PRETRAINED WEIGHT PART!")
            self.net.config_encoder(learnable=False)

        for ei in range(self.p.epoch):
            class_loss_all = []
            total_loss_all = []
            self.curr_phase = Phase.TRAIN
            enc_acc_all = []
            cluster_acc_all = []
            #for di, inputs in enumerate(tqdm(self.data_loader[self.curr_phase], 0)):
            n_batches = int(math.ceil(self.n_batches[self.curr_phase]))
            labels = None
            for di in tqdm(range(n_batches), desc="TRAIN"):
                # form input data
                inputs_cpu = next(iter(self.data_loader[self.curr_phase]))
                inputs, labels = self.form_net_inputs(inputs_cpu)

                # reset gradient
                optimizer.zero_grad()

                # forward
                enc, decoded_output, yhat = self.net(inputs, cluster=True)

                # compute loss
                if labels is not None:
                    class_loss = criterion_class(enc, labels)
                else:
                    class_loss = 0.0
                total_loss = class_loss

                if di == 0:
                    total_loss_all = total_loss.unsqueeze(0)
                    class_loss_all = class_loss.unsqueeze(0)
                else:
                    total_loss_all = torch.cat([total_loss_all, total_loss.unsqueeze(0)], dim=0)
                    class_loss_all = torch.cat([class_loss_all, class_loss.unsqueeze(0)], dim=0)

                # backward prop, and update weights
                total_loss.backward()
                optimizer.step()

                # report accuracy
                if labels is not None:
                    enc_acc, cluster_acc = self.report_accuracy(yhat, enc, labels, debug=False)
                    enc_acc_all.append(enc_acc)
                    cluster_acc_all.append(cluster_acc)


                # (32, 16, 1, 32, 32)
                # visualize
                self.visualize_AE_Training(ei, di, inputs, None, {'total_loss':total_loss, 'class_loss':class_loss}, prefix=ModelType.BMTR)
                resnet_x4_feat = decoded_output[0]
                self.visualize_feat(ei, di, resnet_x4_feat, 'ResNet-feat-x4', labels=labels)

            self.prt.p("AVG class_loss: %.4f, total_loss: %.4f"%(torch.mean(class_loss_all),
                                                                 torch.mean(total_loss_all)), self.prt.STATUS)
            self.prt.p("AVG enc_acc: %.2f%%, clsuter_acc: %.2f%%"%(np.mean(np.array(enc_acc_all)),
                                                                   np.mean(np.array(cluster_acc_all))), self.prt.STATUS)

            # validate only if the dataset has lables
            if validate and (self.dataset['name'] in [ Dataset.MNIST, Dataset.MICROGLIA_LABELED ]):
                self.validate()

            # save model after an epoch
            self.save_model(ei, "AE")


    def train_cluster(self, validate=False):
        """train cluster layer with pretrained autoencoder

        Params
        ----------
        initial_outputs: initial model outputs from autoencoder, as in numpy array
        """
        self.curr_phase = Phase.TRAIN

        learning_rate = 1e-3
        num_epoch = self.p.clustering_epoch
        desc = "ClusterTraining"
        n_batches = int(self.n_batches[self.curr_phase])
        print("n_batches: ", n_batches)
        dname = self.dataset['name']

        kmeans = KMeans(n_clusters=self.p.num_clusters, n_init=20)

        # forward all to get yhat
        if dname in [ Dataset.MNIST, Dataset.MICROGLIA_LABELED ]:
            yhats, zs, gts, _ = self.forward_all(n_batches, return_labels=True)
        else:
            yhats, zs, _ = self.forward_all(n_batches)

        # run KMeans clustering and update cluster_layer's data with KMeans fitted custer centers
        y_pred = kmeans.fit_predict(zs.data.cpu().numpy())
        self.net.cluster_layer.data = torch.from_numpy(np.asarray(kmeans.cluster_centers_, dtype=np.float32)).cuda()

        #self.prt.p("Cluster Centers: {}".format(self.net.cluster_layer.data), self.prt.STATUS)
        y_pred_last = np.copy(y_pred)

        # init optimizer
        #optimizer_c = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=learning_rate)
        optimizer_c = torch.optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=learning_rate, momentum=0.99)
        #criterion_recon = nn.MSELoss(size_average=True)
        criterion_recon = nn.MSELoss()
        criterion_class = nn.CrossEntropyLoss()

        isVal = self.curr_phase == Phase.VAL
        sampled = (isVal and self.p.use_sampler_in_val) or (not isVal and self.p.use_sampler)
        print("sampled? {}".format(sampled))
        f_iter = iter(self.data_loader[self.curr_phase])

        delta_label = 0.99999
        num_batches = self.p.clustering_epoch / self.p.batch_size
        cur_batch_no = 0
        for ei in tqdm(range(num_epoch), desc=desc):
            if ei % self.update_interval == 0:
                self.prt.p("Interval[{}] Cluster Centers: {}".format(ei, self.net.cluster_layer.data), self.prt.STATUS)
                # forward on all data, and update auxiliary target distribution
                if dname in [ Dataset.MNIST, Dataset.MICROGLIA_LABELED ]:
                    soft_labels, zs, gts, sample_images = self.forward_all(n_batches, num_sample_batches=20, return_labels=True)
                else:
                    soft_labels, zs, sample_images = self.forward_all(n_batches, num_sample_batches=20)

                target_p = self.net.target_distribution(soft_labels)

                # get clustering results
                y_pred_cuda = torch.argmax(soft_labels, dim=1)
                y_pred = y_pred_cuda.data.cpu().numpy()
                self.prt.p("y_pred[:20]: {}".format(y_pred[:20]), self.prt.LOG)

                if dname in [ Dataset.MNIST, Dataset.MICROGLIA_LABELED ]:
                    self.prt.p("gts[:20]: {}".format(gts[:20]), self.prt.LOG)
                    self.report_accuracy(soft_labels, zs, gts, prefix="%s_Cluster_Training"%ModelType.BMTR)
                    #acc = bmUtil.calc_acc(gts.data.cpu().numpy(), y_pred)
                    #self.prt.p("acc: %.2f%%"%(acc * 100.), self.prt.LOG)

                    # validate only if the dataset has lables
                    if validate:
                        self.validate()


                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)

                # visualize sample data in embedding
                num_sample_images = len(sample_images)
                self.visualize_feat_embedding(zs[:num_sample_images], y_pred_cuda[:num_sample_images], sample_images, ei)

                # save model after an interval
                self.save_model(ei, desc)


            if sampled:
                inputs_ = self.form_net_inputs(next(iter(self.data_loader[self.curr_phase])))
            else:
                inputs_ = self.form_net_inputs(next(f_iter))
            inputs_ = next(iter(self.data_loader[self.curr_phase]))

            if dname in [ Dataset.MNIST, Dataset.MICROGLIA_LABELED ]:
                inputs, labels = self.form_net_inputs(inputs_)
            else:
                inputs = self.form_net_inputs(inputs_)
                labels = None

            assert len(inputs) != 0

            optimizer_c.zero_grad()

            encoded_feat, decoded_output, yhat = self.net(inputs, cluster=True)

            # Reconstruction loss
            recon_loss = criterion_recon(decoded_output[0], decoded_output[1])
            class_loss = criterion_class(encoded_feat, labels) if labels is not None else 0.0
            total_loss = recon_loss + class_loss

            # KL Distance between distributions
            start_bidx = self.p.batch_size * cur_batch_no
            end_bidx = self.p.batch_size * (cur_batch_no + 1)
            tgt = target_p[start_bidx:end_bidx]
            kld = F.kl_div(yhat.log(), tgt)

            cur_batch_no += 1
            if cur_batch_no == num_batches:
                cur_batch_no = 0

            loss_c = (1. - self.gamma) * total_loss + self.gamma * kld
            print("loss_c: ", loss_c)
            loss_c.backward()
            optimizer_c.step()
            self.visualize_cluster_training(ei, [[loss_c], [kld], [kld]], delta_label)


    def log_desc_to_tfboard(self):
        """log training descriptions and parameters to tensorboard"""

        description = ""
        pdict = vars(self.p)
        for key in pdict.keys():
            if key in ['debug', 'file_ext', 'phase', 'full_resolution', 'force_inference',
                        'timestmp', 'is_eval', 'hostname', 'preproc_map']:
                        continue
            description +="[{}]: {}, ".format(key, pdict[key])

        self.writer.add_text('Description', description, self.w_iter+1)


    def forward_all(self, n_batches, num_sample_batches=0, desc='FORWARD ALL',
                    prefix=ModelType.BMTR, is_validate=False, return_labels=False):
        """forward prop on all available training data and return yhat tensors

        Params
        ----------
        n_batches: number of batches (usually, n_batches = n_iter / batch_size)
        """
        isVal = self.curr_phase == Phase.VAL
        sampled = (isVal and self.p.use_sampler_in_val) or (not isVal and self.p.use_sampler)

        if isVal:
            criterion_recon = nn.MSELoss(size_average=True)      # option: size_average=True
            criterion_class = nn.CrossEntropyLoss()

        f_iter = iter(self.data_loader[self.curr_phase])
        yhats = []
        zs = []
        sample_batches = []
        labels =[]
        #n_batches = int(math.ceil(n_batches))
        n_batches = int(n_batches)

        with torch.no_grad():
            for di in tqdm(range(n_batches), desc=desc):
                if sampled:
                    inputs_ = self.form_net_inputs(next(iter(self.data_loader[self.curr_phase])))
                else:
                    try:
                        inputs_ = self.form_net_inputs(next(iter(f_iter)))
                    except StopIteration:
                        print("EXCEPTION [ StopIteration ] di: ", di, "n_batches: ", n_batches)
                        continue


                if return_labels:
                    inputs, label = inputs_
                else:
                    inputs, _ = inputs_

                sample = inputs.squeeze()[:, int(self.net.zdepth/2), :, :]
                z, decoded_output, yhat = self.net(inputs, deconv=True, cluster=True)

                if isVal:
                    class_loss = criterion_class(z, label)
                    total_loss = class_loss

                    if self.writer:
                        self.writer.add_scalar('%s/%s/class_loss'%(self.curr_phase, prefix), class_loss, self.w_iter_val)
                        self.writer.add_scalar('%s/%s/total_loss'%(self.curr_phase, prefix), total_loss, self.w_iter_val)
                        self.w_iter_val += 1

                if di == 0:
                    if return_labels:
                        labels = label
                    yhats = yhat
                    zs = z
                    if num_sample_batches > 0:
                        sample_batches = sample
                else:
                    if return_labels:
                        labels = torch.cat((labels, label), dim=0)
                    yhats = torch.cat((yhats, yhat), dim=0)
                    zs = torch.cat((zs, z), dim=0)
                    if di < num_sample_batches:
                        sample_batches = torch.cat((sample_batches, sample), dim=0)

        if return_labels:
            return yhats, zs, labels, sample_batches
        else:
            return yhats, zs, sample_batches


    def form_net_inputs(self, _in):
        """form final inputs for the network to be fed, handling differently by dataset"""

        label = None
        dname = self.dataset['name']

        if dname == Dataset.MNIST:
            data, label = _in
            data = data.cuda()
            # as MNIST has only one channel in depth, stack up to lstm depth
            inputs = torch.stack([data]*16, dim=1)

        elif dname in [Dataset.MICROGLIA, Dataset.REALTIME]:
            # unsqueeze and load to gpu
            #inputs = F.interpolate(_in.squeeze(), size=(self.net.in_w, self.net.in_h))
            #inputs = inputs.cuda().unsqueeze(2).unsqueeze(2)
#            inputs = _in.cuda().unsqueeze(2)
            inputs = _in.cuda()
        elif dname == Dataset.MICROGLIA_LABELED:
            inputs, label = _in
            inputs = inputs.cuda().unsqueeze(2)
            label = label.cuda()
        else:
            raise NotImplementedError

        if label is not None:
            return inputs, label.cuda()
        else:
            return inputs, label


    @staticmethod
    def _compute_in_loss(hidden, yhat, threshold, criterion1, criterion2):
        bo, zo, yo, xo = hidden.size()
        inner_tg_dist = torch.cuda.FloatTensor(1).fill_(0)
        c1, c2 = [], []
        for idx, item in enumerate(yhat):
            if item < threshold:
                c1.append(hidden[idx].view(1, zo * yo * xo))
            else:
                c2.append(hidden[idx].view(1, zo * yo * xo))

        c1_loss = None
        c2_loss = None
        if len(c1):
            c1_dist = self._compute_inner_avg_dist(c1)
            c1_loss = criterion1(c1_dist, inner_tg_dist)
        if len(c2):
            c2_dist = self._compute_inner_avg_dist(c2)
            c2_loss = criterion2(c2_dist, inner_tg_dist)

        if c1_loss is None:
            loss = c2_loss
        elif c2_loss is None:
            loss = c1_loss
        else:
            loss = c1_loss + c2_loss

        # out loss
        # build target
#                targets = build_target(last_h)
#                out_loss = criterion_out(yhat, targets)

#        in_loss_weight = 0.5
        #loss = loss * in_loss_weight + out_loss * (1. - in_loss_Weight)

        return loss


    @staticmethod
    def _compute_inner_avg_dist(data):
        d = torch.cuda.FloatTensor(1).fill_(0)
        cnt = 0
        for item in list(itertools.product(data, data)):
            cnt += 1
            d += pdist(item[0], item[1])

        assert cnt != 0

        # avg
        d /= cnt

        return d


    def validata(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError


    def visualize_cluster_training(self, epoch, losses, label_error):

        """visualize results on Tensorboard - for cluster_training

        Params
        ----------
        epoch: current training epoch to visualize
        losses: list of losses to plot (e.g. total_loss, recon_loss, kl_loss)
        """

        #TODO: visualize the encoder/decoder results too
        #TODO: visualize embedding of clustered data
        total_loss, kl_loss, recon_loss = losses

        assert(len(total_loss) == len(recon_loss) == len(kl_loss))

        prefix = "clustering"

        self.writer.add_scalar('%s/%s/label delta'%(self.curr_phase, prefix), label_error, self.w_iter)
        for li in range(len(total_loss)):
            tl = total_loss[li]
            rl = recon_loss[li]
            kll = kl_loss[li]

            self.writer.add_scalar('%s/%s/recon_loss'%(self.curr_phase, prefix), rl, self.w_iter)
            self.writer.add_scalar('%s/%s/total_loss'%(self.curr_phase, prefix), tl, self.w_iter)
            self.writer.add_scalar('%s/%s/kl_loss'%(self.curr_phase, prefix), kll, self.w_iter)

            self.w_iter += 1


    def visualize_feat_embedding(self, features, labels, images, epoch):
        """visualize feature in embedding space, along with predicted labels and images in the middle step

        Params
        ----------
        features: output features of network
        labels: soft-labeled data using cluster_layer
        images: list of images taken from middle depth of 3-D volumetric data -> list of 2-D images
        """


        self.writer.add_embedding(features, metadata=labels, label_img=images.unsqueeze(1), global_step=epoch)


    def visualize_feat(self, epoch, batch_idx, feat, feat_name, labels=None, use_sns=True):
        batch_size = self.p.batch_size
        log_idx = epoch * batch_size + batch_idx

        if use_sns:
            # get dataframe from features
            feat_npy = bmUtil.t2npy(feat).squeeze()
            nr, nc = feat_npy.shape
            ivals = np.arange(nr)
            cvals = ['feat_%d'%i for i in np.arange(nc)]
            df = pd.DataFrame(data=feat_npy, index=ivals, columns=cvals)
            labels_npy = bmUtil.t2npy(labels)

            if labels is not None:
                # sort by predictions
                feat_cols = list(df.columns)
                df['Cell Class'] = labels_npy
                df.reset_index(inplace=True)
                df = df.rename(columns = {'index':'Cell ID'})
                df = df.sort_values(['Cell Class', 'Cell ID'], ascending=True)
                # get sorted labels
                labels_npy = list(df['Cell Class'].values)
                df = df[feat_cols]

            #df_corr = df.corr()

            # Draw figure
            cmp = 'Blues'
            matplotlib.use("Agg")
            fig = plt.figure()
            ax = fig.gca()
            hm = sns.heatmap(df.T, ax=ax, cmap=cmp, xticklabels=labels_npy)
            hm.set_xlabel("Cells")
            hm.set_ylabel("Features")
            fig.canvas.draw()

            # get rendered figure into a numpy array
            img = np.array(fig.canvas.renderer.buffer_rgba(), dtype='uint8')[:, :, :3] # only rgb
            img = np.swapaxes(img, 0, 2)
            img = np.flip(img, 2)
            img = np.rot90(img, axes=(1, 2))

        else:
            feat = feat.squeeze() # if ResNet x4 feat: dim would be [batch_size, 512]
            img = vutils.make_grid(feat, normalize=False, nrow=batch_size)

        self.writer.add_image('%s/Heatmaps/%s'%(self.curr_phase, feat_name), img, log_idx)

        # clear figure
        if use_sns:
            plt.clf()
            plt.close()


    def visualize_AE_Training(self, epoch, batch_idx, data_in, data_out, loss, prefix="AE"):
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

        for key in loss.keys():
            self.writer.add_scalar('%s/%s/%s'%(self.curr_phase, prefix, key), loss[key], log_idx)

        # visualize images
        if data_in is not None and data_out is not None:
            # data_in : (batch_size, z, 1, y, x) (insert a channel)
            if self.p.model_type == ModelType.ALTR:
                bsz, dim, _, _ = data_in.size()
            else:
                bsz, dim, _, _, _ = data_in.size()

            is2d = True if dim == 1 else False
            if is2d:
                a = data_in
                b = data_out
                grid_data = torch.cat((a, b), dim=0) # (bsz, dim, y, x)
                x = vutils.make_grid(grid_data, normalize=False, nrow=bsz)
            else:
                if data_out is not None:
                    grid_data = torch.cat((data_in[0], data_out[0]), dim=0)
                else:
                    grid_data =data_in[0]

                normalize = True if dname == Dataset.MNIST else False
                x = vutils.make_grid(grid_data, normalize=normalize, nrow=16)

            self.writer.add_image('%s/Input_Top_Decoded_Bottom'%self.curr_phase, x, log_idx)
        self.w_iter += 1



    def save_model(self, epoch, prefix):
        """save trained weights to a file"""

        fname = '%s/%s_%s_%05d.pth'%(self.p.save_dir, self.expr_name, prefix, epoch)
        with open(fname, 'wb') as fp:
            torch.save(self.net.state_dict(), fp)

        self.prt.p("saved model to %s"%fname, self.prt.LOG)
