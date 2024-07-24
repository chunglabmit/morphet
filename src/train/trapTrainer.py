"""trapTrainer.py: Model Training Class for TRAP dataset"""
__author__      = "Minyoung Kim"
__license__ = "MIT"
__maintainer__ = "Minyoung Kim"
__email__ = "minykim@mit.edu"
__date__ = "02/20/2020"

import sys
sys.path.append("./")
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


# internal
from train.arch.TRAPNet import TRAP3DResNet
from utils.const import Phase, Dataset
import utils.util as bmUtil

pdist = nn.PairwiseDistance(p=2)

class TRAPTrainer(object):
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

        if self.p.ae_weights is not None:
            self.ae_preloaded = True
            self.load_weights(ae_only=True)

        elif self.p.pretrained_weights_all is not None:
            self.ae_preloaded = True
            self.load_weights()

        try:
            self.init_data_loader()
        except Exception as e:
            self.prt.p("Error: " + str(e), self.prt.ERROR)
            self.prt.p("init_data_loader() failed. skip loading data", self.prt.WARNING)
            pass

        if self.p.save_dir is not None:
            self.save_params()


    def init_network(self, is_eval=False):
        if is_eval:
            return TRAP3DResNet(self.p).cuda().eval()
        else:
            return TRAP3DResNet(self.p).cuda()


    def print_params(self):
        # TODO: make a base class and inherit
        return bmUtil.print_class_params(self.__class__.__name__, vars(self))


    @staticmethod
    def _save_params(msg, fname, save_dir):
        """save parameters to a file

        Params
        ----------
        msg: lines of messages as a list
        fname: base filename
        save_dir: save location
        """

        tb, info, te = msg
        fullpath = '/'.join([save_dir, fname])
        with open(fullpath, 'a') as fp:
            fp.write(tb + '\n')
            fp.write(info + '\n')
            fp.write(te + '\n')


    def save_params(self):
        """save all params information to files"""

        self._save_params(self.net.print_params(), 'NetParams.txt', self.p.save_dir)
        self._save_params(self.p.print_params(), 'Params.txt', self.p.save_dir)
        self._save_params(self.print_params(), 'TrainerParams.txt', self.p.save_dir)


    def init_data_loader(self):
        kwargs = {'num_workers': 16, 'pin_memory': True}        # only for cuda!
        self.data_loader = {}
        self.n_batches = {}

        for key in self.dataset.keys():
            if key == 'name':
                continue

            if key == Phase.TRAIN and self.p.use_sampler:
                weights = self.dataset[key].weights
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
                #if 'fc' in k or 'dfc' in k or 'cluster_layer' in k:
                #    continue
                if k == 'fcconv.weight':
                    fcw = weights['fc.weight']
                    o, d = fcw.size()
                    fcw = fcw.view(o, d, 1, 1, 1)
                    my_dict[k] = fcw
                elif k == 'fcconv.bias':
                    my_dict[k] = weights['fc.bias']
                elif k in weights:
                    #print("k: ", k)
                    my_dict[k] = weights[k]
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


    def validate(self, stage="AE_Training"):
        """validate Autoencoder part of model"""

        self.curr_phase = Phase.VAL
        yhats, zs, gts, _ = self.forward_all(self.n_batches[self.curr_phase], desc="VAL_AE", return_labels=True)

        self.report_accuracy(yhats, zs, gts, stage=stage)


    def report_accuracy(self, yhats, zs, gts, stage="AE_Training", debug=True):
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
            self.writer.add_scalar('%s/%s/enc_acc'%(self.curr_phase, stage), enc_acc, log_iter)
            self.writer.add_scalar('%s/%s/cluster_acc'%(self.curr_phase, stage), cluster_acc, log_iter)

        return enc_acc * 100., cluster_acc * 100.


    def train(self, validate=False):
        """train model by 1) training LSTM Autoencoder, 2) training cluster_layer jointly"""

        # log description to tensorboard
        self.log_desc_to_tfboard()

        # train / load Autoencoder model
        if not self.ae_preloaded:
            self.train_AE(validate=validate)

        # freeze feature extractor?
        #self.net.config_encoder(learnable=False)

    def train_AE(self, validate=False):
        """train Autoencoder"""

        learning_rate = 1e-3
        optimizer = torch.optim.SGD(self.net.parameters(), lr=learning_rate, momentum=0.9)
        #optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        criterion_recon = nn.MSELoss(size_average=True)      # option: size_average=True
        criterion_class = nn.CrossEntropyLoss()


        for ei in range(self.p.epoch):
            recon_loss_all = []
            class_loss_all = []
            total_loss_all = []
            self.curr_phase = Phase.TRAIN
            enc_acc_all = []
            cluster_acc_all = []
            #for di, inputs in enumerate(tqdm(self.data_loader[self.curr_phase], 0)):
            n_batches = int(math.ceil(self.n_batches[self.curr_phase]))
            labels = None

            it = iter(self.data_loader[self.curr_phase])
            for di in tqdm(range(n_batches), desc="TRAIN"):
                # form input data
                #`inputs_cpu = next(iter(self.data_loader[self.curr_phase]))
                inputs_cpu = next(it)
                _, inputs, labels = self.form_net_inputs(inputs_cpu)

                # reset gradient
                optimizer.zero_grad()

                # forward
                enc, decoded_output, yhat = self.net(inputs, cluster=True)

                # compute loss
                ##recon_loss = criterion_recon(decoded_output[0], decoded_output[1])
                #recon_loss.zero_()
                class_loss = criterion_class(enc, labels)
                #total_loss = recon_loss + class_loss
                total_loss = class_loss
                #print("total_loss: {}, recon_loss: {}, class_loss: {}".format(total_loss, recon_loss, class_loss))
                #print("total_loss: {}, class_loss: {}".format(total_loss, class_loss))

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
                self.visualize_AE_Training(ei, di, inputs, None, {'total_loss':total_loss, 'class_loss':class_loss})

            self.prt.p("AVG class_loss: %.4f, total_loss: %.4f"%(torch.mean(class_loss_all),
                                                                 torch.mean(total_loss_all)), self.prt.STATUS)
            self.prt.p("AVG enc_acc: %.2f%%, clsuter_acc: %.2f%%"%(np.mean(np.array(enc_acc_all)),
                                                                   np.mean(np.array(cluster_acc_all))), self.prt.STATUS)

            # validate only if the dataset has lables
            if validate and (self.dataset['name'] in [ Dataset.TRAP, Dataset.MNIST, Dataset.MICROGLIA_LABELED ]):
                self.validate()

            # save model after an epoch
            self.save_model(ei, "AE")


    def log_desc_to_tfboard(self):
        """log training descriptions and parameters to tensorboard"""

        description = "[data_w]: {} [data_h]: {} [data_d]: {} ".format(self.p.data_w, self.p.data_h, self.p.data_d)
        description += "[batch_size]: {}, [ae_weights]: {} ".format(self.p.batch_size, self.p.ae_weights)
        description += "[epoch]: {} [clustering_epoch]: {} ".format(self.p.epoch, self.p.clustering_epoch)
        description += "[num_clusters]: {} ".format(self.p.num_clusters)
        description += "[model_type]: {} ".format(self.p.model_type)
        description += "[description]: {}".format(self.p.description)

        self.writer.add_text('Description', description, self.w_iter+1)


    def forward_all(self, n_batches, num_sample_batches=0, desc='FORWARD ALL', is_validate=False, return_labels=False):
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
                    inputs_cpu = next(f_iter)
                    inputs_ = self.form_net_inputs(inputs_cpu)
                    #inputs_ = self.form_net_inputs(next(iter(self.data_loader[self.curr_phase])))
                else:
                    try:
                        #inputs_ = self.form_net_inputs(next(iter(f_iter)))
                        inputs_ = self.form_net_inputs(next(f_iter))
                    except StopIteration:
                        print("EXCEPTION [ StopIteration ] di: ", di, "n_batches: ", n_batches)
                        continue


                if return_labels:
                    _, inputs, label = inputs_
                else:
                    _, inputs, _ = inputs_

                if len(inputs.size()) > 5:
                    inputs = inputs.squeeze(1)

                sample = inputs[:, :, int(self.net.zdepth/2), :, :]
                z, decoded_output, yhat = self.net(inputs, deconv=True, cluster=True)

                if isVal:
                    class_loss = criterion_class(z, label)
                    total_loss = class_loss

                    if self.writer:
                        self.writer.add_scalar('%s/class_loss'%self.curr_phase, class_loss, self.w_iter_val)
                        self.writer.add_scalar('%s/total_loss'%self.curr_phase, total_loss, self.w_iter_val)
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
        """form final inputs for the network to be fed, handling differently by dataset
           * for TRAP dataset, it's always supervised-learning, and have labels ready
        """
        label = None
        dname = self.dataset['name']

        # get data + label
        fnames, inputs, label = _in

        # check dimension
        slen = len(inputs.size())
        if slen < 5:
            # only expand when necessary
            inputs = inputs.unsqueeze(2)

        return fnames, inputs.cuda(), label.cuda()


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


    def test(self):
        raise NotImplementedError


    def visualize_feat_embedding(self, features, labels, images, epoch):
        """visualize feature in embedding space, along with predicted labels and images in the middle step

        Params
        ----------
        features: output features of network
        labels: soft-labeled data using cluster_layer
        images: list of images taken from middle depth of 3-D volumetric data -> list of 2-D images
        """


        self.writer.add_embedding(features, metadata=labels, label_img=images.unsqueeze(1), global_step=epoch)



    def visualize_AE_Training(self, epoch, batch_idx, data_in, data_out, loss):
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

        prefix = "AE_Training"
        for key in loss.keys():
            self.writer.add_scalar('%s/%s/%s'%(self.curr_phase, prefix, key), loss[key], log_idx)

        # visualize images
        # data_in : (batch_size, z, 1, y, x) (insert a channel)
        if data_out is not None:
            grid_data = torch.cat((data_in[0], data_out[0]), dim=0)
        else:
            grid_data = data_in[0][0].unsqueeze(2)

        normalize = True if dname == Dataset.MNIST else False
        x = vutils.make_grid(grid_data[0], normalize=normalize, nrow=32)
        self.writer.add_image('%s/Input_Top_Decoded_Bottom'%self.curr_phase, x, log_idx)
        self.w_iter += 1



    def save_model(self, epoch, prefix):
        """save trained weights to a file"""

        fname = '%s/%s_%s_%05d.pth'%(self.p.save_dir, self.expr_name, prefix, epoch)
        with open(fname, 'wb') as fp:
            torch.save(self.net.state_dict(), fp)

        self.prt.p("saved model to %s"%fname, self.prt.LOG)
