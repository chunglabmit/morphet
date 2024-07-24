"""main.py: Main entry point for Training"""
__author__      = "Minyoung Kim"
__license__ = "MIT"
__maintainer__ = "Minyoung Kim"
__email__ = "minykim@mit.edu"
__date__ = "08/15/2018"

import sys
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import numpy as np

# internal
from utils.params import TrainParams
from utils.data.dataset import MicrogliaDataset, TrapDataset
from utils.train.preprocessor import TrainPreprocessor
from train.trainer import Trainer
from train.bfmcTrainer import BFMCTrainer
from train.trapTrainer import TRAPTrainer
from train.mTrainer import MTrainer
from utils.const import Dataset, ModelType
from torchvision import datasets, transforms
from utils.util import PRT


def build_dataset(trParams, log=None):
    # get preprocessor
    trPreprocessor = TrainPreprocessor()

    # get dataset
    if trParams.dataset == Dataset.MNIST:
        db = {}
        db['name'] = trParams.dataset
        phases = [Phase.TRAIN, Phase.VAL]
        for p in phases:
            isTrain = True if p == Phase.TRAIN else False
            d = datasets.MNIST('./mnist', train=isTrain, download=True,
                               transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                         ]))
            if log:
                log("Total [ {} ] data found in [ {} ] phase (with_label={}).".format(len(d),
                    d.phase), PRT.STATUS2)
            else:
                print("Total [ {} ] data found in [ {} ] phase.".format(len(d), d.phase))

            db[p] = d
            db[p].phase = p

    elif Dataset.MICROGLIA in trParams.dataset:
        isLabeled = True if trParams.dataset == Dataset.MICROGLIA_LABELED else False
        phases = trParams.phase.split('_')
        nc = trParams.num_clusters
        db = {}
        db['name'] = trParams.dataset
        for p in phases:
            d = MicrogliaDataset(trParams.data_path, trParams.file_ext, p, nc, trPreprocessor, with_labels=isLabeled,
                                 ntype=trParams.norm_type, clip=trParams.clip_data)
            if log:
                log("Total [ {} ] data found in [ {} ] phase (with_label={}).".format(len(d), d.phase, isLabeled), PRT.STATUS2)
            else:
                print("Total [ {} ] data found in [ {} ] phase (with_label={}).".format(len(d), d.phase, isLabeled))
            db[p] = d

    elif Dataset.TRAP in trParams.dataset:
        phases = trParams.phase.split('_')
        db = {}
        nc = trParams.num_clusters
        db['name'] = trParams.dataset
        for p in phases:
            d = TrapDataset(data_path=trParams.data_path, ext=trParams.file_ext, with_labels=True,
                            return_fname=True,
                            phase=p, num_clusters=nc, preprocessor=trPreprocessor,
                            data_size=[trParams.data_d, trParams.data_h, trParams.data_w], # (z, y, x) order
                            ntype=trParams.norm_type, clip=trParams.clip_data)
            if log:
                log("Total [ {} ] data found in [ {} ] phase.".format(len(d), d.phase), PRT.STATUS2)
            else:
                print("Total [ {} ] data found in [ {} ] phase.".format(len(d), d.phase))
            db[p] = d

    else:
        raise DatasetNotRecognizedError

    return db


def build_trainer(trParams, db):
    print("model_type: ", trParams.model_type)
    if trParams.model_type == ModelType.BMTR:
        trainer = Trainer(name="BMTR", params=trParams, dataset=db)
    elif trParams.model_type == ModelType.BFMC:
        trainer = BFMCTrainer(name="BFMC", params=trParams, dataset=db)
    elif trParams.model_type == ModelType.TRAP:
        trainer = TRAPTrainer(name="TRAP", params=trParams, dataset=db)
    elif trParams.model_type == ModelType.ALTR:
        trainer = MTrainer(name="ALTR", params=trParams, dataset=db)
    else:
        raise UnknownModelType

    return trainer


def train(trParams):
    """train entrypoint

    Params
    ---------
    trParams: TrainParams object
    """
    # build dataset
    db = build_dataset(trParams)
    # bulid trainer
    trainer = build_trainer(trParams, db)
    # run training
    trainer.run()


if __name__ == "__main__":
    # build TrainParams object
    runargs = sys.argv
    trParams = TrainParams()
    trParams.build(runargs, "TRParser")

    # run train
    train(trParams)
