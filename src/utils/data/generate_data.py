#!/usr/bin/env python3
"""generate_data.py: script to generate data using DataGenerator()"""

__author__      = "Minyoung Kim"
__license__ = "MIT"
__maintainer__ = "Minyoung Kim"
__email__ = "minykim@mit.edu"
__date__ = "09/20/2018"

from multiprocessing import Pool, freeze_support
from itertools import repeat
from functools import partial

import utils.util as bmUtil
from data_generator import DataGenerator
from utils.params import DataGenParams
import sys

def worker(p, phase, batch):
    p.add_phase_to_savepath(phase)
    dg = DataGenerator(p)
    dg.retrieve_files()
    dg.load_annotations()

    num, start = batch
    dg.generate_data(num_samples=num, jump_to=start)


def main(params):

    if True:
        ncpu = params.num_cpu
        ns = params.num_samples
        phase = ['train'] * ncpu + ['val'] * ncpu + ['test'] * ncpu
        gap_factor = 2
        batch = [(ns, ns * gap_factor * i) for i in range(ncpu * 3)]
        full_args = zip(repeat(params), phase, batch)
        with Pool(processes=ncpu*3) as pool:
            pool.starmap(worker, full_args)

    else:
        worker(params, 'train', [400, 0])


if __name__=="__main__":
    dgParams = DataGenParams()
    dgParams.build(sys.argv, "DataGenParser")
    freeze_support()
    main(dgParams)
