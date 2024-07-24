"""rawData.py: Data Holder Class"""
__author__      = "Minyoung Kim"
__license__ = "MIT"
__maintainer__ = "Minyoung Kim"
__email__ = "minykim@mit.edu"
__date__ = "08/03/2018"


import sys
from torch.utils.data import DataLoader
from utils.data.dataset import MicrogliaDataset


class RawData(object):
    """Raw Data class for holding numpy data, and data loader for annotation"""

    def __init__(self, data_path, ext, phase):
        self.data_path = data_path
        self.ext = ext
        self.phase = phase
        self.batch_size = 100
        self.create_db_and_loader()
        self.cur_bIdx = 0

    def create_db_and_loader(self):
        """create dataset class object and DataLoader"""

        # no preprocess, and return_fname set True
        self.db = MicrogliaDataset(self.data_path, self.ext, self.phase, None, None,
                                    with_labels=False, return_fname=True)
        if len(self.db) == 0:
            raise NoDataAVailableException

        self.data_loader = DataLoader(self.db, batch_size=self.batch_size, shuffle=False, num_workers=4)
        self.total_num_batches = len(self.db) / self.batch_size
        self.dl_iter = iter(self.data_loader)

    def reset(self, new_data_path, new_phase):
        """reset dataset and DataLoader with new_data_path if necessary

        :param new_data_path: new path for data location
        :param new_phase: new phase for data
        """

        if self.data_path == new_data_path and self.phase == new_phase:
            # nothing to do
            return
        else:
            # update data_loader with new data_path
            self.data_path = new_data_path
            self.phase = new_phase
            self.create_db_and_loader()

    def get_a_batch(self):
        """grab a batch of data and return with current batch number"""
        try:
            a_batch = next(self.dl_iter)
        except StopIteration:
            # reset iteration
            self.cur_bIdx = 0
            self.dl_iter = iter(self.data_loader)
            a_batch = next(self.dl_iter)

        last_bIdx = self.cur_bIdx
        self.cur_bIdx += 1

        return last_bIdx, a_batch
