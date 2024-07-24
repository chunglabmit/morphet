"""preprocessor.py: Preprocessing Classes"""
__author__      = "Minyoung Kim"
__license__ = "MIT"
__maintainer__ = "Minyoung Kim"
__email__ = "minykim@mit.edu"
__date__ = "08/15/2018"

import numpy as np

# internal
from utils.const import NormalizationType
from utils.data.preprocessing import BMPreprocessing


class TrainPreprocessor(BMPreprocessing):
    def __init__(self, **args):
        super(TrainPreprocessor, self).__init__(**args)
        self.scale = [1.0, 1.0, 1.0]
        self.intensity_scale = 10.0     # 1.0


    def preprocess_all(self, array, center_crop=False,
                                    norm=True, norm_min=None, norm_max=None,
                                    normByPercentile=None,
                                    clip = False,
                                    rescale=(0.05, 99.9),
                                    resize=None,
                                    sum_limit=0.1,
                                    ntype=NormalizationType.ZERO_AND_ONE):

        """do everything

        Parameters
        ----------
        array: 3-D array (z, y, x)
            input data
        center_crop: boolean
            if to crop from center (deprecated)
        norm: boolean
            if normalize
        norm_min: float
            clim min for normalization
        norm_max: float
            clim max for normalization
        rescale: a tuple or None
            rescale parameter for contrast stretching
        """

        data = array.copy()
        if norm:
            data = self._normalize(data, cmin=norm_min, cmax=norm_max,
                                         percentile=normByPercentile,
                                         clip=clip, ntype=ntype)
        if rescale is not None and np.average(data) > sum_limit:
            data = self._rescale_intensity(data, rescale)

        if resize is not None:
            data = self._resize(data, resize)


        return data
