"""baseAnalyzer.py: contains BaseAnalyzer Class which inherits from BMCanvas"""
__author__      = "Minyoung Kim"
__license__ = "MIT"
__maintainer__ = "Minyoung Kim"
__email__ = "minykim@mit.edu"
__date__ = "10/17/2018"

import numpy as np
from copy  import deepcopy

# internal
from utils.data.data_generator import DataGenerator
from utils.params import DataGenParams
from utils.viz.bmCanvas import BMCanvas
from utils.util import PRT


class BaseAnalyzer(BMCanvas):
    """BaseAnalyzer Class
        - contains base functions to analyze cell centers and visuzlize
    """
    def __init__(self, **args):
        super(BaseAnalyzer, self).__init__(**args)
        self.dg = None


    @staticmethod
    def adjust_coordinates(points, zr=None, yr=None, xr=None, z_damp=0, isRaw=False):
        """adjust coordinates by subracting start index of each range

        Parameters
        ----------
        points: list
            list of coordinates in (z, y, x) order
        zr: tuple
            z-range (zstart, zend)
        yr: tuple
            y-range (ystart, yend)
        xr: tuple
            x-range (xstart, xend)
        isRaw: boolean
            is data raw or downsampled (2x)
        """
        for p in points:
            if zr:
                p[0] -= zr[0]
                p[0] = min(0, p[0] - z_damp)

            if yr:
                p[1] -= yr[0]
            if xr:
                p[2] -= xr[0]

            if isRaw:
                p[1] *= 2
                p[2] *= 2

        points_npy = np.array(points)

        return points_npy


    def global_coordinates(self, points, zr=None, yr=None, xr=None):
        """recover global coordinates from adjusted ones

        Parameters
        ----------
        points: list
            list of coordinates in (z, y, x) order
        zr: tuple
            z-range (zstart, zend)
        yr: tuple
            y-range (ystart, yend)
        xr: tuple
            x-range (xstart, xend)
        """
        for p in points:
            if zr:
                p[0] += zr[0]
            if yr:
                p[1] += yr[0]
            if xr:
                p[2] += xr[0]

        points_npy = np.array(points)

        return points_npy

    @staticmethod
    def find_close_node(node, nodes, k=3):
        """find top-k closest nodes (in Euclidean space)

        Parameters
        ----------
        node: 3D points
            pivot point
        nodes: list
            neighbor points
        k: integer
            number of points to collect (sorted by euclidean distance)
        """

        nodes = np.asarray(nodes)
        dist = [ np.linalg.norm(n - node) for n in nodes]
        indices = np.argsort(dist)

        indices = indices[:k]
        k_close_nodes = nodes[indices]
        k_close_dists = np.array(dist)[indices]

        return k_close_nodes, k_close_dists


    def form_data(self, params=None, name="No-Name", ranges=[None, None, None], show=True, ccd=None,
                  shuffle=True,
                  z_damp=0):
        """formulate data from raw tiff files and annotations

        Parameters
        ------------
        params: list
            set of parameters for DataGenParams
        name: string
            name of data set
        ranges: list of tuples
            zyx ranges to crop, if None no cropping
        show: boolean
        """
        assert (params is not None) or (ccd is not None)

        zr, yr, xr = ranges
        scatter = None

        if ccd is None:
            if self.dg is None:
                dgParams = DataGenParams()
                dgParams.build(params, "DataGenParser-%s"%name)
                self.dg = DataGenerator(dgParams)
            self.dg.retrieve_files()
            vol = self.dg.grab_sub_zyxplane_new(zr, yr, xr)
        else:
            self.dg = ccd.dg
            vol = ccd.vol_raw.copy()
            print("Form_DATA: min: ", np.min(vol), "max: ", np.max(vol))

        self.dg.load_annotations(shuffle=shuffle)

        self.log("vol.shape: {}".format(vol.shape))
        if zr is None and yr is None and xr is None:
            return vol, self.dg.cc_all, self.dg.cc_all, self.dg

        points = self.dg.get_coords_in_range(zr, yr, xr, z_damp=z_damp)
        if points:
            msg = "form_data(): # of coordinates within zr({}), yr({}), xr({}): {}".format(zr, yr, xr, len(points))
            print(msg)
            self.log(msg, PRT.STATUS)
            scatter = BaseAnalyzer.adjust_coordinates(deepcopy(points), zr, yr, xr)

        return vol, points, scatter


    def set_data(self):
        """placeholder for child class"""
        pass
