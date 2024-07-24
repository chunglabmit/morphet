# -*- coding: utf-8 -*-
"""volumeHolder.py: Visualize 3D-volumetric image data, and hold analytic information"""
__author__      = "Minyoung Kim"
__license__ = "MIT"
__maintainer__ = "Minyoung Kim"
__email__ = "minykim@mit.edu"
__date__ = "05/2019"

import sys
import os
import numpy as np
from vispy import app
from vispy.visuals.transforms import STTransform
from copy import deepcopy
# internal
from utils.data.data_generator import DataGenerator
from utils.const import StainChannel, VizMarkers
from utils.params import CellDataParams
from utils.const import VizMarkers
from utils.util import BM_lEval
from utils.viz.bmCanvas import BMCanvas
from utils.data.preprocessing import BMPreprocessing
from analysis.baseAnalyzer import BaseAnalyzer



class VolumeHolder(BMCanvas):
    """VolumeHolder Class"""

    def __init__(self, **args):
        self.p = self.build_param(args.pop('params'))
        super(VolumeHolder, self).__init__(**args)

        self.bmPrep = BMPreprocessing()
        # resize voxel_size
        self.voxel_size = self.recalculate_voxel_size(self.p.voxel_size)
        self.showVoxelResized = False

        self.num_channels = len(self.p.multi_channel)
        self.showScalebar = True
        self.set_data()


    def build_param(self, args):
        p = CellDataParams()
        p.build(args, "VolumeHolderParams")
        p.print_params()

        return p


    def load_data(self):
        self.dg = DataGenerator(self.p)
        self.dg.retrieve_files()
        self.dg.load_annotations()

        zr, yr, xr = self.p.ranges
        self.vol = self.dg.grab_sub_zyxplane(zr, yr, xr)
        self.vol_rsz = None

        if self.voxel_size:
            self.vol_rsz = {}
            for k in sorted(self.vol.keys()):
                self.vol_rsz[k] = self.resize_volume(self.vol[k].copy(), self.voxel_size)


    def show_initial_data(self):
        self.curr_chn_idx = 0
        self.curr_channel = self.p.multi_channel[self.curr_chn_idx]

        self.clims = self.p.clims

        self.viz_markers, self.viz_colors, self.viz_key_binding = self.set_viz_markers()
        self.show_current_volume()

        try:
            self.points = self.grab_points()
            if len(self.points):
                self.overlay_marker(self.points, self.viz_markers[self.curr_channel],
                                    self.viz_key_binding[self.curr_channel],
                                    self.viz_colors[self.curr_channel])

        except Exception as e:
            self.prt.p("Error: " + str(e), self.prt.ERROR)
            self.points = []


    def set_data(self):
        """set data with parameters provided"""

        self.load_data()
        self.show_initial_data()



    def grab_points(self):
        zr, yr, xr = self.p.ranges
        if (not zr) and (not yr) and (not xr):
            return self.dg.cc_all

        points = self.dg.get_coords_in_range(zr, yr, xr)
        if points:
            msg = "grab_points(): # of coordinates within zr({}), yr({}), xr({}): {}".format(zr, yr, xr, len(points))
            print(msg)
            isRaw = True if 'RAW' in self.curr_channel else False
            scatter = BaseAnalyzer.adjust_coordinates(deepcopy(points), zr, yr, xr, isRaw=isRaw)

        return scatter


    @staticmethod
    def recalculate_voxel_size(voxel_size):
        """re-calculate voxel size which is more meaningful (not too small)"""

        factor = 2

        vz, vy, vx = voxel_size
        vz = float(vz)/float(vy) / factor
        vy = 1.0 / factor
        vx = 1.0 / factor

        return [vz, vy, vx]


    @staticmethod
    def resize_volume(vol, voxel_size):
        return BMPreprocessing._resize(vol, voxel_size)


    def show_current_volume(self):
        if self.showVoxelResized:
            v = self.vol_rsz[self.p.multi_channel[self.curr_chn_idx]]
        else:
            v = self.vol[self.p.multi_channel[self.curr_chn_idx]]

        self.show_volume(v, self.curr_chn_idx)


    def step_volume(self, backward=False):
        if backward:
            new_idx = (self.curr_chn_idx - 1) % self.num_channels
        else:
            new_idx = (self.curr_chn_idx + 1) % self.num_channels
        self.curr_chn_idx = new_idx
        self.curr_channel = self.p.multi_channel[self.curr_chn_idx]
        print("converting to chn [ %s ]..."%(self.curr_channel))
        self.show_current_volume()


    def show_volume(self, vol, idx):
        if self.clims[idx] is None:
            self.set_volume(vol, clim=[np.min(vol), np.max(vol)], scalebar=self.showScalebar)
        else:
            self.set_volume(vol, clim=self.clims[idx], scalebar=self.showScalebar)


    def get_curr_vol_key(self):
        return self.p.multi_channel[self.curr_chn_idx]


    def set_viz_markers(self):
        viz_markers = {}
        viz_colors = {}
        key_binding = {}
        colors = [ [1.0, 0.0, 0.0, 0.7],
                   [1.0, 1.0, 0.0, 0.9],
                   [0.0, 1.0, 0.0, 0.9],
                   [0.0, 1.0, 1.0, 0.7],
                   [0.5, 0.0, 0.0, 0.7],
                   [0.5, 0.5, 0.0, 0.7],
                   [0.0, 0.5, 0.0, 0.7],
                   [0.0, 0.5, 0.5, 0.7] ]

        keys = [ "q", "w", "e", "r", "t", "y", "u", "i" ]

        for i, ch in enumerate(self.p.multi_channel):
            viz_markers[ch] = ch
            viz_colors[ch] = colors[i]
            key_binding[ch] = keys[i]

        return viz_markers, viz_colors, key_binding


    def update_clim_max(self, decrease=False):
        if decrease:
            self.clims[self.curr_chn_idx][1] -= 150
        else:
            self.clims[self.curr_chn_idx][1] += 150


    def update_clim_min(self, decrease=False):
        if decrease:
            self.clims[self.curr_chn_idx][0] -= 150
        else:
            self.clims[self.curr_chn_idx][0] += 150



    def on_key_press(self, event):
        """overwrite on_key_press action"""

        for key in self.key_bindings:
            if event.text == key:
                self.toggle_marker(self.key_bindings[key])

        if event.text == '3':
            self.step_volume()

        if event.text == '4':
            self.step_volume(backward=True)

        if event.text == 's':
            self.set_volume_style(cmapToggle=True)

        if event.text == 'v':
            self.showVoxelResized = not self.showVoxelResized
            self.show_current_volume()

        if event.text in ["7", "8", "9", "0"]:
            if event.text == "7":
                self.update_clim_max()
            elif event.text == "8":
                self.update_clim_max(decrease=True)
            elif event.text == "9":
                self.update_clim_min()
            elif event.text == "0":
                self.update_clim_min(decrease=True)
            self.show_current_volume()


if __name__ == '__main__':
    t = sys.argv[2].split('/')[-1]
    sda = VolumeHolder(title='Volume- %s'%t, keys='interactive', size=(800, 600), show=True,
                       logWindow=None, params=sys.argv)
    app.run()
