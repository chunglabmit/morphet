# -*- coding: utf-8 -*-
"""cellCenterAnalyzer.py: Class for Cell Visualization with Markers"""
__author__      = "Minyoung Kim"
__license__ = "MIT"
__maintainer__ = "Minyoung Kim"
__email__ = "minykim@mit.edu"
__date__ = "10/15/2018"

import sys
import numpy as np
from vispy.visuals.transforms import STTransform

# internal
from utils.const import VizMarkers
if sys.version_info >= (3,0):
    from analysis.baseAnalyzer import BaseAnalyzer
else:
    from baseAnalyzer import BaseAnalyzer

from utils.data.preprocessing import BMPreprocessing

class CellCenterAnalyzer(BaseAnalyzer):
    """CellCenterAnalyzer Class"""

    def __init__(self, **args):
        self.gparams = args.pop('gparams')
        self.mparams = args.pop('mparams')
        self.nparams = args.pop('nparams')
        self.clims = args.pop('clims')
        self.ranges = args.pop('ranges')
        self.dist_threshold = args.pop('dist_threshold')
        self.voxelSize = args.pop('voxel_size')
        if self.voxelSize[0] is None:
            self.voxelSize = None

        super(CellCenterAnalyzer, self).__init__(**args)

        self.bmPrep = BMPreprocessing()

        self.drawNuclei = True
        self.showMicroglia = True
        self.showScalebar = True
        self.set_data()


    def set_data(self):
        """set data with parameters provided"""

        # setup Microglia Data
        self.mVol, _, self.mPoints = self.form_data(self.gparams+self.mparams, VizMarkers.GFP_MICROGLIA, self.ranges)
        print("mVol min: {}, max: {}".format(np.min(self.mVol), np.max(self.mVol)))
        self.mVol_rsz = self.resize_volume(self.mVol.copy(), self.voxelSize) if self.voxelSize else None

        self.show_volume(self.mVol.copy(), idx=0)

        self.volume.cmap = 'grays'  # viridis, hot, hsl
        if self.mPoints is not None:
            self.show_chn1_centers()

        # setup Nuclei Data
        if self.nparams:
            self.nVol, _, self.nPoints, self.nDG = self.form_data(self.gparams+self.nparams, VizMarkers.TOPRO3_NUCLEI, self.ranges)
            print("nVol min: {}, max: {}".format(np.min(self.nVol), np.max(self.nVol)))
            self.nVol_rsz = self.resize_volume(self.nVol.copy(), self.voxelSize) if self.voxelSize else None
        else:
            self.nVol = None


    def switch_volume(self):
        if self.showMicroglia:
            self.show_volume(self.mVol.copy(), idx=0)
        else:
            self.show_volume(self.nVol.copy(), idx=1)


    @staticmethod
    def resize_volume(vol, voxelSize):
        return BMPreprocessing._resize(vol, voxelSize)


    def show_volume(self, vol, idx):
        if self.voxelSize:
            vol = self.resize_volume(vol.copy(), self.voxelSize)

        if self.clims is None or self.clims[idx] is None:
            self.set_volume(vol, clim=[np.min(vol), np.max(vol)], scalebar=self.showScalebar)
        else:
            self.set_volume(vol, clim=self.clims[idx], scalebar=self.showScalebar)


    def show_chn1_centers(self, color=[1.0, 0.0, 0.0, 0.7]):
        self.overlay_marker(self.mPoints, VizMarkers.GFP_MICROGLIA, "m", color=color)


    def show_chn2_centers(self, color=[0.0, 0.0, 1.0, 0.5]):
        self.overlay_marker(self.nPoints, VizMarkers.TOPRO3_NUCLEI, "n", color=color)


    def finetune_centers(self):
        """filter center coordinates of microglia, using nuclei points"""

        print("Finetuning centers with threshold %.2f"%self.dist_threshold)
        zs, ys, xs = self.mVol.shape if self.showMicroglia else self.nVol.shape
        self.mainDescription.text = "Axis: (R:x, G:y, B:z), Vol: (%d,%d,%d), Threshold: %.1f"%(xs, ys, zs, self.dist_threshold)
        if self.drawNuclei:
            self.overlay_marker(self.nPoints, VizMarkers.TOPRO3_NUCLEI, 't', color=[0.0, 0.0, 1.0, 0.4])
        good = []
        bad = []
        close_nuclei = {}
        for idx, gp in enumerate(self.mPoints):
            cn, cd = self.find_close_node(gp, self.nPoints, k=1)
            if cd[0] > self.dist_threshold:
                bad.append(idx)
            else:
                good.append(idx)
                key = tuple(cn[0])
                if key in close_nuclei.keys():
                    close_nuclei[key].append(idx)
                else:
                    close_nuclei[key] = [idx]

        print("close_nuclei: {}".format(close_nuclei))
        if len(close_nuclei) and self.drawNuclei:
            self.overlay_marker(np.array(close_nuclei.keys()), VizMarkers.TOPRO3_NUCLEI_NEARBY, 'n', color=[0.0, 1.0, 0.0, 0.9])

        self.overlay_marker(self.mPoints[good, :], VizMarkers.GFP_MICROGLIA_GOOD, "g", color=[1.0, 0.0, 0.0, 0.7])
        self.overlay_marker(self.mPoints[bad, :], VizMarkers.GFP_MICROGLIA_BAD, "b", color=[1.0, 1.0, 0.0, 0.9])

        print("Finetuning centers with threshold %.2f (Done)"%self.dist_threshold)


    def increase_threshold(self):
        """increase threshold by 1.0"""
        self.dist_threshold += 1.0


    def decrease_threshold(self):
        """decrease threshold by 1.0"""
        self.dist_threshold -= 1.0


    def on_key_press(self, event):
        """overwrite on_key_press action"""

        for key in self.key_bindings:
            if event.text == key:
                self.toggle_marker(self.key_bindings[key])

        if event.text == '3':
            if self.nVol is not None:
                self.showMicroglia = not self.showMicroglia
                self.switch_volume()

        if event.text == 's':
            self.set_volume_style(cmapToggle=True)

        if event.text in ['1', '2']:
            if self.nVol is not None:
                if event.text == '1':
                    self.decrease_threshold()
                else:
                    self.increase_threshold()
                self.finetune_centers()

        if event.text == 'd':
            self.drawNuclei = not self.drawNuclei
