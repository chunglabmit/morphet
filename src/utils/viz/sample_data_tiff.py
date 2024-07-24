import numpy as np
import sys
import os
from glob import glob
import pickle

from vispy import app, scene, io
from vispy.scene import visuals
from vispy.visuals.transforms import STTransform

from bmCanvas import BMCanvas
import tifffile

def run(loc, bmc):
    vol = tifffile.imread(loc)
    print("vol.shape: ", vol.shape)
    bmc.set_volume(vol)

if __name__ == '__main__':
    bmc = BMCanvas()
    run(sys.argv[1], bmc)
    app.run()

