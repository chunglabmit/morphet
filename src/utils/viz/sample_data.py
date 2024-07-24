import numpy as np
import os
from glob import glob
import pickle

from vispy import app, scene, io
from vispy.scene import visuals
from vispy.visuals.transforms import STTransform

from bmCanvas import BMCanvas

droot = "/data2/raw/TRAP/SusumuTonegawa_ST-Ctx-1L/cFos/training_data/cc"
date = "020620"
phase = "val"

def run(bmc):
    dpath = os.path.join(droot, date, phase, "labels")
    files = glob("%s/*.p"%dpath)

    imgs = []
    lbls = []
    cnt = 50

    with open(files[0], "rb") as fp:
        bl = pickle.load(fp)
        for key in sorted(bl.keys()):
            imgs.append(key)
            lbls.append(bl[key])

    for item in imgs:
        f = glob("%s/%s"%(os.path.join(droot, date, phase), item))[0]
        print("f: ", f)
        data = np.load(f)
        print("data.shape: ", data.shape)

        bmc.set_volume(data[0])

        break

if __name__ == '__main__':
    bmc = BMCanvas()
    run(bmc)
    app.run()

