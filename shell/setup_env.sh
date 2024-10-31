#!/bin/bash
# ------------------------------------------------------------------
# 07/17/2018
# [Minyoung Kim] setup_env.sh
#          Install required packages
#
#          Anaconda should be installed before running this script
# ------------------------------------------------------------------

#ROOT=`pwd`
#echo "ROOT: ${ROOT}"

pip install numpy
# install required packages with pip
pip install -r requirements.txt --no-cache-dir

#----------------------------
# install pytorch and opencv
#----------------------------
pip3 install torch torchvision torchaudio
pip install opencv-python-headless


# install SimpleITK
conda install -c https://conda.anaconda.org/simpleitk SimpleITK

# install precomputed-tif
pip install https://github.com/chunglabmit/blockfs/archive/master.zip
pip install https://github.com/chunglabmit/precomputed-tif/archive/master.zip
