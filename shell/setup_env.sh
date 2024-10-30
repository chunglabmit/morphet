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

# install required packages with pip
pip install -r requirements.txt

#----------------------------
# install pytorch and opencv
#----------------------------
conda install pytorch torchvision cudatoolkit=11.6 -c pytorch
pip install opencv-python

# install SimpleITK
conda install -c https://conda.anaconda.org/simpleitk SimpleITK

