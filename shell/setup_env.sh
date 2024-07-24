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
#conda install python=3.7
conda install -c menpo opencv
