#!/bin/bash
# ------------------------------------------------------------------
# 04/22/2019
# install_anaconda.sh
#          Install Anaconda2
# ------------------------------------------------------------------

# --- Anaconda 2 ---
# download
VERSION=2020.02
ANACONDA_SHELLSCRIPT="Anaconda3-${VERSION}-Linux-x86_64.sh"
CHECKSUM="fef889d3939132d9caf7f56ac9174ff6"
echo "Downloading ${ANACONDA_SHELLSCRIPT}"
echo "mkdir -p tmp;cd tmp && wget http://repo.continuum.io/archive/${ANACONDA_SHELLSCRIPT}"
mkdir -p tmp && cd tmp && wget http://repo.continuum.io/archive/${ANACONDA_SHELLSCRIPT}
# verify
echo "getting md5sum..."
md5sum ${ANACONDA_SHELLSCRIPT}

read -p "Does checksum match with [${CHECKSUM}]? (y/n) [y]"  yn
yn=${yn:-y}
echo "yn: $yn"
case $yn in
        [Yy]* ) ;;	# continue
        [Nn]* ) echo "Please check the download file";exit;;
        * ) echo "Please answer yes or no.";;
    esac

# install
# IMPORTANT: click [yes] when asked to add PATH to .bashrc!
echo "bash ${ANACONDA_SHELLSCRIPT}"
bash ${ANACONDA_SHELLSCRIPT}
source ~/.bashrc

# delete download file
echo "deleting downloaded file (${ANACONDA_SHELLSCRIPT})..."
cd ../ && rm -rf tmp

