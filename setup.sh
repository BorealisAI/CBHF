#!/bin/bash
# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


VMS_DIR=${PWD}/python-vms
PROJECT=cbhf
mkdir -p ${VMS_DIR}
virtualenv --no-download -p python3 ${VMS_DIR}/${PROJECT}
source ${VMS_DIR}/${PROJECT}/bin/activate
pip install --upgrade pip

#Install libraries
pip install numpy torch scikit-learn gym==0.17 contextualbandits tqdm matplotlib pandas ucimlrepo ipdb requests scikit-image
