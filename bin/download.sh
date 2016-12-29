#!/usr/bin/env bash

if [ -z ${AENET_DATA_DIR+x} ]; then
 export AENET_DATA_DIR='./'
fi
echo "VAR SET TO $AENET_DATA_DIR"

# Create data dir
mkdir $AENET_DATA_DIR

# Download the weights, mean and std of the model
wget -N https://data.vision.ee.ethz.ch/cvl/aenet_feat_data/gstd.npy -P $AENET_DATA_DIR
wget -N https://data.vision.ee.ethz.ch/cvl/aenet_feat_data/gmean.npy -P $AENET_DATA_DIR
wget -N https://data.vision.ee.ethz.ch/cvl/aenet_feat_data/model.pkl -P $AENET_DATA_DIR

# Download HTK config
wget -N https://data.vision.ee.ethz.ch/cvl/aenet_feat_data/configmfb.hcopy -P $AENET_DATA_DIR
