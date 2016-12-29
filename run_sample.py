# Author: Naoya Takahashi
# last modified: 12/28/2016

import os
from aenet import AENet
import numpy as np

# Download the data
os.system('bin/download.sh')

# Define AENet object
ae = AENet()

# Specify wave files here to extract features. wave files should be fs=16kHz, mono
wave_files =['wav/airplane_2.wav', 'wav/acoustic_guitar_60.wav']

# extract AENet features
ae_feat = ae.feat_extract(wave_files, shift=100)


# test if you could successfully extract the AENet feature by comparing with a reference.
ref_feat = np.load('feat/airplane_2.npy')
error = np.max(np.abs(ref_feat -ae_feat[0]))
print ('maximum error ratio %f' %(error/np.mean(np.abs(ref_feat))))

print 'finished'
