# Audio Event Net #

This README shows how to run AENet which classify the audio events.

## Requirements ##

* Python 2
* Lasagne (https://github.com/Lasagne/Lasagne)
* HCopy from HTK (http://htk.eng.cam.ac.uk/)

## How to use ##

#### Prerequisites ####
* Install all requirements. (This repository includes "HCopy" but you probably need to compile on your machine.)
* Define the path where you want to keep your models with
 ``export AENET_DATA_DIR='YOUR_DATA_PATH'``

#### Run example ####
Run ``python run_sample.py`` to test the installation and see how to use the code.

* This sample program reads wave files and stores corresponding AENet features in "ae_feat".
* For testing your installation, the sample program also displays the error between your extraction and a reference.
* If you got a high error, it might be due to the installtion of "HCopy".

#### Install as a package ####
* Make sure to have set the data directory: ``AENET_DATA_DIR``
* Download the model files by running ``bin/download.sh``
* Install the package with ``python setup.py install``
* Now you can use it via ``import aenet``

## Supported format ##
Currently only wave file format with 16kHz sampling rate, 16bit, monoral channel is supported.
If you would like to extract AENet feature from other format audio files, please first convert it.

For convenience the class ``aenet.AENet`` contains the function ``write_wav`` which writes the audio stream of a video
in the correct format using [moviepy](http://zulko.github.io/moviepy/).

## Reference ##
If you end up using this code or the pre-trained network, we ask you to cite the following paper:

**Naoya Takahashi, Michael Gygli, and Luc Van Gool, "AENet: Learning Deep Audio Features for Video Analysis", arXiv preprint arXiv:1701.00599, 2017.**
