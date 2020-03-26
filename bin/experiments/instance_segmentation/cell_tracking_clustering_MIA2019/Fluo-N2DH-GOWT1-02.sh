#!/bin/bash

# Run the tracking routine with three input parameters
# input_folder output_folder weights_file dataset_name
# set PYTHONPATH to root directory of the framework

# Prerequisities: python>=3.4, tensorflow>=1.4.1, numpy>=1.14.0, SimpleITK>=1.0.1, scikit-image>=0.13.1, scikit-learn>=0.19.1, Cython>=0.27.3

export PYTHONPATH=./
python bin/segment_and_track.py /media1/datasets/celltrackingchallenge/trainingdataset/Fluo-N2DH-GOWT1/02/ /media1/datasets/celltrackingchallenge/trainingdataset/Fluo-N2DH-GOWT1/gru_hdbscan/02_RES models/gru/Fluo-N2DH-GOWT1 Fluo-N2DH-GOWT1
