#!/bin/bash

# Run the tracking routine with three input parameters
# input_folder output_folder weights_file dataset_name
# set PYTHONPATH to root directory of the framework

# Prerequisities: python>=3.4, tensorflow==1.4.1, numpy==1.14.0, SimpleITK==1.0.1, scikit-image==0.13.1, scikit-learn==0.19.1, Cython==0.27.3, hdbscan==0.8.4

python bin/segment_and_track.py ../Fluo-C2DL-MSC/01 ../Fluo-C2DL-MSC/01_RES models/Fluo-C2DL-MSC Fluo-C2DL-MSC
