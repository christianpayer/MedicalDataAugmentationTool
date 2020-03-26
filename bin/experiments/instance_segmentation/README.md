# Instance Segmentation and Tracking with Cosine Embeddings and Recurrent Hourglass Networks

## Usage
This example shows how to train networks of the papers [Instance Segmentation and Tracking with Cosine Embeddings and Recurrent Hourglass Networks](https://doi.org/10.1007/978-3-030-00934-2_1) and [Segmenting and Tracking Cell Instances with Cosine Embeddings and Recurrent Hourglass Networks](https://doi.org/10.1016/j.media.2019.06.015).
If you have problems/questions/suggestions about the code, write me a [mail](mailto:christian.payer@gmx.net)!

### Dataset preprocessing
Download the datasets from the [celltracking challenge](http://www.celltrackingchallenge.net/) and extract them. In order to be able to load them with this framework, they need to be preprocessed. Open `main_preprocess.py` and change the variable `dataset_base_folder` to the folder, where you extracted the files from the challenge.
Run `main_preprocess.py`. It should create `.mha` files of the input images and the groundtruth. This program also makes sure, that the IDs from the segmentation and tracking files are consistent throughout the videos.

### Train models
Run `cell_segmentation/main.py` for a non-recurrent version (untested) and `cell_tracking/main.py` for the recurrent version of the network.
The MIA version that allows tiled processing is under `cell_tracking_MIA2019/main.py`. This version also performs instance segmentation during testing.

The MIA version shows how to use a dedicated process for faster preprocessing (by reducing the python GIL). For this, run the script `cell_tracking_MIA2019/server_dataset_loop.py` and set `self.use_pyro_dataset = True` in `cell_tracking_MIA2019/main.py`. You can also run this process on a remote server. You would need to adapt the base_folder and server address for that.

### Perform instance segmentation
The code for performing instance segmentation is under `cell_tracking_clustering_MICCAI2018` and  `cell_tracking_clustering_MIA2019`. The individual results can be generated running the `*.sh` files. The code under this folder is the same that we used to generate the results of the celltracking challenge that we submitted for MICCAI 2018. This folder also contains the trained model files (`/cell_tracking_clustering_MICCAI2018/models/`). Further details for running this code can be found under [celltracking challenge website](http://www.celltrackingchallenge.net/participants/TUG-AT/)).
We additionally created a file that performs instance segmentation with HDBSCAN for the non-recurrent version of the network. Look at the file `cell_segmentation/clustering.py` for more details. Note that the example `cell_segmentation/main.py` was not tested by us and is only used as an example for datasets without video information. You will need to adapt the parameters in order to make it work reasonably.

### Train and test other datasets
In order to train and test on other datasets, modify the `dataset.py` file in the subfolder `cell_tracking` or `cell_segmentation`. See the example files and documentation for the specific file formats. Set the parameter `save_debug_images = True` in order to see, if the network input images are reasonable.

## Citation
If you use this code for your research, please cite our [MICCAI paper](https://doi.org/10.1007/978-3-030-00934-2_1) or [MIA paper](https://doi.org/10.1016/j.media.2019.06.015):

[Segmenting and Tracking Cell Instances with Cosine Embeddings and Recurrent Hourglass Networks](https://doi.org/10.1016/j.media.2019.06.015):

```
@article{Payer2019b,
  title   = {Segmenting and Tracking Cell Instances with Cosine Embeddings and Recurrent Hourglass Networks},
  author  = {Payer, Christian and {\v{S}}tern, Darko and Feiner, Marlies and Bischof, Horst and Urschler, Martin},
  journal = {Medical Image Analysis},
  volume  = {57},
  year    = {2019},
  month   = {oct},
  pages   = {106--119},
  doi     = {10.1016/j.media.2019.06.015},
}
```

```
@inproceedings{Payer2018,
  title     = {Instance Segmentation and Tracking with Cosine Embeddings and Recurrent Hourglass Networks},
  author    = {Payer, Christian and {\v{S}}tern, Darko and Neff, Thomas and Bischof, Horst and Urschler, Martin},
  booktitle = {Medical Image Computing and Computer-Assisted Intervention - {MICCAI} 2018},
  doi       = {10.1007/978-3-030-00934-2_1},
  pages     = {3--11},
  year      = {2018},
}
```
