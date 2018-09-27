# Instance Segmentation and Tracking with Cosine Embeddings and Recurrent Hourglass Networks

## Usage
This example shows how to train networks of the paper [Instance Segmentation and Tracking with Cosine Embeddings and Recurrent Hourglass Networks](https://doi.org/10.1007/978-3-030-00934-2_1).
If you have problems/questions/suggestions about the code, write me a [mail](mailto:christian.payer@gmx.net)!

### Dataset preprocessing
Download the datasets from the [celltracking challenge](http://www.celltrackingchallenge.net/) and extract them. In order to be able to load them with this framework, they need to be preprocessed. Open `main_preprocess.py` and change the variable `dataset_base_folder` to the folder, where you extracted the files from the challenge.
Run `main_preprocess.py`. It should create `.mha` files of the input images and the groundtruth. This program also makes sure, that the IDs from the segmentation and tracking files are consistent throughout the videos.

### Train models
Run `cell_segmentation/main.py` for a non-recurrent version and `cell_tracking/main.py` for the recurrent version of the network.

### Perform instance segmentation
The code for performing instance segmentation is currently not included in this repository. Look into published code on the [celltracking challenge website](http://www.celltrackingchallenge.net/participants/TUG-AT/) for how to perform the final instance segmentation.

### Train and test other datasets
In order to train and test on other datasets, modify the `dataset.py` file in the subfolder `cell_tracking` or `cell_segmentation`. See the example files and documentation for the specific file formats. Set the parameter `save_debug_images = True` in order to see, if the network input images are reasonable.

## Citation
If you use this code for your research, please cite our [paper](https://doi.org/10.1007/978-3-030-00934-2_1):

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
