# Regressing Heatmaps for Multiple Landmark Localization Using CNNs

## Usage
This example implements the networks of the paper [Regressing Heatmaps for Multiple Landmark Localization Using CNNs](https://doi.org/10.1007/978-3-319-46723-8_27). The results when running the scripts may not be competely the same, as we originally trained the network with caffe and not with tensorflow. Also, much more data augmentation is used in this code.
The included version of the spatial configuration net is very basic and just shows the concept.
If you have problems/questions/suggestions about the code, write me a [mail](mailto:christian.payer@gmx.net)!

### Dataset
In the folder `hand_xray_dataset` only 10 images of the [Digital Hand Atlas Database System](http://www.ipilab.org/BAAweb/) are included. This experiment is not included in our paper! It shows, how the training and testing with our framework works.
Download the full dataset, if you want to reproduce our reported results. Write me a message, if you have problems obtaining it.
Unfortunately, we are not allowed to make our 3D hand MRI dataset publicly available. 

### Train models
Run `main.py` to train the network. Adapt parameters in the file to use either MR or CT and to define cross validation or the full training/testing.

### Train and test other datasets
In order to train and test on other datasets, modify the `dataset.py` file. See the example files and documentation for the specific file formats. Set the parameter `save_debug_images = True` in order to see, if the network input images are reasonable.

## Citation
If you use this code for your research, please cite our [paper](https://doi.org/10.1007/978-3-319-75541-0_20):

```
@inproceedings{Payer2016,
  title     = {Regressing Heatmaps for Multiple Landmark Localization Using {CNNs}},
  author    = {Payer, Christian and {\v{S}}tern, Darko and Bischof, Horst and Urschler, Martin},
  booktitle = {Medical Image Computing and Computer-Assisted Intervention - {MICCAI} 2016},
  doi       = {10.1007/978-3-319-46723-8_27},
  pages     = {230--238},
  year      = {2016},
}
```
