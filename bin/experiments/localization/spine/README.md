# Integrating spatial configuration into heatmap regression based CNNs for landmark localization

## Usage
This example implements the SCN and U-Net experiments of the paper [Integrating spatial configuration into heatmap regression based CNNs for landmark localization](https://doi.org/10.1016/j.media.2019.03.007). If you have problems/questions/suggestions about the code, write me a [mail](mailto:christian.payer@gmx.net)!

### Dataset
The folder `spine_localization_dataset` contains the setup files of the [MICCAI CSI 2014 Vertebrae Localization and Identification Challenge](http://csi-workshop.weebly.com/challenges.html). The input images are not included. Ask the challenge organizers for access and copy or link them to the folder `spine_localization_dataset/images`.

### Train models
Run `main.py` to train the networks. Adapt parameters in the file to use either SCN or U-Net and to define cross validation or the full training/testing. If the network loss is nan, either restart network training or reduce the learning rate.

### Train and test other datasets
In order to train and test on other datasets, modify the `dataset.py` file. See the example files and documentation for the specific file formats. Set the parameter `save_debug_images = True` in order to see, if the network input images are reasonable.

## Citation
If you use this code for your research, please cite our [paper](https://doi.org/10.1016/j.media.2019.03.007):

```
@article{Payer2019,
  title   = {Integrating spatial configuration into heatmap regression based {CNNs} for landmark localization},
  author  = {Payer, Christian and {\v{S}}tern, Darko and Bischof, Horst and Urschler, Martin},
  journal = {Medical Image Analysis},
  volume  = {54},
  year    = {2019},
  month   = {may},
  pages   = {207--219},
  doi     = {10.1016/j.media.2019.03.007},
}
```
