# Integrating Spatial Configuration into Heatmap Regression Based CNNs for Landmark Localization

## Usage
This example implements the CEP networks of the paper [Integrating Spatial Configuration into Heatmap Regression Based CNNs for Landmark Localization](https://doi.org/10.1016/j.media.2019.03.007).

Run the file `convert_bmp_to_nii.py` to convert the .bmp files of the original dataset to .nii.gz files with the correct spacing that can be loaded by the framework.

Per default, the network use the groundtruth annotations that were used in the 2015 CEP challenge paper from Wang et al. ([Evaluation and Comparison of Anatomical Landmark Detection Methods for Cephalometric X-Ray Images: A Grand Challenge](https://doi.org/10.1109/TMI.2015.2412951)).
This groundtruth was also used in our MIA paper.
Additionally, the paper Lindner et al. 2016 ([Fully Automatic System for Accurate Localisation and Analysis of Cephalometric Landmarks in Lateral Cephalograms](https://doi.org/10.1038/srep33581)) provided groundtruth annotations from a junior and a senior radiologist.
However, in our MIA paper, we did not use this groundtruth for training or evaluation, but only the groundtruth from Wang et al. 2015.

You can change the groundtruth that is being used for training by setting the `landmark_source` parameter of the MainLoop.
See the file `main.py` for more details.

If you have problems/questions/suggestions about the code, write me a [mail](mailto:christian.payer@gmx.net)!

### Dataset
In the folder `setup`, there are the different landmark groundtruths (`challenge`, `junior`, `senior`) with 0.1 image spacing, as well as lists for training, test1 and test2 set.
Additionally, we included our own 4-fold cross validation setup that was however never use for any of our papers.
In order to load the images with our framework, you need to download the official .bmp images and convert them to .nii.gz images with 0.1 spacing.
You can use the script `convert_bmp_to_nii.py` for this.

### Train models
Run `main.py` to train the network.

### Train and test other datasets
In order to train and test on other datasets, modify the `dataset.py` file. See the example files and documentation for the specific file formats. Set the parameter `save_debug_images = True` in order to see, if the network input images are reasonable.

## Citation
If you use this code for your research, please cite our [paper](https://doi.org/10.1016/j.media.2019.03.007):

[Integrating Spatial Configuration into Heatmap Regression Based CNNs for Landmark Localization](https://doi.org/10.1016/j.media.2019.03.007):

```
@article{Payer2019a,
  title   = {Integrating Spatial Configuration into Heatmap Regression Based {CNNs} for Landmark Localization},
  author  = {Payer, Christian and {\v{S}}tern, Darko and Bischof, Horst and Urschler, Martin},
  journal = {Medical Image Analysis},
  volume  = {54},
  year    = {2019},
  month   = {may},
  pages   = {207--219},
  doi     = {10.1016/j.media.2019.03.007},
}
```
