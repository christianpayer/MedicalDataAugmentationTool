# Multi-label Whole Heart Segmentation Using CNNs and Anatomical Label Configurations

## Usage
This example implements the segmentation network of the paper [Multi-label Whole Heart Segmentation Using CNNs and Anatomical Label Configurations](https://doi.org/10.1007/978-3-319-75541-0_20). The results when running the scripts may not be competely the same, as we originally trained the network with caffe and not with tensorflow. Also, this code only implements the segmentation network and not the localization network. See the localization folder for examples of how to train networks for localization.
If you have problems/questions/suggestions about the code, write me a [mail](mailto:christian.payer@gmx.net)!

### Dataset preprocessing
Download the files from the [challenge website](http://www.sdspeople.fudan.edu.cn/zhuangxiahai/0/mmwhs/). In order for the framework to be able to load the data, they must be preprocessed first with the following commands using the program [c3d](https://sourceforge.net/p/c3d/git/ci/master/tree/doc/c3d.md). Update the paths accordingly:

The labels need to be sorted according to label value ranging from 1 to 7:
`for x in *_label.nii.gz; do c3d ${x} -replace 500 1 600 2 420 3 550 4 205 5 820 6 850 7 -o ${${x:r}:r}_sorted.nii.gz; done`

The images need to be converted to .mha files:
`for x in *.nii.gz; do c3d ${x} -compress -type short -o ../mr_mha/${${x:r}:r}.mha; done`

The images need to be reoriented to RAI:
`for x in *_sorted.mha; do c3d ${x} -compress -type short -orient RAI -o ../mr_mha/${x}; done`

After these commands, copy the resulting files into the folders `mmwhs_dataset/mr_mha` and `mmwhs_dataset/ct_mha` accordingly. For each file from the training dataset, there should be an image and label file (e.g., `ct_train_1001_image.mha` and `ct_train_1001_label.mha`). For each file from the testing dataset, there should be an image file (e.g., `ct_test_2010_image.mha`)

Update: If the program c3d generates error messages, you could also try to run the script `reorient.py`, which should produce the same results.

### Train models
Run `main.py` to train the network. Adapt parameters in the file to use either MR or CT and to define cross validation or the full training/testing.

### Train and test other datasets
In order to train and test on other datasets, modify the `dataset.py` file. See the example files and documentation for the specific file formats. Set the parameter `save_debug_images = True` in order to see, if the network input images are reasonable.

## Citation
If you use this code for your research, please cite our [paper](https://doi.org/10.1007/978-3-319-75541-0_20):

```
@inproceedings{Payer2018a,
  title     = {Multi-label Whole Heart Segmentation Using {CNNs} and Anatomical Label Configurations},
  author    = {Payer, Christian and {\v{S}}tern, Darko and Bischof, Horst and Urschler, Martin},
  booktitle = {Statistical Atlases and Computational Models of the Heart. ACDC and MMWHS Challenges. STACOM 2017},
  doi       = {10.1007/978-3-319-75541-0_20},
  pages     = {190--198},
  year      = {2018},
}
```
