# use tensorflow base image
FROM tensorflow/tensorflow:1.14.0-py3 as base
# install python dependencies
RUN pip3 install SimpleITK itk scikit-image cachetools
# copy MedicalDataAugmentationTool
RUN mkdir /MedicalDataAugmentationTool
RUN mkdir /MedicalDataAugmentationTool/datasets
ADD MedicalDataAugmentationTool/datasets /MedicalDataAugmentationTool/datasets
RUN mkdir /MedicalDataAugmentationTool/datasources
ADD MedicalDataAugmentationTool/datasources /MedicalDataAugmentationTool/datasources
RUN mkdir /MedicalDataAugmentationTool/generators
ADD MedicalDataAugmentationTool/generators /MedicalDataAugmentationTool/generators
RUN mkdir /MedicalDataAugmentationTool/graph
ADD MedicalDataAugmentationTool/graph /MedicalDataAugmentationTool/graph
RUN mkdir /MedicalDataAugmentationTool/iterators
ADD MedicalDataAugmentationTool/iterators /MedicalDataAugmentationTool/iterators
RUN mkdir /MedicalDataAugmentationTool/tensorflow_train
ADD MedicalDataAugmentationTool/tensorflow_train /MedicalDataAugmentationTool/tensorflow_train
RUN mkdir /MedicalDataAugmentationTool/transformations
ADD MedicalDataAugmentationTool/transformations /MedicalDataAugmentationTool/transformations
RUN mkdir /MedicalDataAugmentationTool/utils
ADD MedicalDataAugmentationTool/utils /MedicalDataAugmentationTool/utils
# copy verse files
ADD MedicalDataAugmentationTool/bin/experiments/semantic_segmentation/verse2019/docker/models /models
RUN mkdir /scripts
ADD MedicalDataAugmentationTool/bin/experiments/semantic_segmentation/verse2019/inference/*.py /scripts/
ADD MedicalDataAugmentationTool/bin/experiments/semantic_segmentation/verse2019/other/*.py /scripts/
ADD MedicalDataAugmentationTool/bin/experiments/semantic_segmentation/verse2019/*.py /scripts/
ADD MedicalDataAugmentationTool/bin/experiments/semantic_segmentation/verse2019/docker/predict.sh /

