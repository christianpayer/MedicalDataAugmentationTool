
import SimpleITK as sitk
import numpy as np
import utils.geometry
import utils.sitk_image
import utils.sitk_np
import utils.np_image
import utils.io.image
import utils.io.text
import utils.io.common


class SegmentationTest(object):
    def __init__(self,
                 labels,
                 channel_axis,
                 interpolator='linear',
                 largest_connected_component=False,
                 all_labels_are_connected=False):
        self.labels = labels
        self.channel_axis = channel_axis
        self.interpolator = interpolator
        self.largest_connected_component = largest_connected_component
        self.all_labels_are_connected = all_labels_are_connected
        self.metric_values = {}

    def get_transformed_image_sitk(self, prediction_np, reference_sitk=None, output_spacing=None, transformation=None):
        if transformation is not None:
            predictions_sitk = utils.sitk_image.transform_np_output_to_sitk_input(output_image=prediction_np,
                                                                                  output_spacing=output_spacing,
                                                                                  channel_axis=self.channel_axis,
                                                                                  input_image_sitk=reference_sitk,
                                                                                  transform=transformation,
                                                                                  interpolator=self.interpolator,
                                                                                  output_pixel_type=sitk.sitkFloat32)
        else:
            predictions_np = utils.np_image.split_by_axis(prediction_np, self.channel_axis)
            predictions_sitk = [utils.sitk_np.np_to_sitk(prediction_np) for prediction_np in predictions_np]
        return predictions_sitk

    def get_prediction_labels_list(self, prediction):
        num_labels = len(self.labels)
        prediction_labels = utils.np_image.argmax(prediction, axis=0)
        return utils.np_image.split_label_image(prediction_labels, list(range(num_labels)))

    def filter_largest_connected_component(self, prediction):
        prediction_filtered = prediction.copy()
        while True:
            prediction_labels_list = self.get_prediction_labels_list(prediction_filtered)
            prediction_labels = np.stack(prediction_labels_list, axis=0)
            prediction_labels_largest_cc_list = [utils.np_image.largest_connected_component(prediction_labels)
                                                 for prediction_labels in prediction_labels_list[1:]]
            prediction_labels_largest_cc = np.stack([prediction_labels_list[0]] + prediction_labels_largest_cc_list, axis=0)
            # filter pixels that are in the prediction labels but not in the largest cc
            prediction_filter = prediction_labels != prediction_labels_largest_cc
            # break if no pixels would be filtered
            if not np.any(prediction_filter):
                break
            prediction_filtered[prediction_filter] = -np.inf
        return prediction_filtered

    def filter_all_labels_are_connected(self, prediction):
        # split into background and other labels
        prediction_background, prediction_others = np.split(prediction, [1], axis=0)
        # remove unused dimension in background
        prediction_background = np.squeeze(prediction_background, axis=0)
        # merge other labels by using the max among all labels
        prediction_others = np.max(prediction_others, axis=0)
        # stack background and merged labels
        prediction_background_others = np.stack([prediction_background, prediction_others], axis=0)
        # find arg max -> either background or other labels
        all_labels_prediction = np.argmax(prediction_background_others, axis=0)
        # get largest component of other labels
        all_labels_prediction = utils.np_image.largest_connected_component(all_labels_prediction)
        # filter is the largest component
        prediction_filter = np.stack([all_labels_prediction] * prediction.shape[0], axis=0) == 0
        prediction_filtered = prediction.copy()
        prediction_filtered[prediction_filter] = -np.inf
        return prediction_filtered

    def get_label_image(self, prediction_np, reference_sitk=None, output_spacing=None, transformation=None, return_transformed_sitk=False):
        assert len(self.labels) == prediction_np.shape[self.channel_axis], 'number of labels must be equal to prediction image channel axis'
        prediction_transformed_sitk = self.get_transformed_image_sitk(prediction_np, reference_sitk, output_spacing, transformation)
        prediction_transformed = utils.sitk_np.sitk_list_to_np(prediction_transformed_sitk, axis=0)

        if self.all_labels_are_connected:
            prediction_transformed = self.filter_all_labels_are_connected(prediction_transformed)
        if self.largest_connected_component:
            prediction_transformed = self.filter_largest_connected_component(prediction_transformed)

        prediction_labels_list = self.get_prediction_labels_list(prediction_transformed)

        prediction_labels = utils.np_image.merge_label_images(prediction_labels_list, self.labels)
        prediction_labels_sitk = utils.sitk_np.np_to_sitk(prediction_labels)
        if reference_sitk is not None:
            prediction_labels_sitk.CopyInformation(reference_sitk)

        if return_transformed_sitk:
            return prediction_labels_sitk, prediction_transformed_sitk
        else:
            return prediction_labels_sitk
