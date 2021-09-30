import numpy as np
import utils.geometry
import utils.sitk_image
import utils.sitk_np
import utils.np_image
import utils.landmark.transform
import utils.segmentation.metrics
import utils.io.image
import utils.io.text
import utils.io.common
from collections import OrderedDict
import os
import csv
import copy


class SegmentationStatistics(object):
    def __init__(self,
                 labels,
                 output_folder,
                 metrics=None,
                 save_overlap_image=False):
        self.labels = labels
        self.output_folder = output_folder
        self.metrics = metrics
        self.save_overlap_image = save_overlap_image
        self.metric_values = {}

    def add_labels(self, current_id, prediction_labels, groundtruth_labels):
        current_metric_values = self.get_metric_values(prediction_labels, groundtruth_labels)
        self.metric_values[current_id] = current_metric_values

    def set_metric_values(self, current_id, metric_values):
        self.metric_values[current_id] = metric_values

    def get_metric_mean_list(self, metric_key):
        metric_values_list = [current_metric_values[metric_key] for current_metric_values in self.metric_values.values()]
        metric_mean_list = list(map(lambda x: np.nanmean(x), zip(*metric_values_list)))
        return metric_mean_list

    def print_metric_summary(self, metric_key, values):
        format_string = '{} mean: {:.4f}'
        if len(values) > 1:
            format_string += ', classes: ' + ' '.join(['{:.4f}'] * (len(values) - 1))
            print(format_string.format(metric_key, *values))
        else:
            print(format_string.format(metric_key, *values))

    def print_metric_summaries(self, metric_summaries):
        for key, value in metric_summaries.items():
            self.print_metric_summary(key, value)

    def get_metric_summary(self, metric_key):
        metric_mean_list = self.get_metric_mean_list(metric_key)
        if len(metric_mean_list) > 1:
            metric_mean_total = np.nanmean(metric_mean_list)
            return [metric_mean_total] + metric_mean_list
        else:
            return metric_mean_list

    def finalize(self):
        for metric_key in self.metrics.keys():
            if isinstance(metric_key, tuple):
                for current_metric_key in metric_key:
                    self.save_metric_values(current_metric_key)
            else:
                self.save_metric_values(metric_key)

        metric_summaries = OrderedDict()
        for metric_key in self.metrics.keys():
            if isinstance(metric_key, tuple):
                for current_metric_key in metric_key:
                    metric_summaries[current_metric_key] = self.get_metric_summary(current_metric_key)
            else:
                metric_summaries[metric_key] = self.get_metric_summary(metric_key)

        self.print_metric_summaries(metric_summaries)
        self.save_metric_summaries(metric_summaries)

    def get_metric_values(self, predictions_sitk, groundtruth_sitk):
        current_metric_values = OrderedDict()
        for metric_key, metric in self.metrics.items():
            metric_values = metric(predictions_sitk, groundtruth_sitk, self.labels)
            if isinstance(metric_values, tuple):
                if isinstance(metric_key, tuple):
                    assert len(metric_key) == len(metric_values), 'number of metric key names need to be the same as the returned metric values'
                    for metric_tuple_key, metric_tuple_values in zip(metric_key, metric_values):
                        current_metric_values[metric_tuple_key] = metric_tuple_values
                else:
                    # only take first metric values, for backward compatibility
                    current_metric_values[metric_key] = metric_values[0]
            else:
                # metric values are no tuples
                current_metric_values[metric_key] = metric_values
        return current_metric_values

    def save_metric_values(self, metric_key):
        if self.output_folder is None:
            return
        metric_dict = OrderedDict([(key, value[metric_key]) for key, value in self.metric_values.items()])
        metric_dict = copy.deepcopy(metric_dict)
        num_values = None
        for value in metric_dict.values():
            num_values = len(value)
            if len(value) > 1:
                value.insert(0, np.nanmean(value))
        header = [metric_key, 'mean'] + list(range(num_values))
        utils.io.text.save_dict_csv(metric_dict, os.path.join(self.output_folder, metric_key + '.csv'), header)

    def save_metric_summaries(self, metric_summaries):
        if self.output_folder is None:
            return
        file_name = os.path.join(self.output_folder, 'summary.csv')
        utils.io.common.create_directories_for_file_name(file_name)
        with open(file_name, 'w') as file:
            writer = csv.writer(file)
            for key, value in metric_summaries.items():
                writer.writerow([key])
                writer.writerow(['mean'] + list(range(len(value) - 1)))
                writer.writerow(value)
