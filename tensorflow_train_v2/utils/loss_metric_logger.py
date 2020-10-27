
import datetime

import tensorflow as tf

from collections import OrderedDict
import os
import csv


class LossMetricLogger(object):
    """
    SummaryHandler is used to aggregate loss values and save summary values to a given folder.
    """
    def __init__(self, name, summary_folder, csv_filename, print_format='{0:.4f}'):
        """
        Initializer.
        :param metric_names: The losses dict to save. Key is a string and the name of the loss, value is the loss tensor.
        :param summary_placeholders_dict: The summary/placeholders dict. Key is a string and the name of the summary entry, value is a tuple of summary and placeholder (see create_summary_placeholder).
        :param name: The name of the summary handler. Usually either 'train' or 'test'.
        :param summary_folder: The folder, where to save the summary.
        :param csv_filename: The filename of the generated .csv file
        """
        self.name = name
        self.summary_folder = summary_folder
        self.csv_filename = csv_filename
        self.print_format = print_format
        self.metrics = OrderedDict()
        self.summary_folder = summary_folder
        self.summary_writer = tf.summary.create_file_writer(summary_folder)
        self.last_finalize_time = datetime.datetime.now()
        self.now = None
        self.time_since_last_finalize = None

    def reset_metrics(self):
        """
        Reset the current calculated running mean of the losses.
        """
        for metric in self.metrics.values():
            metric.reset_states()

    def current_metric_values(self):
        """
        Return the current metric values as an OrderedDict.
        :return:
        """
        metric_values = OrderedDict()
        for key, value in self.metrics.items():
            metric_values[key] = value.result()
        return metric_values

    def update_internal_times(self):
        """
        Update the internal time variables used to calculate the time in between self.finalize() calls
        """
        self.now = datetime.datetime.now()
        self.time_since_last_finalize = self.now - self.last_finalize_time
        self.last_finalize_time = self.now

    def update_metrics(self, summary_values):
        """
        Update the metrics fo the given summary_values dictionary. If the metrics did not exist before, create them.
        :param summary_values:
        :return:
        """
        for key, value in summary_values.items():
            if key not in self.metrics:
                self.metrics[key] = tf.keras.metrics.Mean(name=self.name + '/' + key)
            self.metrics[key].update_state(value)

    def write_csv(self, current_iteration, summary_values):
        """
        Write the summary for the given current iteration and summary values to a .csv file.
        :param current_iteration: The current iteration.
        :param summary_values: The current calculated summary values. Keys must be the same as self.summary_placeholders_dict.keys()
        """
        csv_file_exists = os.path.exists(self.csv_filename)
        append_write = 'a' if csv_file_exists else 'w'
        with open(self.csv_filename, append_write) as csv_file:
            writer = csv.writer(csv_file)
            # write the header, if current iteration is the first one or if the csv file did not exist before
            if current_iteration == 0 or not csv_file_exists:
                row = ['iter', 'time'] + list(summary_values.keys())
                writer.writerow(row)
            row = [current_iteration, self.time_since_last_finalize.seconds] + list(map(float, summary_values.values()))
            writer.writerow(list(map(str, row)))

    def write_summary(self, current_iteration, summary_values):
        """
        Write the summary for the given current iteration and summary values.
        :param current_iteration: The current iteration.
        :param summary_values: The current calculated summary values. Keys must be the same as self.summary_placeholders_dict.keys()
        """
        with self.summary_writer.as_default():
            for key, value in summary_values.items():
                tf.summary.scalar(key, value, step=current_iteration)
                self.summary_writer.flush()

    def print(self, current_iteration, summary_values):
        """
        Print the summary for the given current iteration and summary values.
        :param current_iteration: The current iteration.
        :param summary_values: The current calculated summary values. Keys must be the same as self.summary_placeholders_dict.keys()
        """
        date_string = self.now.strftime('%H:%M:%S')
        print_string = date_string + ': ' + self.name + ' iter: ' + str(current_iteration) + ' '
        for key, value in summary_values.items():
            value_string = self.print_format.format(value) if self.print_format is not None else str(value)
            print_string += key + ': ' + value_string + ' '
        print_string += 'seconds: {}.{:03d}'.format(self.time_since_last_finalize.seconds, self.time_since_last_finalize.microseconds // 1000)
        print(print_string)

    def finalize(self, current_iteration):
        """
        Finalize the summary fo the current iteration. Writes summary, .csv file, and prints a short summary string. Additionally resets the internal times and the losses' running mean.
        :param current_iteration: The current iteration.
        :return: Dictionary of all current summary and loss values.
        """
        # update losses and add to current summary_values
        summary_values = self.current_metric_values()

        self.update_internal_times()
        self.write_csv(int(current_iteration), summary_values)
        self.write_summary(int(current_iteration), summary_values)
        self.print(int(current_iteration), summary_values)

        self.reset_metrics()

        return summary_values
