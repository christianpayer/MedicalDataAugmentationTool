
import csv
import datetime
import os

import tensorflow.compat.v1 as tf

from tensorflow_train.utils.tensorflow_util import create_reset_metric
from collections import OrderedDict


def create_summary_placeholder(name):
    """
    Returns a tf.summary.scalar and an empty tf.placeholder with the given name.
    :param name: The name of the summary.
    :return: tf.summary.scalar, tf.placeholder
    """
    placeholder = tf.placeholder(tf.float32, name='summary_placeholder_' + name)
    summary = tf.summary.scalar(name, placeholder)
    return summary, placeholder


class SummaryHandler(object):
    """
    SummaryHandler is used to aggragate loss values and save summary values to a given folder.
    """
    def __init__(self, session, loss_dict, summary_placeholders_dict, name, summary_folder, csv_filename, print_format='{0:.4f}'):
        """
        Initializer.
        :param session: The tf session.
        :param loss_dict: The losses dict to save. Key is a string and the name of the loss, value is the loss tensor.
        :param summary_placeholders_dict: The summary/placeholders dict. Key is a string and the name of the summary entry, value is a tuple of summary and placeholder (see create_summary_placeholder).
        :param name: The name of the summary handler. Usually either 'train' or 'test'.
        :param summary_folder: The folder, where to save the summary.
        :param csv_filename: The filename of the generated .csv file
        """
        self.session = session
        self.loss_dict = loss_dict
        self.name = name
        self.summary_folder = summary_folder
        self.csv_filename = csv_filename
        self.print_format = print_format
        self.summary_placeholders_dict = summary_placeholders_dict
        self.summary = tf.summary.merge(list(zip(*summary_placeholders_dict.values()))[0])
        self.loss_metrics = OrderedDict()
        for key, value in loss_dict.items():
            self.loss_metrics[key] = create_reset_metric(tf.metrics.mean, key + '_' + name, values=value, name=name + '/' + key)
        self.summary_writer = tf.summary.FileWriter(summary_folder, self.session.graph)
        self.last_finalize_time = datetime.datetime.now()
        self.now = None
        self.time_since_last_finalize = None

    def get_update_ops(self):
        """
        Returns a list of tf tensors that need to be evaluated for calculating the running mean of the losses.
        :return: The tf tensors of the loss update ops.
        """
        return tuple(list(zip(*self.loss_metrics.values()))[1])

    def get_current_losses_dict(self):
        """
        Evaluates the current running mean values of the losses and returns them as an OrderedDict (with the same order as self.loss_dict).
        :return: An OrderedDict of the current loss values.
        """
        value_op_list = list(zip(*self.loss_metrics.values()))[0]
        losses = self.session.run(value_op_list)
        return OrderedDict(zip(self.loss_metrics.keys(), losses))

    def reset_current_losses(self):
        """
        Resets the current calculated running mean of the losses.
        """
        reset_op_list = list(zip(*self.loss_metrics.values()))[2]
        self.session.run(reset_op_list)

    def get_summary_feed_dict(self, summary_values):
        """
        Creates the summary feed_dict that will be used for generate the current summary.
        :param summary_values: The individual summary values as a dict. Keys must be the same as self.summary_placeholders_dict.keys()
        :return: The feed_dict that can be used for calculating the summary.
        """
        summary_feed_dict = {}
        for key, value in summary_values.items():
            summary_feed_dict[self.summary_placeholders_dict[key][1]] = value
        return summary_feed_dict

    def write_summary(self, current_iteration, summary_values):
        """
        Writes the summary for the given current iteration and summary values.
        :param current_iteration: The current iteration.
        :param summary_values: The current calculated summary values. Keys must be the same as self.summary_placeholders_dict.keys()
        """
        summary_feed_dict = self.get_summary_feed_dict(summary_values)
        sum = self.session.run(self.summary, feed_dict=summary_feed_dict)
        self.summary_writer.add_summary(sum, current_iteration)

    def print_current_summary(self, current_iteration, summary_values):
        """
        Prints the summary for the given current iteration and summary values.
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

    def write_csv_file(self, current_iteration, summary_values):
        """
        Writes the summary for the given current iteration and summary values to a .csv file.
        :param current_iteration: The current iteration.
        :param summary_values: The current calculated summary values. Keys must be the same as self.summary_placeholders_dict.keys()
        """
        append_write = 'a' if os.path.exists(self.csv_filename) else 'w'
        with open(self.csv_filename, append_write) as csv_file:
            writer = csv.writer(csv_file)
            if current_iteration == 0:
                row = ['iter', 'time'] + list(summary_values.keys())
                writer.writerow(row)
            row = [current_iteration, self.time_since_last_finalize.seconds] + list(summary_values.values())
            writer.writerow(list(map(str, row)))

    def update_internal_times(self):
        """
        Updates the internal time variables used to calculate the time in between self.finalize() calls
        """
        self.now = datetime.datetime.now()
        self.time_since_last_finalize = self.now - self.last_finalize_time
        self.last_finalize_time = self.now

    def finalize(self, current_iteration, summary_values=None):
        """
        Finalizes the summary fo the current iteration. Writes summary, .csv file, and prints a short summary string. Additionally resets the internal times and the losses' running mean.
        :param current_iteration: The current iteration.
        :param summary_values: Additional summary values as a dict. If self.summary_placeholders_dict has additional values that are not in self.loss_dict, these values must be given.
        :return: Dictionary of all current summary and loss values.
        """
        if summary_values is None:
            summary_values = OrderedDict()
        # update losses and add to current summary_values
        loss_dict = self.get_current_losses_dict()
        summary_values.update(loss_dict)

        self.update_internal_times()
        self.write_summary(current_iteration, summary_values)
        self.write_csv_file(current_iteration, summary_values)
        self.print_current_summary(current_iteration, summary_values)

        self.reset_current_losses()

        return summary_values
