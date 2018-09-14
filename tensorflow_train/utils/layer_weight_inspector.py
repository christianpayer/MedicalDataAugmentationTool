
import csv
import datetime
import os

import tensorflow as tf

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


class LayerWeightInspector(object):
    """
    SummaryHandler is used to aggragate loss values and save summary values to a given folder.
    """
    def __init__(self, conv_layer_names, summary_folder):
        """
        Initializer.
        :param summary_folder: The folder, where to save the summary.
        """
        self.conv_layer_names = conv_layer_names
        self.graph = tf.get_default_graph()
        self.tensors = []
        for conv_layer_name in self.conv_layer_names:
            self.tensors += self.get_tensors_for_conv_layer_name(conv_layer_name)
        self.summaries = self.get_histogram_summaries(self.tensors)
        self.summary = tf.summary.merge(self.summaries)
        self.writer = tf.summary.FileWriter(summary_folder)

    def get_tensors_for_conv_layer_name(self, conv_layer_name):
        tensors = []
        layer_postfixes = ['output', 'Conv2D', 'Conv3D', 'activation', 'bias', 'kernel']
        for layer_postfix in layer_postfixes:
            try:
                tensor = self.graph.get_tensor_by_name(conv_layer_name + '/' + layer_postfix + ':0')
                tensors.append(tensor)
            except:
                continue
        return tensors

    def get_histogram_summaries(self, tensors):
        summaries = []
        for tensor in tensors:
            summary = tf.summary.histogram(tensor.name, tensor)
            summaries.append(summary)
        return summaries
