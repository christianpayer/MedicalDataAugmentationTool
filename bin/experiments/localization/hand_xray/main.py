
import os
import numpy as np
import tensorflow as tf
from collections import OrderedDict

import tensorflow_train
import tensorflow_train.utils.tensorflow_util
from tensorflow_train.utils.data_format import get_batch_channel_image_size
import utils.io.image
import utils.io.landmark
import utils.io.text
from tensorflow_train.data_generator import DataGenerator
from tensorflow_train.train_loop import MainLoopBase
from utils.landmark.heatmap_test import HeatmapTest
from utils.landmark.landmark_statistics import LandmarkStatistics


from dataset import Dataset
from network import network_scn, network_unet, network_downsampling, network_conv, network_scn_mmwhs


class MainLoop(MainLoopBase):
    def __init__(self, cv, network_id):
        super().__init__()
        self.cv = cv
        self.network_id = network_id
        self.output_folder = '{}_cv{}'.format(network_id, cv) + '/' + self.output_folder_timestamp()
        self.batch_size = 8
        self.learning_rate = 0.0000001  # TODO adapt learning rates for different networks for faster training
        self.max_iter = 20000
        self.test_iter = 5000
        self.disp_iter = 100
        self.snapshot_iter = self.test_iter
        self.test_initialization = True
        self.current_iter = 0
        self.reg_constant = 0.00005
        self.invert_transformation = False
        image_sizes = {'scn': [256, 256],
                       'unet': [256, 256],
                       'downsampling': [256, 256],
                       'conv': [128, 128],
                       'scn_mmwhs': [256, 256]}
        heatmap_sizes = {'scn': [256, 256],
                         'unet': [256, 256],
                         'downsampling': [64, 64],
                         'conv': [128, 128],
                         'scn_mmwhs': [256, 256]}
        sigmas = {'scn': 3.0,
                  'unet': 3.0,
                  'downsampling': 1.5,
                  'conv': 3.0,
                  'scn_mmwhs': 3.0}
        self.image_size = image_sizes[self.network_id]
        self.heatmap_size = heatmap_sizes[self.network_id]
        self.sigma = sigmas[self.network_id]
        self.image_channels = 1
        self.num_landmarks = 37
        self.data_format = 'channels_first'
        self.save_debug_images = False
        self.base_folder = 'hand_xray_dataset/'
        dataset = Dataset(self.image_size,
                          self.heatmap_size,
                          self.num_landmarks,
                          self.sigma,
                          self.base_folder,
                          self.cv,
                          self.data_format,
                          self.save_debug_images)
        self.dataset_train = dataset.dataset_train()
        self.dataset_train.get_next()
        self.dataset_val = dataset.dataset_val()

        networks = {'scn': network_scn,
                    'unet': network_unet,
                    'downsampling': network_downsampling,
                    'conv': network_conv,
                    'scn_mmwhs': network_scn_mmwhs}
        self.network = networks[self.network_id]
        self.loss_function = lambda x, y: tf.nn.l2_loss(x - y) / get_batch_channel_image_size(x, self.data_format)[0]
        self.files_to_copy = ['main.py', 'network.py', 'dataset.py']

    def initNetworks(self):
        net = tf.make_template('net', self.network)

        if self.data_format == 'channels_first':
            data_generator_entries = OrderedDict([('image', [self.image_channels] + list(reversed(self.image_size))),
                                                  ('landmarks', [self.num_landmarks] + list(reversed(self.heatmap_size)))])
            data_generator_entries_val = OrderedDict([('image', [self.image_channels] + list(reversed(self.image_size))),
                                                      ('landmarks', [self.num_landmarks] + list(reversed(self.heatmap_size)))])
        else:
            raise NotImplementedError

        self.train_queue = DataGenerator(self.dataset_train, self.coord, data_generator_entries, batch_size=self.batch_size, n_threads=8)
        placeholders = self.train_queue.dequeue()
        image = placeholders[0]
        landmarks = placeholders[1]
        prediction = net(image, num_landmarks=self.num_landmarks, is_training=True, data_format=self.data_format)
        self.loss_net = self.loss_function(landmarks, prediction)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            if self.reg_constant > 0:
                reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                self.loss_reg = self.reg_constant * tf.add_n(reg_losses)
                self.loss = self.loss_net + self.loss_reg
            else:
                self.loss_reg = 0
                self.loss = self.loss_net

        self.train_losses = OrderedDict([('loss', self.loss_net), ('loss_reg', self.loss_reg)])
        #self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.99, use_nesterov=True).minimize(self.loss)

        # build val graph
        self.val_placeholders = tensorflow_train.utils.tensorflow_util.create_placeholders(data_generator_entries_val, shape_prefix=[1])
        self.image_val = self.val_placeholders['image']
        self.landmarks_val = self.val_placeholders['landmarks']
        self.prediction_val = net(self.image_val, num_landmarks=self.num_landmarks, is_training=False, data_format=self.data_format)

        # losses
        self.loss_val = self.loss_function(self.landmarks_val, self.prediction_val)
        self.val_losses = OrderedDict([('loss', self.loss_val), ('loss_reg', self.loss_reg)])

    def test_full_image(self, dataset_entry):
        generators = dataset_entry['generators']
        transformations = dataset_entry['transformations']
        feed_dict = {self.val_placeholders['image']: np.expand_dims(generators['image'], axis=0),
                     self.val_placeholders['landmarks']: np.expand_dims(generators['landmarks'], axis=0)}

        # run loss and update loss accumulators
        run_tuple = self.sess.run((self.prediction_val, self.loss_val) + self.val_loss_aggregator.get_update_ops(), feed_dict)
        prediction = np.squeeze(run_tuple[0], axis=0)
        image = generators['image']
        transformation = transformations['image']

        return image, prediction, transformation

    def test(self):
        heatmap_test = HeatmapTest(channel_axis=0, invert_transformation=False)
        landmark_statistics = LandmarkStatistics()

        landmarks = {}
        for i in range(self.dataset_val.num_entries()):
            dataset_entry = self.dataset_val.get_next()
            current_id = dataset_entry['id']['image_id']
            datasources = dataset_entry['datasources']
            reference_image = datasources['image_datasource']
            groundtruth_landmarks = datasources['landmarks_datasource']
            image, prediction, transform = self.test_full_image(dataset_entry)

            utils.io.image.write_np((prediction * 128).astype(np.int8), self.output_file_for_current_iteration(current_id + '_heatmap.mha'))
            predicted_landmarks = heatmap_test.get_landmarks(prediction, reference_image, transformation=transform)
            tensorflow_train.utils.tensorflow_util.print_progress_bar(i, self.dataset_val.num_entries())
            landmarks[current_id] = predicted_landmarks
            landmark_statistics.add_landmarks(current_id, predicted_landmarks, groundtruth_landmarks, normalization_factor=50, normalization_indizes=[1, 5])

        tensorflow_train.utils.tensorflow_util.print_progress_bar(self.dataset_val.num_entries(), self.dataset_val.num_entries())
        print('ipe', landmark_statistics.get_ipe_statistics())
        print('pe', landmark_statistics.get_pe_statistics())
        print('outliers', landmark_statistics.get_num_outliers([2.0, 4.0, 10.0]))

        # finalize loss values
        self.val_loss_aggregator.finalize(self.current_iter)
        utils.io.landmark.save_points_csv(landmarks, self.output_file_for_current_iteration('prediction.csv'))


if __name__ == '__main__':
    # change networks
    networks = ['scn_mmwhs', 'scn', 'unet', 'downsampling', 'conv']
    for network in networks:
        for cv in [1, 2, 3]:
            loop = MainLoop(cv, network)
            loop.run()
