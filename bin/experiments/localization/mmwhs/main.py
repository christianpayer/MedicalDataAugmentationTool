
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
from network import network_unet


class MainLoop(MainLoopBase):
    def __init__(self, cv, modality):
        super().__init__()
        self.cv = cv
        self.output_folder = './mmwhs_localization/{}_{}'.format(modality, cv) + '/' + self.output_folder_timestamp()
        self.batch_size = 1
        self.learning_rate = 0.00001
        self.learning_rates = [self.learning_rate, self.learning_rate * 0.1]
        self.learning_rate_boundaries = [10000]
        self.max_iter = 20000
        self.test_iter = 5000
        self.disp_iter = 100
        self.snapshot_iter = self.test_iter
        self.test_initialization = False
        self.current_iter = 0
        self.reg_constant = 0.00005
        self.invert_transformation = False
        self.image_size = [32] * 3
        if modality == 'ct':
            self.image_spacing = [10] * 3
        else:
            self.image_spacing = [12] * 3
        self.sigma = [1.5] * 3
        self.image_channels = 1
        self.num_landmarks = 1
        self.data_format = 'channels_first'
        self.save_debug_images = False
        self.local_base_folder = '../../semantic_segmentation/mmwhs/mmwhs_dataset'
        dataset_parameters = {'base_folder': self.local_base_folder,
                              'image_size': self.image_size,
                              'image_spacing': self.image_spacing,
                              'cv': cv,
                              'input_gaussian_sigma': 4.0,
                              'modality': modality,
                              'save_debug_images': self.save_debug_images}

        dataset = Dataset(**dataset_parameters)
        self.dataset_train = dataset.dataset_train()
        self.dataset_val = dataset.dataset_val()
        self.network = network_unet
        self.loss_function = lambda x, y: tf.nn.l2_loss(x - y) / get_batch_channel_image_size(x, self.data_format)[0]

    def initNetworks(self):
        net = tf.make_template('net', self.network)

        if self.data_format == 'channels_first':
            data_generator_entries = OrderedDict([('image', [self.image_channels] + list(reversed(self.image_size))),
                                                  ('landmarks', [self.num_landmarks] + list(reversed(self.image_size)))])
            data_generator_entries_val = OrderedDict([('image', [self.image_channels] + list(reversed(self.image_size))),
                                                      ('landmarks', [self.num_landmarks] + list(reversed(self.image_size)))])
        else:
            raise NotImplementedError

        self.train_queue = DataGenerator(self.dataset_train, self.coord, data_generator_entries, batch_size=self.batch_size)
        image, landmarks = self.train_queue.dequeue()
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
        global_step = tf.Variable(self.current_iter)
        learning_rate = tf.train.piecewise_constant(global_step, self.learning_rate_boundaries, self.learning_rates)
        #self.optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.99, use_nesterov=True).minimize(self.loss, global_step=global_step)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss, global_step=global_step)

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
            reference_image = datasources['image']
            groundtruth_landmarks = datasources['landmarks']
            image, prediction, transform = self.test_full_image(dataset_entry)

            utils.io.image.write_np((prediction * 128).astype(np.int8), self.output_file_for_current_iteration(current_id + '_heatmap.mha'))
            predicted_landmarks = heatmap_test.get_landmarks(prediction, reference_image, transformation=transform, output_spacing=self.image_spacing)
            tensorflow_train.utils.tensorflow_util.print_progress_bar(i, self.dataset_val.num_entries())
            landmarks[current_id] = predicted_landmarks
            landmark_statistics.add_landmarks(current_id, predicted_landmarks, groundtruth_landmarks)

        tensorflow_train.utils.tensorflow_util.print_progress_bar(self.dataset_val.num_entries(), self.dataset_val.num_entries())
        overview_string = landmark_statistics.get_overview_string([2.0, 4.0, 10.0])
        print(overview_string)

        # finalize loss values
        self.val_loss_aggregator.finalize(self.current_iter)
        utils.io.landmark.save_points_csv(landmarks, self.output_file_for_current_iteration('prediction.csv'))
        utils.io.text.save_string_txt(overview_string, self.output_file_for_current_iteration('summary.txt'))


if __name__ == '__main__':
    for modality in ['ct']:
        for cv in [1, 2, 3]:
            loop = MainLoop(cv, modality)
            loop.run()
