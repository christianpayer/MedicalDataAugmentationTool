import numpy as np
import tensorflow as tf
from collections import OrderedDict

import tensorflow_train
from tensorflow_train.utils.tensorflow_util import create_placeholders
import utils.io.image
import utils.io.landmark
import utils.io.text
from tensorflow_train.data_generator import DataGenerator
from tensorflow_train.train_loop import MainLoopBase
from tensorflow_train.utils.data_format import get_batch_channel_image_size
from tensorflow_train.utils.heatmap_image_generator import generate_heatmap_target
from transformations.intensity.np.shift_scale_clamp import ShiftScaleClamp
from utils.landmark.heatmap_test import HeatmapTest
from utils.landmark.landmark_statistics import LandmarkStatistics
from utils.image_tiler import ImageTiler, LandmarkTiler
from tensorflow_train.utils.summary_handler import create_summary_placeholder

from dataset import Dataset
from network import network_scn, network_unet


class MainLoop(MainLoopBase):
    def __init__(self, cv, network_id):
        super().__init__()
        self.cv = cv
        self.network_id = network_id
        self.output_folder = network_id
        if cv != -1:
            self.output_folder += '_cv{}'.format(cv)
        self.output_folder += '/' + self.output_folder_timestamp()
        self.batch_size = 1
        learning_rates = {'scn': 0.00000005,
                          'unet': 0.000000005}
        max_iters = {'scn': 40000,
                     'unet': 80000}
        self.learning_rate = learning_rates[self.network_id]
        self.max_iter = max_iters[self.network_id]
        self.test_iter = 2500
        self.disp_iter = 100
        self.snapshot_iter = self.test_iter
        self.test_initialization = False
        self.current_iter = 0
        self.reg_constant = 0.0005
        self.sigma_regularization = 100
        self.sigma_scale = 1000
        self.invert_transformation = False
        self.num_landmarks = 26
        self.image_size = [96, 96, 192]
        self.image_spacing = [2, 2, 2]
        self.heatmap_size = self.image_size
        self.image_channels = 1
        self.heatmap_sigma = 4
        self.data_format = 'channels_first'
        self.save_debug_images = False
        self.base_folder = 'spine_localization_dataset'
        self.generate_landmarks = True
        self.cropped_training = True
        self.cropped_inc = [0, 64, 0, 0]
        if self.cropped_training:
            dataset = Dataset(self.image_size,
                              self.image_spacing,
                              self.heatmap_sigma,
                              self.num_landmarks,
                              self.base_folder,
                              self.cv,
                              self.data_format,
                              self.save_debug_images,
                              generate_heatmaps=not self.generate_landmarks,
                              generate_landmarks=self.generate_landmarks)
            self.dataset_train = dataset.dataset_train()
            dataset = Dataset(self.image_size,
                              self.image_spacing,
                              self.heatmap_sigma,
                              self.num_landmarks,
                              self.base_folder,
                              self.cv,
                              self.data_format,
                              self.save_debug_images,
                              generate_heatmaps=not self.generate_landmarks,
                              generate_landmarks=self.generate_landmarks)
            self.dataset_val = dataset.dataset_val()
        else:
            dataset = Dataset(self.image_size,
                              self.image_spacing,
                              self.heatmap_sigma,
                              self.num_landmarks,
                              self.base_folder,
                              self.cv,
                              self.data_format,
                              self.save_debug_images,
                              generate_heatmaps=not self.generate_landmarks,
                              generate_landmarks=self.generate_landmarks,
                              translate_by_random_factor=False)
            self.dataset_train = dataset.dataset_train()
            self.dataset_val = dataset.dataset_val()

        networks = {'scn': network_scn,
                    'unet': network_unet}
        self.network = networks[self.network_id]

        self.point_statistics_names = ['pe_mean', 'pe_stdev', 'pe_median', 'num_correct']
        self.additional_summaries_placeholders_val = dict([(name, create_summary_placeholder(name)) for name in self.point_statistics_names])

    def loss_function(self, pred, target):
        batch_size, _, _ = get_batch_channel_image_size(pred, self.data_format)
        return tf.nn.l2_loss(pred - target) / batch_size

    def initNetworks(self):
        net = tf.make_template('scn', self.network)

        if self.data_format == 'channels_first':
            if self.generate_landmarks:
                data_generator_entries = OrderedDict([('image', [self.image_channels] + list(reversed(self.image_size))),
                                                      ('landmarks', [self.num_landmarks, 4])])
            else:
                data_generator_entries = OrderedDict([('image', [self.image_channels] + list(reversed(self.image_size))),
                                                      ('heatmaps', [self.num_landmarks] + list(reversed(self.heatmap_size)))])
        else:
            if self.generate_landmarks:
                data_generator_entries = OrderedDict([('image', list(reversed(self.image_size)) + [self.image_channels]),
                                                      ('landmarks', [self.num_landmarks, 4])])
            else:
                data_generator_entries = OrderedDict([('image', list(reversed(self.image_size)) + [self.image_channels]),
                                                      ('heatmaps', list(reversed(self.heatmap_size)) + [self.num_landmarks])])

        sigmas = tf.get_variable('sigmas', [self.num_landmarks], initializer=tf.constant_initializer(self.heatmap_sigma))
        mean_sigmas = tf.reduce_mean(sigmas)
        self.train_queue = DataGenerator(self.dataset_train, self.coord, data_generator_entries, batch_size=self.batch_size)
        #self.train_queue = DataGeneratorDataset(self.dataset_train, data_generator_entries, batch_size=self.batch_size)
        placeholders = self.train_queue.dequeue()
        image = placeholders[0]

        if self.generate_landmarks:
            target_landmarks = placeholders[1]
            target_heatmaps = generate_heatmap_target(list(reversed(self.heatmap_size)), target_landmarks, sigmas, scale=self.sigma_scale, normalize=True, data_format=self.data_format)
            loss_sigmas = self.sigma_regularization * tf.nn.l2_loss(sigmas * target_landmarks[0, :, 0])
        else:
            target_heatmaps = placeholders[1]
            loss_sigmas = self.sigma_regularization * tf.nn.l2_loss(sigmas)
        heatmaps, _, _ = net(image, num_heatmaps=self.num_landmarks, is_training=True, data_format=self.data_format)
        self.loss_net = self.loss_function(heatmaps, target_heatmaps)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            if self.reg_constant > 0:
                reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                self.loss_reg = self.reg_constant * tf.add_n(reg_losses)
                self.loss = self.loss_net + self.loss_reg + loss_sigmas
            else:
                self.loss = self.loss_net + loss_sigmas

        self.train_losses = OrderedDict([('loss', self.loss_net), ('loss_reg', self.loss_reg), ('loss_sigmas', loss_sigmas), ('mean_sigmas', mean_sigmas)])
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.99, use_nesterov=True).minimize(self.loss)

        # build val graph
        val_placeholders = create_placeholders(data_generator_entries, shape_prefix=[1])
        self.image_val = val_placeholders['image']
        if self.generate_landmarks:
            self.target_landmarks_val = val_placeholders['landmarks']
            self.target_heatmaps_val = generate_heatmap_target(list(reversed(self.heatmap_size)), self.target_landmarks_val, sigmas, scale=self.sigma_scale, normalize=True, data_format=self.data_format)
            loss_sigmas_val = self.sigma_regularization * tf.nn.l2_loss(sigmas * self.target_landmarks_val[0, :, 0])
        else:
            self.target_heatmaps_val = val_placeholders['heatmaps']
            loss_sigmas_val = self.sigma_regularization * tf.nn.l2_loss(sigmas)
        self.heatmaps_val, self.heatmals_local_val, self.heatmaps_global_val = net(self.image_val, num_heatmaps=self.num_landmarks, is_training=False, data_format=self.data_format)

        # losses
        self.loss_val = self.loss_function(self.heatmaps_val, self.target_heatmaps_val)
        self.val_losses = OrderedDict([('loss', self.loss_val), ('loss_reg', self.loss_reg), ('loss_sigmas', loss_sigmas_val), ('mean_sigmas', mean_sigmas)])

    def test_full_image(self, dataset_entry):
        generators = dataset_entry['generators']
        transformations = dataset_entry['transformations']
        heatmap_transform = transformations['image']
        feed_dict = {self.image_val: np.expand_dims(generators['image'], axis=0)}
        if self.generate_landmarks:
            feed_dict[self.target_landmarks_val] = np.expand_dims(generators['landmarks'], axis=0)
        else:
            feed_dict[self.target_heatmaps_val] = np.expand_dims(generators['heatmaps'], axis=0)

        # run loss and update loss accumulators
        run_tuple = self.sess.run((self.heatmaps_val, self.target_heatmaps_val, self.loss_val) + self.val_loss_aggregator.get_update_ops(), feed_dict)
        heatmaps = np.squeeze(run_tuple[0], axis=0)
        image = generators['image']

        return image, heatmaps, heatmap_transform

    def test_cropped_image(self, dataset_entry):
        generators = dataset_entry['generators']
        transformations = dataset_entry['transformations']
        heatmap_transform = transformations['image']

        image_size_np = [1] + list(reversed(self.image_size))
        heatmap_size_np = [self.num_landmarks] + list(reversed(self.image_size))
        full_image = generators['image']
        landmarks = generators['landmarks']
        image_tiler = ImageTiler(full_image.shape, image_size_np, self.cropped_inc, True, -1)
        landmark_tiler = LandmarkTiler(full_image.shape, image_size_np, self.cropped_inc)
        heatmap_tiler = ImageTiler((self.num_landmarks,) + full_image.shape[1:], heatmap_size_np, self.cropped_inc, True, 0)

        for image_tiler, landmark_tiler, heatmap_tiler in zip(image_tiler, landmark_tiler, heatmap_tiler):
            current_image = image_tiler.get_current_data(full_image)
            current_landmarks = landmark_tiler.get_current_data(landmarks)
            feed_dict = {self.image_val: np.expand_dims(current_image, axis=0),
                         self.target_landmarks_val: np.expand_dims(current_landmarks, axis=0)}
            run_tuple = self.sess.run((self.heatmaps_val, self.target_heatmaps_val, self.loss_val) + self.val_loss_aggregator.get_update_ops(), feed_dict)
            prediction = np.squeeze(run_tuple[0], axis=0)
            image_tiler.set_current_data(current_image)
            heatmap_tiler.set_current_data(prediction)

        return image_tiler.output_image, heatmap_tiler.output_image, heatmap_transform

    def test(self):
        print('Testing...')
        if self.data_format == 'channels_first':
            np_channel_index = 0
        else:
            np_channel_index = 3
        heatmap_maxima = HeatmapTest(np_channel_index, False)
        landmark_statistics = LandmarkStatistics()
        landmarks = {}
        for i in range(self.dataset_val.num_entries()):
            dataset_entry = self.dataset_val.get_next()
            current_id = dataset_entry['id']['image_id']
            datasources = dataset_entry['datasources']
            image_datasource = datasources['image_datasource']
            landmarks_datasource = datasources['landmarks_datasource']

            if not self.cropped_training:
                image, heatmaps, heatmap_transform = self.test_full_image(dataset_entry)
            else:
                image, heatmaps, heatmap_transform = self.test_cropped_image(dataset_entry)

            utils.io.image.write_np(ShiftScaleClamp(scale=255, clamp_min=0, clamp_max=255)(heatmaps).astype(np.uint8),
                                    self.output_file_for_current_iteration(current_id + '_heatmaps.mha'))
            utils.io.image.write_np(image, self.output_file_for_current_iteration(current_id + '_image.mha'))

            predicted_landmarks = heatmap_maxima.get_landmarks(heatmaps, image_datasource, self.image_spacing, heatmap_transform)
            landmarks[current_id] = predicted_landmarks
            landmark_statistics.add_landmarks(current_id, predicted_landmarks, landmarks_datasource)

            tensorflow_train.utils.tensorflow_util.print_progress_bar(i, self.dataset_val.num_entries(), prefix='Testing ', suffix=' complete')

        tensorflow_train.utils.tensorflow_util.print_progress_bar(self.dataset_val.num_entries(), self.dataset_val.num_entries(), prefix='Testing ', suffix=' complete')
        print(landmark_statistics.get_pe_overview_string())
        print(landmark_statistics.get_correct_id_string(20.0))
        summary_values = OrderedDict(zip(self.point_statistics_names, list(landmark_statistics.get_pe_statistics()) + [landmark_statistics.get_correct_id(20)]))

        # finalize loss values
        self.val_loss_aggregator.finalize(self.current_iter, summary_values)
        utils.io.landmark.save_points_csv(landmarks, self.output_file_for_current_iteration('points.csv'))
        overview_string = landmark_statistics.get_overview_string([2, 2.5, 3, 4, 10, 20], 10, 20.0)
        utils.io.text.save_string_txt(overview_string, self.output_file_for_current_iteration('eval.txt'))



if __name__ == '__main__':
    # TODO: if the loss gets 'nan', either restart the network training or reduce the learning rate
    # change networks
    networks = ['scn', 'unet']
    for network in networks:
        # cv 0, 1 for cross validation
        # cv -1 for training on full training set and testing on test set
        for cv in [-1, 0, 1]:
            loop = MainLoop(cv, network)
            loop.run()
