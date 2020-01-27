
import os
from collections import OrderedDict
from itertools import chain

import numpy as np
import tensorflow as tf
from dataset import Dataset
from network import network_scn_mia, network_unet

import tensorflow_train
import tensorflow_train.utils.tensorflow_util
import utils.io.image
import utils.io.landmark
import utils.io.text
from tensorflow_train.data_generator import DataGenerator
from tensorflow_train.train_loop import MainLoopBase
from tensorflow_train.utils.data_format import get_batch_channel_image_size
from tensorflow_train.utils.heatmap_image_generator import generate_heatmap_target
from tensorflow_train.utils.summary_handler import create_summary_placeholder
from tensorflow_train.utils.tensorflow_util import get_reg_loss
from utils.landmark.heatmap_test import HeatmapTest
from utils.landmark.landmark_statistics import LandmarkStatistics


class MainLoop(MainLoopBase):
    def __init__(self, network_id, cv, landmark_source, sigma_regularization, output_folder_name=''):
        super().__init__()
        self.network_id = network_id
        self.output_folder = os.path.join('output', network_id, landmark_source, cv if cv >= 0 else 'all', output_folder_name, self.output_folder_timestamp())
        self.batch_size = 1
        self.max_iter = 30000
        self.learning_rate = 0.000001
        self.test_iter = 5000
        self.disp_iter = 100
        self.snapshot_iter = self.test_iter
        self.test_initialization = False
        self.current_iter = 0
        self.reg_constant = 0.001
        self.cv = cv
        self.landmark_source = landmark_source
        original_image_extend = [193.5, 240.0]
        image_sizes = {'unet': [512, 512],
                       'scn_mia': [512, 512]}
        heatmap_sizes = {'unet': [512, 512],
                         'scn_mia': [512, 512]}
        sigmas = {'unet': 2.5,
                  'scn_mia': 2.5}
        self.image_size = image_sizes[self.network_id]
        self.heatmap_size = heatmap_sizes[self.network_id]
        self.image_spacing = [float(np.max([e / s for e, s in zip(original_image_extend, self.image_size)]))] * 2
        self.sigma = sigmas[self.network_id]
        self.image_channels = 1
        self.num_landmarks = 19
        self.heatmap_sigma = self.sigma
        self.sigma_regularization = sigma_regularization
        self.sigma_scale = 100.0
        self.data_format = 'channels_first'
        self.save_debug_images = False
        self.base_folder = './'
        dataset_parameters = {'image_size': self.image_size,
                              'heatmap_size': self.heatmap_size,
                              'image_spacing': self.image_spacing,
                              'num_landmarks': self.num_landmarks,
                              'base_folder': self.base_folder,
                              'data_format': self.data_format,
                              'save_debug_images': self.save_debug_images,
                              'cv': self.cv,
                              'landmark_source': self.landmark_source}

        dataset = Dataset(**dataset_parameters)
        self.dataset_train = dataset.dataset_train()
        self.dataset_val = dataset.dataset_val()

        networks = {'unet': network_unet,
                    'scn_mia': network_scn_mia}
        self.network = networks[self.network_id]
        self.landmark_metrics = ['pe_mean', 'pe_std', 'pe_median', 'or2', 'or25', 'or3', 'or4', 'or10']
        self.landmark_metric_prefixes = ['challenge', 'senior', 'junior', 'mean']
        self.additional_summaries_placeholders_val = OrderedDict([(prefix + '_' + name, create_summary_placeholder(prefix + '_' + name)) for name in self.landmark_metrics for prefix in self.landmark_metric_prefixes])

    def loss_function(self, target, prediction):
        return tf.nn.l2_loss(target - prediction) / get_batch_channel_image_size(target, self.data_format)[0]

    def loss_sigmas(self, sigmas, landmarks):
        return self.sigma_regularization * tf.nn.l2_loss(sigmas[None, :] * landmarks[:, :, 0]) / landmarks.get_shape().as_list()[0]

    def initNetworks(self):
        net = tf.make_template('net', self.network)

        if self.data_format == 'channels_first':
            data_generator_entries = OrderedDict([('image', [self.image_channels] + list(reversed(self.image_size))),
                                                  ('landmarks', [self.num_landmarks, 3])])
            data_generator_entries_val = OrderedDict([('image', [self.image_channels] + list(reversed(self.image_size))),
                                                      ('landmarks', [self.num_landmarks, 3])])
        else:
            data_generator_entries = OrderedDict([('image', list(reversed(self.image_size)) + [self.image_channels]),
                                                  ('landmarks', [self.num_landmarks, 3])])
            data_generator_entries_val = OrderedDict([('image', list(reversed(self.image_size)) + [self.image_channels]),
                                                      ('landmarks', [self.num_landmarks, 3])])

        sigmas = tf.get_variable('sigmas', [self.num_landmarks], initializer=tf.constant_initializer(self.heatmap_sigma))
        sigmas_list = [(f's{i}', sigmas[i]) for i in range(self.num_landmarks)]

        # build training graph
        self.train_queue = DataGenerator(self.dataset_train, self.coord, data_generator_entries, batch_size=self.batch_size, n_threads=8)
        placeholders = self.train_queue.dequeue()
        image = placeholders[0]
        target_landmarks = placeholders[1]
        prediction = net(image, num_landmarks=self.num_landmarks, is_training=True, data_format=self.data_format)
        target_heatmaps = generate_heatmap_target(list(reversed(self.heatmap_size)), target_landmarks, sigmas, scale=self.sigma_scale, normalize=True, data_format=self.data_format)
        loss_sigmas = self.loss_sigmas(sigmas, target_landmarks)
        self.loss_reg = get_reg_loss(self.reg_constant)
        self.loss_net = self.loss_function(target_heatmaps, prediction)
        self.loss = self.loss_net + tf.cast(self.loss_reg, tf.float32) + loss_sigmas

        # optimizer
        global_step = tf.Variable(self.current_iter, trainable=False)
        optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.99, use_nesterov=True)
        unclipped_gradients, variables = zip(*optimizer.compute_gradients(self.loss))
        norm = tf.global_norm(unclipped_gradients)
        gradients, _ = tf.clip_by_global_norm(unclipped_gradients, 10000.0)
        self.optimizer = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)
        self.train_losses = OrderedDict([('loss', self.loss_net), ('loss_reg', self.loss_reg), ('loss_sigmas', loss_sigmas), ('norm', norm)] + sigmas_list)

        # build val graph
        self.val_placeholders = tensorflow_train.utils.tensorflow_util.create_placeholders(data_generator_entries_val, shape_prefix=[1])
        self.image_val = self.val_placeholders['image']
        self.target_landmarks_val = self.val_placeholders['landmarks']
        self.prediction_val = net(self.image_val, num_landmarks=self.num_landmarks, is_training=False, data_format=self.data_format)
        self.target_heatmaps_val = generate_heatmap_target(list(reversed(self.heatmap_size)), self.target_landmarks_val, sigmas, scale=self.sigma_scale, normalize=True, data_format=self.data_format)

        # losses
        self.loss_val = self.loss_function(self.target_heatmaps_val, self.prediction_val)
        self.val_losses = OrderedDict([('loss', self.loss_val), ('loss_reg', self.loss_reg), ('loss_sigmas', tf.constant(0, tf.float32)), ('norm', tf.constant(0, tf.float32))] + sigmas_list)

    def test_full_image(self, dataset_entry):
        generators = dataset_entry['generators']
        transformations = dataset_entry['transformations']
        feed_dict = {self.val_placeholders['image']: np.expand_dims(generators['image'], axis=0),
                     self.val_placeholders['landmarks']: np.expand_dims(generators['landmarks'], axis=0)}

        # run loss and update loss accumulators
        run_tuple = self.sess.run((self.prediction_val, self.target_heatmaps_val, self.loss_val) + self.val_loss_aggregator.get_update_ops(), feed_dict)
        prediction = np.squeeze(run_tuple[0], axis=0)
        target_heatmaps = np.squeeze(run_tuple[1], axis=0)
        image = generators['image']
        transformation = transformations['image']

        return image, prediction, target_heatmaps, transformation

    def finalize_landmark_statistics(self, landmark_statistics, prefix):
        pe_mean, pe_std, pe_median = landmark_statistics.get_pe_statistics()
        or2, or25, or3, or4, or10 = landmark_statistics.get_num_outliers([2.0, 2.5, 3.0, 4.0, 10.0], True)
        print(prefix + '_pe', ['{0:.3f}'.format(s) for s in [pe_mean, pe_std, pe_median]])
        print(prefix + '_outliers', ['{0:.3f}'.format(s) for s in [or2, or25, or3, or4, or10]])
        overview_string = landmark_statistics.get_overview_string([2, 2.5, 3, 4, 10, 20], 10, 20.0)
        utils.io.text.save_string_txt(overview_string, self.output_file_for_current_iteration(prefix + '_eval.txt'))
        additional_summaries = {prefix + '_pe_mean': pe_mean,
                                prefix + '_pe_std': pe_std,
                                prefix + '_pe_median': pe_median,
                                prefix + '_or2': or2,
                                prefix + '_or25': or25,
                                prefix + '_or3': or3,
                                prefix + '_or4': or4,
                                prefix + '_or10': or10}
        return additional_summaries

    def test(self):
        heatmap_test = HeatmapTest(channel_axis=0, invert_transformation=False)
        challenge_landmark_statistics = LandmarkStatistics()
        senior_landmark_statistics = LandmarkStatistics()
        junior_landmark_statistics = LandmarkStatistics()
        mean_landmark_statistics = LandmarkStatistics()

        landmarks = {}
        for i in range(self.dataset_val.num_entries()):
            dataset_entry = self.dataset_val.get_next()
            current_id = dataset_entry['id']['image_id']
            datasources = dataset_entry['datasources']
            reference_image = datasources['image_datasource']
            groundtruth_challenge_landmarks = datasources['challenge_landmarks_datasource']
            groundtruth_senior_landmarks = datasources['senior_landmarks_datasource']
            groundtruth_junior_landmarks = datasources['junior_landmarks_datasource']
            groundtruth_mean_landmarks = datasources['mean_landmarks_datasource']
            image, prediction, target_heatmaps, transform = self.test_full_image(dataset_entry)

            utils.io.image.write_np(image, self.output_file_for_current_iteration(current_id + '_image.mha'))
            utils.io.image.write_np(prediction, self.output_file_for_current_iteration(current_id + '_prediction.mha'))
            utils.io.image.write_np(target_heatmaps, self.output_file_for_current_iteration(current_id + '_target_heatmap.mha'))
            predicted_landmarks = heatmap_test.get_landmarks(prediction, reference_image, output_spacing=self.image_spacing, transformation=transform)
            tensorflow_train.utils.tensorflow_util.print_progress_bar(i, self.dataset_val.num_entries())
            landmarks[current_id] = predicted_landmarks
            challenge_landmark_statistics.add_landmarks(current_id, predicted_landmarks, groundtruth_challenge_landmarks)
            senior_landmark_statistics.add_landmarks(current_id, predicted_landmarks, groundtruth_senior_landmarks)
            junior_landmark_statistics.add_landmarks(current_id, predicted_landmarks, groundtruth_junior_landmarks)
            mean_landmark_statistics.add_landmarks(current_id, predicted_landmarks, groundtruth_mean_landmarks)

        tensorflow_train.utils.tensorflow_util.print_progress_bar(self.dataset_val.num_entries(), self.dataset_val.num_entries())
        challenge_summaries = self.finalize_landmark_statistics(challenge_landmark_statistics, 'challenge')
        senior_summaries = self.finalize_landmark_statistics(senior_landmark_statistics, 'senior')
        junior_summaries = self.finalize_landmark_statistics(junior_landmark_statistics, 'junior')
        mean_summaries = self.finalize_landmark_statistics(mean_landmark_statistics, 'mean')
        additional_summaries = OrderedDict(chain(senior_summaries.items(), junior_summaries.items(), challenge_summaries.items(), mean_summaries.items()))

        # finalize loss values
        self.val_loss_aggregator.finalize(self.current_iter, additional_summaries)
        utils.io.landmark.save_points_csv(landmarks, self.output_file_for_current_iteration('prediction.csv'))


if __name__ == '__main__':
    # change parameters
    # possible parameters for landmark_source: 'challenge', 'junior', 'senior', 'mean', 'random'
    # possible parameters for cv: -1: use all training data and test on test1 and test2 set.
    #                              1...4: one of the cross validations.
    loop = MainLoop('scn_mia', cv=-1, landmark_source='challenge', sigma_regularization=20, output_folder_name='baseline')
    loop.run()
