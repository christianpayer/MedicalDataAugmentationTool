#!/usr/bin/python
from collections import OrderedDict
import os
import numpy as np
import tensorflow as tf
from tensorflow_train.losses.instance_segmentation_losses import cosine_embedding_per_instance_loss
import utils.io.image
import tensorflow_train.utils.tensorflow_util
from tensorflow_train.data_generator_padding import DataGeneratorPadding
from tensorflow_train.train_loop import MainLoopBase
from dataset import Dataset
from network import network, network_single_frame_with_lstm_states
import utils.io.text
from iterators.id_list_iterator import IdListIterator


class MainLoop(MainLoopBase):
    def __init__(self, dataset_name):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = 1
        self.learning_rate = 0.0001
        self.learning_rates = [self.learning_rate, self.learning_rate * 0.1]
        self.learning_rate_boundaries = [20000]
        self.max_iter = 40000
        self.test_iter = 5000
        self.disp_iter = 100
        self.snapshot_iter = self.test_iter
        self.test_initialization = True
        self.current_iter = 0
        self.reg_constant = 0.00001
        self.use_batch_norm = False
        self.invert_transformation = False
        self.use_pyro_dataset = False
        self.save_debug_images = False
        self.image_size = [256, 256]
        self.output_size = self.image_size
        self.data_format = 'channels_first'
        self.num_frames = 10
        self.embeddings_dim = 16
        self.test_on_challenge_data = True
        self.challenge_base_folder = '../celltrackingchallenge/'
        self.output_base_folder = '/media1/experiments/cell_tracking/miccai2018/' + self.dataset_name
        self.training_base_folder = os.path.join(self.challenge_base_folder, 'trainingdataset/', self.dataset_name)
        self.testing_base_folder = os.path.join(self.challenge_base_folder, 'challengedataset/', self.dataset_name)
        self.output_folder = os.path.join(self.output_base_folder, self.output_folder_timestamp())
        self.embedding_factors = {'bac': 1, 'tra': 1}
        if self.test_on_challenge_data:
            self.train_id_file = 'tra_all.csv'
            self.val_id_file = 'tra_all.csv'
        else:
            self.train_id_file = 'tra_train.csv'
            self.val_id_file = 'tra_val.csv'
        instance_image_radius_factors = {'DIC-C2DH-HeLa': 0.2,
                                         'Fluo-C2DL-MSC': 0.6,
                                         'Fluo-N2DH-GOWT1': 0.2,
                                         'Fluo-N2DH-SIM+': 0.2,
                                         'Fluo-N2DL-HeLa': 0.1,
                                         'PhC-C2DH-U373': 0.2,
                                         'PhC-C2DL-PSC': 0.1}
        instance_image_radius_factor = instance_image_radius_factors[self.dataset_name]
        label_gaussian_blur_sigmas = {'DIC-C2DH-HeLa': 2.0,
                                      'Fluo-C2DL-MSC': 0,
                                      'Fluo-N2DH-GOWT1': 0,
                                      'Fluo-N2DH-SIM+': 0,
                                      'Fluo-N2DL-HeLa': 0,
                                      'PhC-C2DH-U373': 0,
                                      'PhC-C2DL-PSC': 0}
        label_gaussian_blur_sigma = label_gaussian_blur_sigmas[self.dataset_name]
        crop_image_sizes = {'DIC-C2DH-HeLa': None,
                            'Fluo-C2DL-MSC': [20, 20],
                            'Fluo-N2DH-GOWT1': None,
                            'Fluo-N2DH-SIM+': None,
                            'Fluo-N2DL-HeLa': None,
                            'PhC-C2DH-U373': None,
                            'PhC-C2DL-PSC': None}
        crop_image_size = crop_image_sizes[self.dataset_name]
        normalization_consideration_factors = {'DIC-C2DH-HeLa': (0.2, 0.1),
                                               'Fluo-C2DL-MSC': (0.2, 0.01),
                                               'Fluo-N2DH-GOWT1': (0.2, 0.1),
                                               'Fluo-N2DH-SIM+': (0.2, 0.01),
                                               'Fluo-N2DL-HeLa': (0.2, 0.1),
                                               'PhC-C2DH-U373': (0.2, 0.1),
                                               'PhC-C2DL-PSC': (0.2, 0.1)}
        normalization_consideration_factor = normalization_consideration_factors[self.dataset_name]
        pad_images = {'DIC-C2DH-HeLa': True,
                      'Fluo-C2DL-MSC': False,
                      'Fluo-N2DH-GOWT1': True,
                      'Fluo-N2DH-SIM+': True,
                      'Fluo-N2DL-HeLa': True,
                      'PhC-C2DH-U373': False,
                      'PhC-C2DL-PSC': True}
        pad_image = pad_images[self.dataset_name]
        self.dataset = Dataset(self.image_size,
                               self.num_frames,
                               base_folder=self.training_base_folder,
                               data_format=self.data_format,
                               save_debug_images=self.save_debug_images,
                               instance_image_radius_factor=instance_image_radius_factor,
                               max_num_instances=16,
                               train_id_file=self.train_id_file,
                               val_id_file=self.val_id_file,
                               image_gaussian_blur_sigma=2.0,
                               label_gaussian_blur_sigma=label_gaussian_blur_sigma,
                               normalization_consideration_factors=normalization_consideration_factor,
                               pad_image=pad_image,
                               crop_image_size=crop_image_size)
        self.dataset_train = self.dataset.dataset_train()
        self.dataset_train.get_next()
        
        if self.test_on_challenge_data:
            dataset = Dataset(self.image_size,
                              self.num_frames,
                              base_folder=self.testing_base_folder,
                              data_format=self.data_format,
                              save_debug_images=self.save_debug_images,
                              instance_image_radius_factor=instance_image_radius_factor,
                              max_num_instances=16,
                              train_id_file=self.train_id_file,
                              val_id_file=self.val_id_file,
                              image_gaussian_blur_sigma=2.0,
                              label_gaussian_blur_sigma=label_gaussian_blur_sigma,
                              pad_image=False,
                              crop_image_size=None,
                              load_merged=False,
                              load_has_complete_seg=False,
                              load_seg_loss_mask=False,
                              create_instances_bac=False,
                              create_instances_merged=False)
            self.dataset_val = dataset.dataset_val_single_frame()
            self.setup_base_folder = os.path.join(self.testing_base_folder, 'setup')
            self.video_frame_list_file_name = os.path.join(self.setup_base_folder, 'frames.csv')
            self.iterator_val = IdListIterator(os.path.join(self.setup_base_folder, 'video_only_all.csv'), random=False, keys=['video_id'])
        else:
            self.dataset_val = self.dataset.dataset_val_single_frame()
            self.setup_base_folder = os.path.join(self.training_base_folder, 'setup')
            self.video_frame_list_file_name = os.path.join(self.setup_base_folder, 'frames.csv')
            self.iterator_val = IdListIterator(os.path.join(self.setup_base_folder, 'video_only_all.csv'), random=False, keys=['video_id'])

        self.files_to_copy = ['dataset.py', 'network.py', 'main.py']

    def initNetworks(self):
        network_image_size = self.image_size
        network_output_size = self.output_size
        num_instances = None

        if self.data_format == 'channels_first':
            data_generator_entries = OrderedDict([('image', [1, self.num_frames] + network_image_size),
                                                  ('instances_merged', [num_instances, self.num_frames] + network_output_size),
                                                  ('instances_bac', [1, self.num_frames] + network_output_size)])
            data_generator_entries_single_frame = OrderedDict([('image', [1] + network_image_size),
                                                               ('instances_merged', [num_instances] + network_output_size),
                                                               ('instances_bac', [1] + network_output_size)])
        else:
            data_generator_entries = OrderedDict([('image', [self.num_frames] + network_image_size + [1]),
                                                  ('instances_merged', [self.num_frames] + network_output_size + [num_instances]),
                                                  ('instances_bac', [self.num_frames] + network_output_size + [1])])
            data_generator_entries_single_frame = OrderedDict([('image', network_image_size + [1]),
                                                               ('instances_merged', network_output_size + [num_instances]),
                                                               ('instances_bac', network_output_size + [1])])

        # create model with shared weights between train and val
        lstm_net = network
        training_net = tf.make_template('net', lstm_net)
        loss_function = lambda prediction, groundtruth: cosine_embedding_per_instance_loss(prediction,
                                                                                           groundtruth,
                                                                                           data_format=self.data_format,
                                                                                           normalize=True,
                                                                                           term_1_squared=True,
                                                                                           l=1.0)

        # build train graph
        self.train_queue = DataGeneratorPadding(self.dataset_train, self.coord, data_generator_entries, batch_size=self.batch_size, queue_size=64)
        data, tracking, instances_bac = self.train_queue.dequeue()
        embeddings, embeddings_2 = training_net(data, num_outputs_embedding=self.embeddings_dim, is_training=True, data_format=self.data_format)

        # losses, first and second hourglass
        with tf.variable_scope('loss'):
            tracking_embedding_loss = self.embedding_factors['tra'] * loss_function(embeddings, tracking)
            bac_embedding_loss = self.embedding_factors['bac'] * loss_function(embeddings, instances_bac)

        with tf.variable_scope('loss_2'):
            tracking_embedding_loss_2 = self.embedding_factors['tra'] * loss_function(embeddings_2, tracking)
            bac_embedding_loss_2 = self.embedding_factors['bac'] * loss_function(embeddings_2, instances_bac)

        self.loss_net = tracking_embedding_loss + bac_embedding_loss + tracking_embedding_loss_2 + bac_embedding_loss_2

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if self.reg_constant > 0:
                regularization_variables = []
                for tf_var in tf.trainable_variables():
                    if 'kernel' in tf_var.name:
                        regularization_variables.append(tf.nn.l2_loss(tf_var))
                self.loss_reg = self.reg_constant * tf.add_n(regularization_variables)
                self.loss = self.loss_net + self.loss_reg
            else:
                self.loss = self.loss_net

        self.train_losses = OrderedDict([('loss_tra_emb', tracking_embedding_loss), ('loss_bac_emb', bac_embedding_loss), ('loss_reg', self.loss_reg),
                                         ('loss_tra_emb_2', tracking_embedding_loss_2), ('loss_bac_emb_2', bac_embedding_loss_2)])

        # solver
        global_step = tf.Variable(self.current_iter)
        learning_rate = tf.train.piecewise_constant(global_step, self.learning_rate_boundaries, self.learning_rates)
        self.optimizer = tf.contrib.opt.NadamOptimizer(learning_rate=learning_rate).minimize(self.loss, global_step=global_step)

        # initialize variables
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        print('Variables')
        for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            print(i)

        # build val graph
        val_placeholders = tensorflow_train.utils.tensorflow_util.create_placeholders(data_generator_entries_single_frame, shape_prefix=[1])
        self.data_val = val_placeholders['image']
        self.tracking_val = val_placeholders['instances_merged']
        self.instances_bac_val = val_placeholders['instances_bac']

        with tf.variable_scope('net/rnn', reuse=True):
            self.embeddings_val, self.embeddings_2_val, self.lstm_input_states_val, self.lstm_output_states_val = network_single_frame_with_lstm_states(self.data_val, num_outputs_embedding=self.embeddings_dim, data_format=self.data_format)
            self.embeddings_normalized_val = tf.nn.l2_normalize(self.embeddings_val, dim=1)
            self.embeddings_normalized_2_val = tf.nn.l2_normalize(self.embeddings_2_val, dim=1)

        with tf.variable_scope('loss'):
            self.tracking_embedding_loss_val = self.embedding_factors['tra'] * loss_function(self.embeddings_val, self.tracking_val)
            self.bac_embedding_loss_val = self.embedding_factors['bac'] * loss_function(self.embeddings_val, self.instances_bac_val)

        with tf.variable_scope('loss_2'):
            self.tracking_embedding_loss_2_val = self.embedding_factors['tra'] * loss_function(self.embeddings_2_val, self.tracking_val)
            self.bac_embedding_loss_2_val = self.embedding_factors['bac'] * loss_function(self.embeddings_2_val, self.instances_bac_val)

        self.loss_val = self.tracking_embedding_loss_val + self.bac_embedding_loss_val + self.tracking_embedding_loss_2_val + self.bac_embedding_loss_2_val
        self.val_losses = OrderedDict([('loss_tra_emb', self.tracking_embedding_loss_val), ('loss_bac_emb', self.bac_embedding_loss_val), ('loss_reg', self.loss_reg),
                                       ('loss_tra_emb_2', self.tracking_embedding_loss_2_val), ('loss_bac_emb_2', self.bac_embedding_loss_2_val)])

    def test(self):
        if self.test_on_challenge_data:
            self.test_test()
        else:
            self.test_val()

    def test_val(self):
        print('Testing val dataset...')
        video_id_frames = utils.io.text.load_dict_csv(self.video_frame_list_file_name)

        channel_axis = 0
        if self.data_format == 'channels_last':
            channel_axis = 3

        num_entries = self.iterator_val.num_entries()
        for current_entry_index in range(num_entries):
            video_id = self.iterator_val.get_next_id()['video_id']
            # print(video_id)
            video_frames = video_id_frames[video_id]
            # first lstm round
            first = True
            current_predictions = []
            current_predictions_2 = []
            for video_frame in video_frames:
                current_id = video_id + '_' + video_frame
                dataset_entry = self.dataset_val.get({'video_id': video_id, 'frame_id': video_frame, 'unique_id': current_id})
                datasources = dataset_entry['datasources']
                generators = dataset_entry['generators']
                transformations = dataset_entry['transformations']
                feed_dict = {self.data_val: np.expand_dims(generators['image'], axis=0),
                             self.tracking_val: np.expand_dims(generators['instances_merged'], axis=0),
                             self.instances_bac_val: np.expand_dims(generators['instances_bac'], axis=0)}
                # run loss and update loss accumulators
                if not first:
                    for i in range(len(self.lstm_input_states_val)):
                        feed_dict[self.lstm_input_states_val[i]] = current_lstm_states[i]

                run_tuple = self.sess.run([self.loss_val, self.embeddings_normalized_val, self.embeddings_normalized_2_val] + list(self.lstm_output_states_val) + list(self.val_loss_aggregator.get_update_ops()),
                                          feed_dict=feed_dict)
                embeddings_softmax = np.squeeze(run_tuple[1], axis=0)
                embeddings_softmax_2 = np.squeeze(run_tuple[2], axis=0)
                current_lstm_states = run_tuple[3:-len(self.val_loss_aggregator.get_update_ops())]
                current_predictions.append(embeddings_softmax)
                current_predictions_2.append(embeddings_softmax_2)
                first = False

            prediction = np.stack(current_predictions, axis=1)
            current_predictions = []
            utils.io.image.write_np(prediction, os.path.join(self.output_folder, 'out_first/iter_' + str(self.current_iter) + '/' + video_id + '_embeddings.mha'))
            prediction = None
            prediction_2 = np.stack(current_predictions_2, axis=1)
            current_predictions_2 = []
            utils.io.image.write_np(prediction_2, os.path.join(self.output_folder, 'out_first/iter_' + str(self.current_iter) + '/' + video_id + '_embeddings_2.mha'))
            prediction_2 = None
            tensorflow_train.utils.tensorflow_util.print_progress_bar(current_entry_index, num_entries, prefix='Testing ', suffix=' complete')

        # finalize loss values
        self.val_loss_aggregator.finalize(self.current_iter)

    def test_test(self):
        print('Testing test dataset...')
        video_id_frames = utils.io.text.load_dict_csv(self.video_frame_list_file_name)
        
        channel_axis = 0
        if self.data_format == 'channels_last':
            channel_axis = 3

        num_entries = self.iterator_val.num_entries()
        for current_entry_index in range(num_entries):
            video_id = self.iterator_val.get_next_id()['video_id']
            video_frames = video_id_frames[video_id]
            # first lstm round
            first = True
            current_predictions = []
            current_predictions_2 = []
            current_images = []
            for video_frame in video_frames:
                current_id = video_id + '_' + video_frame
                dataset_entry = self.dataset_val.get({'video_id': video_id, 'frame_id': video_frame, 'unique_id': current_id})
                datasources = dataset_entry['datasources']
                generators = dataset_entry['generators']
                transformations = dataset_entry['transformations']
                feed_dict = {self.data_val: np.expand_dims(generators['image'], axis=0)}
                # run loss and update loss accumulators
                if not first:
                    for i in range(len(self.lstm_input_states_val)):
                        feed_dict[self.lstm_input_states_val[i]] = current_lstm_states[i]

                run_tuple = self.sess.run([self.embeddings_normalized_val, self.embeddings_normalized_2_val] + list(self.lstm_output_states_val), feed_dict=feed_dict)
                # print(iv[0].decode())
                embeddings_softmax = np.squeeze(run_tuple[0], axis=0)
                embeddings_softmax_2 = np.squeeze(run_tuple[1], axis=0)
                current_lstm_states = run_tuple[2:]
                current_predictions.append(embeddings_softmax)
                current_predictions_2.append(embeddings_softmax_2)
                current_images.append(generators['image'])
                #current_instances.append(instance_segmentation_test.get_instances_cosine_kmeans_2d(embeddings_softmax))
                first = False

            prediction = np.stack(current_predictions, axis=1)
            current_predictions = []
            utils.io.image.write_np(prediction, os.path.join(self.output_folder, 'out_first/iter_' + str(self.current_iter) + '/' + video_id + '_embeddings.mha'))
            prediction = None
            prediction_2 = np.stack(current_predictions_2, axis=1)
            current_predictions_2 = []
            utils.io.image.write_np(prediction_2, os.path.join(self.output_folder, 'out_first/iter_' + str(self.current_iter) + '/' + video_id + '_embeddings_2.mha'))
            prediction_2 = None
            images = np.stack(current_images, axis=1)
            current_images = []
            utils.io.image.write_np(images, os.path.join(self.output_folder, 'out_first/iter_' + str(self.current_iter) + '/' + video_id + '_image.mha'))
            images = None
            tensorflow_train.utils.tensorflow_util.print_progress_bar(current_entry_index, num_entries, prefix='Testing ', suffix=' complete')

if __name__ == '__main__':
    datasets = ['DIC-C2DH-HeLa',
                'Fluo-C2DL-MSC',
                'Fluo-N2DH-GOWT1',
                'Fluo-N2DH-SIM+',
                'Fluo-N2DL-HeLa',
                'PhC-C2DH-U373']
    dataset = datasets[0]
    loop = MainLoop(dataset)
    loop.run()
