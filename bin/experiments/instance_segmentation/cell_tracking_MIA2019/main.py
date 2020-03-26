#!/usr/bin/python
from collections import OrderedDict
import os
import numpy as np
import tensorflow as tf

from tensorflow_train.losses.instance_segmentation_losses_MIA import cosine_embedding_per_instance_batch_loss
import utils.io.image
from tensorflow_train.utils.tensorflow_util import get_reg_loss, create_placeholders, print_progress_bar
from tensorflow_train.data_generator_padding import DataGeneratorPadding
from tensorflow_train.data_generator_dataset import DataGeneratorDataset
from tensorflow_train.train_loop import MainLoopBase
from dataset import Dataset
from datasets.pyro_dataset import PyroClientDataset
from network import network, network_single_frame_with_lstm_states, UnetIntermediateGruWithStates2D
from utils.image_tiler import ImageTiler
import utils.io.text
from iterators.id_list_iterator import IdListIterator
from clustering import InstanceImageCreator, InstanceMerger, InstanceTracker
import utils.sitk_np


def get_dataset_parameters(dataset_name):
    instance_image_radius_factors = {'DIC-C2DH-HeLa': 0.2,
                                     'Fluo-C2DL-MSC': 0.6,
                                     'Fluo-N2DH-GOWT1': 0.2,
                                     'Fluo-N2DH-SIM+': 0.2,
                                     'Fluo-N2DL-HeLa': 0.03,
                                     'PhC-C2DH-U373': 0.2,
                                     'PhC-C2DL-PSC': 0.03}
    label_gaussian_blur_sigmas = {'DIC-C2DH-HeLa': 2.0,
                                  'Fluo-C2DL-MSC': 0,
                                  'Fluo-N2DH-GOWT1': 0,
                                  'Fluo-N2DH-SIM+': 0,
                                  'Fluo-N2DL-HeLa': 0,
                                  'PhC-C2DH-U373': 0,
                                  'PhC-C2DL-PSC': 1.0}
    crop_image_sizes = {'DIC-C2DH-HeLa': None,
                        'Fluo-C2DL-MSC': [20, 20],
                        'Fluo-N2DH-GOWT1': None,
                        'Fluo-N2DH-SIM+': None,
                        'Fluo-N2DL-HeLa': None,
                        'PhC-C2DH-U373': None,
                        'PhC-C2DL-PSC': None}
    normalization_consideration_factors = {'DIC-C2DH-HeLa': (0.001, 0.001),
                                           'Fluo-C2DL-MSC': (0.001, 0.001),
                                           'Fluo-N2DH-GOWT1': (0.001, 0.001),
                                           'Fluo-N2DH-SIM+': (0.001, 0.001),
                                           'Fluo-N2DL-HeLa': (0.001, 0.001),
                                           'PhC-C2DH-U373': (0.001, 0.001),
                                           'PhC-C2DL-PSC': (0.001, 0.001)}
    pad_images = {'DIC-C2DH-HeLa': True,
                  'Fluo-C2DL-MSC': False,
                  'Fluo-N2DH-GOWT1': True,
                  'Fluo-N2DH-SIM+': True,
                  'Fluo-N2DL-HeLa': True,
                  'PhC-C2DH-U373': False,
                  'PhC-C2DL-PSC': True}

    image_gaussian_blur_sigma = {'DIC-C2DH-HeLa': 1.0,
                                 'Fluo-C2DL-MSC': 1.0,
                                 'Fluo-N2DH-GOWT1': 1.0,
                                 'Fluo-N2DH-SIM+': 1.0,
                                 'Fluo-N2DL-HeLa': 1.0,
                                 'PhC-C2DH-U373': 1.0,
                                 'PhC-C2DL-PSC': 1.0}
    return {'instance_image_radius_factor': instance_image_radius_factors[dataset_name],
            'label_gaussian_blur_sigma': label_gaussian_blur_sigmas[dataset_name],
            'crop_image_size': crop_image_sizes[dataset_name],
            'normalization_consideration_factors': normalization_consideration_factors[dataset_name],
            'pad_image': pad_images[dataset_name],
            'image_gaussian_blur_sigma': image_gaussian_blur_sigma[dataset_name] }


def get_folder_string(image_size, num_embeddings, network_string, optimizer_string):
    return '{}_{}/{}/{}'.format(image_size, num_embeddings, network_string, optimizer_string)


def get_network_string(network_name, network_parameters):
    return network_name + '/' + '_'.join(map(lambda a: a[0] + str(a[1]), network_parameters.items()))


def image_sizes_for_dataset_name(dataset_name):
    train_image_size = {'DIC-C2DH-HeLa': [256, 256],
                        'Fluo-C2DL-MSC':  [256, 256],
                        'Fluo-N2DH-GOWT1': [256, 256],
                        'Fluo-N2DH-SIM+': [256, 256],
                        'Fluo-N2DL-HeLa': [256, 256],
                        'PhC-C2DH-U373': [256, 256],
                        'PhC-C2DL-PSC': [256, 256]}

    test_image_size = {'DIC-C2DH-HeLa': [256, 256],  # * 2
                        'Fluo-C2DL-MSC':  [384, 256],  # uneven
                        'Fluo-N2DH-GOWT1': [512, 512],  # * 2
                        'Fluo-N2DH-SIM+': [512, 512],  # uneven
                        'Fluo-N2DL-HeLa': [1024, 640],  # exact
                        'PhC-C2DH-U373': [512, 384],  # uneven
                        'PhC-C2DL-PSC': [720, 576]}  # exact

    return train_image_size[dataset_name], test_image_size[dataset_name]


def get_instance_image_creator_parameters(dataset_name):
    bandwidth = {'DIC-C2DH-HeLa': 0.1,
                      'Fluo-C2DL-MSC': 0.1,
                      'Fluo-N2DH-GOWT1': 0.1,
                      'Fluo-N2DH-SIM+': 0.1,
                      'Fluo-N2DL-HeLa': 0.1,
                      'PhC-C2DH-U373': 0.1,
                      'PhC-C2DL-PSC': 0.1}
    min_label_size = {'DIC-C2DH-HeLa': 500,
                      'Fluo-C2DL-MSC': 100,
                      'Fluo-N2DH-GOWT1': 100,
                      'Fluo-N2DH-SIM+': 100,
                      'Fluo-N2DL-HeLa': 50,
                      'PhC-C2DH-U373': 500,
                      'PhC-C2DL-PSC': 50}
    coord_factors = {'DIC-C2DH-HeLa': 0.005,
                      'Fluo-C2DL-MSC': 0.005,
                      'Fluo-N2DH-GOWT1': 0.005,
                      'Fluo-N2DH-SIM+': 0.005,
                      'Fluo-N2DL-HeLa': 0.005,
                      'PhC-C2DH-U373': 0.005,
                      'PhC-C2DL-PSC': 0.005}
    return {'bandwidth': bandwidth[dataset_name],
            'min_label_size': min_label_size[dataset_name],
            'coord_factors': coord_factors[dataset_name]}


def get_instance_tracker_parameters(dataset_name):
    image_ignore_border = {'DIC-C2DH-HeLa': (50, 50),
              'Fluo-C2DL-MSC': (50, 50),
              'Fluo-N2DH-GOWT1': (50, 50),
              'Fluo-N2DH-SIM+': (0, 0),
              'Fluo-N2DL-HeLa': (25, 25),
              'PhC-C2DH-U373': (50, 50),
              'PhC-C2DL-PSC': (25, 25)}
    parent_search_dilation_size = {'DIC-C2DH-HeLa': 0,
                       'Fluo-C2DL-MSC': 0,
                       'Fluo-N2DH-GOWT1': 5,
                       'Fluo-N2DH-SIM+': 5,
                       'Fluo-N2DL-HeLa': 3,
                       'PhC-C2DH-U373': 0,
                       'PhC-C2DL-PSC': 0}
    max_parent_search_frames = {'DIC-C2DH-HeLa': 8,
                           'Fluo-C2DL-MSC': 8,
                           'Fluo-N2DH-GOWT1': 2,
                           'Fluo-N2DH-SIM+': 2,
                           'Fluo-N2DL-HeLa': 2,
                           'PhC-C2DH-U373': 8,
                           'PhC-C2DL-PSC': 2}
    max_merge_search_frames = {'DIC-C2DH-HeLa': 10,
                               'Fluo-C2DL-MSC': 10,
                               'Fluo-N2DH-GOWT1': 10,
                               'Fluo-N2DH-SIM+': 10,
                               'Fluo-N2DL-HeLa': 10,
                               'PhC-C2DH-U373': 10,
                               'PhC-C2DL-PSC': 10}
    min_track_length = {'DIC-C2DH-HeLa': 2,
                        'Fluo-C2DL-MSC': 2,
                        'Fluo-N2DH-GOWT1': 2,
                        'Fluo-N2DH-SIM+': 2,
                        'Fluo-N2DL-HeLa': 2,
                        'PhC-C2DH-U373': 2,
                        'PhC-C2DL-PSC': 5}
    return {'image_ignore_border': image_ignore_border[dataset_name],
            'parent_search_dilation_size': parent_search_dilation_size[dataset_name],
            'max_parent_search_frames': max_parent_search_frames[dataset_name],
            'max_merge_search_frames': max_merge_search_frames[dataset_name],
            'min_track_length': min_track_length[dataset_name]}


class MainLoop(MainLoopBase):
    def __init__(self, cv, dataset_name, num_embeddings, actual_network, network_parameters, num_frames, learning_rate=0.00001):
        super().__init__()
        self.cv = cv
        self.network = network
        self.actual_network = actual_network
        self.network_parameters = network_parameters

        self.batch_size = 1
        if self.cv == 'train_all':
            self.max_iter = 60000
        else:
            self.max_iter = 40000
        self.test_iter = self.max_iter // 2
        self.disp_iter = 100
        self.learning_rate = learning_rate
        self.learning_rates = [learning_rate, learning_rate * 0.1]
        self.learning_rate_boundaries = [self.max_iter // 2]

        self.l = 1.0
        self.l2_factor = 0.0
        self.term_1_2_normalization = 'individual'

        self.image_prefix = 'mask'
        self.track_file_name = 'res_track.txt'
        self.save_all_embeddings = False
        self.save_instances = True
        self.save_intermediate_instance_images = True
        self.save_challenge_instance_images = True
        self.save_overlapping_embeddings = True
        self.use_pyro_dataset = True

        self.image_size, self.test_image_size = image_sizes_for_dataset_name(dataset_name)
        self.tiled_processing = True
        self.tiled_increment = [128, 128]
        self.instances_ignore_border = [48, 48]
        self.scale_factor = [self.image_size[i] / self.test_image_size[i] for i in range(2)]
        self.normalized_embeddings = False
        self.use_seg_file = False
        self.num_frames = num_frames

        network_string = get_network_string(actual_network.__name__, network_parameters)
        optimizer_string = 'nadam_lr{}_b{}_t12norm{}_embnorm{}_f{}'.format(learning_rate, self.batch_size, self.term_1_2_normalization, self.normalized_embeddings, self.num_frames)
        folder_string = get_folder_string(self.image_size[0], num_embeddings, network_string, optimizer_string)

        self.snapshot_iter = 10000
        self.test_initialization = True
        self.current_iter = 0
        self.reg_constant = 0.00001
        self.bitwise_instance_image = True
        self.max_num_instances = 31
        if self.max_num_instances <= 15:
            self.bitwise_instances_image_type = tf.int32
        elif self.max_num_instances <= 31:
            self.bitwise_instances_image_type = tf.int64
        else:
            assert 'max_num_instances is too large'
        self.invert_transformation = False
        self.save_debug_images = False
        self.data_format = 'channels_first'
        if self.data_format == 'channels_first':
            self.channel_axis = 1
            self.time_stack_axis = 1
        else:
            self.channel_axis = 3
            self.time_stack_axis = 0
        self.num_embeddings = num_embeddings
        self.dataset_name = dataset_name
        self.base_folder = os.path.join('/media1/datasets/celltrackingchallenge/trainingdataset/mha/', self.dataset_name)
        self.base_folder_test = os.path.join('/media1/datasets/celltrackingchallenge/challengedataset/mha/', self.dataset_name)
        self.output_base_folder = os.path.join('/media0/experiments/cell_tracking_github_test/', self.dataset_name)
        self.output_folder = os.path.join(self.output_base_folder, folder_string + '_' + self.output_folder_timestamp())
        self.embedding_factors = {'bac': 1.0,
                                  'tra': 1.0}

        self.instance_image_creator_parameters = get_instance_image_creator_parameters(dataset_name)
        self.instance_tracker_parameters = get_instance_tracker_parameters(dataset_name)

        if self.cv == 'train_all':
            train_id_file = 'tra_seg_equal_all.csv'
            val_id_file = 'seg_all.csv'
        elif self.cv == 0:
            train_id_file = 'tra_seg_equal_train.csv'
            val_id_file = 'tra_val.csv'
        elif self.cv == 1:
            train_id_file = 'tra_seg_equal_val.csv'
            val_id_file = 'tra_train.csv'

        # train dataset
        dataset_parameters = get_dataset_parameters(dataset_name)
        dataset_parameters.update({'image_size': self.image_size,
                                   'num_frames': self.num_frames,
                                   'base_folder': self.base_folder,
                                   'data_format': self.data_format,
                                   'save_debug_images': self.save_debug_images,
                                   'max_num_instances': self.max_num_instances,
                                   'train_id_file': train_id_file,
                                   'val_id_file': val_id_file,
                                   'scale_factor': self.scale_factor,
                                   'bitwise_instance_image': self.bitwise_instance_image})
        dataset = Dataset(**dataset_parameters)
        if self.use_pyro_dataset:
            server_name = '@localhost:47832'
            uri = 'PYRO:cell_tracking' + server_name
            print('using pyro uri', uri)
            self.dataset_train = PyroClientDataset(uri, dataset_name=self.dataset_name, **dataset_parameters)
        else:
            self.dataset_train = dataset.dataset_train()
        self.dataset_train.get_next()

        # val dataset
        dataset_val_parameters = dataset_parameters.copy()
        dataset_val_parameters.update({'image_size': self.test_image_size,
                                       'num_frames': None,
                                       'scale_factor': [1.0, 1.0],
                                       'bitwise_instance_image': False,
                                       'max_num_instances': 32,
                                       'label_gaussian_blur_sigma': 0.0,
                                       'pad_image': False,
                                       'crop_image_size': None,
                                       'load_merged': False,
                                       'load_has_complete_seg': False,
                                       'load_seg_loss_mask': False,
                                       'create_instances_bac': False,
                                       'create_instances_merged': False,
                                       })
        dataset = Dataset(**dataset_val_parameters)
        self.dataset_val = dataset.dataset_val_single_frame()

        # test dataset
        dataset_test_parameters = dataset_parameters.copy()
        dataset_test_parameters.update({'image_size': self.test_image_size,
                                        'num_frames': None,
                                        'base_folder': self.base_folder_test,
                                        'pad_image': False,
                                        'crop_image_size': None,
                                        'load_merged': False,
                                        'load_has_complete_seg': False,
                                        'load_seg_loss_mask': False,
                                        'create_instances_bac': False,
                                        'create_instances_merged': False,
                                        'scale_factor': [1.0, 1.0]})
        dataset = Dataset(**dataset_test_parameters)
        self.dataset_test = dataset.dataset_val_single_frame()


    def loss_function(self, prediction, groundtruth, bitwise_instances):
        """
        Loss function for the tracking instances.
        :param prediction: The predicted embeddings.
        :param groundtruth: The groundtruth instances.
        :param bitwise_instances: If true, instances are bitwise and not channelwise.
        :return: The cosine embedding loss.
        """
        return cosine_embedding_per_instance_batch_loss(prediction, groundtruth,
                                                        normalized_embeddings=self.normalized_embeddings,
                                                        data_format=self.data_format,
                                                        term_1_2_normalization=self.term_1_2_normalization,
                                                        term_0_squared=True,
                                                        term_1_squared=True,
                                                        l=self.l,
                                                        l2=self.l2_factor,
                                                        bitwise_instances=bitwise_instances)

    def loss_function_background(self, prediction, groundtruth, bitwise_instances):
        """
        Loss function for the background instances.
        :param prediction: The predicted embeddings.
        :param groundtruth: The groundtruth instances.
        :param bitwise_instances: If true, instances are bitwise and not channelwise.
        :return: The cosine embedding loss.
        """
        return cosine_embedding_per_instance_batch_loss(prediction, groundtruth,
                                                        normalized_embeddings=self.normalized_embeddings,
                                                        data_format=self.data_format,
                                                        term_1_2_normalization=self.term_1_2_normalization,
                                                        term_0_squared=True,
                                                        term_1_squared=True,
                                                        l=self.l,
                                                        l2=self.l2_factor,
                                                        is_background=True,
                                                        bitwise_instances=bitwise_instances)


    def losses(self, embeddings_tuple, instances_tra, instances_bac, bitwise_instances):
        """
        Returns the losses for the given embeddings and instances.
        :param embeddings_tuple: The embedding tuples. Each tuple entry is a network output on which the losses will be calculated.
        :param instances_tra: The tracking instances.
        :param instances_bac: The background instances.
        :param bitwise_instances: If true, instances are in bitwise and not channelwise form. Bitwise produces the same results with less memory, but could be a little slower.
        :return: OrderedDict of all losses.
        """
        if not isinstance(embeddings_tuple, tuple):
            embeddings_tuple = (embeddings_tuple, )
        losses_list = []
        for i, embeddings in enumerate(embeddings_tuple):
            with tf.name_scope('losses' + str(i)):
                background_embedding_loss = self.embedding_factors['bac'] * self.loss_function_background(embeddings, instances_bac, bitwise_instances)
                losses_list.append(('loss_bac_emb_' + str(i), background_embedding_loss))
                tra_embedding_loss = self.embedding_factors['tra'] * self.loss_function(embeddings, instances_tra, bitwise_instances)
                losses_list.append(('loss_tra_emb_' + str(i), tra_embedding_loss))
        return OrderedDict(losses_list)

    def init_networks(self):
        """
        Init training and validation networks.
        """
        network_image_size = list(reversed(self.image_size))
        num_instances = 1 if self.bitwise_instance_image else None
        num_instances_val = None

        if self.data_format == 'channels_first':
            data_generator_entries = OrderedDict([('image', [1, self.num_frames] + network_image_size),
                                                  ('instances_merged', [num_instances, self.num_frames] + network_image_size),
                                                  ('instances_bac', [num_instances, self.num_frames] + network_image_size)])
            data_generator_entries_test_cropped_single_frame = OrderedDict([('image', [1] + network_image_size),
                                                                    ('instances_merged', [num_instances_val] + network_image_size),
                                                                    ('instances_bac', [num_instances_val] + network_image_size)])
            embedding_normalization_function = lambda x: tf.nn.l2_normalize(x, dim=self.channel_axis)
        else:
            assert 'channels_last not supported'
        data_generator_types = {'image': tf.float32,
                                'instances_merged': self.bitwise_instances_image_type,
                                'instances_bac': self.bitwise_instances_image_type}

        # create model with shared weights between train and val
        training_net = tf.make_template('net', self.network)

        # build train graph
        self.train_queue = DataGeneratorPadding(self.dataset_train, self.coord, data_generator_entries, batch_size=self.batch_size, data_types=data_generator_types, n_threads=4)

        # build train graph
        data, instances_tra, instances_bac = self.train_queue.dequeue()
        embeddings_tuple = training_net(data,
                                        num_outputs_embedding=self.num_embeddings,
                                        is_training=True,
                                        data_format=self.data_format,
                                        actual_network=self.actual_network,
                                        **self.network_parameters)

        if not isinstance(embeddings_tuple, tuple):
            embeddings_tuple = (embeddings_tuple, )

        loss_reg = get_reg_loss(self.reg_constant, True)

        with tf.name_scope('train_loss'):
            train_losses_dict = self.losses(embeddings_tuple, instances_tra, instances_bac, bitwise_instances=self.bitwise_instance_image)
            train_losses_dict['loss_reg'] = loss_reg
            self.loss = tf.reduce_sum(list(train_losses_dict.values()))
            self.train_losses = train_losses_dict

        # solver
        global_step = tf.Variable(self.current_iter)
        learning_rate = tf.train.piecewise_constant(global_step, self.learning_rate_boundaries, self.learning_rates)
        optimizer = tf.contrib.opt.NadamOptimizer(learning_rate=learning_rate)
        self.optimizer = optimizer.minimize(self.loss, global_step=global_step)

        # initialize variables
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        print('Variables')
        for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            print(i)

        # build val graph
        val_placeholders_cropped = create_placeholders(data_generator_entries_test_cropped_single_frame, shape_prefix=[1])
        self.data_cropped_val = val_placeholders_cropped['image']
        self.instances_cropped_tra_val = val_placeholders_cropped['instances_merged']
        self.instances_cropped_bac_val = val_placeholders_cropped['instances_bac']
        with tf.variable_scope('net/rnn', reuse=True):
            output_tuple = network_single_frame_with_lstm_states(self.data_cropped_val,
                                                   num_outputs_embedding=self.num_embeddings,
                                                   data_format=self.data_format,
                                                    actual_network=self.actual_network,
                                                   **self.network_parameters)
            self.lstm_input_states_cropped_val = output_tuple[0]
            self.lstm_output_states_cropped_val = output_tuple[1]
            self.embeddings_cropped_val = output_tuple[2:]

        if not isinstance(self.embeddings_cropped_val, tuple):
            self.embeddings_cropped_val = (self.embeddings_cropped_val,)

        with tf.variable_scope('loss'):
            val_losses_dict = self.losses(self.embeddings_cropped_val, self.instances_cropped_tra_val, self.instances_cropped_bac_val, bitwise_instances=False)
            val_losses_dict['loss_reg'] = loss_reg
            self.loss_val = tf.reduce_sum(list(val_losses_dict.values()))
            self.val_losses = val_losses_dict

        if not self.normalized_embeddings:
            self.embeddings_cropped_val = tuple([embedding_normalization_function(e) for e in self.embeddings_cropped_val])

    def test_cropped_image(self, dataset_entry, current_lstm_states, return_all_intermediate_embeddings=False):
        """
        Tests the whole image by cropping the input image.
        :param dataset_entry: The dataset entry.
        :param current_lstm_states: The current lstm states per tile.
        :param return_all_intermediate_embeddings: If true, return embeddings for all tiles.
        :return: merged embeddings, (list of all intermediate embeddings), list of next lstm states per tile
        """
        generators = dataset_entry['generators']
        full_image = generators['image']
        # initialize sizes based on data_format
        fetches = self.embeddings_cropped_val + self.lstm_output_states_cropped_val
        if self.data_format == 'channels_first':
            image_size_np = [1] + list(reversed(self.image_size))
            full_image_size_np = list(full_image.shape)
            embeddings_size_np = [self.num_embeddings] + list(reversed(self.image_size))
            full_embeddings_size_np = [self.num_embeddings] + list(full_image.shape[1:])
            inc = [0] + list(reversed(self.tiled_increment))
        else:
            image_size_np = list(reversed(self.image_size)) + [1]
            full_image_size_np = list(full_image.shape)
            embeddings_size_np = list(reversed(self.image_size)) + [self.num_embeddings]
            full_embeddings_size_np = list(full_image.shape[0:2]) + [self.num_embeddings]
            inc = list(reversed(self.tiled_increment)) + [0]
        # initialize on image tiler for the input and a list of image tilers for the embeddings
        image_tiler = ImageTiler(full_image_size_np, image_size_np, inc, True, -1)
        embeddings_tilers = tuple([ImageTiler(full_embeddings_size_np, embeddings_size_np, inc, True, -1) for _ in range(len(self.embeddings_cropped_val))])

        next_lstm_states = []
        all_intermediate_embeddings = []
        for state_index, all_tilers in enumerate(zip(*((image_tiler,) + embeddings_tilers))):
            image_tiler = all_tilers[0]
            embeddings_tilers = all_tilers[1:]
            current_image = image_tiler.get_current_data(full_image)
            feed_dict = {self.data_cropped_val: np.expand_dims(current_image, axis=0)}
            if len(current_lstm_states) > 0:
                for i in range(len(self.lstm_input_states_cropped_val)):
                    feed_dict[self.lstm_input_states_cropped_val[i]] = current_lstm_states[state_index][i]
            run_tuple = self.sess.run(fetches, feed_dict)
            image_tiler.set_current_data(current_image)
            for i, embeddings_tiler in enumerate(embeddings_tilers):
                embeddings = np.squeeze(run_tuple[i], axis=0)
                if return_all_intermediate_embeddings and i == len(embeddings_tilers) - 1:
                    all_intermediate_embeddings.append(embeddings)
                embeddings_tiler.set_current_data(embeddings)
            current_next_lstm_states = run_tuple[len(self.embeddings_cropped_val):len(self.embeddings_cropped_val)+len(self.lstm_output_states_cropped_val)]
            next_lstm_states.append(current_next_lstm_states)

        embeddings = [embeddings_tiler.output_image for embeddings_tiler in embeddings_tilers]

        if return_all_intermediate_embeddings:
            return embeddings, all_intermediate_embeddings, next_lstm_states
        else:
            return embeddings, next_lstm_states

    def merge_tiled_instances(self, tiled_instances):
        """
        Merges neighboring tiled instances.
        :param tiled_instances: list of instances.
        :return: Merged instances.
        """
        # initialize sizes based on data_format
        instances_size_np = [2] + list(reversed(self.image_size))
        full_instances_size_np = [2] + list(reversed(self.test_image_size))
        inc = [0] + list(reversed(self.tiled_increment))
        # initialize on image tiler for the input and a list of image tilers for the embeddings
        instance_tiler = ImageTiler(full_instances_size_np, instances_size_np, inc, True, 0, output_image_dtype=np.uint16)
        instance_merger = InstanceMerger(ignore_border=self.instances_ignore_border)

        for i, instance_tiler in enumerate(instance_tiler):
            current_instance_pair = tiled_instances[i]
            instance_tiler.set_current_data(current_instance_pair, instance_merger.merge_as_larger_instances, merge_whole_image=True)
        instances = instance_tiler.output_image

        return instances

    def get_instances(self, stacked_two_embeddings):
        """
        Return instances for subsequent embeddings.
        :param stacked_two_embeddings: Two stacked embedding frames.
        :return: The instances.
        """
        clusterer = InstanceImageCreator(**self.instance_image_creator_parameters)
        clusterer.create_instance_image(stacked_two_embeddings)
        instances = clusterer.label_image
        return instances

    def get_merged_instances(self, stacked_two_embeddings_tile_list):
        tiled_instances = []
        for stacked_two_embeddings in stacked_two_embeddings_tile_list:
            if self.data_format == 'channels_last':
                stacked_two_embeddings = np.transpose(stacked_two_embeddings, [3, 0, 1, 2])
            instances = self.get_instances(stacked_two_embeddings)
            tiled_instances.append(instances)
        merged_instances = self.merge_tiled_instances(tiled_instances)
        return merged_instances

    def test(self):
        """
        Test validation and test dataset.
        """
        self.test_folder(self.base_folder, self.dataset_val, 'train', self.cv != 'train_all')
        self.test_folder(self.base_folder_test, self.dataset_test, 'test', False)

    def test_folder(self, base_folder, dataset, name, update_loss):
        """
        Test dataset folder. Creates embeddings and also performs instance segmentation (if parameters are set in __init__)
        :param base_folder: The base folder of the images to test.
        :param dataset: The dataset used for data preprocessing.
        :param name: The name of the dataset.
        :param update_loss: If true, update loss values.
        """
        setup_base_folder = os.path.join(base_folder, 'setup')
        video_frame_list_file_name = os.path.join(setup_base_folder, 'frames.csv')
        iterator = IdListIterator(os.path.join(setup_base_folder, 'video_only_all.csv'), random=False, keys=['video_id'])
        video_id_frames = utils.io.text.load_dict_csv(video_frame_list_file_name)

        num_entries = iterator.num_entries()
        for current_entry_index in range(num_entries):
            video_id = iterator.get_next_id()['video_id']
            video_frames_all = video_id_frames[video_id]
            frame_index = 0
            instance_tracker = InstanceTracker(**self.instance_tracker_parameters)
            current_all_embeddings_cropped = [[] for _ in range(len(self.embeddings_cropped_val))]
            for j in range(len(video_frames_all) - self.num_frames + 1):
                print('Processing frame', j)
                current_lstm_states_cropped = []
                video_frames = video_frames_all[j:j + self.num_frames]
                current_all_intermediate_embeddings = []
                for k, video_frame in enumerate(video_frames):
                    current_id = video_id + '_' + video_frame
                    dataset_entry = dataset.get({'video_id': video_id, 'frame_id': video_frame, 'unique_id': current_id})
                    embeddings_cropped, all_intermediate_embeddings, current_lstm_states_cropped = self.test_cropped_image(dataset_entry, current_lstm_states_cropped, return_all_intermediate_embeddings=True)
                    current_all_intermediate_embeddings.append(all_intermediate_embeddings)
                    if self.save_overlapping_embeddings:
                        if j == 0 or k >= self.num_frames - 2:
                            for i, e in enumerate(embeddings_cropped):
                                current_all_embeddings_cropped[i].append((e * 128).astype(np.int8))

                if j == 0:
                    for i in range(self.num_frames - 1):
                        stacked_two_embeddings_tile_list = []
                        for tile_i in range(len(current_all_intermediate_embeddings[0])):
                            stacked_two_embeddings_tile_list.append(np.stack([current_all_intermediate_embeddings[i][tile_i], current_all_intermediate_embeddings[i + 1][tile_i]], axis=self.time_stack_axis))
                        if self.save_instances:
                            instances = self.get_merged_instances(stacked_two_embeddings_tile_list)
                            instance_tracker.add_new_label_image(instances)
                        if self.save_all_embeddings:
                            for tile_i, e in enumerate(stacked_two_embeddings_tile_list):
                                utils.io.image.write_np(e, self.output_file_for_current_iteration(name + '_' + video_id, 'embeddings', 'frame_' + str(j + i).zfill(3) + '_tile_' + str(tile_i).zfill(2) + '.mha'), compress=False)
                else:
                    stacked_two_embeddings_tile_list = []
                    for tile_i in range(len(current_all_intermediate_embeddings[0])):
                        stacked_two_embeddings_tile_list.append(np.stack([current_all_intermediate_embeddings[num_frames - 2][tile_i], current_all_intermediate_embeddings[num_frames - 1][tile_i]], axis=self.time_stack_axis))
                    if self.save_instances:
                        instances = self.get_merged_instances(stacked_two_embeddings_tile_list)
                        instance_tracker.add_new_label_image(instances)
                    if self.save_all_embeddings:
                        for tile_i, e in enumerate(stacked_two_embeddings_tile_list):
                            utils.io.image.write_np(e, self.output_file_for_current_iteration(name + '_' + video_id, 'embeddings', 'frame_' + str(frame_index).zfill(3) + '_tile_' + str(tile_i).zfill(2) + '.mha'), compress=False)
                if j == 0:
                    frame_index += self.num_frames - 1
                else:
                    frame_index += 1

            if self.save_overlapping_embeddings:
                for i in range(len(current_all_embeddings_cropped)):
                    if len(current_all_embeddings_cropped[i]) > 0:
                        current_embeddings = np.stack(current_all_embeddings_cropped[i], axis=1)
                        utils.io.image.write_np(current_embeddings, self.output_file_for_current_iteration(name + '_' + video_id, 'embeddings_cropped_' + str(i) + '.mha'))

            if self.save_instances:
                if self.save_intermediate_instance_images:
                    utils.io.image.write_np(instance_tracker.stacked_label_image.astype(np.uint16), self.output_file_for_current_iteration(name + '_' + video_id, 'merged_instances.mha'))

                to_size = dataset_entry['datasources']['image'].GetSize()
                transformation = dataset_entry['transformations']['image_transformation']
                # transformation = scale_transformation_for_image_sizes(from_size, to_size, [0.95, 0.95] if self.dataset_name == 'Fluo-C2DL-MSC' else [1.0, 1.0])
                instance_tracker.resample_stacked_label_image(to_size, transformation, 1.0)
                if self.save_intermediate_instance_images:
                    utils.io.image.write_np(instance_tracker.stacked_label_image.astype(np.uint16), self.output_file_for_current_iteration(name + '_' + video_id, 'merged_instances_resampled.mha'))

                instance_tracker.finalize()
                if self.save_intermediate_instance_images:
                    utils.io.image.write_np(instance_tracker.stacked_label_image.astype(np.uint16), self.output_file_for_current_iteration(name + '_' + video_id, 'merged_instances_final.mha'))

                if self.save_challenge_instance_images:
                    track_tuples = instance_tracker.track_tuples
                    final_track_image_np = instance_tracker.stacked_label_image

                    print('Saving output images and tracks...')
                    final_track_images_sitk = [utils.sitk_np.np_to_sitk(np.squeeze(im)) for im in np.split(final_track_image_np, final_track_image_np.shape[0], axis=0)]
                    for i, final_track_image_sitk in enumerate(final_track_images_sitk):
                        video_frame = str(i).zfill(3)
                        utils.io.image.write(final_track_image_sitk, self.output_file_for_current_iteration(name + '_' + video_id, 'instances', self.image_prefix + video_frame + '.tif'))
                    utils.io.text.save_list_csv(track_tuples, self.output_file_for_current_iteration(name + '_' + video_id, 'instances', self.track_file_name), delimiter=' ')

        # finalize loss values
        if update_loss:
            self.val_loss_aggregator.finalize(self.current_iter)


if __name__ == '__main__':
    datasets = [#'DIC-C2DH-HeLa',
                #'Fluo-N2DL-HeLa',
                'Fluo-N2DH-GOWT1',
                #'Fluo-N2DH-SIM+',
                #'PhC-C2DH-U373',
                #'Fluo-C2DL-MSC',
                ]
    network_parameters = {'stacked_hourglass': True,
                          'filters': 64,
                          'levels': 7,
                          'activation': 'relu',
                          'normalize': False,
                          'padding': 'same'}
    num_frames = 8
    for dataset in datasets:
        loop = MainLoop('train_all', dataset, 16, UnetIntermediateGruWithStates2D, network_parameters, num_frames, 0.0001)
        loop.run()

