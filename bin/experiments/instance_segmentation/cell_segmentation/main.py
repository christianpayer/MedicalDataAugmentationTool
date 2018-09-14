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
from network import network
import utils.io.text
import utils.sitk_np
import utils.sitk_image
from iterators.id_list_iterator import IdListIterator


class MainLoop(MainLoopBase):
    def __init__(self, dataset_name):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = 10
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
        self.output_base_folder = '/media1/experiments/cell_tracking/miccai2018_segmentation/' + self.dataset_name
        self.training_base_folder = os.path.join(self.challenge_base_folder, 'trainingdataset/', self.dataset_name)
        self.testing_base_folder = os.path.join(self.challenge_base_folder, 'challengedataset/', self.dataset_name)
        self.output_folder = os.path.join(self.output_base_folder, self.output_folder_timestamp())
        self.embedding_factors = {'bac': 1, 'tra': 1}
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
        self.bitwise_instance_image = False
        self.dataset = Dataset(self.image_size, num_frames=1,
                               base_folder=self.training_base_folder,
                               data_format=self.data_format,
                               save_debug_images=True,
                               instance_image_radius_factor=instance_image_radius_factor,
                               max_num_instances=32,
                               train_id_file='tra_all.csv',
                               val_id_file='tra_all.csv',
                               image_gaussian_blur_sigma=1.0,
                               label_gaussian_blur_sigma=label_gaussian_blur_sigma,
                               normalization_consideration_factors=normalization_consideration_factor,
                               pad_image=pad_image,
                               crop_image_size=crop_image_size)
        self.dataset_train = self.dataset.dataset_train_single_frame()
        self.dataset_val = self.dataset.dataset_val_single_frame()
        self.dataset_train.get_next()
        self.setup_base_folder = os.path.join(self.training_base_folder, 'setup')
        self.video_frame_list_file_name = os.path.join(self.setup_base_folder, 'frames.csv')
        self.iterator_val = IdListIterator(os.path.join(self.setup_base_folder, 'video_only_all.csv'), random=False, keys=['video_id'])

    def initNetworks(self):
        network_image_size = self.image_size

        if self.data_format == 'channels_first':
            data_generator_entries = OrderedDict([('image', [1] + network_image_size),
                                                  ('instances_merged', [None] + network_image_size),
                                                  ('instances_bac', [None] + network_image_size)])
        else:
            data_generator_entries = OrderedDict([('image', network_image_size + [1]),
                                                  ('instances_merged', network_image_size + [None]),
                                                  ('instances_bac', network_image_size + [None])])

        # create model with shared weights between train and val
        training_net = tf.make_template('net', network)
        loss_function = lambda prediction, groundtruth: cosine_embedding_per_instance_loss(prediction,
                                                                                           groundtruth,
                                                                                           data_format=self.data_format,
                                                                                           normalize=True,
                                                                                           term_1_squared=True,
                                                                                           l=1.0)

        # build train graph
        self.train_queue = DataGeneratorPadding(self.dataset_train, self.coord, data_generator_entries, batch_size=self.batch_size)
        data, tracking, instances_bac = self.train_queue.dequeue()
        embeddings_0, embeddings_1 = training_net(data,
                                  num_outputs_embedding=self.embeddings_dim,
                                  is_training=True,
                                  data_format=self.data_format)
        # losses
        background_embedding_loss_0 = self.embedding_factors['bac'] * loss_function(embeddings_0, instances_bac)
        tra_embedding_loss_0 = self.embedding_factors['tra'] * loss_function(embeddings_0, tracking)
        background_embedding_loss_1 = self.embedding_factors['bac'] * loss_function(embeddings_1, instances_bac)
        tra_embedding_loss_1 = self.embedding_factors['tra'] * loss_function(embeddings_1, tracking)
        self.loss_net = background_embedding_loss_0 + tra_embedding_loss_0 + background_embedding_loss_1 + tra_embedding_loss_1

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if self.reg_constant > 0:
                reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                self.loss_reg = self.reg_constant * tf.add_n(reg_losses)
                self.loss = self.loss_net + self.loss_reg
            else:
                self.loss_reg = 0
                self.loss = self.loss_net
        self.train_losses = OrderedDict([('loss_bac_emb_0', background_embedding_loss_0), ('loss_tra_emb_0', tra_embedding_loss_0),
                                         ('loss_bac_emb_1', background_embedding_loss_1), ('loss_tra_emb_1', tra_embedding_loss_1),
                                         ('loss_reg', self.loss_reg)])

        # solver
        global_step = tf.Variable(self.current_iter)
        learning_rate = tf.train.piecewise_constant(global_step, self.learning_rate_boundaries, self.learning_rates)
        self.optimizer = tf.contrib.opt.NadamOptimizer(learning_rate=learning_rate).minimize(self.loss, global_step=global_step)

        # build val graph
        val_placeholders = tensorflow_train.utils.tensorflow_util.create_placeholders(data_generator_entries, shape_prefix=[1])
        self.data_val = val_placeholders['image']
        self.tracking_val = val_placeholders['instances_merged']
        self.instances_bac_val = val_placeholders['instances_bac']

        self.embeddings_0_val, self.embeddings_1_val = training_net(self.data_val,
                                                                    num_outputs_embedding=self.embeddings_dim,
                                                                    is_training=False,
                                                                    data_format=self.data_format)
        self.embeddings_normalized_1_val = tf.nn.l2_normalize(self.embeddings_1_val, dim=1)
        # losses
        self.background_embedding_loss_0_val = self.embedding_factors['bac'] * loss_function(self.embeddings_0_val, self.instances_bac_val)
        self.tra_embedding_loss_0_val = self.embedding_factors['tra'] * loss_function(self.embeddings_0_val, self.tracking_val)
        self.background_embedding_loss_1_val = self.embedding_factors['bac'] * loss_function(self.embeddings_1_val, self.instances_bac_val)
        self.tra_embedding_loss_1_val = self.embedding_factors['tra'] * loss_function(self.embeddings_1_val, self.tracking_val)
        self.loss_val = self.background_embedding_loss_0_val + self.tra_embedding_loss_0_val # + self.background_embedding_loss_1_val + self.tra_embedding_loss_1_val
        self.val_losses = OrderedDict([('loss_bac_emb_0', self.background_embedding_loss_0_val), ('loss_tra_emb_0', self.tra_embedding_loss_0_val),
                                       ('loss_bac_emb_1', self.background_embedding_loss_1_val), ('loss_tra_emb_1', self.tra_embedding_loss_1_val),
                                       ('loss_reg', self.loss_reg)])


    def test(self):
        print('Testing...')

        channel_axis = 0
        if self.data_format == 'channels_last':
            channel_axis = 3
        interpolator = 'cubic'
        video_id_frames = utils.io.text.load_dict_csv(self.video_frame_list_file_name)
        num_entries = self.iterator_val.num_entries()
        for current_entry_index in range(num_entries):
            video_id = self.iterator_val.get_next_id()['video_id']
            video_frames = video_id_frames[video_id]
            current_embeddings = []
            current_embeddings_softmax = []
            for video_frame in video_frames:
                current_id = video_id + '_' + video_frame
                dataset_entry = self.dataset_val.get({'video_id': video_id, 'frame_id': video_frame, 'unique_id': current_id})
                datasources = dataset_entry['datasources']
                generators = dataset_entry['generators']
                transformations = dataset_entry['transformations']
                feed_dict = {self.data_val: np.expand_dims(generators['image'], axis=0),
                             self.tracking_val: np.expand_dims(generators['instances_merged'], axis=0),
                             self.instances_bac_val: np.expand_dims(generators['instances_bac'], axis=0)}

                run_tuple = self.sess.run((self.embeddings_1_val, self.embeddings_normalized_1_val, self.loss_val) + self.val_loss_aggregator.get_update_ops(),
                                          feed_dict=feed_dict)
                embeddings = np.squeeze(run_tuple[0], axis=0)
                embeddings_softmax = np.squeeze(run_tuple[1], axis=0)

                current_embeddings.append(embeddings)
                current_embeddings_softmax.append(embeddings_softmax)

                if self.invert_transformation:
                    input_sitk = datasources['tra']
                    transformation = transformations['image']
                    _ = utils.sitk_image.transform_np_output_to_sitk_input(output_image=embeddings, output_spacing=None, channel_axis=channel_axis, input_image_sitk=input_sitk, transform=transformation, interpolator=interpolator)

            current_embeddings_softmax = np.stack(current_embeddings_softmax, axis=1)
            utils.io.image.write_np(current_embeddings_softmax, os.path.join(self.output_folder, 'out/iter_' + str(self.current_iter) + '/' + video_id + '_embeddings_softmax.mha'))
            tensorflow_train.utils.tensorflow_util.print_progress_bar(current_entry_index, num_entries, prefix='Testing ', suffix=' complete')

        # finalize loss values
        self.val_loss_aggregator.finalize(self.current_iter)

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
