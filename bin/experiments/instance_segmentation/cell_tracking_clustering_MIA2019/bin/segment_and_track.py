#!/usr/bin/python
import sys
from collections import OrderedDict
import os
import numpy as np
import tensorflow as tf

import utils.io.image
from utils.io.common import create_directories
from tensorflow_train.utils.tensorflow_util import create_placeholders
from bin.dataset import Dataset
from tensorflow_train.networks.unet_lstm_dynamic import network_single_frame_with_lstm_states
from utils.image_tiler import ImageTiler
import utils.io.text
import utils.sitk_np
from bin.clustering import InstanceImageCreator, InstanceMerger, InstanceTracker
from glob import glob


def get_dataset_parameters(dataset_name):
    normalization_consideration_factors = {'DIC-C2DH-HeLa': (0.001, 0.001),
                                           'Fluo-C2DL-MSC': (0.001, 0.001),
                                           'Fluo-N2DH-GOWT1': (0.001, 0.001),
                                           'Fluo-N2DH-SIM+': (0.001, 0.001),
                                           'Fluo-N2DL-HeLa': (0.001, 0.001),
                                           'PhC-C2DH-U373': (0.001, 0.001),
                                           'PhC-C2DL-PSC': (0.001, 0.001)}
    image_gaussian_blur_sigma = {'DIC-C2DH-HeLa': 1.0,
                                 'Fluo-C2DL-MSC': 1.0,
                                 'Fluo-N2DH-GOWT1': 1.0,
                                 'Fluo-N2DH-SIM+': 1.0,
                                 'Fluo-N2DL-HeLa': 1.0,
                                 'PhC-C2DH-U373': 1.0,
                                 'PhC-C2DL-PSC': 1.0}
    additional_scale = {'DIC-C2DH-HeLa': [1, 1],
                        'Fluo-C2DL-MSC': [0.95, 0.95],
                        'Fluo-N2DH-GOWT1': [1, 1],
                        'Fluo-N2DH-SIM+': [1, 1],
                        'Fluo-N2DL-HeLa': [1, 1],
                        'PhC-C2DH-U373': [1, 1],
                        'PhC-C2DL-PSC': [1, 1]}
    return {'normalization_consideration_factors': normalization_consideration_factors[dataset_name],
            'image_gaussian_blur_sigma': image_gaussian_blur_sigma[dataset_name],
            'additional_scale': additional_scale[dataset_name]}


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



def image_sizes_for_dataset_name(dataset_name):
    train_image_size = {'DIC-C2DH-HeLa': [256, 256],
                        'Fluo-C2DL-MSC':  [256, 256],
                        'Fluo-N2DH-GOWT1': [256, 256],
                        'Fluo-N2DH-SIM+': [256, 256],
                        'Fluo-N2DL-HeLa': [256, 256],
                        'PhC-C2DH-U373': [256, 256],
                        'PhC-C2DL-PSC': [256, 256]}

    test_image_size = {'DIC-C2DH-HeLa': [256, 256],  # * 2
                       'Fluo-C2DL-MSC': [256, 256],  # uneven
                       'Fluo-N2DH-GOWT1': [512, 512],  # * 2
                       'Fluo-N2DH-SIM+': [512, 512],  # uneven
                       'Fluo-N2DL-HeLa': [1100, 700],  # exact
                       'PhC-C2DH-U373': [512, 384],  # uneven
                       'PhC-C2DL-PSC': [720, 576]}  # exact

    return train_image_size[dataset_name], test_image_size[dataset_name]


class MainLoop(object):
    def __init__(self, dataset_name, input_image_folder, output_folder, model_file_name, num_embeddings, num_frames):
        self.dataset_name = dataset_name
        self.sess = tf.Session()
        self.coord = tf.train.Coordinator()
        self.image_prefix = 'mask'
        self.track_file_name = 'res_track.txt'

        self.save_all_embeddings = False
        self.save_all_input_images = False
        self.save_all_predictions = True
        self.hdbscan = False
        self.image_size, self.test_image_size = image_sizes_for_dataset_name(dataset_name)
        self.tiled_increment = [128, 128]
        self.instances_ignore_border = [32, 32]
        self.num_frames = num_frames
        self.data_format = 'channels_last'
        self.output_size = self.image_size
        self.num_embeddings = num_embeddings
        self.input_image_folder = input_image_folder
        self.output_folder = output_folder
        self.load_model_filename = model_file_name
        self.dataset_parameters = get_dataset_parameters(dataset_name)
        self.instance_image_creator_parameters = get_instance_image_creator_parameters(dataset_name)
        if self.hdbscan == True:
            if self.dataset_name == 'DIC-C2DH-HeLa':
                min_cluster_size = 1000
                coord_factors = 0.02
            if self.dataset_name == 'Fluo-N2DH-GOWT1':
                min_cluster_size = 1000
                coord_factors = 0.001
            if self.dataset_name == 'PhC-C2DH-U373':
                min_cluster_size = 500
                coord_factors = 0.005
            if self.dataset_name == 'Fluo-N2DL-HeLa':
                min_cluster_size = 25
                coord_factors = 0.01
            self.instance_image_creator_parameters['min_cluster_size'] = min_cluster_size
            self.instance_image_creator_parameters['coord_factors'] = coord_factors
            self.instance_image_creator_parameters['hdbscan'] = True
        self.instance_tracker_parameters = get_instance_tracker_parameters(dataset_name)
        self.dataset = Dataset(self.test_image_size,
                               base_folder=self.input_image_folder,
                               data_format=self.data_format,
                               debug_image_folder=os.path.join(self.output_folder, 'input_images') if self.save_all_input_images else None,
                               **self.dataset_parameters)

        self.dataset_val = self.dataset.dataset_val_single_frame()
        self.video_frames = glob(self.input_image_folder + '/*.tif')
        self.video_frames = sorted([os.path.splitext(os.path.basename(frame))[0][1:] for frame in self.video_frames])

        if self.data_format == 'channels_first':
            self.channel_axis = 1
            self.time_stack_axis = 1
        else:
            self.channel_axis = 3
            self.time_stack_axis = 0

    def create_output_folder(self):
        create_directories(self.output_folder)

    def load_model(self):
        self.saver = tf.train.Saver()
        print('Restoring model ' + self.load_model_filename)
        self.saver.restore(self.sess, self.load_model_filename)

    def init_all(self):
        self.init_networks()
        self.load_model()
        self.create_output_folder()

    def run_test(self):
        self.init_all()
        print('Starting main test loop')
        self.test()

    def init_networks(self):
        network_image_size = self.image_size
        network_output_size = self.output_size

        if self.data_format == 'channels_first':
            data_generator_entries = OrderedDict([('image', [1] + network_image_size),
                                                  ('instances_merged', [None] + network_output_size),
                                                  ('instances_bac', [None] + network_output_size)])
        else:
            data_generator_entries = OrderedDict([('image', network_image_size + [1]),
                                                  ('instances_merged', network_output_size + [None]),
                                                  ('instances_bac', network_output_size + [None])])

        # build val graph
        val_placeholders = create_placeholders(data_generator_entries, shape_prefix=[1])
        self.data_val = val_placeholders['image']

        with tf.variable_scope('net/rnn'):
            self.lstm_input_states, self.lstm_output_states, self.embeddings_0, self.embeddings_1 = network_single_frame_with_lstm_states(self.data_val, num_outputs_embedding=self.num_embeddings, data_format=self.data_format)
            self.embeddings_normalized_0 = tf.nn.l2_normalize(self.embeddings_0, dim=self.channel_axis)
            self.embeddings_normalized_1 = tf.nn.l2_normalize(self.embeddings_1, dim=self.channel_axis)
            self.embeddings_cropped_val = (self.embeddings_normalized_0, self.embeddings_normalized_1)
            self.lstm_output_states_cropped_val = self.lstm_output_states
            self.lstm_input_states_cropped_val = self.lstm_input_states

    def test_cropped_image(self, dataset_entry, current_lstm_states, return_all_intermediate_embeddings=False):
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
            feed_dict = {self.data_val: np.expand_dims(current_image, axis=0)}
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
        if len(self.video_frames) == 0:
            print('No images found!')
            return
        print('Testing...', self.input_image_folder)

        video_frames_all = self.video_frames
        frame_index = 0
        instance_tracker = InstanceTracker(**self.instance_tracker_parameters)
        for j in range(len(video_frames_all)-self.num_frames + 1):
            print('Processing frame', j)
            current_lstm_states_cropped = []
            video_frames = video_frames_all[j:j+self.num_frames]
            current_all_intermediate_embeddings = []
            for k, video_frame in enumerate(video_frames):
                dataset_entry = self.dataset_val.get({'image_id': video_frame})
                embeddings_cropped, all_intermediate_embeddings, current_lstm_states_cropped = self.test_cropped_image(dataset_entry, current_lstm_states_cropped, return_all_intermediate_embeddings=True)
                current_all_intermediate_embeddings.append(all_intermediate_embeddings)

            if j == 0:
                for i in range(self.num_frames - 1):
                    stacked_two_embeddings_tile_list = []
                    for tile_i in range(len(current_all_intermediate_embeddings[0])):
                        stacked_two_embeddings_tile_list.append(np.stack([current_all_intermediate_embeddings[i][tile_i], current_all_intermediate_embeddings[i+1][tile_i]], axis=self.time_stack_axis))
                    instances = self.get_merged_instances(stacked_two_embeddings_tile_list)
                    instance_tracker.add_new_label_image(instances)
                    if self.save_all_embeddings:
                        for tile_i, e in enumerate(stacked_two_embeddings_tile_list):
                            utils.io.image.write_np(e, os.path.join(self.output_folder, 'embeddings', 'frame_' + str(j + i).zfill(3) + '_tile_' + str(tile_i).zfill(2) + '.mha'), compress=False)
            else:
                stacked_two_embeddings_tile_list = []
                for tile_i in range(len(current_all_intermediate_embeddings[0])):
                    stacked_two_embeddings_tile_list.append(np.stack([current_all_intermediate_embeddings[num_frames-2][tile_i], current_all_intermediate_embeddings[num_frames-1][tile_i]], axis=self.time_stack_axis))
                instances = self.get_merged_instances(stacked_two_embeddings_tile_list)
                instance_tracker.add_new_label_image(instances)
                if self.save_all_embeddings:
                    for tile_i, e in enumerate(stacked_two_embeddings_tile_list):
                        utils.io.image.write_np(e, os.path.join(self.output_folder, 'embeddings', 'frame_' + str(frame_index).zfill(3) + '_tile_' + str(tile_i).zfill(2) + '.mha'), compress=False)
            if j == 0:
                frame_index += self.num_frames - 1
            else:
                frame_index += 1

            if self.save_all_predictions:
                utils.io.image.write_np(instance_tracker.stacked_label_image.astype(np.uint16), os.path.join(self.output_folder, 'predictions', 'merged_instances.mha'))

        to_size = dataset_entry['datasources']['image'].GetSize()
        transformation = dataset_entry['transformations']['image']
        #transformation = scale_transformation_for_image_sizes(from_size, to_size, [0.95, 0.95] if self.dataset_name == 'Fluo-C2DL-MSC' else [1.0, 1.0])
        instance_tracker.resample_stacked_label_image(to_size, transformation, 1.0)
        if self.save_all_predictions:
            utils.io.image.write_np(instance_tracker.stacked_label_image.astype(np.uint16), os.path.join(self.output_folder, 'predictions', 'merged_instances_resampled.mha'))

        instance_tracker.finalize()
        if self.save_all_predictions:
            utils.io.image.write_np(instance_tracker.stacked_label_image.astype(np.uint16), os.path.join(self.output_folder, 'predictions', 'merged_instances_final.mha'))

        track_tuples = instance_tracker.track_tuples
        final_track_image_np = instance_tracker.stacked_label_image

        print('Saving output images and tracks...')
        final_track_images_sitk = [utils.sitk_np.np_to_sitk(np.squeeze(im)) for im in np.split(final_track_image_np, final_track_image_np.shape[0], axis=0)]
        for i, final_track_image_sitk in enumerate(final_track_images_sitk):
            video_frame = str(i).zfill(3)
            utils.io.image.write(final_track_image_sitk, os.path.join(self.output_folder, self.image_prefix + video_frame + '.tif'))
        utils.io.text.save_list_csv(track_tuples, os.path.join(self.output_folder, self.track_file_name), delimiter=' ')


if __name__ == '__main__':
    embedding_size = 16
    num_frames = 8

    if len(sys.argv) != 5:
        print('Usage: segment_and_track.py input_folder output_folder weights_file dataset_name')
    input_image_folder = sys.argv[1]
    output_folder = sys.argv[2]
    model_file_name = sys.argv[3]
    dataset_name = sys.argv[4]

    loop = MainLoop(dataset_name,
                    input_image_folder,
                    output_folder,
                    model_file_name,
                    embedding_size,
                    num_frames)
    loop.run_test()

