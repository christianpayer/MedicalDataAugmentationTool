#!/usr/bin/python
import sys

from collections import OrderedDict
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
#import SimpleITK as sitk
import tensorflow as tf
import glob

import utils.io.image
import tensorflow_train.utils.tensorflow_util
from utils.io.common import create_directories
from bin.dataset import Dataset
from bin.network_hourglass_dynamic import network_single_frame_with_lstm_states
from bin.embedding_tracker import EmbeddingTracker
import utils.io.text
import utils.sitk_image
import utils.np_image
import utils.sitk_np
from utils.timer import Timer


class MainLoop(object):
    def __init__(self, dataset_name, input_image_folder, output_folder, model_file_name, num_embeddings, image_size, additional_scale, normalization_consideration_factors,
                 coord_factors, min_samples, sigma, border_size, parent_dilation, parent_frame_search):
        self.coord_factors = coord_factors
        self.min_samples = min_samples
        self.sigma = sigma
        self.border_size = border_size
        self.parent_dilation = parent_dilation
        self.parent_frame_search = parent_frame_search
        self.sess = tf.Session()
        self.coord = tf.train.Coordinator()
        self.image_prefix = 'mask'
        self.track_file_name = 'res_track.txt'

        self.save_all_embeddings = True
        self.save_all_input_images = True
        self.save_all_predictions = True
        self.save_debug_images = False
        self.image_size = image_size
        self.data_format = 'channels_last'
        self.output_size = self.image_size
        self.num_embeddings = num_embeddings
        self.input_image_folder = input_image_folder
        self.output_folder = output_folder
        self.load_model_filename = model_file_name
        self.dataset = Dataset(self.image_size,
                               base_folder=self.input_image_folder,
                               data_format=self.data_format,
                               save_debug_images=self.save_debug_images,
                               normalization_consideration_factors=normalization_consideration_factors,
                               additional_scale=additional_scale,
                               image_gaussian_blur_sigma=2.0,
                               pad_image=False)

        self.dataset_val = self.dataset.dataset_val_single_frame()
        self.video_frames = glob.glob(self.input_image_folder + '/*.tif')
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
        val_placeholders = tensorflow_train.utils.tensorflow_util.create_placeholders(data_generator_entries, shape_prefix=[1])
        self.data_val = val_placeholders['image']
        self.tracking_val = val_placeholders['instances_merged']
        self.instances_bac_val = val_placeholders['instances_bac']

        with tf.variable_scope('net/rnn'):
            self.embeddings_0, self.embeddings_1, self.lstm_input_states, self.lstm_output_states = network_single_frame_with_lstm_states(self.data_val, num_outputs_embedding=self.num_embeddings, data_format=self.data_format)
            self.embeddings_normalized_0 = tf.nn.l2_normalize(self.embeddings_0, dim=self.channel_axis)
            self.embeddings_normalized_1 = tf.nn.l2_normalize(self.embeddings_1, dim=self.channel_axis)

    def test(self):
        #label_stack = utils.sitk_np.sitk_to_np(utils.io.image.read(os.path.join(self.output_folder, 'label_stack.mha'), sitk_pixel_type=sitk.sitkVectorUInt16))
        #label_stack = np.transpose(label_stack, [0, 3, 1, 2])

        if len(self.video_frames) == 0:
            print('No images found!')
            return
        print('Testing...', self.input_image_folder)
        tracker = EmbeddingTracker(coord_factors=self.coord_factors,
                                   stack_neighboring_slices=2,
                                   min_cluster_size=self.min_samples,
                                   min_samples=self.min_samples,
                                   min_label_size_per_stack=self.min_samples / 2,
                                   save_label_stack=True,
                                   image_ignore_border=self.border_size,
                                   parent_search_dilation_size=self.parent_dilation,
                                   max_parent_search_frames=self.parent_frame_search)
        #tracker.set_label_stack(label_stack)
        first = True
        current_images = []
        current_lstm_states = []
        for i, video_frame in enumerate(self.video_frames):
            with Timer('processing video frame ' + str(video_frame)):
                dataset_entry = self.dataset_val.get({'image_id': video_frame})
                generators = dataset_entry['generators']
                feed_dict = {self.data_val: np.expand_dims(generators['image'], axis=0)}
                if not first:
                    for j in range(len(self.lstm_input_states)):
                        feed_dict[self.lstm_input_states[j]] = current_lstm_states[j]
                run_tuple = self.sess.run([self.embeddings_normalized_1] + list(self.lstm_output_states), feed_dict=feed_dict)
                embeddings_normalized_1 = np.squeeze(run_tuple[0], axis=0)
                current_lstm_states = run_tuple[1:]
                if self.data_format == 'channels_last':
                    embeddings_normalized_1 = np.transpose(embeddings_normalized_1, [2, 0, 1])
                tracker.add_slice(embeddings_normalized_1)
                if self.save_all_input_images:
                    current_images.append(generators['image'])
                first = False

        # finalize tracker and resample to input resolution
        transformations = dataset_entry['transformations']
        transformation = transformations['image']
        datasources = dataset_entry['datasources']
        input_image = datasources['image']

        #utils.io.image.write_np(np.stack(tracker.label_stack_list, axis=1), os.path.join(self.output_folder, 'label_stack.mha'))
        tracker.finalize()
        tracker.resample_stacked_label_image(input_image, transformation, self.sigma)
        tracker.fix_tracks_after_resampling()
        track_tuples = tracker.track_tuples
        final_track_image_np = tracker.stacked_label_image

        print('Saving output images and tracks...')
        final_track_images_sitk = [utils.sitk_np.np_to_sitk(np.squeeze(im)) for im in np.split(final_track_image_np, final_track_image_np.shape[0], axis=0)]
        for video_frame, final_track_image_sitk in zip(self.video_frames, final_track_images_sitk):
            utils.io.image.write(final_track_image_sitk, os.path.join(self.output_folder, self.image_prefix + video_frame + '.tif'))
        utils.io.text.save_list_csv(track_tuples, os.path.join(self.output_folder, self.track_file_name), delimiter=' ')

        if self.save_all_embeddings:
            # embeddings are always 'channels_first'
            embeddings = np.stack(tracker.embeddings_slices, axis=1)
            utils.io.image.write_np(embeddings, os.path.join(self.output_folder, 'embeddings.mha'), 'channels_first')
        if self.save_all_input_images:
            images = np.stack(current_images, axis=self.time_stack_axis)
            utils.io.image.write_np(images, os.path.join(self.output_folder, 'image.mha'), self.data_format)
        if self.save_all_predictions:
            predictions = np.stack(tracker.stacked_label_image, axis=self.time_stack_axis)
            utils.io.image.write_np(predictions, os.path.join(self.output_folder, 'predictions.mha'), self.data_format)


if __name__ == '__main__':
    datasets = ['DIC-C2DH-HeLa',
                'Fluo-C2DL-MSC',
                'Fluo-N2DH-GOWT1',
                'Fluo-N2DH-SIM+',
                'Fluo-N2DL-HeLa',
                'PhC-C2DH-U373']
    c = {'DIC-C2DH-HeLa': 0.02,
         'Fluo-C2DL-MSC': 0.01,
         'Fluo-N2DH-GOWT1': 0.001,
         'Fluo-N2DH-SIM+': 0.001,
         'Fluo-N2DL-HeLa': 0.01,
         'PhC-C2DH-U373': 0.005}
    min_samples = {'DIC-C2DH-HeLa': 1000,
                   'Fluo-C2DL-MSC': 500,
                   'Fluo-N2DH-GOWT1': 50,
                   'Fluo-N2DH-SIM+': 100,
                   'Fluo-N2DL-HeLa': 25,
                   'PhC-C2DH-U373': 500}
    sigma = {'DIC-C2DH-HeLa': 2,
             'Fluo-C2DL-MSC': 2,
             'Fluo-N2DH-GOWT1': 2,
             'Fluo-N2DH-SIM+': 2,
             'Fluo-N2DL-HeLa': 1,
             'PhC-C2DH-U373': 2}
    border = {'DIC-C2DH-HeLa': (25, 25),
              'Fluo-C2DL-MSC': (10, 16),
              'Fluo-N2DH-GOWT1': (12, 12),
              'Fluo-N2DH-SIM+': (0, 0),
              'Fluo-N2DL-HeLa': (12, 18),
              'PhC-C2DH-U373': (18, 24)}
    parent_dilation = {'DIC-C2DH-HeLa': 0,
                       'Fluo-C2DL-MSC': 0,
                       'Fluo-N2DH-GOWT1': 5,
                       'Fluo-N2DH-SIM+': 5,
                       'Fluo-N2DL-HeLa': 3,
                       'PhC-C2DH-U373': 0}
    parent_frame_search = {'DIC-C2DH-HeLa': 8,
                           'Fluo-C2DL-MSC': 8,
                           'Fluo-N2DH-GOWT1': 2,
                           'Fluo-N2DH-SIM+': 2,
                           'Fluo-N2DL-HeLa': 2,
                           'PhC-C2DH-U373': 8}
    additional_scale = {'DIC-C2DH-HeLa': [1, 1],
                        'Fluo-C2DL-MSC': [0.95, 0.95],
                        'Fluo-N2DH-GOWT1': [1, 1],
                        'Fluo-N2DH-SIM+': [1, 1],
                        'Fluo-N2DL-HeLa': [1, 1],
                        'PhC-C2DH-U373': [1, 1]}
    normalization_consideration_factors = {'DIC-C2DH-HeLa': (0.2, 0.1),
                                           'Fluo-C2DL-MSC': (0.2, 0.01),
                                           'Fluo-N2DH-GOWT1': (0.2, 0.1),
                                           'Fluo-N2DH-SIM+': (0.2, 0.01),
                                           'Fluo-N2DL-HeLa': (0.2, 0.1),
                                           'PhC-C2DH-U373': (0.2, 0.1)}
    embedding_size = 16
    network_image_size = [256, 256]

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
                    network_image_size,
                    additional_scale[dataset_name],
                    normalization_consideration_factors[dataset_name],
                    c[dataset_name],
                    min_samples[dataset_name],
                    sigma[dataset_name],
                    border[dataset_name],
                    parent_dilation[dataset_name],
                    parent_frame_search[dataset_name])
    loop.run_test()
