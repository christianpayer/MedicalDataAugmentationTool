#!/usr/bin/python
import sys
import math

sys.path.append('/home/chris/work/CellTracking/')
from collections import OrderedDict
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import tensorflow as tf
import SimpleITK as sitk
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
    def __init__(self, dataset_name, base_folder, output_folder, model_file_name, num_embeddings, image_size,
                 coord_factors, min_samples, sigma, border_size, parent_dilation, parent_frame_search):
        self.coord_factors = coord_factors
        self.min_samples = min_samples
        self.sigma = sigma
        self.border_size = border_size
        self.parent_dilation = parent_dilation
        self.parent_frame_search = parent_frame_search
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.coord = tf.train.Coordinator()
        self.image_prefix = 'mask'
        self.track_file_name = 'res_track.txt'

        self.save_debug_images = False
        self.image_size = image_size
        self.data_format = 'channels_last'
        self.output_size = self.image_size
        self.num_embeddings = num_embeddings
        self.base_folder = base_folder
        self.output_folder = output_folder
        self.load_model_filename = model_file_name
        additional_scale = {'DIC-C2DH-HeLa': [1, 1],
                           'Fluo-C2DL-MSC': [0.95, 0.95],
                           'Fluo-N2DH-GOWT1': [1, 1],
                           'Fluo-N2DH-SIM+': [1, 1],
                           'Fluo-N2DL-HeLa': [1, 1],
                           'PhC-C2DH-U373': [1, 1],
                           'PhC-C2DL-PSC': [1, 1]}
        additional_scale = additional_scale[dataset_name]
        normalization_consideration_factors = {'DIC-C2DH-HeLa': (0.2, 0.1),
                                               'Fluo-C2DL-MSC': (0.2, 0.01),
                                               'Fluo-N2DH-GOWT1': (0.2, 0.1),
                                               'Fluo-N2DH-SIM+': (0.2, 0.01),
                                               'Fluo-N2DL-HeLa': (0.2, 0.1),
                                               'PhC-C2DH-U373': (0.2, 0.1),
                                               'PhC-C2DL-PSC': (0.2, 0.1)}
        normalization_consideration_factor = normalization_consideration_factors[dataset_name]
        self.dataset = Dataset(self.image_size,
                               base_folder=self.base_folder,
                               data_format=self.data_format,
                               save_debug_images=self.save_debug_images,
                               normalization_consideration_factors=normalization_consideration_factor,
                               additional_scale=additional_scale,
                               image_gaussian_blur_sigma=2.0,
                               pad_image=False)

        self.dataset_val = self.dataset.dataset_val_single_frame()

        if self.data_format == 'channels_first':
            self.channel_axis = 1
            self.time_stack_axis = 1
        else:
            self.channel_axis = 3
            self.time_stack_axis = 0

    def __del__(self):
        self.sess.close()
        tf.reset_default_graph()

    # def loadModel(self):
    #    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='net/first_frame_net') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='net/data_conv'))
    #    model_filename = 'weights/model-' + str(self.current_iter)
    #    print('Restoring model ' + model_filename)
    #    saver.restore(self.sess, model_filename)

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
            self.embeddings_val, self.embeddings_2_val, self.lstm_input_states_val, self.lstm_output_states_val = network_single_frame_with_lstm_states(self.data_val, num_outputs_embedding=self.num_embeddings, data_format=self.data_format)
            self.embeddings_normalized_val = tf.nn.l2_normalize(self.embeddings_val, dim=self.channel_axis)
            self.embeddings_normalized_2_val = tf.nn.l2_normalize(self.embeddings_2_val, dim=self.channel_axis)

    def testxx(self):
        print('Testing...', self.base_folder)
        video_frames = glob.glob(self.base_folder + '*.tif')
        video_frames = sorted([os.path.splitext(os.path.basename(frame))[0][1:] for frame in video_frames])
        #video_frames = video_frames[100:]

        #coord_factors = 0.001
        #min_cluster_size = 100
        #min_samples = 100
        #min_label_size_per_stack = 100
        tracker = EmbeddingTracker(coord_factors=self.coord_factors,
                                   stack_neighboring_slices=2,
                                   min_cluster_size=self.min_samples,
                                   min_samples=self.min_samples,
                                   min_label_size_per_stack=self.min_samples / 2,
                                   save_label_stack=True,
                                   image_ignore_border=self.border_size,
                                   parent_search_dilation_size=self.parent_dilation,
                                   max_parent_search_frames=self.parent_frame_search)

        first = True
        current_predictions = []
        current_predictions_2 = []
        current_images = []
        # reset_every_frames = 20
        for i, video_frame in enumerate(video_frames):
            #if int(video_frame) < 150 or int(video_frame) > 250:
            #    continue
            with Timer('processing video frame ' + str(video_frame)):
                dataset_entry = self.dataset_val.get({'image_id': video_frame})
                datasources = dataset_entry['datasources']
                generators = dataset_entry['generators']
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
                #current_predictions.append(embeddings_softmax)
                #current_predictions_2.append(embeddings_softmax_2)
                current_images.append(generators['image'])
                # current_instances.append(instance_segmentation_test.get_instances_cosine_kmeans_2d(embeddings_softmax))
                first = False

                datasources = dataset_entry['datasources']
                input_image = datasources['image']
                transformations = dataset_entry['transformations']
                transformation = transformations['image']
                # embeddings_original = utils.sitk_image.transform_np_output_to_sitk_input(embeddings_softmax_2,
                #                                                                        output_spacing=None,
                #                                                                        channel_axis=2,
                #                                                                        input_image_sitk=input_image,
                #                                                                        transform=transformation,
                #                                                                        interpolator='linear',
                #                                                                        output_pixel_type=sitk.sitkFloat32)
                # embeddings_softmax_2 = utils.sitk_np.sitk_list_to_np(embeddings_original, axis=2)

                current_predictions_2.append(embeddings_softmax_2)

                # if not first and i % reset_every_frames != 0:
                #     run_tuple = self.sess.run([self.embeddings_normalized_val, self.embeddings_normalized_2_val] + list(self.lstm_output_states_val), feed_dict=feed_dict)
                #     embeddings_softmax_2 = np.squeeze(run_tuple[1], axis=0)
                #     tracker.add_reset_slice(np.transpose(embeddings_softmax_2, [2, 0, 1]))


        # prediction = np.stack(current_predictions, axis=self.time_stack_axis)
        # del current_predictions
        # utils.io.image.write_np(prediction, os.path.join(self.output_folder, 'embeddings.mha'), self.data_format)
        # del prediction
        prediction_2 = np.stack(current_predictions_2, axis=self.time_stack_axis)
        del current_predictions_2
        utils.io.image.write_np(prediction_2, os.path.join(self.output_folder, 'embeddings_2.mha'), self.data_format)
        del prediction_2
        images = np.stack(current_images, axis=self.time_stack_axis)
        del current_images
        utils.io.image.write_np(images, os.path.join(self.output_folder, 'image.mha'), self.data_format)
        del images
        transformations = dataset_entry['transformations']
        transformation = transformations['image']
        sitk.WriteTransform(transformation, os.path.join(self.output_folder, 'transform.txt'))


    def test(self):
        print('Testing...', self.base_folder)
        video_frames = glob.glob(self.base_folder + '*.tif')
        video_frames = sorted([os.path.splitext(os.path.basename(frame))[0][1:] for frame in video_frames])
        video_frames = video_frames[:5]

        #coord_factors = 0.001
        #min_cluster_size = 100
        #min_samples = 100
        #min_label_size_per_stack = 100
        tracker = EmbeddingTracker(coord_factors=self.coord_factors,
                                   stack_neighboring_slices=2,
                                   min_cluster_size=self.min_samples,
                                   min_samples=self.min_samples,
                                   min_label_size_per_stack=self.min_samples / 2,
                                   save_label_stack=True,
                                   image_ignore_border=self.border_size,
                                   parent_search_dilation_size=self.parent_dilation,
                                   max_parent_search_frames=self.parent_frame_search)

        first = True
        current_predictions = []
        current_predictions_2 = []
        current_images = []
        # reset_every_frames = 20
        for i, video_frame in enumerate(video_frames):
            #if int(video_frame) < 150 or int(video_frame) > 250:
            #    continue
            with Timer('processing video frame ' + str(video_frame)):
                dataset_entry = self.dataset_val.get({'image_id': video_frame})
                datasources = dataset_entry['datasources']
                generators = dataset_entry['generators']
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
                #current_predictions.append(embeddings_softmax)
                #current_predictions_2.append(embeddings_softmax_2)
                current_images.append(generators['image'])
                # current_instances.append(instance_segmentation_test.get_instances_cosine_kmeans_2d(embeddings_softmax))
                first = False

                datasources = dataset_entry['datasources']
                input_image = datasources['image']
                transformations = dataset_entry['transformations']
                transformation = transformations['image']
                # embeddings_original = utils.sitk_image.transform_np_output_to_sitk_input(embeddings_softmax_2,
                #                                                                        output_spacing=None,
                #                                                                        channel_axis=2,
                #                                                                        input_image_sitk=input_image,
                #                                                                        transform=transformation,
                #                                                                        interpolator='linear',
                #                                                                        output_pixel_type=sitk.sitkFloat32)
                # embeddings_softmax_2 = utils.sitk_np.sitk_list_to_np(embeddings_original, axis=2)

                current_predictions_2.append(embeddings_softmax_2)

                tracker.add_slice(np.transpose(embeddings_softmax_2, [2, 0, 1]))

                if tracker.stacked_label_image is not None:
                    utils.io.image.write_np(tracker.stacked_label_image, os.path.join(self.output_folder, 'merged.mha'))

                # if not first and i % reset_every_frames != 0:
                #     run_tuple = self.sess.run([self.embeddings_normalized_val, self.embeddings_normalized_2_val] + list(self.lstm_output_states_val), feed_dict=feed_dict)
                #     embeddings_softmax_2 = np.squeeze(run_tuple[1], axis=0)
                #     tracker.add_reset_slice(np.transpose(embeddings_softmax_2, [2, 0, 1]))


        # prediction = np.stack(current_predictions, axis=self.time_stack_axis)
        # del current_predictions
        # utils.io.image.write_np(prediction, os.path.join(self.output_folder, 'embeddings.mha'), self.data_format)
        # del prediction
        prediction_2 = np.stack(current_predictions_2, axis=self.time_stack_axis)
        del current_predictions_2
        utils.io.image.write_np(prediction_2, os.path.join(self.output_folder, 'embeddings_2.mha'), self.data_format)
        del prediction_2
        images = np.stack(current_images, axis=self.time_stack_axis)
        del current_images
        utils.io.image.write_np(images, os.path.join(self.output_folder, 'image.mha'), self.data_format)
        del images
        transformations = dataset_entry['transformations']
        transformation = transformations['image']
        sitk.WriteTransform(transformation, os.path.join(self.output_folder, 'transform.txt'))

        #if self.data_format == 'channels_last':
        #    prediction_2 = np.transpose(prediction_2, [3, 0, 1, 2])


        # two_slices = tracker.get_instances_cosine_dbscan_slice_by_slice(prediction_2)
        # utils.io.image.write_np(two_slices, os.path.join(self.output_folder, 'two_slices.mha'))
        # merged = tracker.merge_consecutive_slices(two_slices, slice_neighbour_size=2)
        # utils.io.image.write_np(merged, os.path.join(self.output_folder, 'merged.mha'), self.data_format)


        datasources = dataset_entry['datasources']
        input_image = datasources['image']
        if self.sigma == 1:
            interpolator = 'label_gaussian'
        else:
            interpolator = 'nearest'

        merged = tracker.stacked_label_image
        final_predictions = utils.sitk_image.transform_np_output_to_sitk_input(merged,
                                                                               output_spacing=None,
                                                                               channel_axis=0,
                                                                               input_image_sitk=input_image,
                                                                               transform=transformation,
                                                                               interpolator=interpolator,
                                                                               output_pixel_type=sitk.sitkUInt16)
        tracker.stacked_label_image = np.stack([utils.sitk_np.sitk_to_np(sitk_im) for sitk_im in final_predictions], axis=0)
        tracker.finalize()
        final_predictions = [utils.sitk_np.np_to_sitk(sitk_im) for sitk_im in tracker.stacked_label_image]
        track_tuples = tracker.track_tuples

        #final_predictions = [utils.sitk_np.np_to_sitk(np.squeeze(im), type=np.uint16) for im in np.split(merged, merged.shape[0], axis=0)]
        #final_predictions_smoothed_2 = [utils.sitk_image.apply_np_image_function(im, lambda x: self.label_smooth(x, sigma=2)) for im in final_predictions]
        if self.sigma > 1:
            final_predictions = [utils.sitk_image.apply_np_image_function(im, lambda x: self.label_smooth(x, sigma=self.sigma)) for im in final_predictions]

        for video_frame, final_prediction in zip(video_frames, final_predictions):
            utils.io.image.write(final_prediction, os.path.join(self.output_folder, self.image_prefix + video_frame + '.tif'))

        utils.io.image.write_np(np.stack(tracker.label_stack_list, axis=1), os.path.join(self.output_folder, 'label_stack.mha'))

        final_predictions_stacked = utils.sitk_image.accumulate(final_predictions)
        utils.io.image.write(final_predictions_stacked, os.path.join(self.output_folder, 'stacked.mha'))
        #utils.io.image.write(utils.sitk_image.accumulate(final_predictions_smoothed_2), os.path.join(self.output_folder, 'stacked_2.mha'))
        #utils.io.image.write(utils.sitk_image.accumulate(final_predictions_smoothed_4), os.path.join(self.output_folder, 'stacked_4.mha'))

        print(track_tuples)
        utils.io.text.save_list_csv(track_tuples, os.path.join(self.output_folder, self.track_file_name), delimiter=' ')

    def label_smooth(self, im, sigma):
        label_images, labels = utils.np_image.split_label_image_with_unknown_labels(im, dtype=np.float32)
        smoothed_label_images = utils.np_image.smooth_label_images(label_images, sigma=sigma, dtype=im.dtype)
        return utils.np_image.merge_label_images(smoothed_label_images, labels)

    def testabc(self):
        label_stack = utils.sitk_np.sitk_to_np(utils.io.image.read(os.path.join(self.output_folder, 'label_stack.mha'), sitk_pixel_type=sitk.sitkVectorUInt16))
        label_stack = np.transpose(label_stack, [0, 3, 1, 2])
        tracker = EmbeddingTracker(coord_factors=self.coord_factors,
                                   stack_neighboring_slices=2,
                                   min_cluster_size=self.min_samples,
                                   min_samples=self.min_samples,
                                   min_label_size_per_stack=self.min_samples / 2,
                                   save_label_stack=True,
                                   image_ignore_border=self.border_size,
                                   parent_search_dilation_size=self.parent_dilation,
                                   max_parent_search_frames=self.parent_frame_search)
        tracker.set_label_stack(label_stack)




        video_frames = glob.glob(self.base_folder + '*.tif')
        video_frames = sorted([os.path.splitext(os.path.basename(frame))[0][1:] for frame in video_frames])

        dataset_entry = self.dataset_val.get({'image_id': video_frames[0]})
        datasources = dataset_entry['datasources']
        input_image = datasources['image']
        transformations = dataset_entry['transformations']
        transformation = transformations['image']

        datasources = dataset_entry['datasources']
        input_image = datasources['image']
        if self.sigma == 1:
            interpolator = 'label_gaussian'
        else:
            interpolator = 'nearest'

        #merged = tracker.stacked_label_image
        #final_predictions = utils.sitk_image.transform_np_output_to_sitk_input(merged,
                                                                               # output_spacing=None,
                                                                               # channel_axis=0,
                                                                               # input_image_sitk=input_image,
                                                                               # transform=transformation,
                                                                               # interpolator=interpolator,
                                                                               # output_pixel_type=sitk.sitkUInt16)
        #tracker.stacked_label_image = np.stack([utils.sitk_np.sitk_to_np(sitk_im) for sitk_im in final_predictions], axis=0)
        tracker.finalize()
        merged = tracker.stacked_label_image
        final_predictions = utils.sitk_image.transform_np_output_to_sitk_input(merged,
                                                                               output_spacing=None,
                                                                               channel_axis=0,
                                                                               input_image_sitk=input_image,
                                                                               transform=transformation,
                                                                               interpolator=interpolator,
                                                                               output_pixel_type=sitk.sitkUInt16)
        tracker.stacked_label_image = np.stack([utils.sitk_np.sitk_to_np(sitk_im) for sitk_im in final_predictions], axis=0)
        tracker.fix_tracks_after_resampling()
        final_predictions = [utils.sitk_np.np_to_sitk(sitk_im) for sitk_im in tracker.stacked_label_image]
        track_tuples = tracker.track_tuples


        # final_predictions = [utils.sitk_np.np_to_sitk(np.squeeze(im), type=np.uint16) for im in np.split(merged, merged.shape[0], axis=0)]
        # final_predictions_smoothed_2 = [utils.sitk_image.apply_np_image_function(im, lambda x: self.label_smooth(x, sigma=2)) for im in final_predictions]
        if self.sigma > 1:
            final_predictions = [utils.sitk_image.apply_np_image_function(im, lambda x: self.label_smooth(x, sigma=self.sigma)) for im in final_predictions]

        for video_frame, final_prediction in zip(video_frames, final_predictions):
            utils.io.image.write(final_prediction, os.path.join(self.output_folder, self.image_prefix + video_frame + '.tif'))

        utils.io.image.write_np(np.stack(tracker.label_stack_list, axis=1), os.path.join(self.output_folder, 'label_stack.mha'))

        final_predictions_stacked = utils.sitk_image.accumulate(final_predictions)
        utils.io.image.write(final_predictions_stacked, os.path.join(self.output_folder, 'stacked.mha'))
        # utils.io.image.write(utils.sitk_image.accumulate(final_predictions_smoothed_2), os.path.join(self.output_folder, 'stacked_2.mha'))
        # utils.io.image.write(utils.sitk_image.accumulate(final_predictions_smoothed_4), os.path.join(self.output_folder, 'stacked_4.mha'))

        print(track_tuples)
        utils.io.text.save_list_csv(track_tuples, os.path.join(self.output_folder, self.track_file_name), delimiter=' ')

    def create_output_folder(self):
        create_directories(self.output_folder)

    def load_model(self):
        self.saver = tf.train.Saver()
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


if __name__ == '__main__':
    datasets = ['DIC-C2DH-HeLa',
                'Fluo-C2DL-MSC',
                'Fluo-N2DH-GOWT1',
                'Fluo-N2DH-SIM+',
                'Fluo-N2DL-HeLa',
                'PhC-C2DH-U373',
                'PhC-C2DL-PSC']
    e = 16
    f = 10
    i = 2
    s = 256
    sequences = ['01']
    cluster_size_ranges = {'DIC-C2DH-HeLa': [100, 500, 1000],
                           'Fluo-C2DL-MSC': [100, 500, 1000],
                           'Fluo-N2DH-GOWT1': [10, 20, 50, 100, 200],
                           'Fluo-N2DH-SIM+': [10, 20, 50, 100, 200],
                           'Fluo-N2DL-HeLa': [10, 20, 50, 100, 200],
                           'PhC-C2DH-U373': [100, 500, 1000],
                           'PhC-C2DL-PSC': [100, 500, 1000]}
    c = {'DIC-C2DH-HeLa': 0.005,
         'Fluo-C2DL-MSC': 0.01,
         'Fluo-N2DH-GOWT1': 0.001,
         'Fluo-N2DH-SIM+': 0.005,
         'Fluo-N2DL-HeLa': 0.01,
         'PhC-C2DH-U373': 0.001,
         'PhC-C2DL-PSC': 0.001}
    min_samples = {'DIC-C2DH-HeLa': 250,
                   'Fluo-C2DL-MSC': 500,
                   'Fluo-N2DH-GOWT1': 100,
                   'Fluo-N2DH-SIM+': 200,
                   'Fluo-N2DL-HeLa': 25,
                   'PhC-C2DH-U373': 500,
                   'PhC-C2DL-PSC': 20}
    sigma = {'DIC-C2DH-HeLa': 2,
             'Fluo-C2DL-MSC': 2,
             'Fluo-N2DH-GOWT1': 1,
             'Fluo-N2DH-SIM+': 2,
             'Fluo-N2DL-HeLa': 1,
             'PhC-C2DH-U373': 1,
             'PhC-C2DL-PSC': 4}
    border = {'DIC-C2DH-HeLa': (25, 25),
              'Fluo-C2DL-MSC': (10, 16),
              'Fluo-N2DH-GOWT1': (12, 12),
              'Fluo-N2DH-SIM+': (0, 0),
              'Fluo-N2DL-HeLa': (12, 18),
              'PhC-C2DH-U373': (18, 25),
              'PhC-C2DL-PSC': (9, 12)}
    parent_dilation = {'DIC-C2DH-HeLa': 0,
                       'Fluo-C2DL-MSC': 0,
                       'Fluo-N2DH-GOWT1': 5,
                       'Fluo-N2DH-SIM+': 5,
                       'Fluo-N2DL-HeLa': 3,
                       'PhC-C2DH-U373': 0,
                       'PhC-C2DL-PSC': 0}
    parent_frame_search = {'DIC-C2DH-HeLa': 2,
                           'Fluo-C2DL-MSC': 2,
                           'Fluo-N2DH-GOWT1': 4,
                           'Fluo-N2DH-SIM+': 2,
                           'Fluo-N2DL-HeLa': 2,
                           'PhC-C2DH-U373': 2,
                           'PhC-C2DL-PSC': 2}
    for i in [4]:
        for seq in sequences:
            dataset_name = datasets[i]
            base_folder = '/run/media/chris/media1/datasets/celltrackingchallenge/trainingdataset/' + datasets[i] + '/' + seq + '/'
            #base_folder = '/run/media/chris/media1/datasets/celltrackingchallenge/challengedataset/' + datasets[i] + '/' + seq + '/'
            #output_folder = '/run/media/chris/media1/experiments/cell_tracking_cluster_track_test/' + datasets[i] + '/' + seq + '_RES_val_250'
            output_folder = '/run/media/chris/media1/experiments/cell_tracking_cluster_grid/' + datasets[i] + '/' + seq + '_RES_val'
            #output_folder = '../embeddings/' + datasets[i] + '/' + seq + '_test'
            #model_file_name = '/run/media/chris/media1/experiments/cell_tracking_output/' + datasets[i] + '/double_u_l7_64_64_s' + str(s) + '_f' + str(f) + '_e' + str(e) + datasets[i] + '/weights/model-20000'
            #model_file_name = '/run/media/chris/media1/experiments/cell_tracking_output/' + datasets[i] + '/double_u_l7_64_64_s' + str(s) + '_f' + str(f) + '_e' + str(e) + datasets[i] + '/weights/model-40000'
            model_file_name = '../weights/' + dataset_name
            loop = MainLoop(datasets[i],
                            base_folder,
                            output_folder,
                            model_file_name,
                            e,
                            [256, 256],
                            c[dataset_name],
                            min_samples[dataset_name],
                            sigma[dataset_name],
                            border[dataset_name],
                            parent_dilation[dataset_name],
                            parent_frame_search[dataset_name])
            loop.run_test()
            del loop

# import cProfile
#
# dataset_train, dataset_test = dataset()
#
# def run():
#     timer = Timer('preprocess')
#     timer.start()
#     for i in range(100):
#         data_entry = dataset_train.get_next()
#     timer.stop()
#
# if __name__ == '__main__':
#     cProfile.run('run()', sort='cumtime')
