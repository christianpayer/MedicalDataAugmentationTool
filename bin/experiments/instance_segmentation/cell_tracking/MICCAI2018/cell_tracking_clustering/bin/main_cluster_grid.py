#!/usr/bin/python
import sys
import math

sys.path.append('/home/chris/work/CellTracking/')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import SimpleITK as sitk

import utils.io.image
from bin.embedding_tracker import EmbeddingTracker
import utils.io.text
import utils.sitk_image
import utils.np_image


def label_smooth(im, sigma):
    label_images, labels = utils.np_image.split_label_image_with_unknown_labels(im, dtype=np.float32)
    smoothed_label_images = utils.np_image.smooth_label_images(label_images, sigma=sigma, dtype=im.dtype)
    return utils.np_image.merge_label_images(smoothed_label_images, labels)

def grid_search(embeddings, transformation, input_image, coord_factors_list, min_cluster_list, min_samples_list, output_base_folder, data_format, sigma, border_size, parent_dilation):
    image_prefix = 'mask'
    track_file_name = 'res_track.txt'
    for coord_factors in coord_factors_list:
        for min_cluster in min_cluster_list:
            for min_samples in min_samples_list:
                folder = 'c' + str(coord_factors) + 'mc' + str(min_cluster) + 'ms' + str(min_samples)
                print(folder)
                tracker = EmbeddingTracker(coord_factors=coord_factors,
                                           stack_neighboring_slices=2,
                                           min_cluster_size=50,
                                           min_samples=50,
                                           min_label_size_per_stack=10,
                                           save_label_stack=True,
                                           image_ignore_border=border_size,
                                           parent_search_dilation_size=parent_dilation)
                output_folder = os.path.join(output_base_folder, folder)

                for i in range(embeddings.shape[1]):
                    tracker.add_slice(embeddings[:, i, :, :])

                if sigma == 1:
                    interpolator = 'label_gaussian'
                else:
                    interpolator = 'nearest'

                tracker.finalize()
                track_tuples = tracker.track_tuples
                merged = tracker.stacked_label_image

                final_predictions = utils.sitk_image.transform_np_output_to_sitk_input(merged,
                                                                                       output_spacing=None,
                                                                                       channel_axis=0,
                                                                                       input_image_sitk=input_image,
                                                                                       transform=transformation,
                                                                                       interpolator=interpolator,
                                                                                       output_pixel_type=sitk.sitkUInt16)
                # final_predictions = [utils.sitk_np.np_to_sitk(np.squeeze(im), type=np.uint16) for im in np.split(merged, merged.shape[0], axis=0)]
                # final_predictions_smoothed_2 = [utils.sitk_image.apply_np_image_function(im, lambda x: self.label_smooth(x, sigma=2)) for im in final_predictions]
                if sigma > 1:
                    final_predictions = [utils.sitk_image.apply_np_image_function(im, lambda x: label_smooth(x, sigma=sigma)) for im in final_predictions]

                for i, final_prediction in enumerate(final_predictions):
                    utils.io.image.write(final_prediction, os.path.join(output_folder, image_prefix + 't' + str(i).zfill(3) + '.tif'))

                utils.io.image.write_np(np.stack(tracker.label_stack_list, axis=1), os.path.join(output_folder, 'label_stack.mha'))

                final_predictions_stacked = utils.sitk_image.accumulate(final_predictions)
                utils.io.image.write(final_predictions_stacked, os.path.join(output_folder, 'stacked.mha'))
                # utils.io.image.write(utils.sitk_image.accumulate(final_predictions_smoothed_2), os.path.join(self.output_folder, 'stacked_2.mha'))
                # utils.io.image.write(utils.sitk_image.accumulate(final_predictions_smoothed_4), os.path.join(self.output_folder, 'stacked_4.mha'))

                print(track_tuples)
                utils.io.text.save_list_csv(track_tuples, os.path.join(output_folder, track_file_name), delimiter=' ')


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
    sequences = ['01', '02']
    cluster_size_ranges = {'DIC-C2DH-HeLa': [100, 500, 1000],
                           'Fluo-C2DL-MSC': [100, 500, 1000],
                           'Fluo-N2DH-GOWT1': [10, 20, 50, 100, 200],
                           'Fluo-N2DH-SIM+': [10, 20, 50, 100, 200],
                           'Fluo-N2DL-HeLa': [10, 20, 50, 100, 200],
                           'PhC-C2DH-U373': [100, 500, 1000],
                           'PhC-C2DL-PSC': [100, 500, 1000]}

    cluster_size_ranges = {'DIC-C2DH-HeLa': [100, 500, 1000],
                           'Fluo-C2DL-MSC': [100, 500, 1000],
                           'Fluo-N2DH-GOWT1': [10, 20, 50, 100, 200],
                           'Fluo-N2DH-SIM+': [10, 20, 50, 100, 200],
                           'Fluo-N2DL-HeLa': [10, 20, 50, 100, 200],
                           'PhC-C2DH-U373': [100, 500, 1000],
                           'PhC-C2DL-PSC': [100, 500, 1000]}
    c = {'DIC-C2DH-HeLa': 0.02,
         'Fluo-C2DL-MSC': 0.01,
         'Fluo-N2DH-GOWT1': 0.001,
         'Fluo-N2DH-SIM+': 0.005,
         'Fluo-N2DL-HeLa': 0.005,
         'PhC-C2DH-U373': 0.001,
         'PhC-C2DL-PSC': 0.001}
    min_samples = {'DIC-C2DH-HeLa': 500,
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

    for i in [0, 1, 2, 3, 4, 5]:
        coord_factors_list = [0.01, 0.001]
        min_cluster_list = cluster_size_ranges[datasets[i]]
        min_samples_list = cluster_size_ranges[datasets[i]]
        data_format = 'channels_first'
        for seq in sequences:
            net_output_folder = '/run/media/chris/media1/experiments/cell_tracking_cluster_grid/' + datasets[i] + '/' + seq + '_RES_test'
            embeddings_filename = os.path.join(net_output_folder, 'embeddings_2.mha')
            input_image_filename = '/run/media/chris/media1/datasets/celltrackingchallenge/trainingdataset/' + datasets[i] + '/' + seq + '/t000.tif'
            transformation_filename = os.path.join(net_output_folder, 'transform.txt')
            output_base_folder = '/run/media/chris/media1/experiments/cell_tracking_cluster_grid_output/' + datasets[i] + '/' + seq + '/'
            embeddings = utils.io.image.read(embeddings_filename, sitk.sitkVectorFloat32)
            embeddings = sitk.GetArrayFromImage(embeddings)
            embeddings = np.transpose(embeddings, [3, 0, 1, 2])
            embeddings = embeddings.astype(np.float64)
            embeddings = embeddings[:, :10, :, :]
            input_image = utils.io.image.read(input_image_filename, sitk.sitkVectorFloat32)
            input_image.SetSpacing([1] * input_image.GetDimension())
            input_image.SetOrigin([0] * input_image.GetDimension())
            input_image.SetDirection(np.eye(input_image.GetDimension()).flatten())
            transformation = sitk.ReadTransform(transformation_filename)
            grid_search(embeddings, transformation, input_image, coord_factors_list, min_cluster_list, min_samples_list, output_base_folder, data_format,
                        sigma[datasets[i]], border[datasets[i]], parent_dilation[datasets[i]])
