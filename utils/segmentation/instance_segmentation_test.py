
import SimpleITK as sitk
import numpy as np
import utils.geometry
import utils.sitk_image
import utils.sitk_np
import utils.np_image
import utils.landmark.transform
import utils.segmentation.metrics
import utils.io.image
import utils.io.text
import utils.io.common
from collections import OrderedDict
import os
import csv
import copy
import sklearn.cluster
import sklearn.mixture
#from utils.segmentation.pyxmeans.xmeans import XMeans


class InstanceSegmentationTest(object):
    def __init__(self,
                 output_folder,
                 output_extension='.mha',
                 metrics=None,
                 interpolator='linear',
                 save_overlap_image=False,
                 largest_connected_component=False):
        self.output_folder = output_folder
        self.output_extension = output_extension
        self.metrics = metrics
        self.interpolator = interpolator
        self.save_overlap_image = save_overlap_image
        self.largest_connected_component = largest_connected_component
        self.metric_values = {}

    def get_instances(self, class_labels, embeddings):
        classes = utils.np_image.argmax(class_labels)
        labels, num_labels = utils.np_image.connected_component(classes)
        #labels_split = utils.np_image.split_label_image(labels, range(1, num_labels + 1))
        hist_range = (-100, 100)
        hist_bin_size = 0.5
        hist_num_bins = int((hist_range[1] - hist_range[0]) / hist_bin_size)
        current_labels = []
        for label in range(1, num_labels + 1):
            current_label = np.expand_dims(labels, 0) == label
            embeddings_inside_label = embeddings[current_label]
            h, edges = np.histogram(embeddings_inside_label, bins=hist_num_bins, range=hist_range)
            nms = np.bitwise_and(h > np.pad(h[1:], (0, 1), 'constant'), h > np.pad(h[:-1], (1, 0), 'constant'))
            num_labels = np.sum(nms)
            if num_labels > 1:
                instance_labels = [(edges[i] + edges[i + 1]) / 2 for i in np.flatnonzero(nms)]
                embedding_distances = [np.where(current_label, np.abs(embeddings - i), 0) for i in instance_labels]
                new_labels = np.argmax(embedding_distances, axis=0)
                for i in range(len(instance_labels)):
                    new_label_image = np.bitwise_and(new_labels == i, current_label).astype(np.int16)
                    current_labels.append(new_label_image)
            else:
                current_labels.append(current_label.astype(np.int16))
        instances = utils.np_image.merge_label_images(current_labels)
        return instances


    def get_instances_cosine_kmeans_with_coordinates_2d(self, embeddings_normalized):
        y, x = np.meshgrid(range(embeddings_normalized.shape[1]), range(embeddings_normalized.shape[2]), indexing='ij')
        y = y.astype(np.float32)
        x = x.astype(np.float32)
        y = y / 256
        x = x / 256
        coordinates = np.stack([y, x], axis=0)
        embeddings_normalized_with_meshgrid = np.concatenate([coordinates, embeddings_normalized], axis=0)
        codes = np.transpose(embeddings_normalized_with_meshgrid, [1, 2, 0])
        codes = np.reshape(codes, [-1, codes.shape[2]])
        #bandwidth = sklearn.cluster.estimate_bandwidth(codes, quantile=0.1, n_samples=10000)
        bandwidth = 0.5
        #bandwidth=0.7
        print(bandwidth)

        ms = sklearn.cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(codes)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        output = np.reshape(labels, embeddings_normalized_with_meshgrid.shape[1:3])
        print("number of estimated clusters : %d" % n_clusters_)

        return output

    def get_instances_cosine_kmeans_2d(self, embeddings_normalized, coord_factors=0, bandwidth=0.5, min_label_size=100):
        if coord_factors > 0:
            y, x = np.meshgrid(range(embeddings_normalized.shape[1]), range(embeddings_normalized.shape[2]), indexing='ij')
            y = y.astype(np.float32)
            x = x.astype(np.float32)
            y = y * coord_factors
            x = x * coord_factors
            coordinates = np.stack([y, x], axis=0)
            embeddings_normalized = np.concatenate([coordinates, embeddings_normalized], axis=0)
        codes = np.transpose(embeddings_normalized, [1, 2, 0])
        codes = np.reshape(codes, [-1, codes.shape[2]])
        #bandwidth = sklearn.cluster.estimate_bandwidth(codes, quantile=0.1, n_samples=10000)
        #bandwidth = 0.6
        #bandwidth=0.7
        print(bandwidth)

        ms = sklearn.cluster.MeanShift(bandwidth=bandwidth, bin_seeding=False, cluster_all=False, n_jobs=-2)
        ms.fit(codes)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        output = np.reshape(labels, embeddings_normalized.shape[1:3])
        for label in range(n_clusters_):
            label_sum = np.sum(output==label)
            if label_sum < min_label_size:
                output[output == label] = -1
        print("number of estimated clusters : %d" % n_clusters_)

        return output

    def get_instances_cosine_kmeans_3d(self, embeddings_normalized, coord_factors=0, bandwidth=0.5, min_label_size=100):
        # if coord_factors > 0:
        #     z, y, x = np.meshgrid(range(embeddings_normalized.shape[1]), range(embeddings_normalized.shape[2]), indexing='ij')
        #     y = y.astype(np.float32)
        #     x = x.astype(np.float32)
        #     y = y * coord_factors
        #     x = x * coord_factors
        #     coordinates = np.stack([y, x], axis=0)
        #     embeddings_normalized = np.concatenate([coordinates, embeddings_normalized], axis=0)
        codes = np.transpose(embeddings_normalized, [1, 2, 3, 0])
        codes = np.reshape(codes, [-1, codes.shape[3]])
        #bandwidth = sklearn.cluster.estimate_bandwidth(codes, quantile=0.1, n_samples=10000)
        #bandwidth = 0.6
        #bandwidth=0.7
        print(bandwidth)

        seeds = sklearn.cluster.get_bin_seeds(codes, 0.1, 1)
        ms = sklearn.cluster.MeanShift(bandwidth=bandwidth, bin_seeding=False, cluster_all=False, n_jobs=-2, min_bin_freq=100, seeds=seeds)
        ms.fit(codes)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        output = np.reshape(labels, embeddings_normalized.shape[1:4])
        for label in range(n_clusters_):
            label_sum = np.sum(output==label)
            if label_sum < min_label_size:
                output[output == label] = -1
        print("number of estimated clusters : %d" % n_clusters_)

        return output

    def get_instances_cosine_kmeans_slice_by_slice(self, embeddings_normalized, coord_factors=0, bandwidth=0.5, min_label_size=100):
        # if coord_factors > 0:
        #     z, y, x = np.meshgrid(range(embeddings_normalized.shape[1]), range(embeddings_normalized.shape[2]), indexing='ij')
        #     y = y.astype(np.float32)
        #     x = x.astype(np.float32)
        #     y = y * coord_factors
        #     x = x * coord_factors
        #     coordinates = np.stack([y, x], axis=0)
        #     embeddings_normalized = np.concatenate([coordinates, embeddings_normalized], axis=0)
        outputs = np.zeros(embeddings_normalized.shape[1:4])
        for i in range(embeddings_normalized.shape[1]):
            current_embeddings_normalized = embeddings_normalized[:, i, :, :]
            codes = np.transpose(current_embeddings_normalized, [1, 2, 0])
            codes = np.reshape(codes, [-1, codes.shape[2]])
            #bandwidth = sklearn.cluster.estimate_bandwidth(codes, quantile=0.1, n_samples=10000)
            #bandwidth = 0.6
            #bandwidth=0.7
            print(bandwidth)

            seeds = sklearn.cluster.get_bin_seeds(codes, 0.1, 1)
            ms = sklearn.cluster.MeanShift(bandwidth=bandwidth, bin_seeding=False, cluster_all=False, n_jobs=-2, seeds=seeds)
            ms.fit(codes)
            labels = ms.labels_
            cluster_centers = ms.cluster_centers_
            labels_unique = np.unique(labels)
            n_clusters_ = len(labels_unique)
            output = np.reshape(labels, current_embeddings_normalized.shape[1:4])
            outputs[i, ...] = output

        for label in range(n_clusters_):
            label_sum = np.sum(output==label)
            if label_sum < min_label_size:
                output[output == label] = -1
        print("number of estimated clusters : %d" % n_clusters_)

        return outputs

    def get_instances_cosine_gmm_3d(self, embeddings_normalized, coord_factors=0, bandwidth=0.5, min_label_size=0):
        if coord_factors > 0:
            z, y, x = np.meshgrid(range(embeddings_normalized.shape[1]), range(embeddings_normalized.shape[2]), range(embeddings_normalized.shape[3]), indexing='ij')
            y = y.astype(np.float32)
            x = x.astype(np.float32)
            y = y * coord_factors
            x = x * coord_factors
            coordinates = np.stack([y, x], axis=0)
            embeddings_normalized = np.concatenate([coordinates, embeddings_normalized], axis=0)
        codes = np.transpose(embeddings_normalized, [1, 2, 3, 0])
        codes = np.reshape(codes, [-1, codes.shape[3]])

        # for n_components in range(15, 25):
        #     ms = sklearn.mixture.GaussianMixture(n_components=n_components, covariance_type='tied')
        #     ms.fit(codes)
        #     print(ms.bic(codes))
        #     labels = ms.predict(codes)
        #     labels_unique = np.unique(labels)
        #     n_clusters_ = len(labels_unique)
        #     output = np.reshape(labels, embeddings_normalized.shape[1:4])
        #     for label in range(n_clusters_):
        #         label_sum = np.sum(output==label)
        #         if label_sum < min_label_size:
        #             output[output == label] = -1
        #     print("number of estimated clusters : %d" % n_clusters_)
        #ms = sklearn.mixture.GaussianMixture(n_components=11, covariance_type='full')
        #ms = sklearn.cluster.AgglomerativeClustering(n_clusters=11)
        ms = sklearn.cluster.DBSCAN(eps=0.2, leaf_size=5)
    #     ms = sklearn.mixture.BayesianGaussianMixture(
    # n_components=100, covariance_type='diag',
    # init_params="random", max_iter=100, random_state=2)
        labels = ms.fit_predict(codes)
        #print(ms.bic(codes))
        #labels = ms.predict(codes)
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        output = np.reshape(labels, embeddings_normalized.shape[1:4])
        for label in range(n_clusters_):
            label_sum = np.sum(output == label)
            if label_sum < min_label_size:
                output[output == label] = -1
        print("number of estimated clusters : %d" % n_clusters_)

        return output

    def get_instances_cosine_xmeans(self, embeddings_normalized, coord_factors=0, bandwidth=0.5, min_label_size=0):
        if coord_factors > 0:
            y, x = np.meshgrid(range(embeddings_normalized.shape[1]), range(embeddings_normalized.shape[2]), indexing='ij')
            y = y.astype(np.float32)
            x = x.astype(np.float32)
            y = y * coord_factors
            x = x * coord_factors
            coordinates = np.stack([y, x], axis=0)
            embeddings_normalized = np.concatenate([coordinates, embeddings_normalized], axis=0)
        codes = np.transpose(embeddings_normalized, [1, 2, 0])
        codes = np.reshape(codes, [-1, codes.shape[2]])

        ms = XMeans()
        labels = ms.fit_predict(codes)
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        output = np.reshape(labels, embeddings_normalized.shape[1:3])
        for label in range(n_clusters_):
            label_sum = np.sum(output == label)
            if label_sum < min_label_size:
                output[output == label] = -1
        print("number of estimated clusters : %d" % n_clusters_)

        return output

    def get_instances_cosine_gmm(self, embeddings_normalized, coord_factors=0, bandwidth=0.5, min_label_size=100):
        # if coord_factors > 0:
        #     z, y, x = np.meshgrid(range(embeddings_normalized.shape[1]), range(embeddings_normalized.shape[2]), indexing='ij')
        #     y = y.astype(np.float32)
        #     x = x.astype(np.float32)
        #     y = y * coord_factors
        #     x = x * coord_factors
        #     coordinates = np.stack([y, x], axis=0)
        #     embeddings_normalized = np.concatenate([coordinates, embeddings_normalized], axis=0)
        codes = np.transpose(embeddings_normalized, [1, 2, 3, 0])
        codes = np.reshape(codes, [-1, codes.shape[3]])
        #bandwidth = sklearn.cluster.estimate_bandwidth(codes, quantile=0.1, n_samples=10000)
        #bandwidth = 0.6
        #bandwidth=0.7
        print(bandwidth)

        ms = sklearn.cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=False, n_jobs=-2)
        ms.fit(codes)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        output = np.reshape(labels, embeddings_normalized.shape[1:4])
        for label in range(n_clusters_):
            label_sum = np.sum(output==label)
            if label_sum < min_label_size:
                output[output==label] = -1
        print("number of estimated clusters : %d" % n_clusters_)

        return output


    # def get_instances_cosine_kmeans_3d(self, embeddings_normalized):
    #     z, y, x = np.meshgrid(range(embeddings_normalized.shape[1]), range(embeddings_normalized.shape[2]), range(embeddings_normalized.shape[3]), indexing='ij')
    #     z = z.astype(np.float32)
    #     y = y.astype(np.float32)
    #     x = x.astype(np.float32)
    #     z = z * 0
    #     y = y * 0.01
    #     x = x * 0.01
    #     coordinates = np.stack([z, y, x], axis=0)
    #     embeddings_normalized_with_meshgrid = np.concatenate([coordinates, embeddings_normalized], axis=0)
    #     codes = np.transpose(embeddings_normalized_with_meshgrid, [1, 2, 3, 0])
    #     codes = np.reshape(codes, [-1, codes.shape[3]])
    #     #bandwidth = sklearn.cluster.estimate_bandwidth(codes, quantile=0.2, n_samples=10000)
    #     bandwidth=0.7
    #     print(bandwidth)
    #
    #     ms = sklearn.cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    #     ms.fit(codes)
    #     labels = ms.labels_
    #     cluster_centers = ms.cluster_centers_
    #     labels_unique = np.unique(labels)
    #     n_clusters_ = len(labels_unique)
    #     output = np.reshape(labels, embeddings_normalized_with_meshgrid.shape[1:4])
    #     print("number of estimated clusters : %d" % n_clusters_)
    #
    #     return output

    def get_instances_cosine_pca(self, embeddings_normalized):
        original_shape = embeddings_normalized.shape
        transposed_shape = [embeddings_normalized.shape[1], embeddings_normalized.shape[2], embeddings_normalized.shape[0]]
        codes = np.transpose(embeddings_normalized, [1, 2, 0])
        codes = np.reshape(codes, [-1, codes.shape[2]])
        pca = sklearn.decomposition.PCA(n_components=codes.shape[1])
        output = pca.fit_transform(codes)
        #print(pca.explained_variance_ratio_)
        #print(pca.singular_values_)
        output = np.reshape(output, transposed_shape)
        output = np.transpose(output, [2, 0, 1])

        return output

    def get_instances_cosine_ica(self, embeddings_normalized):
        original_shape = embeddings_normalized.shape
        transposed_shape = [embeddings_normalized.shape[1], embeddings_normalized.shape[2], embeddings_normalized.shape[0]]
        codes = np.transpose(embeddings_normalized, [1, 2, 0])
        codes = np.reshape(codes, [-1, codes.shape[2]])
        pca = sklearn.decomposition.FastICA()
        output = pca.fit_transform(codes)
        #print(pca.explained_variance_ratio_)
        #print(pca.singular_values_)
        output = np.reshape(output, transposed_shape)
        output = np.transpose(output, [2, 0, 1])

        return output


    def get_instances_cosine_svd(self, embeddings_normalized):
        original_shape = embeddings_normalized.shape
        n_components = 15
        codes = np.transpose(embeddings_normalized, [1, 2, 0])
        codes = np.reshape(codes, [-1, codes.shape[2]])
        svd = sklearn.decomposition.TruncatedSVD(n_components=n_components, n_iter=10, random_state=42)
        output = svd.fit_transform(codes)
        # print(svd.explained_variance_ratio_)
        # print(svd.singular_values_)
        transposed_shape = [embeddings_normalized.shape[1], embeddings_normalized.shape[2], embeddings_normalized.shape[0]]
        transposed_shape[2] = n_components
        output = np.reshape(output, transposed_shape)
        output = np.transpose(output, [2, 0, 1])

        return output





