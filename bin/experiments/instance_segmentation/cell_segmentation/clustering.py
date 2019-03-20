
import numpy as np
import hdbscan
import utils.np_image
from utils.timer import Timer
import utils.sitk_image
import utils.sitk_np


class InstanceImageCreator(object):
    def __init__(self,
                 coord_factors=0.0,
                 min_label_size=100,
                 image_ignore_border=None,
                 hdbscan=True):
        self.coord_factors = coord_factors
        self.min_label_size = min_label_size
        self.label_image = None
        self.background_embedding = None
        self.current_max_label = 0
        self.max_background_cosine_distance = 0.5
        self.image_ignore_border = image_ignore_border or [0, 0]
        self.hdbscan = hdbscan
        self.use_mask = True

    def get_codes(self, embeddings):
        embedding_size = embeddings.shape[0]
        if self.coord_factors > 0:
            y, x = np.meshgrid(range(embeddings.shape[1]), range(embeddings.shape[2]), indexing='ij')
            y = y.astype(np.float32)
            x = x.astype(np.float32)
            y = y * self.coord_factors
            x = x * self.coord_factors
            coordinates = np.stack([y, x], axis=0)
            #coordinates = np.stack([coordinates] * embeddings.shape[1], axis=1)
            embeddings = np.concatenate([embeddings, coordinates], axis=0)
            embedding_size += 2
        codes = np.reshape(np.transpose(embeddings, [1, 2, 0]), [-1, embedding_size])
        return codes

    def get_codes_with_mask(self, embeddings):
        reshaped_embedding = np.transpose(embeddings, [1, 2, 0])
        mask = np.square(np.dot(reshaped_embedding, self.background_embedding)) < self.max_background_cosine_distance
        embedding_size = embeddings.shape[0]
        if self.coord_factors > 0:
            y, x = np.meshgrid(range(embeddings.shape[1]), range(embeddings.shape[2]), indexing='ij')
            y = y.astype(np.float32)
            x = x.astype(np.float32)
            y = y * self.coord_factors
            x = x * self.coord_factors
            coordinates = np.stack([y, x], axis=0)
            #coordinates = np.stack([coordinates] * embeddings.shape[1], axis=1)
            embeddings = np.concatenate([embeddings, coordinates], axis=0)
            embedding_size += 2
        #background_embedding = np.median(embeddings)
        mask_b = np.broadcast_to(mask, embeddings.shape)
        codes = np.reshape(embeddings[mask_b], [embedding_size, -1])
        codes = np.transpose(codes)
        return codes, mask

    def cluster_codes(self, codes, max_distance):
        labels_list = []
        labels = []
        for i in range(codes.shape[0]):
            min_distance = 1e10
            best_index = -1
            current_embedding = codes[i]
            for j, (embedding, list) in enumerate(labels_list):
                distance = np.dot(embedding, current_embedding)
                if distance < min_distance:
                    min_distance = distance
                    best_index = j
            if min_distance > max_distance:
                new_label = len(labels_list)
                labels_list.append((current_embedding, [i]))
                labels.append(new_label)
            else:
                labels_list[best_index][1].append(i)
                labels.append(best_index)
        return np.array(labels)

    def get_cluster_labels(self, codes, shape, mask):
        cluster = hdbscan.HDBSCAN(min_cluster_size=self.min_label_size,
                                  min_samples=self.min_label_size,
                                  metric='l2',
                                  core_dist_n_jobs=8,
                                  algorithm='boruvka_kdtree',
                                  leaf_size=25)
        cluster.fit(codes)
        labels = cluster.labels_
        labels_unique = np.unique(labels[labels >= 0])
        if self.use_mask:
            output = np.zeros(shape, np.int32)
            output[mask] = labels + 1
        else:
            output = np.reshape(labels, shape)
        current_label_list = utils.np_image.split_label_image(output, range(1, len(labels_unique) + 1), dtype=np.uint8)

        final_label_list = []
        for current_label in current_label_list:
            cc, num_labels = utils.np_image.connected_component(current_label, dtype=np.uint8)
            if num_labels == 1:
                final_label_list.append(current_label)
            else:
                for label in range(1, num_labels + 1):
                    final_label_list.append((cc == label).astype(np.uint8))
        return final_label_list

    def create_instance_image(self, embeddings_normalized_slice):
        with Timer('add slice'):
            self.calculate_background_embedding_as_largest_median(embeddings_normalized_slice)
            if self.use_mask:
                codes, mask = self.get_codes_with_mask(embeddings_normalized_slice)
            else:
                codes = self.get_codes(embeddings_normalized_slice)
                mask = None

            current_label_list = self.get_cluster_labels(codes, embeddings_normalized_slice.shape[1:3], mask)

            current_label_list = self.relabel_nonoverlapping_labels(current_label_list, min_label_size=self.min_label_size)
            label_image = utils.np_image.merge_label_images(current_label_list)
            self.label_image = label_image
            self.remove_segmentations_outside_border()

    def calculate_background_embedding_as_constant(self, embeddings):
        if self.background_embedding is not None:
            return
        self.background_embedding = np.concatenate([np.ones((1), dtype=embeddings.dtype), np.zeros((embeddings.shape[0] - 1), dtype=embeddings.dtype)], axis=0)

    def calculate_background_embedding_as_largest_median(self, embeddings):
        if self.background_embedding is not None:
            return
        self.background_embedding = np.median(embeddings, axis=(1, 2), keepdims=False)

    def remove_background_label(self, label_list, mean_embeddings):
        new_label_list = [label for label_index, label in enumerate(label_list)
                          if np.dot(mean_embeddings[label_index], self.background_embedding) < self.max_background_cosine_distance]
        return new_label_list

    def relabel_nonoverlapping_labels(self, label_list, min_label_size):
        new_label_list = []
        for l, current_label in enumerate(label_list):
            cc = utils.np_image.split_connected_components(*utils.np_image.connected_component(current_label))
            if len(cc) > 1:
                print('cc of temp label', l, len(cc))
            for comp in cc:
                comp_label_size = np.sum(comp)
                if comp_label_size < min_label_size:
                    print('removed label with size', comp_label_size)
                    continue
                new_label_list.append(comp)
        return new_label_list

    def get_next_free_label(self):
        self.current_max_label += 1
        return self.current_max_label

    def remove_segmentations_outside_border(self):
        if self.image_ignore_border[0] == 0 and self.image_ignore_border[1] == 0:
            return
        border_image = np.zeros(self.label_image.shape, np.bool)
        border_image[self.image_ignore_border[1]:-self.image_ignore_border[1], self.image_ignore_border[0]:-self.image_ignore_border[0]] = True

        label_images, labels = utils.np_image.split_label_image_with_unknown_labels(self.label_image)
        for label_image, label in zip(label_images, labels):
            if not np.any(np.bitwise_and(border_image, label_image)):
                print('removing border track', label)
                self.label_image[self.label_image == label] = 0
