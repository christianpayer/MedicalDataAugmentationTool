
import SimpleITK as sitk
import numpy as np
import hdbscan
import utils.np_image
from utils.timer import Timer
import utils.sitk_image
import utils.sitk_np


def calculate_label_overlap(labels, other_labels, ignore_zero_label=False):
    max_previous_label = np.max(other_labels) + 1
    max_label = np.max(labels) + 1
    distance_overlap_matrix = np.zeros([max_label, max_previous_label])
    if ignore_zero_label:
        start = 1
    else:
        start = 0
    for previous_label in range(start, max_previous_label):
        previous_label_indizes = other_labels == previous_label
        if not np.any(previous_label_indizes):
            continue
        for label in range(start, max_label):
            #print(label, max_label, max_previous_label)
            current_label_indizes = labels == label
            # intersection over union
            iou = np.count_nonzero(np.bitwise_and(current_label_indizes, previous_label_indizes)) / (.1e-8 + np.count_nonzero(np.bitwise_or(current_label_indizes, previous_label_indizes)))
            distance_overlap_matrix[label, previous_label] = iou
    # print(distance_overlap_matrix)
    return distance_overlap_matrix


class EmbeddingTracker(object):
    def __init__(self,
                 coord_factors=0.0,
                 min_cluster_size=500,
                 min_samples=500,
                 stack_neighboring_slices=2,
                 min_label_size_per_stack=100,
                 save_label_stack=True,
                 image_ignore_border=0,
                 parent_search_dilation_size=0,
                 max_parent_search_frames=1):
        self.coord_factors = coord_factors
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.stack_neighboring_slices = stack_neighboring_slices
        self.min_label_size_per_stack = min_label_size_per_stack
        self.embeddings_slices = []
        self.stacked_label_image = None
        self.background_embedding = None
        self.current_max_label = 0
        self.max_background_cosine_distance = 0.95
        self.max_merge_distance = 0.9
        self.max_parent_search_frames = max_parent_search_frames
        self.min_parent_overlap = 0.01
        self.save_label_stack = save_label_stack
        self.image_ignore_border = image_ignore_border
        self.parent_search_dilation_size = parent_search_dilation_size
        self.label_stack_list = []
        self.use_mask = True

    def calculate_mean_embeddings(self, label_list, embeddings):
        embedding_size = embeddings.shape[0]
        embeddings_flattened = np.reshape(np.transpose(embeddings, [1, 2, 3, 0]), [-1, embedding_size])
        mean_embeddings = [np.mean(embeddings_flattened[np.reshape(label, [-1]) > 0], axis=0) for label in label_list]
        return [mean_embedding / np.linalg.norm(mean_embedding, ord=2) for mean_embedding in mean_embeddings]

    def get_codes(self, embeddings):
        embedding_size = embeddings.shape[0]
        if self.coord_factors > 0:
            y, x = np.meshgrid(range(embeddings.shape[2]), range(embeddings.shape[3]), indexing='ij')
            y = y.astype(np.float32)
            x = x.astype(np.float32)
            y = y * self.coord_factors
            x = x * self.coord_factors
            coordinates = np.stack([y, x], axis=0)
            coordinates = np.stack([coordinates] * embeddings.shape[1], axis=1)
            embeddings = np.concatenate([embeddings, coordinates], axis=0)
            embedding_size += 2
        codes = np.reshape(np.transpose(embeddings, [1, 2, 3, 0]), [-1, embedding_size])
        return codes

    def get_codes_with_mask(self, embeddings):
        reshaped_embedding = np.transpose(embeddings, [1, 2, 3, 0])
        mask = np.dot(reshaped_embedding, self.background_embedding) < self.max_background_cosine_distance
        embedding_size = embeddings.shape[0]
        if self.coord_factors > 0:
            y, x = np.meshgrid(range(embeddings.shape[2]), range(embeddings.shape[3]), indexing='ij')
            y = y.astype(np.float32)
            x = x.astype(np.float32)
            y = y * self.coord_factors
            x = x * self.coord_factors
            coordinates = np.stack([y, x], axis=0)
            coordinates = np.stack([coordinates] * embeddings.shape[1], axis=1)
            embeddings = np.concatenate([embeddings, coordinates], axis=0)
            embedding_size += 2
        #background_embedding = np.median(embeddings)
        mask_b = np.broadcast_to(mask, embeddings.shape)
        codes = np.reshape(embeddings[mask_b], [embedding_size, -1])
        codes = np.transpose(codes)
        return codes, mask

    def add_reset_slice(self, embeddings_normalized_slice):
        self.embeddings_slices.append(embeddings_normalized_slice)

    def add_slice(self, embeddings_normalized_slice):
        with Timer('add slice'):
            self.embeddings_slices.append(embeddings_normalized_slice)

            if len(self.embeddings_slices) < self.stack_neighboring_slices:
                return

            current_embeddings_normalized = np.stack(self.embeddings_slices[-self.stack_neighboring_slices:], axis=1)
            self.calculate_background_embedding_as_largest_median(current_embeddings_normalized)
            #codes = self.get_codes(current_embeddings_normalized)
            if self.use_mask:
                codes, mask = self.get_codes_with_mask(current_embeddings_normalized)
            else:
                codes = self.get_codes(current_embeddings_normalized)
                mask = None
            cluster = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size,
                                      min_samples=self.min_samples,
                                      metric='l2',
                                      core_dist_n_jobs=8,
                                      algorithm='boruvka_kdtree',
                                      leaf_size=25)
            cluster.fit(codes)
            labels = cluster.labels_
            labels_unique = np.unique(labels[labels >= 0])
            if self.use_mask:
                output = np.zeros(current_embeddings_normalized.shape[1:4], np.int32)
                output[mask] = labels + 1
            else:
                output = np.reshape(labels, current_embeddings_normalized.shape[1:4])
            current_label_list = utils.np_image.split_label_image(output, range(1, len(labels_unique) + 1))

            mean_embeddings = self.calculate_mean_embeddings(current_label_list, current_embeddings_normalized)
            current_label_list = self.remove_background_label(current_label_list, mean_embeddings)
            current_label_list = self.relabel_nonoverlapping_labels(current_label_list, min_label_size=self.min_label_size_per_stack)
            label_image = utils.np_image.merge_label_images(current_label_list)
            self.merge_new_label_image(label_image)

    def set_label_stack(self, label_stack):
        for i in range(label_stack.shape[0]):
            label_image = label_stack[i, ...]
            current_label_list, _ = utils.np_image.split_label_image_with_unknown_labels(label_image)
            del current_label_list[0]
            current_label_list = self.relabel_nonoverlapping_labels(current_label_list, min_label_size=self.min_label_size_per_stack)
            label_image = utils.np_image.merge_label_images(current_label_list)
            self.merge_new_label_image(label_image)

    def calculate_background_embedding(self, label_list, mean_embeddings):
        if self.background_embedding is not None:
            return
        background_index, value = max(enumerate(label_list), key=lambda x: np.count_nonzero(x[1]))
        self.background_embedding = mean_embeddings[background_index]

    def calculate_background_embedding_as_largest_median(self, embeddings):
        if self.background_embedding is not None:
            return
        self.background_embedding = np.median(embeddings, axis=(1, 2, 3), keepdims=False)

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

    def merge_new_label_image(self, label_image):
        if self.save_label_stack:
            self.label_stack_list.append(label_image)
        # create initial label image
        if self.stacked_label_image is None:
            self.stacked_label_image = label_image
            self.current_max_label = np.max(self.stacked_label_image)
            return

        previous_label_image = self.stacked_label_image[-self.stack_neighboring_slices:, :, :]
        relabeled_label_image = self.relabel_next_slices(previous_label_image, label_image)
        self.stacked_label_image = np.append(self.stacked_label_image, np.expand_dims(relabeled_label_image, axis=0), axis=0)

    def get_next_free_label(self):
        self.current_max_label += 1
        return self.current_max_label

    def relabel_next_slices(self, current_slice_stack, next_slice_stack, min_overlap=0.25):
        current_slice_stack_for_overlap = current_slice_stack[1:, ...]
        next_slice_stack_for_overlap = next_slice_stack[:-1, ...]
        next_slice = next_slice_stack[-1, ...]
        # print(current_shape.shape)
        # print(next_shape.shape)
        distance_overlap_matrix = calculate_label_overlap(current_slice_stack_for_overlap, next_slice_stack_for_overlap)
        # print(distance_overlap_matrix)
        next_shape_merged = np.zeros_like(next_slice)
        done_labels = [0]
        while distance_overlap_matrix.shape[0] > 0 and distance_overlap_matrix.shape[1] > 0:
            max_index = np.unravel_index(np.argmax(distance_overlap_matrix, axis=None), distance_overlap_matrix.shape)
            label_from, label_to = max_index
            max_value = distance_overlap_matrix[label_from, label_to]
            print(label_from, label_to, max_value)
            if max_value < min_overlap:
                break
            next_shape_merged[next_slice == label_to] = label_from
            done_labels.append(label_to)
            distance_overlap_matrix[:, label_to] = 0
            distance_overlap_matrix[label_from, :] = 0

        #print(done_labels)
        max_new_labels = np.max(next_slice) + 1
        for j in range(max_new_labels):
            if j not in done_labels:
                indizes = next_slice == j
                if np.any(indizes):
                    new_label = self.get_next_free_label()
                    print('new label', new_label)
                    next_shape_merged[indizes] = new_label
        return next_shape_merged

    def remove_segmentations_outside_border(self):
        if self.image_ignore_border[0] == 0 and self.image_ignore_border[1] == 0:
            return
        border_image = np.zeros(self.stacked_label_image.shape[1:], np.bool)
        border_image[self.image_ignore_border[1]:-self.image_ignore_border[1], self.image_ignore_border[0]:-self.image_ignore_border[0]] = True
        for i in range(self.stacked_label_image.shape[0] - 1, -1, -1):
            current_label_image = self.stacked_label_image[i, ...]
            label_images, labels = utils.np_image.split_label_image_with_unknown_labels(current_label_image)
            for label_image, label in zip(label_images, labels):
                if not np.any(np.bitwise_and(border_image, label_image)):
                    print('removing border track', label, 'from frame', i)
                    new_track_id = self.get_next_free_label()
                    print('new id', new_track_id)
                    current_label_image[current_label_image == label] = 0
                    if 0 == self.relabel_track_from_to_frame(label, new_track_id, i + 1, self.stacked_label_image.shape[0] - 1):
                        self.current_max_label -= 1


    def relabel_stacked_label_image(self):
        label_images, labels = utils.np_image.split_label_image_with_unknown_labels(self.stacked_label_image)
        del label_images[0]
        #label_images.sort(key=lambda x: self.get_min_max_frame(x)[0])
        self.stacked_label_image = utils.np_image.merge_label_images(label_images)
        self.current_max_label = len(label_images)

    def finalize(self):
        print('Finalizing')
        self.remove_segmentations_outside_border()
        self.relabel_stacked_label_image()
        self.relabel_track_splits()

    def resample_stacked_label_image(self, input_image, transformation, smoothing_sigma):
        if smoothing_sigma == 1:
            interpolator = 'label_gaussian'
        else:
            interpolator = 'nearest'
        resampled_stacked_predictions_sitk = utils.sitk_image.transform_np_output_to_sitk_input(self.stacked_label_image,
                                                                                                output_spacing=None,
                                                                                                channel_axis=0,
                                                                                                input_image_sitk=input_image,
                                                                                                transform=transformation,
                                                                                                interpolator=interpolator,
                                                                                                output_pixel_type=sitk.sitkUInt16)
        if smoothing_sigma > 1:
            resampled_stacked_predictions_sitk = [utils.sitk_image.apply_np_image_function(im, lambda x: self.label_smooth(x, sigma=smoothing_sigma)) for im in resampled_stacked_predictions_sitk]
        self.stacked_label_image = np.stack([utils.sitk_np.sitk_to_np(sitk_im) for sitk_im in resampled_stacked_predictions_sitk], axis=0)

    def label_smooth(self, im, sigma):
        label_images, labels = utils.np_image.split_label_image_with_unknown_labels(im, dtype=np.float32)
        smoothed_label_images = utils.np_image.smooth_label_images(label_images, sigma=sigma, dtype=im.dtype)
        return utils.np_image.merge_label_images(smoothed_label_images, labels)

    def get_track_tuples(self):
        track_tuples = []
        for i in range(1, self.current_max_label + 1):
            print('track tuples for', i)
            track_tuple = (i,) + self.get_from_to_parent_tuple(i)
            print(track_tuple)
            track_tuples.append(track_tuple)
        return track_tuples

    def get_parent_id(self, track_image, min_track_frame):
        if min_track_frame == 0:
            return 0
        first_track_image = track_image[min_track_frame, ...]
        if self.parent_search_dilation_size > 0:
            first_track_image = utils.np_image.dilation_circle(first_track_image, [self.parent_search_dilation_size, self.parent_search_dilation_size])
        first_track_image = np.stack([first_track_image] * (min_track_frame - max(0, min_track_frame - self.max_parent_search_frames)), axis=0)
        #for frame in range(min_track_frame - 1, max(-1, min_track_frame - self.max_parent_search_frames), -1):
        current_frame = self.stacked_label_image[max(0, min_track_frame - self.max_parent_search_frames):min_track_frame, ...]
        overlaps = calculate_label_overlap(first_track_image, current_frame, ignore_zero_label=True)
        max_overlap_index = np.argmax(overlaps[1])
        max_overlap = overlaps[1, max_overlap_index]
        if max_overlap > self.min_parent_overlap:
            print('parent', max_overlap_index, 'overlap', max_overlap)
            return max_overlap_index
        print('found no parent')
        return 0

    def get_min_max_frame(self, track_image):
        indizes = np.any(track_image, axis=(1, 2))
        track_range = np.where(indizes)
        return track_range[0][[0, -1]]

    def is_consecutive(self, track_image):
        indizes = np.any(track_image, axis=(1, 2))
        track_range = np.where(indizes)
        num_frames = len(track_range[0])
        first = track_range[0][0]
        last = track_range[0][-1]
        return num_frames == (last - first + 1)

    def get_from_to_parent_tuple(self, track_id):
        track_image = self.stacked_label_image == track_id
        min_track_frame, max_track_frame = self.get_min_max_frame(track_image)
        parent_id = self.get_parent_id(track_image, min_track_frame)
        return min_track_frame, max_track_frame, parent_id

    # def merge_invalid_children(self):
    #     for track_id in range(self.current_max_label):
    #         track_image = self.stacked_label_image == track_id
    #         min_frame, max_frame = self.get_min_max_frame(track_image)
    #         if min_frame <= 0 or max_frame - 1 >= self.stacked_label_image.shape[0]:
    #             continue
    #         first_image = track_image[min_frame, ...]
    #         first_image_size = np.count_nonzero(first_image)
    #         first_other_image = self.stacked_label_image[min_frame - 1, ...] * first_image
    #         labels_first = [(label, np.count_nonzero(first_other_image == label)) for label in range(1, np.max(first_other_image))]
    #         labels_first_max = max(labels_first, key=lambda x: x[1])
    #         ratio_first = labels_first_max[1] / first_image_size
    #
    #         last_image = track_image[max_frame, ...]
    #         last_image_size = np.count_nonzero(last_image)
    #         last_other_image = self.stacked_label_image[max_frame + 1, ...] * last_image
    #         labels_last = [(label, np.count_nonzero(last_other_image == label)) for label in range(1, np.max(last_other_image))]
    #         labels_last_max = max(labels_last, key=lambda x: x[1])
    #         ratio_last = labels_last_max[1] / last_image_size
    #
    #         if labels_first_max[0] == labels_last_max[0] and ratio_first > self.max_merge_distance and ratio_last > self.max_merge_distance:
    #             print('merging wrong child')
    #             self.relabel_track_from_to_frame(track_id, labels_first_max[0], min_frame, max_frame)

    def relabel_track_from_to_frame(self, track_id, new_track_id, from_frame, to_frame):
        #for frame_index in range(from_frame, to_frame + 1):
        label_indizes = self.stacked_label_image[from_frame:to_frame + 1] == track_id
        self.stacked_label_image[from_frame:to_frame + 1][label_indizes] = new_track_id
        return np.count_nonzero(label_indizes)

    def split_track_tuples(self, track_tuple_index, split_point):
        track_id, min_frame, max_frame, parent_id = self.track_tuples[track_tuple_index]
        if min_frame > max_frame:
            print('error, this should not happen')
        self.track_tuples[track_tuple_index] = (track_id, min_frame, split_point, parent_id)
        new_track_id = self.get_next_free_label()
        new_frame_min = split_point + 1
        self.relabel_track_from_to_frame(track_id, new_track_id, new_frame_min, max_frame)
        print('split track', track_id, min_frame, max_frame, parent_id, 'into', track_id, min_frame, split_point, parent_id, 'and', new_track_id, new_frame_min, max_frame, track_id)
        for i, (other_track_id, other_min_frame, other_max_frame, other_parent_id) in enumerate(self.track_tuples):
            if other_min_frame > new_frame_min:
                break
        self.track_tuples.insert(i, (new_track_id, new_frame_min, max_frame, track_id))

        # relabel parents
        for i, (other_track_id, other_min_frame, other_max_frame, other_parent_id) in enumerate(self.track_tuples):
            if other_parent_id == track_id and other_track_id != new_track_id and other_min_frame > split_point + 1:
                print('relabel track', other_track_id, other_min_frame, other_max_frame, other_parent_id, 'into', other_track_id, other_min_frame, other_max_frame, new_track_id)
                self.track_tuples[i] = (other_track_id, other_min_frame, other_max_frame, new_track_id)



    def relabel_track_splits(self):
        # tuple (split_frame_index, parent, child_0, child_1)
        self.track_tuples = self.get_track_tuples()

        # check for invalid splits -> parent used after a split
        changes = True
        while changes:
            changes = False
            for track_id, min_frame, max_frame, parent_id in self.track_tuples:
                if parent_id == 0:
                    continue
                for other_i, (other_track_id, other_min_frame, other_max_frame, other_parent_id) in enumerate(self.track_tuples):
                    if parent_id == other_track_id:
                        if other_max_frame >= min_frame:
                            print('invalid track', track_id, 'with parent', parent_id)
                            print(track_id, min_frame, max_frame, parent_id)
                            print(other_track_id, other_min_frame, other_max_frame, other_parent_id)
                            self.split_track_tuples(other_i, min_frame - 1)
                            changes = True
                            break
                if changes:
                    break
                        # new_track_id = self.get_next_free_label()
                        # print('old track', track_id, min_frame, max_frame, new_track_id)
                        # track_tuples[i] = (track_id, min_frame, max_frame, new_track_id)
                        # print('new track', new_track_id, other_min_frame, min_frame - 1, other_parent_id)
                        # track_tuples.append((new_track_id, other_min_frame, min_frame - 1, other_parent_id))
                        # print('split track', track_id, min_frame, max_frame, new_track_id)
                        # track_tuples[other_i] = (other_track_id, min_frame, other_max_frame, new_track_id)
                        # self.relabel_track_from_to_frame(other_track_id, new_track_id, other_min_frame, min_frame - 1)

    def fix_tracks_after_resampling(self):
        fixed_tracks = []
        removed_tracks = []
        for track_id, min_frame, max_frame, parent_id in self.track_tuples:
            track_image = self.stacked_label_image == track_id
            if np.count_nonzero(track_image) == 0:
                print('remove track with no pixels', track_id)
                removed_tracks.append(track_id)
                continue
            if not self.is_consecutive(track_image):
                print('removed non-consecutive track')
                self.relabel_track_from_to_frame(track_id, 0, min_frame, max_frame)
                removed_tracks.append(track_id)
            new_min_frame, new_max_frame = self.get_min_max_frame(track_image)
            if new_min_frame != min_frame or new_max_frame != max_frame:
                print('change track min max', track_id, 'min', min_frame, new_min_frame, 'max', max_frame, new_max_frame)
                fixed_tracks.append((track_id, new_min_frame, new_max_frame, parent_id))
                continue
            fixed_tracks.append((track_id, min_frame, max_frame, parent_id))

        for i, (track_id, min_frame, max_frame, parent_id) in enumerate(fixed_tracks):
            if parent_id in removed_tracks:
                fixed_tracks[i] = (track_id, min_frame, max_frame, 0)
        self.track_tuples = fixed_tracks
        #self.relabel_tracks_ascending()

    # def relabel_tracks_ascending(self):
    #     fixed_tracks = []
    #     track_mapping = {0: 0}
    #     next_free_track_id = 1
    #     for track_id, min_frame, max_frame, parent_id in self.track_tuples:
    #         new_track_id = next_free_track_id
    #         new_parent_id = track_mapping[parent_id]
    #         self.relabel_track_from_to_frame(track_id, new_track_id, min_frame, max_frame)
    #         fixed_tracks.append((new_track_id, min_frame, max_frame, new_parent_id))
    #         track_mapping[track_id] = new_track_id
    #         next_free_track_id += 1
    #     self.track_tuples = fixed_tracks
