
import SimpleITK as sitk
import numpy as np
import hdbscan
from utils.mean_shift_cosine import MeanShiftCosine
import utils.np_image
import utils.sitk_image
import utils.sitk_np
import utils.io.image
from random import shuffle


class InstanceMerger(object):
    def __init__(self, ignore_border=None, remove_instances_on_inner_border=True):
        self.current_max_label = 0
        self.remove_instances_on_inner_border = remove_instances_on_inner_border
        self.ignore_border = ignore_border or [0, 0]

    def get_next_free_label(self):
        self.current_max_label += 1
        return self.current_max_label

    def outer_borders(self, old_image, old_slice):
        is_top_outer = old_slice[1].start == 0
        is_bottom_outer = old_slice[1].stop == old_image.shape[1]
        is_left_outer = old_slice[2].start == 0
        is_right_outer = old_slice[2].stop == old_image.shape[2]
        return is_top_outer, is_bottom_outer, is_left_outer, is_right_outer

    def border_mask(self, old_image, old_slice):
        is_top_outer, is_bottom_outer, is_left_outer, is_right_outer = self.outer_borders(old_image, old_slice)
        mask = np.ones([old_slice[1].stop - old_slice[1].start, old_slice[2].stop - old_slice[2].start], dtype=np.bool)
        if not is_top_outer:
            mask[:self.ignore_border[0], :] = False
        if not is_bottom_outer:
            mask[-self.ignore_border[0]:, :] = False
        if not is_left_outer:
            mask[:, self.ignore_border[1]] = False
        if not is_right_outer:
            mask[:, -self.ignore_border[1]:] = False
        return mask

    def merge_as_larger_instances(self, old_image, new_image, old_slice, new_slice):
        new_image_cropped = new_image[new_slice]
        old_image_cropped = old_image[old_slice]
        merged_image = np.copy(old_image)
        labels = np.unique(new_image)
        border_mask = self.border_mask(old_image, old_slice)
        cc_old, _ = utils.np_image.connected_component(old_image > 0, connectivity=1)
        if self.remove_instances_on_inner_border:
            for label in labels[1:]:
                current_label_image = new_image == label
                is_top_outer, is_bottom_outer, is_left_outer, is_right_outer = self.outer_borders(old_image, old_slice)
                a = np.where(current_label_image != 0)
                label_top, label_bottom, label_left, label_right = np.min(a[1]), np.max(a[1]), np.min(a[2]), np.max(a[2])
                if not is_top_outer and label_top == 0:
                    new_image[current_label_image] = 0
                    continue
                if not is_bottom_outer and label_bottom == new_image.shape[1] - 1:
                    new_image[current_label_image] = 0
                    continue
                if not is_left_outer and label_left == 0:
                    new_image[current_label_image] = 0
                    continue
                if not is_right_outer and label_right == new_image.shape[2] - 1:
                    new_image[current_label_image] = 0
                    continue
        labels = np.unique(new_image)
        cc_new, _ = utils.np_image.connected_component(new_image > 0, connectivity=1)
        for label in labels[1:]:
            current_label_image = new_image == label
            if np.count_nonzero(np.bitwise_and(border_mask, current_label_image)) == 0:
                continue
            masked_other_labels = merged_image[old_slice][current_label_image]
            other_labels = np.unique(masked_other_labels)
            if len(other_labels) == 1 and other_labels[0] == 0:
                #other_label_image = old_image == other_labels[1]
                next_label = self.get_next_free_label()
                merged_image[old_slice][current_label_image] = next_label
            else:
                cc_new_labels = np.unique(cc_new[current_label_image])
                cc_old_labels = np.unique(cc_old[old_slice][current_label_image])

                # if len(cc_new_label) != 1:
                #     utils.io.image.write_np(cc_new.astype(np.int16), 'cc.mha')
                #     utils.io.image.write_np(current_label_image.astype(np.int16), 'label.mha')
                #     print(cc_new_label)
                #
                # assert len(cc_new_label) == 1, 'Should not happen'
                #
                # cc_new_label_image = cc_new == cc_new_labels[0]
                cc_new_label_image = np.zeros_like(new_image, dtype=np.bool)
                for cc_new_label in cc_new_labels:
                    if cc_new_label == 0:
                        continue
                    cc_new_label_image[cc_new == cc_new_label] = 1
                cc_old_label_image = np.zeros_like(merged_image, dtype=np.bool)
                for cc_old_label in cc_old_labels:
                    if cc_old_label == 0:
                        continue
                    cc_old_label_image[cc_old == cc_old_label] = 1

                cc_new_label_size = np.count_nonzero(cc_new_label_image)
                cc_old_label_size = np.count_nonzero(cc_old_label_image)

                if cc_new_label_size > cc_old_label_size:
                    merged_image[cc_old_label_image] = 0
                    new_labels = np.unique(new_image_cropped[cc_new_label_image])
                    for new_label in new_labels:
                        next_label = self.get_next_free_label()
                        merged_image[old_slice][new_image_cropped == new_label] = next_label

            # else:
            #     current_label_size = np.count_nonzero(current_label_image)
            #     other_label_images = [merged_image == other_label for other_label in other_labels if other_label != 0]
            #     other_label_sizes = [np.count_nonzero(other_label_image) for other_label_image in other_label_images]
            #     other_label_size = np.sum(other_label_sizes)
            #     if current_label_size > other_label_size:
            #         merged_other_label_images = np.zeros_like(merged_image, dtype=np.bool)
            #         for other_label_image in other_label_images:
            #             merged_other_label_images[other_label_image] = 1
            #             merged_image[other_label_image] = 0
            #
            #         masked_new_image = new_image_cropped[merged_other_label_images[old_slice]]
            #         all_new_labels = np.unique(masked_new_image)
            #         for all_new_label in all_new_labels:
            #             if all_new_label == 0:
            #                 continue
            #             next_label = self.get_next_free_label()
            #             merged_image[old_slice][new_image_cropped == all_new_label] = next_label

            # else:
            #     #current_label_size = np.count_nonzero(current_label_image)
            #     #other_label_images = [merged_image == other_label for other_label in other_labels if other_label != 0]
            #     region_of_other_labels = np.zeros_like(old_image, dtype=np.bool)
            #     for other_label in other_labels:
            #         region_of_other_labels = np.bitwise_or(region_of_other_labels, merged_image == other_label)
            #     touching_labels_on_new_image = np.unique(np.bitwise_and(region_of_other_labels[old_slice], new_image_cropped))
            #     touching_labels_on_new_image_labels = np.zeros_like(new_image_cropped, dtype=np.bool)
            #     for other_label in touching_labels_on_new_image:
            #         touching_labels_on_new_image_labels = np.bitwise_or(touching_labels_on_new_image_labels, new_image_cropped == other_label)
            #     #other_label_sizes = [np.count_nonzero(other_label_image) for other_label_image in other_label_images]
            #     #other_label_size = np.sum(other_label_sizes)
            #     other_label_size = np.count_nonzero(region_of_other_labels[old_slice])
            #     new_label_sizes = np.count_nonzero(touching_labels_on_new_image_labels)
            #     if new_label_sizes > other_label_size:
            #         #for other_label_image in other_label_images:
            #         merged_image[region_of_other_labels] = 0
            #         next_label = self.get_next_free_label()
            #         merged_image[old_slice][current_label_image] = next_label
        return merged_image


def intersection(gt_label, predicted_label):
    return np.count_nonzero(np.bitwise_and(gt_label, predicted_label))


def union(gt_label, predicted_label):
    return np.count_nonzero(np.bitwise_or(gt_label, predicted_label))


def intersection_over_union(gt_label, predicted_label):
    i = intersection(gt_label, predicted_label)
    u = union(gt_label, predicted_label)
    return i / u


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
            i = np.count_nonzero(np.bitwise_and(current_label_indizes, previous_label_indizes))
            if i == 0 :
                distance_overlap_matrix[label, previous_label] = 0.0
            else:
                u = np.count_nonzero(np.bitwise_or(current_label_indizes, previous_label_indizes))
                iou = i / (.1e-8 + u)
                distance_overlap_matrix[label, previous_label] = iou
    # print(distance_overlap_matrix)
    return distance_overlap_matrix


class InstanceImageCreator(object):
    def __init__(self,
                 coord_factors=0.0,
                 min_label_size=10,
                 bandwidth=0.1,
                 hdbscan=False,
                 min_cluster_size=50):
        self.coord_factors = coord_factors
        self.min_label_size = min_label_size
        self.bandwidth = bandwidth
        self.label_image = None
        self.background_embedding = None
        self.current_max_label = 0
        self.max_background_cosine_distance = 0.5
        self.use_mask = True
        self.hdbscan = hdbscan
        self.min_cluster_size = min_cluster_size

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
        mask = np.square(np.dot(reshaped_embedding, self.background_embedding)) < self.max_background_cosine_distance
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
        if self.hdbscan:
            cluster = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size,
                                      min_samples=self.min_cluster_size,
                                      metric='l2',
                                      core_dist_n_jobs=8,
                                      algorithm='boruvka_kdtree',
                                      leaf_size=25)
        else:
            seed_indizes = list(range(codes.shape[0]))
            shuffle(seed_indizes)
            seed_indizes = seed_indizes[:128]
            seeds = codes[seed_indizes]
            cluster = MeanShiftCosine(bandwidth=self.bandwidth, seeds=seeds, min_bin_freq=25, cluster_all=True)
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

    def create_instance_image(self, embeddings_normalized_slice_pair):
        #with Timer('add slice'):
            self.calculate_background_embedding_as_constant(embeddings_normalized_slice_pair)
            if self.use_mask:
                codes, mask = self.get_codes_with_mask(embeddings_normalized_slice_pair)
            else:
                codes = self.get_codes(embeddings_normalized_slice_pair)
                mask = None

            current_label_list = self.get_cluster_labels(codes, embeddings_normalized_slice_pair.shape[1:4], mask)
            current_label_list = self.relabel_nonoverlapping_labels(current_label_list, min_label_size=self.min_label_size)
            if len(current_label_list) == 0:
                self.label_image = np.zeros(embeddings_normalized_slice_pair.shape[1:4], dtype=np.uint16)
            else:
                label_image = utils.np_image.merge_label_images(current_label_list, dtype=np.uint16)
                self.label_image = label_image
                #self.remove_segmentations_outside_border()

    def calculate_background_embedding_as_constant(self, embeddings):
        if self.background_embedding is not None:
            return
        self.background_embedding = np.concatenate([np.ones((1), dtype=embeddings.dtype), np.zeros((embeddings.shape[0] - 1), dtype=embeddings.dtype)], axis=0)

    def remove_background_label(self, label_list, mean_embeddings):
        new_label_list = [label for label_index, label in enumerate(label_list)
                          if np.dot(mean_embeddings[label_index], self.background_embedding) < self.max_background_cosine_distance]
        return new_label_list

    def relabel_nonoverlapping_labels(self, label_list, min_label_size):
        new_label_list = []
        for l, current_label in enumerate(label_list):
            cc = utils.np_image.split_connected_components(*utils.np_image.connected_component(current_label))
            #if len(cc) > 1:
            #    print('cc of temp label', l, len(cc))
            for comp in cc:
                comp_label_size = np.sum(comp)
                if comp_label_size < min_label_size:
                    #print('removed label with size', comp_label_size)
                    continue
                new_label_list.append(comp)
        return new_label_list

    def get_next_free_label(self):
        self.current_max_label += 1
        return self.current_max_label

    #def remove_segmentations_outside_border(self):
    #    if self.image_ignore_border[0] == 0 and self.image_ignore_border[1] == 0:
    #        return
    #    border_image = np.zeros(self.label_image.shape, np.bool)
    #    border_image[self.image_ignore_border[1]:-self.image_ignore_border[1], self.image_ignore_border[0]:-self.image_ignore_border[0]] = True
    #
    #    label_images, labels = utils.np_image.split_label_image_with_unknown_labels(self.label_image)
    #    for label_image, label in zip(label_images, labels):
    #        if not np.any(np.bitwise_and(border_image, label_image)):
    #            print('removing border track', label)
    #            self.label_image[self.label_image == label] = 0



class InstanceTracker(object):
    def __init__(self,
                 stack_neighboring_slices=2,
                 save_label_stack=True,
                 image_ignore_border=None,
                 parent_search_dilation_size=0,
                 max_parent_search_frames=1,
                 max_merge_search_frames=10,
                 min_track_length=2):
        self.stack_neighboring_slices = stack_neighboring_slices
        self.embeddings_slices = []
        self.stacked_label_image = None
        self.background_embedding = None
        self.current_max_label = 0
        self.max_merge_distance = 0.9
        self.max_parent_search_frames = max_parent_search_frames
        self.max_merge_search_frames = max_merge_search_frames
        self.min_track_length = min_track_length
        self.min_parent_overlap = 0.01
        self.save_label_stack = save_label_stack
        self.image_ignore_border = image_ignore_border or [0, 0]
        self.parent_search_dilation_size = parent_search_dilation_size
        self.label_stack_list = []
        self.use_mask = True

    def add_new_label_image(self, label_image_pair):
        #with Timer('add label image'):
            self.merge_new_label_image(label_image_pair)

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
        self.search_and_merge_invalid_splits()

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

    def get_label_with_max_metric(self, binary_label, other_labels, metric):
        masked_other_labels = other_labels[binary_label]
        labels = np.unique(masked_other_labels)
        max_metric_value = 0
        max_label = 0
        for label in labels:
            if label == 0:
                continue
            other_binary_label = other_labels == label
            metric_value = metric(binary_label, other_binary_label)
            if max_metric_value < metric_value:
                max_metric_value = metric_value
                max_label = label
        return max_label, max_metric_value


    def search_and_merge_invalid_splits(self):
        if self.max_merge_search_frames == 0:
            return
        last_image = self.stacked_label_image[-1, ...]
        previous_image = self.stacked_label_image[-2, ...]
        previous_labels = np.unique(previous_image)

        for label in previous_labels:
            if label == 0:
                continue
            previous_label_image = previous_image == label
            max_label, max_overlap = self.get_label_with_max_metric(previous_label_image, last_image, intersection)
            if max_label != label and max_overlap > self.min_parent_overlap:
                min_frame, max_frame = self.get_min_max_frame(self.stacked_label_image == label)
                if max_frame - min_frame > self.max_merge_search_frames:
                    continue
                merge = False
                if min_frame == 0:
                    merge = True
                else:
                    first_image = self.stacked_label_image[min_frame - 1, ...]
                    next_image = self.stacked_label_image[min_frame, ...]
                    next_label_image = next_image == label
                    first_max_label, max_overlap = self.get_label_with_max_metric(next_label_image, first_image, intersection)
                    if first_max_label == max_label and max_overlap > self.min_parent_overlap:
                        merge = True
                if merge:
                    print('found merge, relabel', label, 'to', max_label, 'from frame', min_frame, 'to frame', max_frame)
                    self.relabel_track_from_to_frame(label, max_label, min_frame, max_frame)

        #for i in range(self.current_max_label):
        #    current_track_image = self.stacked_label_image == i
        #    self.get_min_max_frame(current_track_image)

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
        self.stacked_label_image = utils.np_image.relabel_ascending(self.stacked_label_image)
        self.current_max_label = np.max(self.stacked_label_image)

    def finalize(self):
        print('Finalizing')
        self.remove_too_short_tracks()
        self.remove_segmentations_outside_border()
        self.relabel_stacked_label_image()
        self.relabel_track_splits()

    def remove_too_short_tracks(self):
        labels = np.unique(self.stacked_label_image)
        for label in labels:
            if label == 0:
                continue
            track_image = self.stacked_label_image == label
            min_track_frame, max_track_frame = self.get_min_max_frame(track_image)
            length = max_track_frame - min_track_frame + 1
            if length < self.min_track_length:
                parent_id = self.get_parent_id(track_image, min_track_frame)
                if parent_id == 0:
                    print('remove too short track', label)
                    self.stacked_label_image[track_image] = 0

    def resample_stacked_label_image(self, input_image_size, transformation, smoothing_sigma):
        if smoothing_sigma == 1:
            interpolator = 'label_gaussian'
        else:
            interpolator = 'nearest'
        resampled_stacked_predictions_sitk = utils.sitk_image.transform_np_output_to_input(self.stacked_label_image,
                                                                                           output_spacing=None,
                                                                                           channel_axis=0,
                                                                                           input_image_size=input_image_size,
                                                                                           input_image_spacing=[1.0, 1.0],
                                                                                           input_image_origin=[0.0, 0.0],
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
