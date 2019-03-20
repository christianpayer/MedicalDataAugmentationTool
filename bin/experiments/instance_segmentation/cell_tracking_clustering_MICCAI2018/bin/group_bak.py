
import SimpleITK as sitk
import numpy as np
import sklearn.cluster
import sklearn.decomposition
import hdbscan


def get_instances_cosine_kmeans_slice_by_slice(embeddings_normalized, coord_factors=0, bandwidth=0.5,
                                               min_label_size=100):
    print(embeddings_normalized.shape)
    # if coord_factors > 0:
    #     z, y, x = np.meshgrid(range(embeddings_normalized.shape[1]), range(embeddings_normalized.shape[2]), indexing='ij')
    #     y = y.astype(np.float32)
    #     x = x.astype(np.float32)
    #     y = y * coord_factors
    #     x = x * coord_factors
    #     coordinates = np.stack([y, x], axis=0)
    #     embeddings_normalized = np.concatenate([coordinates, embeddings_normalized], axis=0)
    outputs = np.zeros(embeddings_normalized.shape[1:4])
    previous_label_cluster_centers = []
    next_free_label = 1
    for i in range(embeddings_normalized.shape[1]):
        current_embeddings_normalized = embeddings_normalized[:, i, :, :]
        codes = np.transpose(current_embeddings_normalized, [1, 2, 0])
        codes = np.reshape(codes, [-1, codes.shape[2]])
        # bandwidth = sklearn.cluster.estimate_bandwidth(codes, quantile=0.1, n_samples=10000)
        # bandwidth = 0.6
        # bandwidth=0.7
        print(i, bandwidth)

        # if len(previous_label_cluster_centers) == 0:
        #     seeds = sklearn.cluster.get_bin_seeds(codes, 0.1, 1)
        # else:
        #     seeds = [cluster_center for _, cluster_center in previous_label_cluster_centers]
        seeds = sklearn.cluster.get_bin_seeds(codes, 0.1, 1)
        ms = sklearn.cluster.MeanShift(bandwidth=bandwidth, bin_seeding=False, cluster_all=False, n_jobs=-2, seeds=seeds)
        ms.fit(codes)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        output = np.reshape(labels, current_embeddings_normalized.shape[1:4])

        if i > 10:
            break

        mapped_labels = np.ones_like(output)
        current_label_cluster_centers = []
        for label in range(n_clusters_ - 1):
            label_sum = np.sum(output == label)
            if label_sum < min_label_size:
                continue
            current_cluster_center = cluster_centers[label, :]

            use_new_label = True
            if len(previous_label_cluster_centers) > 0:
                min_cluster = max(previous_label_cluster_centers, key=lambda i: np.dot(i[1], current_cluster_center))
                if np.dot(min_cluster[1], current_cluster_center) > 0.9:
                    use_new_label = False
                    actual_label = min_cluster[0]
            if use_new_label:
                actual_label = next_free_label
                print(actual_label)
                next_free_label += 1
            current_label_cluster_centers.append((actual_label, current_cluster_center))
            mapped_labels[output == label] = actual_label
        previous_label_cluster_centers = current_label_cluster_centers
        outputs[i, ...] = mapped_labels
    print("number of estimated clusters : %d" % n_clusters_)

    return outputs


def get_instances_cosine_kmeans_3d(embeddings_normalized, coord_factors=0, bandwidth=0.5,
                                               min_label_size=100):
    print(embeddings_normalized.shape)
    # if coord_factors > 0:
    #     z, y, x = np.meshgrid(range(embeddings_normalized.shape[1]), range(embeddings_normalized.shape[2]), indexing='ij')
    #     y = y.astype(np.float32)
    #     x = x.astype(np.float32)
    #     y = y * coord_factors
    #     x = x * coord_factors
    #     coordinates = np.stack([y, x], axis=0)
    #     embeddings_normalized = np.concatenate([coordinates, embeddings_normalized], axis=0)
    outputs = np.zeros(embeddings_normalized.shape[1:4])
    current_embeddings_normalized = embeddings_normalized
    codes = np.transpose(current_embeddings_normalized, [1, 2, 3, 0])
    codes = np.reshape(codes, [-1, codes.shape[3]])
    # bandwidth = sklearn.cluster.estimate_bandwidth(codes, quantile=0.1, n_samples=10000)
    # bandwidth = 0.6
    # bandwidth=0.7
    print(bandwidth)

    # if len(previous_label_cluster_centers) == 0:
    #     seeds = sklearn.cluster.get_bin_seeds(codes, 0.1, 1)
    # else:
    #     seeds = [cluster_center for _, cluster_center in previous_label_cluster_centers]
    seeds = sklearn.cluster.get_bin_seeds(codes, 0.1, 1)
    print('after_seeds')
    ms = sklearn.cluster.MeanShift(bandwidth=bandwidth, bin_seeding=False, cluster_all=False, n_jobs=-2, seeds=seeds)
    ms.fit(codes)
    print('after ms')
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    outputs = np.reshape(labels, current_embeddings_normalized.shape[1:4]).astype(np.float32)

    return outputs


def get_instances_cosine_dbscan_3d(embeddings_normalized, coord_factors=0.001, bandwidth=0.5,
                                               min_label_size=100):
    print(embeddings_normalized.shape)
    if coord_factors > 0:
        z, y, x = np.meshgrid(range(embeddings_normalized.shape[1]), range(embeddings_normalized.shape[2]), range(embeddings_normalized.shape[3]), indexing='ij')
        y = y.astype(np.float32)
        x = x.astype(np.float32)
        y = y * coord_factors
        x = x * coord_factors
        coordinates = np.stack([y, x], axis=0)
        embeddings_normalized = np.concatenate([coordinates, embeddings_normalized], axis=0)
    outputs = np.zeros(embeddings_normalized.shape[1:4])
    current_embeddings_normalized = embeddings_normalized
    codes = np.transpose(current_embeddings_normalized, [1, 2, 3, 0])
    codes = np.reshape(codes, [-1, codes.shape[3]])
    # bandwidth = sklearn.cluster.estimate_bandwidth(codes, quantile=0.1, n_samples=10000)
    # bandwidth = 0.6
    # bandwidth=0.7

    # if len(previous_label_cluster_centers) == 0:
    #     seeds = sklearn.cluster.get_bin_seeds(codes, 0.1, 1)
    # else:
    #     seeds = [cluster_center for _, cluster_center in previous_label_cluster_centers]
    #seeds = sklearn.cluster.get_bin_seeds(codes, 0.1, 1)
    #print('after_seeds')

    #ms = sklearn.cluster.DBSCAN(eps=0.05, metric='l2', algorithm='ball_tree')#, leaf_size=100)
    ms = hdbscan.HDBSCAN(min_cluster_size=100, metric='l2', core_dist_n_jobs=8)
    ms.fit(codes)
    print('after ms')
    labels = ms.labels_
    #cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    outputs = np.reshape(labels, current_embeddings_normalized.shape[1:4]).astype(np.float32)

    # dbscan_instance = dbscan(codes.tolist(), 0.05, 50, ccore=True)
    # dbscan_instance.process()
    # print('done')
    # clusters = dbscan_instance.get_clusters()
    #
    # outputs = np.zeros(current_embeddings_normalized.shape[1] * current_embeddings_normalized.shape[2] * current_embeddings_normalized.shape[3])
    # for label, cluster in enumerate(clusters):
    #     outputs[cluster] = label
    #
    # outputs = np.reshape(outputs, current_embeddings_normalized.shape[1:4]).astype(np.float32)

    return outputs

def calculate_label_overlap(labels, previous_labels):
    max_previous_label = np.max(previous_labels) + 1
    max_label = np.max(labels) + 1
    distance_overlap_matrix = np.zeros([max_label, max_previous_label])
    for label in range(max_label):
        current_label_indizes = labels == label
        current_label_size = np.sum(current_label_indizes)
        for previous_label in range(max_previous_label):
            previous_label_indizes = previous_labels == previous_label
            overlap = np.sum(np.bitwise_and(current_label_indizes, previous_label_indizes)) / np.sum(np.bitwise_or(current_label_indizes, previous_label_indizes))
            distance_overlap_matrix[label, previous_label] = overlap
    print(distance_overlap_matrix)
    return distance_overlap_matrix


def merge_slice_by_slice_overlap(labels, min_overlap=0.25):
    merged_labels = np.zeros_like(labels)
    merged_labels[0, ...] = labels[0, ...]
    current_max_label = 0
    current_shape = np.zeros_like(labels[0, ...])
    for i in range(labels.shape[0]):
        print('slice', i)
        next_shape = labels[i, ...]
        next_shape_merged = np.zeros_like(next_shape)
        distance_overlap_matrix = calculate_label_overlap(next_shape, current_shape)
        done_labels = []
        while (distance_overlap_matrix.shape[0] > 0 and distance_overlap_matrix.shape[1] > 0):
            max_index = np.unravel_index(np.argmax(distance_overlap_matrix, axis=None), distance_overlap_matrix.shape)
            label_to, label_from = max_index
            max_value = distance_overlap_matrix[label_to, label_from]
            if max_value < min_overlap:
                break
            next_shape_merged[next_shape == label_to] = label_from
            done_labels.append(label_to)
            distance_overlap_matrix[label_to, :] = 0
            distance_overlap_matrix[:, label_from] = 0

        max_new_labels = np.max(next_shape) + 1
        for j in range(max_new_labels):
            if j not in done_labels:
                indizes = next_shape == j
                if np.any(indizes):
                    current_max_label += 1
                    print('new label', current_max_label)
                    next_shape_merged[indizes] = current_max_label

        current_shape = next_shape_merged
        merged_labels[i, ...] = next_shape_merged
    return merged_labels

def get_background_dbscan(embeddings_slice_normalized, min_label_size=100):
    codes = np.transpose(embeddings_slice_normalized, [1, 2, 0])
    codes = np.reshape(codes, [-1, codes.shape[2]])
    ms = hdbscan.HDBSCAN(min_cluster_size=min_label_size, metric='l2', core_dist_n_jobs=8, algorithm='best')
    ms.fit(codes)
    labels = ms.labels_
    output = np.reshape(labels, embeddings_slice_normalized.shape[1:3])
    labels_unique = np.unique(labels[labels >= 0])
    label_sizes = []
    for label in range(len(labels_unique)):
        label_indizes = output == label
        label_size = np.sum(label_indizes)
        label_sizes.append((label, label_size))
    max_label = max(label_sizes, key=lambda x: x[1])
    background = output == max_label[0]
    return background



def get_instances_cosine_dbscan_slice_by_slice(embeddings_normalized, coord_factors=0.0, bandwidth=0.5,
                                               min_label_size=500, neighboring_slices=1):
    outputs = np.zeros(embeddings_normalized.shape[1:4], np.int32)
    outputs = np.concatenate([outputs] * neighboring_slices, axis=0)
    print(embeddings_normalized.shape)
    embedding_size = embeddings_normalized.shape[0]

    for i in range(embeddings_normalized.shape[1] - neighboring_slices):
        print(i)
        current_embeddings_normalized = embeddings_normalized[:, i:i+neighboring_slices, :, :]
        #background = get_background_dbscan(current_embeddings_normalized)
        #outputs[i, ...] = background + 1
        #continue
        if coord_factors > 0:
            y, x = np.meshgrid(range(current_embeddings_normalized.shape[1]), range(current_embeddings_normalized.shape[2]), indexing='ij')
            y = y.astype(np.float32)
            x = x.astype(np.float32)
            y = y * coord_factors
            x = x * coord_factors
            coordinates = np.stack([y, x], axis=0)
            current_embeddings_normalized = np.concatenate([coordinates, current_embeddings_normalized], axis=0)
        codes = np.transpose(current_embeddings_normalized, [1, 2, 3, 0])
        codes = np.reshape(codes, [-1, embedding_size])
        # bandwidth = sklearn.cluster.estimate_bandwidth(codes, quantile=0.1, n_samples=10000)
        # bandwidth = 0.6
        # bandwidth=0.7

        # if len(previous_label_cluster_centers) == 0:
        #     seeds = sklearn.cluster.get_bin_seeds(codes, 0.1, 1)
        # else:
        #     seeds = [cluster_center for _, cluster_center in previous_label_cluster_centers]
        #ms = hdbscan.HDBSCAN(min_cluster_size=min_label_size, metric='l2', core_dist_n_jobs=8)
        ms = hdbscan.HDBSCAN(min_cluster_size=min_label_size, metric='l2', core_dist_n_jobs=8, algorithm='best')
        ms.fit(codes)
        labels = ms.labels_
        #cluster_centers = ms.cluster_centers_
        labels_unique = np.unique(labels[labels >= 0])
        #n_clusters_ = len(labels_unique)
        output = np.reshape(labels, current_embeddings_normalized.shape[1:4])

        mean_embeddings = [np.mean(codes[labels == label], axis=0) for label in range(len(labels_unique))]
        mean_embeddings = [mean_embedding / np.linalg.norm(mean_embedding, ord=2) for mean_embedding in mean_embeddings]
        #
        # for j, jv in enumerate(mean_embeddings):
        #     print(j, jv)
        #     for k, kv in enumerate(mean_embeddings):
        #         print(j, k, np.arccos(np.minimum(np.dot(jv, kv), 1)) / np.pi * 180)

        # for label in range(len(labels_unique)):
        #     label_indizes = output == label
        #     label_size = np.sum(label_indizes)
        #     #max_label_size = 0.1 * current_embeddings_normalized.shape[1] * current_embeddings_normalized.shape[2]
        #     background_overlap = np.sum(np.bitwise_and(label_indizes, background))
        #     background_overlap_ratio = background_overlap / label_size
        #     if background_overlap_ratio > 0.5:
        #         print('removed label', label)
        #         output[label_indizes] = -1

        outputs[i*neighboring_slices:i*neighboring_slices+neighboring_slices, ...] = output + 1

    #outputs = merge_slice_by_slice_overlap(outputs)
    print('?')

    return outputs


# def get_acyclic_graph(outputs):
#     all_ids = np.unique(outputs)
#     id_tracks = {}
#     for current_id in all_ids:
#         if current_id == 0:
#             continue


np.set_printoptions(suppress=True)
x = sitk.ReadImage('/run/media/chris/media1/experiments/cell_tracking_working/Fluo-N2DH-GOWT1/double_u_l7_64_64_s256_f10_e16Fluo-N2DH-GOWT1/out_first/iter_5000/02_embeddings_2.mha', sitk.sitkVectorFloat32)
x = sitk.GetArrayFromImage(x)
x = np.transpose(x, [3,0,1,2])

#output = get_instances_cosine_kmeans_slice_by_slice(x)
output = get_instances_cosine_dbscan_slice_by_slice(x[:, 0:10, :, :], neighboring_slices=1)
#output = get_instances_cosine_dbscan_slice_by_slice(x[:, 0:10, :, :])
print(output.shape)
sitk.WriteImage(sitk.GetImageFromArray(output), 'out_single.mha')
print('Done!')

# current_slice = x[0, ...]
# embedding_size = current_slice.shape[-1]
# current_code = np.reshape(current_slice, [-1, embedding_size])
#
# sklearn.decomposition.MeanShift
#
#
# # current_code_py = current_code.tolist()
# # initial_centers = kmeans_plusplus_initializer(current_code_py, 15).initialize()
# # #xmeans_instance = xmeans(current_code, initial_centers, kmax=20)
# # xmeans_instance = xmeans(current_code_py, initial_centers, 100, 0.025, splitting_type.BAYESIAN_INFORMATION_CRITERION, ccore=True)
# # xmeans_instance.process()
# # clusters = xmeans_instance.get_clusters()
# # centers = xmeans_instance.get_centers()
# #
# # print(len(clusters))
# # for i in range(len(clusters)):
# #     label = np.zeros([128*128])
# #     label[clusters[i]] = 255
# #     label = np.reshape(label, [128, 128])
# #     slice_np = label.astype(np.uint8)
# #     sitk.WriteImage(sitk.GetImageFromArray(slice_np), 'out' + str(i) + '.png')
# #q, r = np.linalg.qr(current_code.T, mode='full')
# #u, s, v = np.linalg.svd(current_code, compute_uv=True, full_matrices=True)
#
#
#
# #svd = sklearn.decomposition.TruncatedSVD()
# #out = svd.fit_transform(current_code)
#
# #q, r = np.linalg.qr(current_code.T)
# u, s, v = np.linalg.svd(current_code, compute_uv=True, full_matrices=False)
#
# print(s)
# print(u.shape)
# print(s.shape)
# print(v.shape)
#
# new_slice = np.reshape(u, current_slice.shape)
#
# for i in range(new_slice.shape[2]):
#     print(np.min(np.abs(new_slice[:, :, i])), np.max(np.abs(new_slice[:, :, i])))
#     slice_np = (np.minimum(255, np.maximum(0, np.abs(new_slice[:, :, i]) * 1000))).astype(np.uint8)
#     sitk.WriteImage(sitk.GetImageFromArray(slice_np), 'out' + str(i) + '.png')
