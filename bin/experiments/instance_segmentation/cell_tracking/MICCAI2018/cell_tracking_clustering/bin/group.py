
import SimpleITK as sitk
import numpy as np
from tensorflow_train.experiments.cell_tracking.embedding_tracker import EmbeddingTracker

np.set_printoptions(suppress=True)
#folder = 'dic'
#output_folder = 'output/' + folder + '/'
#x = sitk.ReadImage('input/' + folder + '/iter_60000/02_embeddings_2.mha', sitk.sitkVectorFloat32)
input = '/run/media/chris/media1/experiments/cell_tracking_output/DIC-C2DH-HeLa/outDIC-C2DH-HeLa/out_first/iter_60000/01_embeddings_2.mha'
output_folder = './'
x = sitk.ReadImage(input, sitk.sitkVectorFloat32)
x = sitk.GetArrayFromImage(x)
x = np.transpose(x, [3,0,1,2])
#x = x[:, :, ::2, ::2]
x = x.astype(np.float64)
coord_factors=0.01
#output = get_instances_cosine_kmeans_slice_by_slice(x)
for min_cluster_size in [500]: #[500, 1000]: #[100, 500, 1000]:
    for min_samples in [500]:
        tracker = EmbeddingTracker(coord_factors=coord_factors, stack_neighboring_slices=2, min_cluster_size=min_cluster_size, min_samples=min_samples, min_label_size_per_stack=1)
        output = tracker.get_instances_cosine_dbscan_slice_by_slice(x[:, :, :, :])
        #output = get_instances_cosine_dbscan_slice_by_slice(x[:, 0:10, :, :])
        print(output.shape)
        sitk.WriteImage(sitk.GetImageFromArray(output), output_folder + '01_out_2_slice_c' + str(min_cluster_size) + 's' + str(min_samples) + 'c' + str(coord_factors) + '.mha')
        merged = tracker.merge_consecutive_slices(output, slice_neighbour_size=2)
        sitk.WriteImage(sitk.GetImageFromArray(merged), output_folder + '01_merged_c' + str(min_cluster_size) + 's' + str(min_samples) + 'c' + str(coord_factors) + '.mha')
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
