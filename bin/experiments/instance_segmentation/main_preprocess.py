import SimpleITK as sitk
import numpy as np
import glob
import os
import utils.io.image
import utils.sitk_np
from utils.np_image import split_label_image_with_unknown_labels, merge_label_images
from utils.sitk_image import copy_information, copy_information_additional_dim


def get_seg_tra_image_list_2d(folder):
    index = 0
    seg_list = []
    tra_list = []
    while True:
        num_str = f'{index:03}'
        file_name_seg = os.path.join(folder, 'SEG/man_seg' + num_str + '.tif')
        try:
            seg = utils.io.image.read(file_name_seg, sitk.sitkUInt16)
        except:
            seg = None

        file_name_tra = os.path.join(folder, 'TRA/man_track' + num_str + '.tif')
        try:
            tra = utils.io.image.read(file_name_tra, sitk.sitkUInt16)
        except:
            break

        seg_list.append(seg)
        tra_list.append(tra)
        index += 1

    return seg_list, tra_list


def get_input_image_list_2d(folder):
    index = 0
    im_list = []
    while True:
        num_str = f'{index:03}'
        file_name = os.path.join(folder, 't' + num_str + '.tif')
        try:
            im = utils.io.image.read(file_name, sitk.sitkUInt16)
        except:
            break

        im_list.append(im)
        index += 1

    return im_list


def relabel_seg(seg, tra):
    seg_label_images, _ = split_label_image_with_unknown_labels(seg)
    tra_label_images, tra_labels = split_label_image_with_unknown_labels(tra)
    del tra_label_images[0]
    del tra_labels[0]
    new_seg_labels = [0]
    for i in range(1, len(seg_label_images)):
        seg_label_image = seg_label_images[i]
        tra_label_overlap = [(tra_label, np.sum(np.bitwise_and(seg_label_image == 1, tra_label_image == 1))) for tra_label_image, tra_label in zip(tra_label_images, tra_labels)]
        best_tra_label, best_overlap = max(tra_label_overlap, key=lambda x: x[1])
        new_seg_labels.append(best_tra_label)
    assert len(np.unique(new_seg_labels)) == len(new_seg_labels), 'duplicate labels'
    new_seg = merge_label_images(seg_label_images, new_seg_labels)
    return new_seg


def process_seg_folder(folder, output_folder_prefix):
    seg, tra = get_seg_tra_image_list_2d(folder)
    folder = os.path.normpath(folder)
    split = folder.split(os.sep)
    print(split)
    output_folder = os.path.join(output_folder_prefix, split[-3], split[-2], split[-1], 'MERGED')
    print(output_folder)

    stacked_np = []
    for i, (seg_im, tra_im) in enumerate(zip(seg, tra)):
        j = i + 0
        file_name = f'{j:03}' + '.mha'
        if seg_im is not None:
            seg_im_np = utils.sitk_np.sitk_to_np(seg_im)
            tra_im_np = utils.sitk_np.sitk_to_np(tra_im)
            new_seg_im_np = relabel_seg(seg_im_np, tra_im_np)
            merged_im_np = tra_im_np.copy()
            merged_im_np[new_seg_im_np > 0] = new_seg_im_np[new_seg_im_np > 0]
            merged_im = utils.sitk_np.np_to_sitk(merged_im_np)
            copy_information(tra_im, merged_im)
            stacked_np.append(merged_im_np)

            utils.io.image.write(merged_im, os.path.join(output_folder, file_name))
        else:
            tra_im_np = utils.sitk_np.sitk_to_np(tra_im)
            stacked_np.append(tra_im_np)
            utils.io.image.write(tra_im, os.path.join(output_folder, file_name))

    print('stack')
    stacked_np = np.stack(stacked_np, axis=0)
    print(stacked_np.shape)
    stacked = utils.sitk_np.np_to_sitk(stacked_np)
    copy_information_additional_dim(tra[0], stacked)
    print('save')
    utils.io.image.write(stacked, os.path.join(output_folder, 'stacked.mha'))


def process_input_folder(folder, output_folder_prefix):
    im_list = get_input_image_list_2d(folder)
    folder = os.path.normpath(folder)
    split = folder.split(os.sep)
    print(split)
    output_folder = os.path.join(output_folder_prefix, split[-3], split[-2], split[-1])
    print(output_folder)

    stacked_np = []
    for i, im in enumerate(im_list):
        file_name = f't{i:03}' + '.mha'
        im_np = utils.sitk_np.sitk_to_np(im)
        stacked_np.append(im_np)
        utils.io.image.write(im, os.path.join(output_folder, file_name))

    if len(stacked_np) == 0:
        return

    print('stack')
    stacked_np = np.stack(stacked_np, axis=0)
    print(stacked_np.shape)
    stacked = utils.sitk_np.np_to_sitk(stacked_np)
    copy_information_additional_dim(im_list[0], stacked)
    print('save')
    utils.io.image.write(stacked, os.path.join(output_folder, 'stacked.mha'))


def main():
    dataset_base_folder = 'CHANGE_FOLDER'

    seg_folders = glob.glob(dataset_base_folder + 'trainingdataset/*2D*/0?_GT/')
    seg_folders = [folder for folder in seg_folders if not 'PSC' in folder]
    print(seg_folders)
    for folder in sorted(seg_folders):
        process_seg_folder(folder, 'celltrackingchallenge')

    image_folders = sorted(glob.glob(dataset_base_folder + '*/*2D*/0?/'))
    image_folders = [folder for folder in image_folders if not 'PSC' in folder]
    print(image_folders)
    for folder in image_folders:
        process_input_folder(folder, 'celltrackingchallenge')


if __name__ == '__main__':
    main()
