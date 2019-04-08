
import numpy as np
import scipy.ndimage
import math
import skimage.morphology
import skimage.measure
import skimage.draw
from transformations.intensity.np.smooth import gaussian


def find_maximum_coord_in_image(image):
    # calculate maximum
    max_index = np.argmax(image)
    coord = np.array(np.unravel_index(max_index, image.shape), np.int32)
    return coord


def find_maximum_in_image(image):
    # calculate maximum
    coord = find_maximum_coord_in_image(image)
    max_value = image[tuple(coord)]
    # flip indizes from [y,x] to [x,y]
    return max_value, coord


def find_quadratic_subpixel_maximum_in_image(image):
    coord = find_maximum_coord_in_image(image)
    max_value = image[tuple(coord)]
    refined_coord = coord.astype(np.float32)
    dim = coord.size
    for i in range(dim):
        if int(coord[i]) - 1 < 0 or int(coord[i]) + 1 >= image.shape[i]:
            continue
        before_coord = coord.copy()
        before_coord[i] -= 1
        after_coord = coord.copy()
        after_coord[i] += 1
        pa = image[tuple(before_coord)]
        pb = image[tuple(coord)]
        pc = image[tuple(after_coord)]
        diff = 0.5 * (pa - pc) / (pa - 2 * pb + pc)
        refined_coord[i] += diff
    return max_value, refined_coord


def nms_2d(image, min_value):
    neigh8 = np.asarray([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])
    maxima_indizes = (image > scipy.ndimage.maximum_filter(image, footprint=neigh8, mode='constant', cval=-np.inf)) * (image > min_value)
    maxima_values = image[maxima_indizes]
    maxima_coords = np.nonzero(maxima_indizes)
    maxima_value_coord_pairs = [(maxima_values[i], np.array((maxima_coords[1][i], maxima_coords[0][i]), np.float)) for i in range(len(maxima_values))]
    maxima_value_coord_pairs.sort(key=lambda tup: tup[0], reverse=True)
    return maxima_value_coord_pairs


def split_by_axis(image, axis=0):
    image_list = np.split(image, image.shape[axis], axis=axis)
    return [np.squeeze(image, axis=axis) for image in image_list]


def split_label_image(image, labels, dtype=None):
    if dtype is None:
        dtype = image.dtype
    image_list = []
    for label in labels:
        image_list.append((image == label).astype(dtype))
    return image_list


def split_label_image_with_unknown_labels(image, dtype=None):
    if dtype is None:
        dtype = image.dtype
    image_list = []
    labels = np.unique(image).tolist()
    for label in labels:
        label_image = (image == label).astype(dtype)
        image_list.append(label_image)
    return image_list, labels


def merge_label_images(images, labels=None, dtype=None):
    if labels == None:
        labels = list(range(1, len(images) + 1))
    image = np.zeros_like(images[0], dtype=dtype)
    for i, label in enumerate(labels):
        if dtype is not None:
            image = image + images[i].astype(dtype) * label
        else:
            image = image + images[i] * label
    return image


def relabel_ascending(image, dtype=None):
    if dtype is None:
        dtype = image.dtype
    labels = np.unique(image).tolist()
    relabeled = np.zeros_like(image, dtype=dtype)
    next_free_number = 0
    for label in labels:
        label_image = image == label
        relabeled[label_image] = next_free_number
        next_free_number += 1
    return relabeled


def smooth_label_images(images, sigma=1, dtype=None):
    if dtype is None:
        dtype = images[0].dtype
    smoothed_images = [gaussian(image, sigma=sigma) for image in images]
    smoothed = np.stack(smoothed_images, 0)
    label_images = np.argmax(smoothed, axis=0)
    return split_label_image(label_images, range(0, len(images)), dtype=dtype)


def argmax(image, axis=0, dtype=np.uint8):
    shape = image.shape[:axis] + image.shape[axis+1:]
    max_index_np = np.zeros(shape, dtype=dtype)
    np.argmax(image, axis=axis, out=max_index_np)
    return max_index_np


def gallery(images, num_cols=None):
    assert len(images) > 0
    shape = images[0].shape
    assert all([np.allclose(image.shape, shape) for image in images]), 'gallery only works for image list with equal shape'
    if num_cols is None:
        num_cols = math.ceil(math.sqrt(len(images)))
    num_rows = math.ceil(len(images) / num_cols)
    if len(shape) == 2:
        output_shape = (shape[0] * num_rows, shape[1] * num_cols)
    elif len(shape) == 3:
        output_shape = (shape[0], shape[1] * num_rows, shape[2] * num_cols)
    output = np.zeros(output_shape, np.float32)

    for i, image in enumerate(images):
        col_index = int(i % num_cols)
        row_index = int(i / num_cols)
        if len(output_shape) == 2:
            output[row_index*shape[0]:(row_index+1)*shape[0],col_index*shape[1]:(col_index+1)*shape[1]] = image
        if len(output_shape) == 3:
            output[:,row_index*shape[1]:(row_index+1)*shape[1],col_index*shape[2]:(col_index+1)*shape[2]] = image

    return output


def convex_hull(image):
    if np.sum(image) == 0:
        return image
    return skimage.morphology.convex_hull_image(image) > 0


def connected_component(image, dtype=np.uint8, connectivity=2):
    labels, num = skimage.measure.label(image, connectivity=connectivity, return_num=True, background=0)
    if dtype is not np.int64:
        labels = labels.astype(dtype)
    return labels, num


def split_connected_components(labels, num):
    labels_list = []
    for i in range(1, num+1):
        current_label = (labels == i)
        labels_list.append(current_label)
    return labels_list


def largest_connected_component(image):
    labels, num = connected_component(image)
    if num == 0:
        return np.zeros_like(image)
    counts = np.bincount(labels.flatten())
    largest_label = np.argmax(counts[1:]) + 1
    lcc = (labels == largest_label)
    return lcc


def dilation_square(image, kernel_size):
    kernel = np.ones(kernel_size)
    return scipy.ndimage.binary_dilation(image, kernel).astype(image.dtype)


def dilation_circle(image, kernel_size):
    assert kernel_size[0] == kernel_size[1], 'currently only circles are supported'
    center = (kernel_size[0] - 1) / 2
    radius = (kernel_size[0] - 1) / 2 + 1e-5
    kernel = np.zeros(kernel_size)
    r, c = skimage.draw.circle(center, center, radius, shape=kernel_size)
    kernel[r, c] = 1
    return scipy.ndimage.binary_dilation(image, kernel).astype(image.dtype)


def erosion_square(image, kernel_size):
    kernel = np.ones(kernel_size)
    return scipy.ndimage.binary_erosion(image, kernel).astype(image.dtype)


def center_of_mass(image):
    return scipy.ndimage.measurements.center_of_mass(image)


def draw_circle(image, center_rc, radius, value=1):
    r, c = skimage.draw.circle(center_rc[0], center_rc[1], radius, image.shape)
    image[r, c] = value
    return image


def distance_transform(image):
    return scipy.ndimage.morphology.distance_transform_edt(image)


def roll_with_pad(image, shift, mode='constant'):
    pad_list = []
    shift_list = []
    for s in shift:
        if s == 0:
            pad_list.append((0, 0))
            shift_list.append(slice(None, None))
        elif s < 0:
            pad_list.append((0, -s))
            shift_list.append(slice(-s, None))
        else:
            pad_list.append((s, -0))
            shift_list.append(slice(None, -s))
    padded = np.pad(image, pad_list, mode=mode)
    cropped = padded[tuple(shift_list)]
    return cropped
