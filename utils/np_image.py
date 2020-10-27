
import numpy as np
import scipy.ndimage
import math
import skimage.morphology
import skimage.measure
import skimage.draw
import itertools
from transformations.intensity.np.smooth import gaussian


def find_maximum_coord_in_image(image):
    """
    Return the max coordinate from an image.
    :param image: The np image.
    :return: The coordinate as np array.
    """
    max_index = np.argmax(image)
    coord = np.array(np.unravel_index(max_index, image.shape), np.int32)
    return coord


def find_maximum_in_image(image):
    """
    Return the max value and coordinate from an image.
    :param image: The np image.
    :return: A tuple of the max value and coordinate as np array.
    """
    coord = find_maximum_coord_in_image(image)
    max_value = image[tuple(coord)]
    return max_value, coord


def refine_coordinate_subpixel(image, coord):
    """
    Refine a local maximum coordinate to the subpixel maximum.
    :param image: The np image.
    :param coord: The coordinate to refine
    :return: The refined coordinate as np array.
    """
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
    return refined_coord


def find_quadratic_subpixel_maximum_in_image(image):
    """
    Return the max value and the subpixel refined coordinate from an image.
    Refine a local maximum coordinate to the subpixel maximum.
    :param image: The np image.
    :return: A tuple of the max value and the refined coordinate as np array.
    """
    coord = find_maximum_coord_in_image(image)
    max_value = image[tuple(coord)]
    refined_coord = refine_coordinate_subpixel(image, coord)
    return max_value, refined_coord


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
    """
    Return the argmax over the given axis.
    Performance tip: If axis == len(image.shape) - 1, no internal copy operation is needed, otherwise, this operation may need lots of memory.
    :param image: The np image.
    :param axis: The axis to take the argmax from.
    :param dtype: The output dtype.
    :return: The np array of the argmax.
    """
    if axis < 0:
        axis = len(image.shape) + axis
    shape = image.shape[:axis] + image.shape[axis+1:]
    max_index_np = np.zeros(shape, dtype=dtype)
    np.argmax(image, axis=axis, out=max_index_np)
    return max_index_np


def gallery(images, num_cols=None):
    assert len(images) > 0
    shape = np.max([image.shape for image in images], axis=0)
    if num_cols is None:
        num_cols = math.ceil(math.sqrt(len(images)))
    num_rows = math.ceil(len(images) / num_cols)
    if len(shape) == 2:
        output_shape = (shape[0] * num_rows, shape[1] * num_cols)
    elif len(shape) == 3:
        output_shape = (shape[0], shape[1] * num_rows, shape[2] * num_cols)
    output = np.zeros(output_shape, images[0].dtype)

    for i, image in enumerate(images):
        col_index = int(i % num_cols)
        row_index = int(i / num_cols)
        current_shape = image.shape
        if len(output_shape) == 2:
            output[row_index*shape[0]:row_index*shape[0]+current_shape[0],col_index*shape[1]:col_index*shape[1]+current_shape[1]] = image
        if len(output_shape) == 3:
            output[:current_shape[0],row_index*shape[1]:row_index*shape[1]+current_shape[1],col_index*shape[2]:col_index*shape[2]+current_shape[2]] = image

    return output


def convex_hull(image):
    if np.sum(image) == 0:
        return image
    return skimage.morphology.convex_hull_image(image) > 0


def connected_component(image, dtype=np.uint8, connectivity=2, calculate_bounding_box=True):
    """
    Calculate connected components of pixels != 0 (see skimage.measure.label) and return labels image and number of labels tuple
    :param image: The np image.
    :param dtype: The dtype of the output label image.
    :param connectivity: The connectivity (num hops). See skimage.measure.label.
    :param calculate_bounding_box: If true, calculate the connected components inside a bounding box, which speeds up the the function
                                   in cases, where the target structures are only a small fraction of the overal input image.
    :return: Tuple of np image representing labels, and number of labels.
    """
    labels = np.zeros(image.shape, dtype=dtype)
    if calculate_bounding_box:
        # find the bounding_box, which makes the function typically faster
        start, end = bounding_box(image)
        if np.any(np.isnan(start)) or np.any(np.isnan(end)):
            # if bounding box is not found, return a zeroes image with 0 number of labels
            return labels, 0
        slices = tuple([slice(s, e + 1) for s, e in zip(start, end)])  # +1 because bounding_box is inclusive
    else:
        slices = tuple([slice(0, s) for s in image.shape])
    image_cropped = image[slices]
    labels_cropped, num = skimage.measure.label(image_cropped, connectivity=connectivity, return_num=True, background=0)
    labels = np.zeros(image.shape, dtype=dtype)
    labels[slices] = labels_cropped
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


def binary_fill_holes(image):
    """
    Perform morphological hole filling of a binary image.
    :param image: The np image.
    :return: The image with filled holes.
    """
    return scipy.ndimage.binary_fill_holes(image)


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
    coords = skimage.draw.circle(center_rc[0], center_rc[1], radius, image.shape)
    skimage.draw.set_color(image, coords, value)
    return image


def draw_sphere(image, center, radius, value=1):
    z, y, x = np.meshgrid(range(image.shape[0]), range(image.shape[1]), range(image.shape[2]), indexing='ij')
    d = np.sqrt((z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2)
    image[d <= radius] = value
    return image


def draw_line(image, coords_from, coords_to, value=1):
    coords = skimage.draw.line(int(coords_from[0]), int(coords_from[1]), int(coords_to[0]), int(coords_to[1]))
    skimage.draw.set_color(image, coords, value)
    return image


def distance_transform(image):
    return scipy.ndimage.morphology.distance_transform_edt(image)


def roll_with_pad(image, shift, mode='constant'):
    """
    Roll axis by given shifts and padding. (see np.roll)
    :param image: The np image.
    :param shift: List of shifts per axis.
    :param mode: Padding mode. See np.pad.
    :return: The rolled np.array. Has same shape and type of image.
    """
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


def image_projection(image, projection_function):
    """
    Performs a projection function per view/axis and returns the lower dimensional outputs next to each other.
    :param image: The image.
    :param projection_function: The projection function per axis.
    :return: Projected images next to each other.
    """
    dim = image.ndim
    projection = [projection_function(image, axis) for axis in range(dim)]
    return gallery(projection, dim)


def max_projection(image):
    """
    Performs a max projection per view/axis and returns the lower dimensional outputs next to each other.
    :param image: The image.
    :return: Projected images next to each other.
    """
    return image_projection(image, np.max)


def avg_projection(image):
    """
    Performs an average projection per view/axis and returns the lower dimensional outputs next to each other.
    :param image: The image.
    :return: Projected images next to each other.
    """
    return image_projection(image, np.mean)


def center_slice_projection(image):
    """
    Performs a center slice projection per view/axis and returns the lower dimensional outputs next to each other.
    :param image: The image.
    :return: Projected images next to each other.
    """
    return image_projection(image, lambda current_image, axis: np.squeeze(current_image[tuple([slice(None) if i != axis else slice(current_image.shape[axis] // 2, current_image.shape[axis] // 2 + 1)
                                                                                               for i in range(current_image.ndim)])], axis=axis))


def bounding_box(image):
    """
    Calculate the bounding box of an image of pixels != 0. Both start and end index are inclusive, i.e., contain the image value.
    If the image is all zeroes, return np arrays of np.nan.
    :param image: The image.
    :return: The bounding box as start and end tuple.
    """
    dim = image.ndim
    start = []
    end = []
    for ax in itertools.combinations(reversed(range(dim)), dim - 1):
        nonzero = np.any(image, axis=ax)
        nonzero_where = np.where(nonzero)[0]
        if len(nonzero_where) > 0:
            curr_start, curr_end = nonzero_where[[0, -1]]
        else:
            curr_start, curr_end = np.nan, np.nan
        start.append(curr_start)
        end.append(curr_end)
    return np.array(start), np.array(end)


def local_maxima(image):
    """
    Calculate all local maxima of an image. A local maxima is a pixel that is larger than each of its neighbors.
    Uses 2 (1D), 8 (2D), or 27 (3D) neighborhood.
    :param image: The np image.
    :return: tuple of indizes array and corresponding values array.
    """
    dim = len(image.shape)
    neigh = np.ones([3] * dim)
    neigh[tuple([1] * dim)] = 0
    maxima = (image > scipy.ndimage.maximum_filter(image, footprint=neigh, mode='constant', cval=np.inf))
    maxima_indizes = np.array(np.where(maxima))
    maxima_values = image[tuple([maxima_indizes[i] for i in range(dim)])]
    return maxima_indizes.T, maxima_values
    # TODO: the following code is a different implementation of local_maxima, which could be faster or slower -> check
    # dim = len(image.shape)
    # image_local_maxima = np.ones(image.shape - np.array([2] * dim), np.bool)
    # image_cropped = image[tuple([slice(1, image.shape[i] - 1) for i in range(dim)])]
    # # iterate over all neighbors by shifting the image for each coordinate
    # for shifts in itertools.product(*([[-1, 0, 1]] * dim)):
    #     # if shifts == [0] * dim, the shifted image is equal to the cropped (unshifted) image
    #     if np.count_nonzero(shifts) == 0:
    #         # do not compare
    #         continue
    #     slices = [slice(1 + shifts[i], image.shape[i] - 1 + shifts[i]) for i in range(dim)]
    #     # shift image
    #     image_shifted = image[tuple(slices)]
    #     # compare with cropped (unshifted) image
    #     image_local_maxima = image_local_maxima & (image_cropped > image_shifted)
    # # calculate indizes of local maxima and shift by [1] * dim due to cropping
    # indizes = np.array(np.where(image_local_maxima)) + np.array([[1]] * dim)
    # values = image[tuple([indizes[i] for i in range(dim)])]
    # return indizes.T, values
