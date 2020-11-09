
import SimpleITK as sitk
import utils.sitk_np
import numpy as np
import utils.np_image
import utils.sitk_np


def get_sitk_interpolator(interpolator):
    """
    Return an sitk interpolator object for the given string.
    :param interpolator: Interpolator type as string.
                         'nearest': sitk.sitkNearestNeighbor
                         'linear': sitk.sitkLinear
                         'cubic': sitk.sitkBSpline
                         'label_gaussian': sitk.sitkLabelGaussian
                         'gaussian': sitk.sitkGaussian
                         'lanczos': sitk.sitkLanczosWindowedSinc
    :return: The sitk interpolator object.
    """
    if interpolator == 'nearest':
        return sitk.sitkNearestNeighbor
    elif interpolator == 'linear':
        return sitk.sitkLinear
    elif interpolator == 'cubic':
        return sitk.sitkBSpline
    elif interpolator == 'label_gaussian':
        return sitk.sitkLabelGaussian
    elif interpolator == 'gaussian':
        return sitk.sitkGaussian
    elif interpolator == 'lanczos':
        return sitk.sitkLanczosWindowedSinc
    else:
        raise Exception('invalid interpolator type')


def resample(input_image,
             transform,
             output_size,
             output_spacing=None,
             output_origin=None,
             output_direction=None,
             interpolator=None,
             output_pixel_type=None,
             default_pixel_value=None):
    """
    Resample a given input image according to a transform.
    :param input_image: The input sitk image.
    :param transform: The sitk transformation to apply to the resample filter
    :param output_size: The image size in pixels of the output image.
    :param output_spacing: The spacing in mm of the output image.
    :param output_direction: The direction matrix of the output image.
    :param default_pixel_value: The pixel value of pixels outside the image region.
    :param output_origin: The output origin.
    :param interpolator: The interpolation function. See get_sitk_interpolator() for possible values.
    :param output_pixel_type: The output pixel type.
    :return: The resampled image.
    """
    image_dim = input_image.GetDimension()
    transform_dim = transform.GetDimension()
    assert image_dim == transform_dim, 'image and transform dim must be equal, are ' + str(image_dim) + ' and ' + str(transform_dim)
    output_spacing = output_spacing or [1] * image_dim
    output_origin = output_origin or [0] * image_dim
    output_direction = output_direction or np.eye(image_dim).flatten().tolist()
    interpolator = interpolator or 'linear'

    sitk_interpolator = get_sitk_interpolator(interpolator)

    # resample the image
    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetSize(output_size)
    resample_filter.SetInterpolator(sitk_interpolator)
    resample_filter.SetOutputSpacing(output_spacing)
    resample_filter.SetOutputOrigin(output_origin)
    resample_filter.SetOutputDirection(output_direction)
    resample_filter.SetTransform(transform)
    if default_pixel_value is not None:
        resample_filter.SetDefaultPixelValue(default_pixel_value)
    if output_pixel_type is None:
        resample_filter.SetOutputPixelType(input_image.GetPixelID())
    else:
        resample_filter.SetOutputPixelType(output_pixel_type)

    # perform resampling
    output_image = resample_filter.Execute(input_image)

    return output_image


def resample_to_spacing(image, new_spacing, interpolator=None):
    """
    Resamples a given image to a given spacing. (see resample)
    :param image: The image.
    :param new_spacing: The spacing.
    :param interpolator: The interpolator. Default is linear.
    :return: The resampled image.
    """
    # TODO: origin should be adapted to incorporate new spacing
    #  origin is at the center of the first pixel -> it has a 'half' spacing
    old_spacing = image.GetSpacing()
    old_size = image.GetSize()
    old_origin = image.GetOrigin()
    old_direction = image.GetDirection()
    new_size = [int(old_sp * old_si / new_sp) for old_sp, old_si, new_sp in zip(old_spacing, old_size, new_spacing)]
    return resample(image, sitk.AffineTransform(len(new_spacing)), new_size, new_spacing, old_origin, old_direction, interpolator)


def split_vector_components(image):
    """
    Split vector image into list of vector components.
    :param image: sitk image with vector image type.
    :return: list of sitk images with scalar image type.
    """
    filter = sitk.VectorIndexSelectionCastImageFilter()
    output = []
    for i in range(image.GetNumberOfComponentsPerPixel()):
        filter.SetIndex(i)
        output.append(filter.Execute(image))
    return output


def merge_vector_components(images):
    """
    Merge sitk images into an sitk image with vector image type.
    :param images: list of sitk images with scalar image type.
    :return: sitk image with vector image type.
    """
    filter = sitk.ComposeImageFilter()
    output = filter.Execute(images)
    return output

def rgba_to_rgb(image):
    components = split_vector_components(image)
    assert len(components) == 4, 'wrong number of components'
    del components[3]
    return merge_vector_components(components)

def copy_information_additional_dim(src, dst):
    src_dim = src.GetDimension()
    src_origin = src.GetOrigin()
    src_spacing = src.GetSpacing()
    src_direction = np.array(src.GetDirection(), np.float32).reshape((src_dim, src_dim))

    dst_origin = src_origin + (0,)
    dst_spacing = src_spacing + (1,)
    dst_direction = np.zeros((src_dim + 1, src_dim + 1), np.float32)
    dst_direction[:src_dim, :src_dim] = src_direction
    dst_direction[src_dim, src_dim] = 1

    dst.SetOrigin(dst_origin)
    dst.SetSpacing(dst_spacing)
    dst.SetDirection(dst_direction.flatten().tolist())


def copy_information(src, dst):
    src_origin = src.GetOrigin()
    src_spacing = src.GetSpacing()
    src_direction = src.GetDirection()

    dst.SetOrigin(src_origin)
    dst.SetSpacing(src_spacing)
    dst.SetDirection(src_direction)


def accumulate(images, origin=0.0, spacing=1.0):
    return sitk.JoinSeries(images, origin, spacing)


def reduce_dimension(image, axis=None):
    dim = image.GetDimension()
    if axis is None:
        axis = dim - 1
    size = list(image.GetSize())
    assert size[axis] == 1, 'size in dimension to reduce must be 1'
    size[axis] = 0
    index = [0] * dim
    return sitk.Extract(image, size, index)


def image4Dto3D(image):
    dim = image.GetDimension()
    assert dim == 4, 'dimension must be 4'
    axis = dim - 1
    size = list(image.GetSize())
    num_splits = size[axis]
    size[axis] = 0
    images = []
    for i in range(num_splits):
        index = [0] * dim
        index[axis] = i
        images.append(sitk.Extract(image, size, index))
    return images

# def accumulate(images):
#     #num_images = len(images)
#     layout = [1] * images[0].GetDimension() + [len(images)]
#     num_components = images[0].GetNumberOfComponentsPerPixel()
#     if num_components == 1:
#         filter = sitk.TileImageFilter()
#         filter.SetLayout(layout)
#         return filter.Execute(images)
#     else:
#         images_per_component = [split_vector_components(image) for image in images]
#         filter = sitk.TileImageFilter()
#         filter.SetLayout(layout)
#         volumes_per_component = [filter.Execute(im) for im in zip(*images_per_component)]
#         #for i in range(num_components):
#         #    filter = sitk.TileImageFilter()
#         #    filter.SetLayout(layout)
#         #    volumes_per_component.append(filter.Execute(list(zip(*images_per_component))[i]))
#         return merge_vector_components(volumes_per_component)

def argmax(images):
    # create np images, but do not copy
    images_np = [utils.sitk_np.sitk_to_np_no_copy(image) for image in images]
    # stack along first axis
    image_np = np.stack(images_np, axis=0)
    max_index_np = utils.np_image.argmax(image_np, axis=0, dtype=np.uint8)
    # convert back to sitk, and copy image information of first image
    max_index = utils.sitk_np.np_to_sitk(max_index_np)
    max_index.CopyInformation(images[0])
    return max_index


def split_label_image(image, labels):
    splits_np = utils.np_image.split_label_image(utils.sitk_np.sitk_to_np_no_copy(image), labels)
    splits = []
    for split_np in splits_np:
        split = utils.sitk_np.np_to_sitk(split_np)
        split.CopyInformation(image)
        splits.append(split)
    return splits


def split_multi_label_image(image, labels):
    dim_to_split = -1 #collapse channel dimension (expects 'channels_first'; numpy and itk dimension order is reversed!)
    size = list(image.GetSize())
    size[dim_to_split] = 0
    splits = []
    index = [0] * len(size)
    for label in labels:
        index[dim_to_split] = label
        split = sitk.Extract(image, size, index)
        splits.append(split)
    return splits


def merge_label_images(images, labels):
    images_np = [utils.sitk_np.sitk_to_np_no_copy(image) for image in images]
    merged_np = utils.np_image.merge_label_images(images_np, labels)
    merged = utils.sitk_np.np_to_sitk(merged_np)
    merged.CopyInformation(images[0])
    return merged


def transform_np_output_to_sitk_input(output_image, output_spacing, channel_axis, input_image_sitk, transform, interpolator='linear', output_pixel_type=None):
    input_image_size = input_image_sitk.GetSize()
    input_image_spacing = input_image_sitk.GetSpacing()
    input_image_origin = input_image_sitk.GetOrigin()
    input_image_direction = input_image_sitk.GetDirection()

    return transform_np_output_to_input(output_image=output_image,
                                        output_spacing=output_spacing,
                                        channel_axis=channel_axis,
                                        input_image_size=input_image_size,
                                        input_image_spacing=input_image_spacing,
                                        input_image_origin=input_image_origin,
                                        input_image_direction=input_image_direction,
                                        transform=transform,
                                        interpolator=interpolator,
                                        output_pixel_type=output_pixel_type)


def transform_single_np_output_to_input(output_image, output_spacing, input_image_size, input_image_spacing, input_image_origin, input_image_direction, transform, interpolator='linear', output_pixel_type=None):
    output_image_sitk = utils.sitk_np.np_to_sitk(output_image)
    if output_spacing is not None:
        output_image_sitk.SetSpacing(output_spacing)
    transformed_output_image_sitk = resample(output_image_sitk,
                                             transform.GetInverse(),
                                             input_image_size,
                                             input_image_spacing,
                                             input_image_origin,
                                             input_image_direction,
                                             interpolator,
                                             output_pixel_type)
    return transformed_output_image_sitk


def transform_np_output_to_input(output_image, output_spacing, channel_axis, input_image_size, input_image_spacing, input_image_origin, input_image_direction, transform, interpolator='linear', output_pixel_type=None):
    if channel_axis is not None:
        output_images = utils.np_image.split_by_axis(output_image, axis=channel_axis)
    else:
        output_images = [output_image]

    return list(map(lambda output_image: transform_single_np_output_to_input(output_image,
                                                                             output_spacing,
                                                                             input_image_size,
                                                                             input_image_spacing,
                                                                             input_image_origin,
                                                                             input_image_direction,
                                                                             transform,
                                                                             interpolator,
                                                                             output_pixel_type), output_images))


def connected_component(image):
    filter = sitk.ConnectedComponentImageFilter()
    filter.FullyConnectedOn()
    output_image = filter.Execute(image)
    num_components = filter.GetObjectCount()
    return output_image, num_components


def largest_connected_component(image):
    cc_image, num_components = connected_component(image)
    if num_components == 1:
        return image

    filter = sitk.LabelShapeStatisticsImageFilter()
    filter.Execute(cc_image)

    largest_label = 0
    largest_count = 0
    for i in range(1, num_components + 1):
        current_count = filter.GetNumberOfPixels(i)
        if current_count > largest_count:
            largest_count = current_count
            largest_label = i

    change_map = {}
    for i in range(1, num_components + 1):
        change_map[i] = 0

    change_map[largest_label] = 1
    filter = sitk.ChangeLabelImageFilter()
    filter.SetChangeMap(change_map)
    return filter.Execute(cc_image)


def distance_transform(image, squared_distance=False, use_image_spacing=False):
    return sitk.DanielssonDistanceMap(image, inputIsBinary=True, squaredDistance=squared_distance, useImageSpacing=use_image_spacing)


def apply_np_image_function(image, f):
    image_np = utils.sitk_np.sitk_to_np(image)
    output_np = f(image_np)
    output = utils.sitk_np.np_to_sitk(output_np)
    copy_information(image, output)
    return output

def hausdorff_distances(image_0, image_1, labels, multi_label=False):
    if multi_label:
        label_images_0 = utils.sitk_image.split_vector_components(image_0)
        label_images_1 = utils.sitk_image.split_vector_components(image_1)
    else:
        label_images_0 = utils.sitk_image.split_label_image(image_0, labels)
        label_images_1 = utils.sitk_image.split_label_image(image_1, labels)
    hausdorff_distance_list = []
    average_hausdorff_distance_list = []
    for label_image_0, label_image_1 in zip(label_images_0, label_images_1):
        assert label_image_0.GetPixelIDValue() != -1, "ITK PixelIDValue: -1 == 'Unknown'"
        assert label_image_1.GetPixelIDValue() != -1, "ITK PixelIDValue: -1 == 'Unknown'"
        assert label_image_0.GetPixelIDValue() == label_image_1.GetPixelIDValue(),\
            "ITK PixelIDValue has to be the same for both images, otherwise HausdorffDistanceImageFilter results in nan"
        try:
            filter = sitk.HausdorffDistanceImageFilter()
            filter.Execute(label_image_0, label_image_1)
            current_hausdorff_distance = filter.GetHausdorffDistance()
            current_average_hausdorff_distance = filter.GetAverageHausdorffDistance()
        except:
            current_hausdorff_distance = np.nan
            current_average_hausdorff_distance = np.nan
            pass
        hausdorff_distance_list.append(current_hausdorff_distance)
        average_hausdorff_distance_list.append(current_average_hausdorff_distance)
    return hausdorff_distance_list, average_hausdorff_distance_list

def surface_distance(label_image_0, label_image_1):
    # code adapted from https://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/34_Segmentation_Evaluation.html
    try:
        # calculate distances on label contours
        reference_distance_map = sitk.SignedMaurerDistanceMap(label_image_1, squaredDistance=False, useImageSpacing=True)
        reference_distance_map_arr = sitk.GetArrayViewFromImage(reference_distance_map)
        reference_surface = sitk.LabelContour(label_image_1)
        reference_surface_arr = sitk.GetArrayViewFromImage(reference_surface)

        segmented_distance_map = sitk.SignedMaurerDistanceMap(label_image_0, squaredDistance=False, useImageSpacing=True)
        segmented_distance_map_arr = sitk.GetArrayViewFromImage(segmented_distance_map)
        segmented_surface = sitk.LabelContour(label_image_0)
        segmented_surface_arr = sitk.GetArrayViewFromImage(segmented_surface)

        seg2ref_distances = np.abs(reference_distance_map_arr[segmented_surface_arr == 1])
        ref2seg_distances = np.abs(segmented_distance_map_arr[reference_surface_arr == 1])

        all_surface_distances = np.concatenate([seg2ref_distances, ref2seg_distances])

        # # Multiply the binary surface segmentations with the distance maps. The resulting distance
        # # maps contain non-zero values only on the surface (they can also contain zero on the surface)
        # seg2ref_distance_map = reference_distance_map * sitk.Cast(segmented_surface, sitk.sitkFloat32)
        # ref2seg_distance_map = segmented_distance_map * sitk.Cast(reference_surface, sitk.sitkFloat32)
        #
        # statistics_image_filter = sitk.StatisticsImageFilter()
        # # Get the number of pixels in the reference surface by counting all pixels that are 1.
        # statistics_image_filter.Execute(reference_surface)
        # num_reference_surface_pixels = int(statistics_image_filter.GetSum())
        # # Get the number of pixels in the reference surface by counting all pixels that are 1.
        # statistics_image_filter.Execute(segmented_surface)
        # num_segmented_surface_pixels = int(statistics_image_filter.GetSum())
        #
        # # Get all non-zero distances and then add zero distances if required.
        # seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
        # seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0])
        # seg2ref_distances = seg2ref_distances + list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
        # ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
        # ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0])
        # ref2seg_distances = ref2seg_distances + list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))
        #
        # all_surface_distances = seg2ref_distances + ref2seg_distances

        current_mean_surface_distance = np.mean(all_surface_distances)
        current_median_surface_distance = np.median(all_surface_distances)
        current_std_surface_distance = np.std(all_surface_distances)
        current_max_surface_distance = np.max(all_surface_distances)
    except:
        current_mean_surface_distance = np.nan
        current_median_surface_distance = np.nan
        current_std_surface_distance = np.nan
        current_max_surface_distance = np.nan
        pass

    return current_mean_surface_distance, current_median_surface_distance, current_std_surface_distance, current_max_surface_distance

def surface_distances(image_0, image_1, labels, calculate_mean=True, calculate_median=True, calculate_std=True, calculate_max=True):
    label_images_0 = utils.sitk_image.split_label_image(image_0, labels)
    label_images_1 = utils.sitk_image.split_label_image(image_1, labels)
    mean_surface_distance_list = []
    median_surface_distance_list = []
    std_surface_distance_list = []
    max_surface_distance_list = []

    for current_mean_surface_distance, current_median_surface_distance, current_std_surface_distance, current_max_surface_distance in map(surface_distance, label_images_0, label_images_1):
        mean_surface_distance_list.append(current_mean_surface_distance)
        median_surface_distance_list.append(current_median_surface_distance)
        std_surface_distance_list.append(current_std_surface_distance)
        max_surface_distance_list.append(current_max_surface_distance)
    return_tuple = tuple()
    if calculate_mean:
        return_tuple += (mean_surface_distance_list,)
    if calculate_median:
        return_tuple += (median_surface_distance_list,)
    if calculate_std:
        return_tuple += (std_surface_distance_list,)
    if calculate_max:
        return_tuple += (max_surface_distance_list,)
    return return_tuple


def label_to_rgb(label, float_range=True):
    """
    Converts a label index to a color. Uses lookup table from ITK.
    :param label: The label index.
    :param float_range: If true, RGB values are in float, i.e., (1.0, 0.5, 0.0), otherwise in byte (255, 128, 0)
    :return: RGB color as a list.
    """
    colormap = [[255, 0, 0], [0, 205, 0], [0, 0, 255], [0, 255, 255],
        [255, 0, 255], [255, 127, 0], [0, 100, 0], [138, 43, 226],
        [139, 35, 35], [0, 0, 128], [139, 139, 0], [255, 62, 150],
        [139, 76, 57], [0, 134, 139], [205, 104, 57], [191, 62, 255],
        [0, 139, 69], [199, 21, 133], [205, 55, 0], [32, 178, 170],
        [106, 90, 205], [255, 20, 147], [69, 139, 116], [72, 118, 255],
        [205, 79, 57], [0, 0, 205], [139, 34, 82], [139, 0, 139],
        [238, 130, 238], [139, 0, 0]]
    color = colormap[label % len(colormap)]
    if float_range:
        color = [c / 255.0 for c in color]
    return color


def set_spacing_origin_direction(image, spacing, origin, direction):
    if spacing is not None:
        image.SetSpacing(spacing)
    if origin is not None:
        image.SetOrigin(origin)
    if direction is not None:
        image.SetDirection(direction)
    return image
