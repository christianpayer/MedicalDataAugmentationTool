
import SimpleITK as sitk
import utils.sitk_np
import numpy as np
import utils.np_image
import utils.sitk_np


def get_sitk_interpolator(interpolator):
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

    :param input_image: ITK image
        the input image
    :param transform: SimpleITK.Transform
        the (composite) transform to be applied
    :param output_size: list of int
        default is same as input image
    :param output_spacing: list of float
        default is input spacing from input_image
    :param output_direction: list of float
        default is input direction from input_image
    :param default_pixel_value:
        default is zero
    :param output_origin: list of int
        Default is zero-origin for each dimension
    :param interpolator: SimpleITK.InterpolatorEnum
        Default is SimpleITK.sitkLinear.
    :return: the resampled image
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

def split_vector_components(image):
    filter = sitk.VectorIndexSelectionCastImageFilter()
    output = []
    for i in range(image.GetNumberOfComponentsPerPixel()):
        filter.SetIndex(i)
        output.append(filter.Execute(image))
    return output

def merge_vector_components(images):
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


def transform_np_output_to_input(output_image, output_spacing, channel_axis, input_image_size, input_image_spacing, input_image_origin, input_image_direction, transform, interpolator='linear', output_pixel_type=None):
    if channel_axis is not None:
        output_images = utils.np_image.split_by_axis(output_image, axis=channel_axis)
    else:
        output_images = [output_image]

    transformed_output_images_sitk = []
    for output_image in output_images:
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
        transformed_output_images_sitk.append(transformed_output_image_sitk)

    return transformed_output_images_sitk


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
