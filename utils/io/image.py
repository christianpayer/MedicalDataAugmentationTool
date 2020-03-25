
from utils.io.common import create_directories_for_file_name
from transformations.intensity.np.normalize import scale_min_max, scale
from utils.sitk_image import label_to_rgb, set_spacing_origin_direction
import SimpleITK as sitk
import numpy as np
import utils.np_image
import os


def write_nd_np(img, path, compress=True):
    write(sitk.GetImageFromArray(np.transpose(img, (1, 2, 3, 0))), path, compress)

def write_np(img, path, compress=True):
    if len(img.shape) == 4:
        write(sitk.GetImageFromArray(np.transpose(img, (1, 2, 3, 0))), path, compress)
    else:
        write(sitk.GetImageFromArray(img), path, compress)


def write(img, path, compress=True):
    """
    Write a volume to a file path.

    :param img: the volume
    :param path: the target path
    :return:
    """
    create_directories_for_file_name(path)
    writer = sitk.ImageFileWriter()
    writer.Execute(img, path, compress)


def write_np_rgb(img, path, compress=True):
    assert(img.shape[0] == 3)
    rgb_components = [sitk.GetImageFromArray(img[i, :, :]) for i in range(img.shape[0])]
    filter = sitk.ComposeImageFilter()
    rgb = filter.Execute(rgb_components[0], rgb_components[1], rgb_components[2])
    write(rgb, path, compress)


def write_np_rgba(img, path, compress=True):
    assert(img.shape[0] == 4)
    rgb_components = [sitk.GetImageFromArray(img[i, :, :]) for i in range(img.shape[0])]
    filter = sitk.ComposeImageFilter()
    rgb = filter.Execute(rgb_components[0], rgb_components[1], rgb_components[2], rgb_components[3])
    write(rgb, path, compress)


def read(path, sitk_pixel_type=sitk.sitkInt16):
    image = sitk.ReadImage(path, sitk_pixel_type)
    x = image.GetNumberOfComponentsPerPixel()
    # TODO every sitkVectorUInt8 image is converted to have 3 channels (RGB) -> we may not want that
    if sitk_pixel_type == sitk.sitkVectorUInt8 and x == 1:
        image_single = sitk.VectorIndexSelectionCast(image)
        image = sitk.Compose(image_single, image_single, image_single)
    return image


def read_meta_data(path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    return reader


def create_layout_image(image, mode, split_channel_axis, data_format):
    """
    Convert the np input image to a given layout.
    :param image: The np image.
    :param mode: One of the following:
                'default': Do not preprocess the image layout, just stack the input images.
                'max_projection': Create max projection images of every input image and axis. Layout them the same as 'gallery'.
                'avg_projection': Create avg projection images of every input image and axis. Layout them the same as 'gallery'.
                'label_rgb': Create RGB outputs of the integer label input images.
                'channel_rgb': Multiply each input label image with a label color and take the maximum response over all images.
                'channel_rgb_no_first': Same as 'channel_rgb', but ignore image of first channel.
                'gallery': Create a gallery, where each input image is next to each other on a square 2D grid.
    :param split_channel_axis: If true, split by channel axis.
    :param data_format: The data_format. Either 'channels_first' or 'channels_last'.
    :return: The layouted np image.
    """
    if split_channel_axis:
        image_axis = 0 if data_format == 'channels_first' else len(image.shape) - 1
        split_list = np.split(image, image.shape[image_axis], axis=image_axis)
        split_list = [np.squeeze(split, axis=image_axis) for split in split_list]
    else:
        split_list = [image]
    if mode == 'default':
        if len(split_list) == 1:
            return split_list[0]
        return np.stack(split_list, axis=0)
    if mode == 'max_projection':
        dim = len(split_list[0].shape)
        max_projections = [np.max(image, axis=axis) for image in split_list for axis in range(dim)]
        return utils.np_image.gallery(max_projections, dim)
    if mode == 'avg_projection':
        dim = len(split_list[0].shape)
        max_projections = [np.mean(image, axis=axis) for image in split_list for axis in range(dim)]
        return utils.np_image.gallery(max_projections, dim)
    if mode == 'center_slice_0':
        #dim = len(split_list[0].shape)
        center_slices = [image[image.shape[0] // 2, ...] for image in split_list]
        return center_slices
    if mode == 'label_rgb':
        rgb_image = None
        for i, image in enumerate(split_list):
            labels = np.unique(image).tolist()
            current_rgb_image = np.zeros(image.shape + (3,), dtype=image.dtype)
            for label in labels:
                if label == 0:
                    continue
                current_rgb_image[image == label] = label_to_rgb(label)
            if rgb_image is None:
                rgb_image = current_rgb_image
            else:
                rgb_image = np.max([rgb_image, current_rgb_image], axis=0)
        return np.stack([rgb_image[..., 0], rgb_image[..., 1], rgb_image[..., 2]], axis=0)
    if mode == 'channel_rgb' or mode == 'channel_rgb_no_first':
        rgb_image = None
        for i, image in enumerate(split_list):
            if mode == 'channel_rgb_no_first' and i == 0:
                continue
            current_rgb_image = np.expand_dims(image, axis=-1) * label_to_rgb(i)
            if rgb_image is None:
                rgb_image = current_rgb_image
            else:
                rgb_image = np.max([rgb_image, current_rgb_image], axis=0)
        return np.stack([rgb_image[..., 0], rgb_image[..., 1], rgb_image[..., 2]], axis=0)
    if mode == 'gallery':
        return utils.np_image.gallery(split_list)
    raise Exception('Unsupported layout mode.')


def normalize_image_to_np_range(image, mode, dtype):
    """
    Normalize the given np image for the given mode and dtype.
    :param image: The np image.
    :param mode: One of the following:
                'default': Perform no normalization.
                'min_max': Scale min and max of input image such that the output image covers the whole possible range of the output image type (dtype_min and dtype_max).
                tuple(min, max): Scale the input image such that the old range (min, max) is scaled to the output image type range (dtype_min and dtype_max).
    :param dtype: The output image dtype. Needed for some normalization modes.
    :return: The normalized np image.
    """
    if mode == 'default':
        return image
    # get allowed min and max value of numpy image type
    if dtype == np.float16 or dtype == np.float32 or dtype == np.float64:
        dtype_min, dtype_max = 0, 1
    else:
        dtype_info = np.iinfo(dtype)
        dtype_min, dtype_max = dtype_info.min, dtype_info.max
    if mode == 'min_max':
        return scale_min_max(image, (dtype_min, dtype_max))
    if isinstance(mode, tuple) or isinstance(mode, list):
        assert len(mode) == 2, 'Normalization value list must have 2 entries.'
        image = scale(image, mode, (dtype_min, dtype_max))
        return np.clip(image, dtype_min, dtype_max)
    raise Exception('Unsupported normalization mode.')


def create_sitk_image(image, mode, spacing=None, origin=None, direction=None):
    """
    Creates a sitk image from the given np image and mode.
    :param image: The np image.
    :param mode: One of the following:
                 'default': Treat the channel dimension as an additional spatial dimension.
                 'vector': Treat the channel dimension not as a spatial dimension, but as a vector dimension.
                 'split': Split the channel dimension and create output images for each individual image of the channel dimension.
    :param spacing: The output spacing.
    :param origin: The output origin.
    :param direction: The output direction.
    :return: The sitk image.
    """
    if mode == 'default':
        sitk_image = sitk.GetImageFromArray(image, isVector=False)
        set_spacing_origin_direction(sitk_image, spacing, origin, direction)
        return sitk_image
    if mode == 'vector':
        components = [sitk.GetImageFromArray(image[i, ...], isVector=False) for i in range(image.shape[0])]
        sitk_image = sitk.ComposeImageFilter().Execute(components)
        set_spacing_origin_direction(sitk_image, spacing, origin, direction)
        return sitk_image
    if mode == 'split':
        image_list = []
        for i in range(image.shape[0]):
            sitk_image = sitk.GetImageFromArray(image[i, ...], isVector=False)
            set_spacing_origin_direction(sitk_image, spacing, origin, direction)
            image_list.append(sitk_image)
        return image_list
    raise Exception('Unsupported save mode.')


def write_multichannel_np(image,
                          path,
                          split_channel_axis=False,
                          layout_mode=None,
                          normalization_mode=None,
                          sitk_image_mode=None,
                          image_type=None,
                          compress=True,
                          data_format='channels_first',
                          spacing=None,
                          origin=None,
                          direction=None):
    """
    Writes a np image to the given path. Allows various output modes.
    :param image: The np image.
    :param path: The output path.
    :param split_channel_axis: If true, split the input image by the channel axis.
    :param layout_mode: One of the following:
                        'default': Do not preprocess the image layout, just stack the input images.
                        'max_projection': Create max projection images of every input image and axis. Layout them the same as 'gallery'.
                        'avg_projection': Create avg projection images of every input image and axis. Layout them the same as 'gallery'.
                        'label_rgb': Create RGB outputs of the integer label input images.
                        'channel_rgb': Multiply each input label image with a label color and take the maximum response over all images.
                        'channel_rgb_no_first': Same as 'channel_rgb', but ignore image of first channel.
                        'gallery': Create a gallery, where each input image is next to each other on a square 2D grid.
                        None: Take 'default'.
    :param normalization_mode: One of the following:
                               'default': Perform no normalization.
                               'min_max': Scale min and max of input image such that the output image covers the whole possible range of the output image type (dtype_min and dtype_max).
                               tuple(min, max): Scale the input image such that the old range (min, max) is scaled to the output image type range (dtype_min and dtype_max).
                               None: Take 'vector' if split_channel_axis == True and layout_mode == 'default', otherwise take 'default'
    :param sitk_image_mode: One of the following:
                            'default': Treat the channel dimension as an additional spatial dimension.
                            'vector': Treat the channel dimension not as a spatial dimension, but as a vector dimension.
                            'split': Split the channel dimension and create output images for each individual image of the channel dimension.
                             None: Take 'default'.
    :param image_type: The output image type.
    :param compress: If true, compress the output image.
    :param data_format: The data_format. Either 'channels_first' or 'channels_last'.
    :param spacing: The output spacing.
    :param origin: The output origin.
    :param direction: The output direction.
    """
    layout_mode = layout_mode or 'default'
    normalization_mode = normalization_mode or 'default'
    if sitk_image_mode is None:
        if split_channel_axis == True and layout_mode == 'default':
            sitk_image_mode = 'vector'
        else:
            sitk_image_mode = 'default'
    image_type = image_type or image.dtype
    image = normalize_image_to_np_range(image, normalization_mode, image_type)
    #image = image.astype(image_type)
    image = create_layout_image(image, layout_mode, split_channel_axis, data_format)
    image = image.astype(image_type)
    image_sitk = create_sitk_image(image, sitk_image_mode, spacing, origin, direction)
    if isinstance(image_sitk, list):
        for i, current_image_sitk in enumerate(image_sitk):
            current_path_parts = os.path.splitext(path)
            current_path = current_path_parts[0] + '_' + str(i) + current_path_parts[1]
            write(current_image_sitk, current_path, compress)
    else:
        write(image_sitk, path, compress)


def write_multichannel_sitk(image, *args, **kwargs):
    """
    Writes a multichannel sitk image with the given parameters. See write_multichannel_np for individual parameters.
    :param image: The sitk image.
    :param args: *args of write_multichannel_np
    :param kwargs: **kwargs of write_multichannel_np
    """
    image_np = sitk.GetArrayFromImage(image)
    write_multichannel_np(image_np, spacing=image.GetSpacing(), origin=image.GetOrigin(), direction=image.GetDirection(), *args, **kwargs)
