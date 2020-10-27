
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
    Write an sitk image to a file path.
    :param img: The sitk image.
    :param path: The target path.
    :param compress: If true, compress the file.
    """
    create_directories_for_file_name(path)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(path)
    writer.SetUseCompression(compress)
    writer.Execute(img)


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
    """
    Read the metadata of an sitk image. The functions GetOrigin(), GetDirection(), and GetSpacing() of
    the resulting image work, the rest does not work.
    :param path: The path of the metadata to read.
    :return: An sitk image where only GetOrigin(), GetDirection(), and GetSpacing() work.
    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    return reader


def create_channel_layout_image(image, mode, data_format):
    """
    Convert the np input image to a given layout.
    :param image: The np image.
    :param mode: One of the following:
                 'default': Do not preprocess the image layout, just stack the input images.
                 'max': Take the maximum over all channels.
                 'avg': Take the mean over all channels.
                 'label_rgb': Create RGB outputs of the integer label input images.
                 'channel_rgb': Multiply each input label image with a label color and take the maximum response over all images.
                 'channel_rgb_no_first': Same as 'channel_rgb', but ignore image of first channel.
                 None: Take 'default'.
    :param data_format: The data_format. Either 'channels_first' or 'channels_last'.
    :return: The layouted np image.
    """
    channel_axis = 0 if data_format == 'channels_first' else -1
    if mode is None:
        return image
    if mode == 'max':
        return np.max(image, axis=channel_axis, keepdims=True)
    if mode == 'avg':
        return np.mean(image, axis=channel_axis, keepdims=True)
    if mode == 'label_rgb':
        if image.shape[channel_axis] != 1:
            raise ValueError('For mode == \'label_rgb\' only single channel images are allowed')
        current_image = np.squeeze(image, axis=channel_axis)
        labels = np.unique(current_image).tolist()
        rgb_image = np.zeros(current_image.shape + (3,), dtype=np.float32)
        for label in labels:
            if label == 0:
                continue
            rgb_image[current_image == label] = label_to_rgb(label)
        return np.stack([rgb_image[..., 0], rgb_image[..., 1], rgb_image[..., 2]], axis=0)
    if mode == 'channel_rgb' or mode == 'channel_rgb_no_first':
        rgb_image = None
        for i, current_image in enumerate(np.rollaxis(image, channel_axis)):
            if mode == 'channel_rgb_no_first' and i == 0:
                continue
            if rgb_image is None:
                rgb_image = np.zeros(current_image.shape + (3,), dtype=np.float32)
            current_rgb_image = np.expand_dims(current_image, axis=-1) * label_to_rgb(i)
            rgb_image = np.max([rgb_image, current_rgb_image], axis=0)
        return np.stack([rgb_image[..., 0], rgb_image[..., 1], rgb_image[..., 2]], axis=channel_axis)
    raise ValueError('Unsupported layout mode: ' + mode)


def create_image_layout_image(image, mode, data_format):
    """
    Convert the np input image to a given layout.
    :param image: The np image.
    :param mode: One of the following:
                'default': Do not preprocess the image layout, just stack the input images.
                'max_projection': Create max projections for every view.
                'avg_projection': Create avg projections for every view.
                'center_slice_projection': Take the center slice for each view.
    :param data_format: The data_format. Either 'channels_first' or 'channels_last'.
    :return: The layouted np image.
    """
    channel_axis = 0 if data_format == 'channels_first' else -1
    if mode is None:
        return image
    if mode == 'max_projection' or mode == 'avg_projection' or mode == 'center_slice_projection':
        projection_function = None
        if mode == 'max_projection':
            projection_function = utils.np_image.max_projection
        if mode == 'avg_projection':
            projection_function = utils.np_image.avg_projection
        if mode == 'center_slice_projection':
            projection_function = utils.np_image.center_slice_projection
        projections = [projection_function(current_image) for current_image in np.rollaxis(image, axis=channel_axis)]
        return np.stack(projections, axis=channel_axis)
    raise ValueError('Unsupported layout mode: ' + mode)


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
    if mode is None:
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
    raise ValueError('Unsupported normalization mode: ' + mode)


def create_sitk_image(image, mode, data_format, spacing=None, origin=None, direction=None):
    """
    Creates a sitk image from the given np image and mode.
    :param image: The np image.
    :param mode: One of the following:
                 'additional_dimension': Treat the channel dimension as an additional spatial dimension.
                 'vector': Treat the channel dimension not as a spatial dimension, but as a vector dimension.
                 'split': Split the channel dimension and create output images for each individual image of the channel dimension.
                 'gallery': Create a gallery, where each input image is next to each other on a square 2D grid.
    :param data_format: The data_format. Either 'channels_first' or 'channels_last'.
    :param spacing: The output spacing.
    :param origin: The output origin.
    :param direction: The output direction.
    :return: The sitk image.
    """
    channel_axis = 0 if data_format == 'channels_first' else -1
    # if image contains only one channel, just squeeze the channel and return the image without the channel dimension.
    if image.shape[channel_axis] == 1:
        sitk_image = sitk.GetImageFromArray(np.squeeze(image, axis=channel_axis), isVector=False)
        set_spacing_origin_direction(sitk_image, spacing, origin, direction)
        return sitk_image
    if mode == 'additional_dimension':
        if data_format != 'channels_first':
            image = np.transpose(image, [image.ndim - 1] + list(range(0, image.ndim - 1)))
        sitk_image = sitk.GetImageFromArray(image, isVector=False)
        set_spacing_origin_direction(sitk_image, spacing, origin, direction)
        return sitk_image
    if mode == 'vector':
        components = [sitk.GetImageFromArray(current_image, isVector=False) for current_image in np.rollaxis(image, axis=channel_axis)]
        sitk_image = sitk.ComposeImageFilter().Execute(components)
        set_spacing_origin_direction(sitk_image, spacing, origin, direction)
        return sitk_image
    if mode == 'split':
        image_list = []
        for current_image in np.rollaxis(image, axis=channel_axis):
            sitk_image = sitk.GetImageFromArray(current_image, isVector=False)
            set_spacing_origin_direction(sitk_image, spacing, origin, direction)
            image_list.append(sitk_image)
        return image_list
    if mode == 'gallery':
        return sitk.GetImageFromArray(utils.np_image.gallery(list(np.rollaxis(image, axis=channel_axis))), isVector=False)
    raise ValueError('Unsupported save mode: ' + mode)


def write_multichannel_np(image,
                          path,
                          channel_layout_mode=None,
                          image_layout_mode=None,
                          output_normalization_mode=None,
                          sitk_image_output_mode='vector',
                          image_type=None,
                          compress=True,
                          data_format='channels_first',
                          is_single_channel=False,
                          spacing=None,
                          origin=None,
                          direction=None,
                          reference_image=None):
    """
    Writes a np image to the given path. Allows various output modes.
    :param image: The np image.
    :param path: The output path.
    :param channel_layout_mode: One of the following:
                                None: Do not preprocess the image layout, just stack the input images.
                                'max': Take the maximum over all channels.
                                'avg': Take the mean over all channels.
                                'label_rgb': Create RGB outputs of the integer label input images.
                                'channel_rgb': Multiply each input label image with a label color and take the maximum response over all images.
                                'channel_rgb_no_first': Same as 'channel_rgb', but ignore image of first channel.
    :param image_layout_mode: One of the following:
                              None: Do not preprocess the image layout, just stack the input images.
                              'max_projection': Create max projections for every view.
                              'avg_projection': Create avg projections for every view.
                              'center_slice_projection': Take the center slice for each view.
    :param output_normalization_mode: One of the following:
                                      None: Perform no normalization.
                                      'min_max': Scale min and max of input image such that the output image covers the whole possible range of the output image type (dtype_min and dtype_max).
                                      tuple(min, max): Scale the input image such that the old range (min, max) is scaled to the output image type range (dtype_min and dtype_max).
    :param sitk_image_output_mode: One of the following:
                                   'additional_dimension': Treat the channel dimension as an additional spatial dimension.
                                   'vector': Treat the channel dimension not as a spatial dimension, but as a vector dimension.
                                   'split': Split the channel dimension and create output images for each individual image of the channel dimension.
                                   'gallery': Create a gallery, where each input image is next to each other on a square 2D grid.
    :param image_type: The output image type.
    :param compress: If true, compress the output image.
    :param data_format: The data_format. Either 'channels_first' or 'channels_last'.
    :param is_single_channel: If true, the input is treated as an image, where the channel dimension is missing. Therefore, a dimension will be added before processing.
    :param spacing: The output spacing.
    :param origin: The output origin.
    :param direction: The output direction.
    :param reference_image: If set, take spacing, origin, and direction from this sitk image.
    """
    # set default parameters
    image_type = image_type or image.dtype
    if reference_image is not None:
        if spacing is not None or origin is not None or direction is not None:
            raise ValueError('If reference image is given, spacing or origin or direction may not be given.')
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        direction = image.GetDirection()

    # perform processing
    if is_single_channel:
        image = np.expand_dims(image, axis=0 if data_format == 'channels_first' else -1)
    image = create_channel_layout_image(image, channel_layout_mode, data_format)
    image = create_image_layout_image(image, image_layout_mode, data_format)
    image = normalize_image_to_np_range(image, output_normalization_mode, image_type)
    image = image.astype(image_type)
    image_sitk = create_sitk_image(image, sitk_image_output_mode, data_format, spacing, origin, direction)

    # write output
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
