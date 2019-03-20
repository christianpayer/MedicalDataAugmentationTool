
from utils.io.common import create_directories_for_file_name
import SimpleITK as sitk
import numpy as np


def write_np(img, path, data_format='channels_first'):
    if len(img.shape) == 4:
        if data_format == 'channels_first':
            write(sitk.GetImageFromArray(np.transpose(img, (1, 2, 3, 0))), path)
        else:
            write(sitk.GetImageFromArray(img), path)
    else:
        write(sitk.GetImageFromArray(img), path)


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


def read(path, sitk_pixel_type=sitk.sitkInt16):
    image = sitk.ReadImage(path, sitk_pixel_type)
    x = image.GetNumberOfComponentsPerPixel()
    # TODO every sitkVectorUInt8 image is converted to have 3 channels (RGB) -> we may not want that
    if sitk_pixel_type == sitk.sitkVectorUInt8 and x == 1:
        image_single = sitk.VectorIndexSelectionCast(image)
        image = sitk.Compose(image_single, image_single, image_single)
    return image

