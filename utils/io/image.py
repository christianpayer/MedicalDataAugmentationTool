
from utils.io.common import create_directories_for_file_name
import SimpleITK as sitk
import numpy as np


def write_nd_np(img, path):
    write(sitk.GetImageFromArray(np.transpose(img, (1, 2, 3, 0))), path)

def write_np(img, path):
    if len(img.shape) == 4:
        write(sitk.GetImageFromArray(np.transpose(img, (1, 2, 3, 0))), path)
    else:
        write(sitk.GetImageFromArray(img), path)


def write(img, path):
    """
    Write a volume to a file path.

    :param img: the volume
    :param path: the target path
    :return:
    """
    create_directories_for_file_name(path)
    writer = sitk.ImageFileWriter()
    writer.Execute(img, path, True)


def write_np_rgb(img, path):
    assert(img.shape[0] == 3)
    rgb_components = [sitk.GetImageFromArray(img[i, :, :]) for i in range(img.shape[0])]
    filter = sitk.ComposeImageFilter()
    rgb = filter.Execute(rgb_components[0], rgb_components[1], rgb_components[2])
    write(rgb, path)


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
