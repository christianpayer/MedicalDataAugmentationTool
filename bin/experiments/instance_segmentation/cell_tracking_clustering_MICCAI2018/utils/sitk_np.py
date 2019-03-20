
import numpy as np
import SimpleITK as sitk


def sitk_to_np_no_copy(image_sitk):
    return sitk.GetArrayViewFromImage(image_sitk)


def sitk_to_np(image_sitk, type=None):
    if type is None:
        return sitk.GetArrayFromImage(image_sitk)
    else:
        return sitk.GetArrayViewFromImage(image_sitk).astype(type)


def np_to_sitk(image_np, type=None, is_vector=False):
    if type is None:
        return sitk.GetImageFromArray(image_np, is_vector)
    else:
        return sitk.GetImageFromArray(image_np.astype(type), is_vector)


def sitk_list_to_np(image_list_sitk, type=None, axis=0):
    image_list_np = []
    for image_sitk in image_list_sitk:
        image_list_np.append(sitk_to_np(image_sitk, type))
    return np.stack(image_list_np, axis)
