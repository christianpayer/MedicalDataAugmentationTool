import os
from glob import glob

import SimpleITK as sitk

from utils.io.image import read, write


def main():
    """
    Convert bmp images to nii.gz images and set spacing to 0.1.
    """
    output_folder = 'images'
    bmp_filenames = glob('TODO_CHANGE_PATH/RawImage/*/*.bmp')
    for bmp_filename in sorted(bmp_filenames):
        image_id = os.path.splitext(os.path.basename(bmp_filename))[0]
        print(image_id)
        image = read(bmp_filename, sitk.sitkUInt8)
        image.SetSpacing([0.1, 0.1])
        write(image, os.path.join(output_folder, image_id + '.nii.gz'))


main()
