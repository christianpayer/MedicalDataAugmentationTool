
from datasets.dataset_base import DatasetBase
import numpy as np
import utils.io.image
import utils.np_image
import os


class DebugImageDataset(DatasetBase):
    """
    Basic dataset consisting of multiple datasources, datagenerators and an iterator.
    """
    def __init__(self,
                 debug_image_folder=None,
                 debug_image_type='default',
                 *args, **kwargs):
        """
        Initializer.
        :param debug_image_folder: debug image folder for saving debug images
        :param debug_image_type: debug image output, 'default' - channels are additional dimension, 'gallery' - channels are saved in a tiled image next to each other
        :param args: Arguments passed to super init.
        :param kwargs: Keyword arguments passed to super init.
        """
        super(DebugImageDataset, self).__init__(*args, **kwargs)
        self.debug_image_folder = debug_image_folder
        self.debug_image_type = debug_image_type
        # TODO: use split_axis based on channel index
        self.split_axis = 0

    def get_debug_image(self, image):
        """
        Returns the debug image from the given np array.
        if self.debug_image_type == 'default': channels are additional image dimension.
        elif self.debug_image_type == 'gallery': channels are saved in a tiled image next to each other.
        :param image: The np array from which the debug image should be created.
        :return: The debug image.
        """
        if self.debug_image_type == 'default':
            return image
        elif self.debug_image_type == 'gallery':
            split_list = np.split(image, image.shape[self.split_axis], axis=self.split_axis)
            split_list = [np.squeeze(split, axis=self.split_axis) for split in split_list]
            return utils.np_image.gallery(split_list)

    def save_debug_image(self, image, file_name):
        """
        Saves the given image at the given file_name. Images with 3 and 4 dimensions are supported.
        :param image: The np array to save.
        :param file_name: The file name where to save the image.
        """
        if len(image.shape) == 3:
            utils.io.image.write_np(image, file_name)
        if len(image.shape) == 4:
            utils.io.image.write_nd_np(image, file_name)

    def save_debug_images(self, entry_dict):
        """
        Saves all debug images for a given entry_dict, to self.debug_image_folder, if self.debug_image_folder is not None.
        All images of entry_dict['generators'] will be saved.
        :param entry_dict: The dictionary of the generated entries. Must have a key 'generators'.
        """
        if self.debug_image_folder is None:
            return

        generators = entry_dict['generators']

        for key, value in generators.items():
            if not isinstance(value, np.ndarray):
                continue
            if not len(value.shape) in [3, 4]:
                continue
            if isinstance(entry_dict['id'], list):
                id_dict = entry_dict['id'][0]
            else:
                id_dict = entry_dict['id']
            if 'unique_id' in id_dict:
                current_id = id_dict['unique_id']
            else:
                current_id = '_'.join(map(str, id_dict.values()))
            file_name = os.path.join(self.debug_image_folder, current_id + '_' + key + '.mha')
            image = self.get_debug_image(value)
            self.save_debug_image(image, file_name)
