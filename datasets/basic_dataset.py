
from datasets.dataset_base import DatasetBase
import numpy as np
import utils.io.image
import utils.np_image
import os

class BasicDataset(DatasetBase):
    """
    Basic dataset consisting of multiple datasources, datagenerators and an iterator.
    """
    def __init__(self,
                 dim,
                 datasources,
                 data_generators,
                 data_generator_sources,
                 iterator,
                 all_generators_post_processing=None,
                 debug_image_folder=None,
                 debug_image_type='default'):
        """
        Initializer
            Example:
            data_sources = {'image_datasource': ImageDataSource(...),
                            'label_datasource': LabelDataSource(...)}
            data_generators = {'image_generator': ImageGenerator(...),
                               'label_generator': LabelDataGenerator(...)}
            data_generator_sources = {'image_generator': ('image', 'image_datasource'),
                                      'label_generator': ('label', 'label_datasource')}
            This example has 2 datasources and 2 data_generators. The sources and the generators are connected as follows:
            The output of 'image_datasource' is the input of the 'image' parameter of the function get() of 'image_generator'.
            The output of 'label_datasource' is the input of the 'label' parameter of the function get() of 'label_generator'.
        :param dim: image dimension
        :param datasources: dict of datasources
        :param data_generators: dict of data_generators
        :param data_generator_sources: dict of tuples that define the inputs to the data_generator get() functions
        :param all_generators_post_processing: function that will be called, if each data generator is calculated; takes generator dict as input and returns new generator dict
        :param iterator: iterator that generates next id tuples
        :param debug_image_folder: debug image folder for saving debug images
        :param debug_image_type: debug image output, 'default' - channels are additional dimension, 'gallery' - channels are saved in a tiled image next to each other
        """
        self.dim = dim
        self.datasources = datasources
        self.data_generators = data_generators
        self.data_generator_sources = data_generator_sources
        self.iterator = iterator
        self.all_generators_post_processing = all_generators_post_processing
        self.debug_image_folder = debug_image_folder
        self.debug_image_type = debug_image_type
        self.split_axis = 0

    def num_entries(self):
        """
        Returns the number of entries of the iterator.
        :return: the number of entries of the iterator
        """
        assert self.iterator is not None, 'iterator is not set'
        return self.iterator.num_entries()

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

    def get(self, id_dict):
        """
        Generates datasource_values and generated_values for a given id_dict.
        :param id_dict: dict of id for datasources
        :return: dict of datasource_values, generated_values and transformations for a given id_dict
            {'id', id_dict,
             'datasources': generated datasource values
             'generators': output of generators as np arrays}
        """
        # load data form datasources
        datasource_values = {}
        for key, datasource in self.datasources.items():
            datasource_values[key] = datasource.get(id_dict)

        # update generators
        generated_values = {}
        for key, data_generator in self.data_generators.items():
            kwarguments = dict([(key, datasource_values[value]) for key, value in self.data_generator_sources[key].items()])
            current_generated_values = data_generator.get(**kwarguments)
            generated_values[key] = current_generated_values

        if self.all_generators_post_processing is not None:
            generated_values = self.all_generators_post_processing(generated_values)

        entry = {'id': id_dict,
                 'datasources': datasource_values,
                 'generators': generated_values}

        self.save_debug_images(entry)

        return entry

    def get_next(self):
        """
        Returns the dict of id, datasources and datagenerators for the next id of the iterator.
        :return: see get(id_dict)
        """
        assert self.iterator is not None, 'iterator is not set'
        return self.get(self.iterator.get_next_id())