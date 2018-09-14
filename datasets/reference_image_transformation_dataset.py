
from datasets.basic_dataset import BasicDataset


class ReferenceTransformationDataset(BasicDataset):
    """
    Dataset consisting of multiple datasources, datagenerators a reference spatial transformation and an iterator.
    The reference transformation is used for all datagenerators. Usually image to image networks need this dataset.
    This dataset is used for segmentation/localization tasks, where the generated outputs must have the same spatial transformations,
    e.g., for segmentation, the input and the mask image.
    """
    def __init__(self,
                 dim,
                 reference_datasource_keys,
                 reference_transformation,
                 datasources,
                 data_generators,
                 data_generator_sources,
                 iterator,
                 all_generators_post_processing=None,
                 debug_image_folder=None,
                 debug_image_type='default',
                 use_only_first_reference_datasource_entry=False):
        """
        Initializer.
            Example:
            reference_datasource_keys = {'image': 'image_datasource'}
            data_sources = {'image_datasource': ImageDataSource(...),
                            'segmentation_datasource': ImageDataSource(...)}
            data_generators = {'image_generator': ImageGenerator(...),
                               'segmentation_generator': ImageGenerator(...)}
            data_generator_sources = {'image_generator': ('image', 'image_datasource'),
                                      'segmentation_generator': ('image', 'segmentation_generator')}
            This example has 2 datasources and 2 data_generators. The sources and the generators are connected as follows:
            The output of 'image_datasource' is the input of the 'image' parameter of the function get() of 'image_generator'.
            The output of 'segmentation_datasource' is the input of the 'image' parameter of the function get() of 'segmentation_generator'.
            When the reference_transformation is initialized with get(), the 'image' parameter is set as the output of 'image_datasource'.
        :param dim: image dimension
        :param reference_datasource_keys: dict of strings that define the datasource inputs to the reference_transformation get() functions
        :param reference_transformation: the reference transformation that will be shared by all data_generators
        :param datasources: dict of datasources
        :param data_generators: dict of data_generators
        :param data_generator_sources: dict of tuples that define the inputs to the data_generator get() functions
        :param all_generators_post_processing: function that will be called, if each data generator is calculated; takes generator dict as input and returns new generator dict
        :param iterator: iterator that generates next id tuples
        :param debug_image_folder: debug image folder for saving debug images
        :param debug_image_type: debug image output, 'default' - channels are additional dimension, 'gallery' - channels are saved in a tiled image next to each other
        :param use_only_first_reference_datasource_entry: if true, extracts the first entry of the reference datasource entry before creating the transformation
        """
        super(ReferenceTransformationDataset, self).__init__(dim,
                                                             datasources,
                                                             data_generators,
                                                             data_generator_sources,
                                                             iterator,
                                                             all_generators_post_processing,
                                                             debug_image_folder,
                                                             debug_image_type)
        self.reference_datasource_keys = reference_datasource_keys
        self.reference_transformation = reference_transformation
        self.use_only_first_reference_datasource_entry = use_only_first_reference_datasource_entry

    def get_reference_datasource_dict(self, datasource_values):
        """
        Returns the reference datasource entries that will be given to the reference transformation.
        The resulting dictionary will have all keys defined by self.reference_datasource_keys, the values are the
        entries of datasource_values for the corresponding keys of self.reference_datasource_keys.
        :param datasource_values: All current datasource_values.
        :return: The reference datasource values.
        """
        reference_datasource_dict = {}
        for reference_datasource_key_key, reference_datasource_key_value in self.reference_datasource_keys.items():
            current_value = datasource_values[reference_datasource_key_value]
            if self.use_only_first_reference_datasource_entry:
                current_value = current_value[0]
            reference_datasource_dict[reference_datasource_key_key] = current_value
        return reference_datasource_dict

    def get(self, id_dict):
        """
        Generates datasource_values, generated_values and transformations for a given id_dict.
        :param id_dict: dict of id for datasources
        :return: dict of datasource_values, generated_values and transformations for a given id_dict
            {'id', id_dict,
             'datasources': generated datasource values
             'generators': output of generators as np arrays,
             'transformations': dict of transformations used for each generator}
        """
        # load data form datasources
        datasource_values = {}
        for key, datasource in self.datasources.items():
            datasource_values[key] = datasource.get(id_dict)

        reference_datasource_dict = self.get_reference_datasource_dict(datasource_values)
        base_transformation = self.reference_transformation.get(**reference_datasource_dict)

        # update generators
        generated_values = {}
        transformations = {}
        for key, data_generator in self.data_generators.items():
            kwarguments = dict([(key, datasource_values[value]) for key, value in self.data_generator_sources[key].items()])
            current_transformation = data_generator.get_transformation(base_transformation=base_transformation, **reference_datasource_dict)
            if current_transformation is not None:
                current_generated_values = data_generator.get(transformation=current_transformation, **kwarguments)
                transformations[key] = current_transformation
            else:
                current_generated_values = data_generator.get(**kwarguments)
            generated_values[key] = current_generated_values

        if self.all_generators_post_processing is not None:
            generated_values = self.all_generators_post_processing(generated_values)

        entry = {'id': id_dict,
                 'datasources': datasource_values,
                 'generators': generated_values,
                 'transformations': transformations}

        self.save_debug_images(entry)

        return entry
