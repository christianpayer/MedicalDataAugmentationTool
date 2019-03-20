
from datasets.basic_dataset import BasicDataset


class ReferenceTransformationDataset(BasicDataset):
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
                 debug_image_type='default'):
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

    """
    Dataset consisting of multiple datasources, datagenerators a reference spatial transformation and an iterator.
    The reference segmentation is used for all datagenerators. Usually image to image networks need this dataset.
    For example image segmentation: input image and target segmentation have the same spatial transformation.
    """
    def get_reference_transformation(self, datasource_values):
        """
        Abstract method that returns the current spatial reference transformation for the given datasources.
        :param datasource_values: a dict of datasource values
        :return: reference sitk transform
        """
        raise NotImplementedError()

    def get_reference_datasource_dict(self, datasource_values):
        reference_datasource_dict = {}
        for reference_datasource_key_key, reference_datasource_key_value in self.reference_datasource_keys.items():
            current_value = datasource_values[reference_datasource_key_value]
            if isinstance(current_value, list):
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
