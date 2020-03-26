
from datasets.debug_image_dataset import DebugImageDataset


class BasicDataset(DebugImageDataset):
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
                 *args, **kwargs):
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
        :param args: Arguments passed to super init.
        :param kwargs: Keyword arguments passed to super init.
        """
        super(BasicDataset, self).__init__(*args, **kwargs)
        self.dim = dim
        self.datasources = datasources
        self.data_generators = data_generators
        self.data_generator_sources = data_generator_sources
        self.iterator = iterator
        self.all_generators_post_processing = all_generators_post_processing

    def num_entries(self):
        """
        Returns the number of entries of the iterator.
        :return: the number of entries of the iterator
        """
        assert self.iterator is not None, 'iterator is not set'
        return self.iterator.num_entries()

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