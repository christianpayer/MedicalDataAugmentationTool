
from datasets.debug_image_dataset import DebugImageDataset
from graph.run_graph import run_graph


class GraphDataset(DebugImageDataset):
    """
    Dataset that runs a node graph.
    """
    def __init__(self,
                 data_generators,
                 data_sources=None,
                 transformations=None,
                 iterator=None,
                 *args, **kwargs):
        """
        Initializer.
        :param dim: image dimension
        :param data_generators: list of data_generators
        :param data_sources: list of datasources
        :param transformations: list of transformations
        :param iterator: iterator that generates next id tuples
        :param args: Arguments passed to super init.
        :param kwargs: Keyword arguments passed to super init.
        """
        super(GraphDataset, self).__init__(*args, **kwargs)
        self.data_generators = data_generators
        self.data_sources = data_sources
        self.transformations = transformations
        self.iterator = iterator

    def num_entries(self):
        """
        Returns the number of entries of the iterator.
        :return: the number of entries of the iterator
        """
        assert self.iterator is not None, 'iterator is not set'
        return self.iterator.num_entries()

    def get_graph_entries(self, feed_dict=None):
        """
        Runs the graph, i.e., data_generators, data_sources and transformations. Takes an optional feed_dict for precalculated values.
        :param feed_dict: Optional dict of precalculated values.
        :return: dict of datasource_values, generated_values and transformations.
            {'id', id_dict,
             'datasources': generated datasource values
             'generators': output of generators as np arrays
             'transformations': output of transformations as sitk transformations}
        """
        # create a list of all nodes to fetch
        all_fetches = [] + self.data_generators
        # optionally add data_sources and transformations
        if self.transformations is not None:
            all_fetches += self.transformations
        if self.data_sources is not None:
            all_fetches += self.data_sources
        if self.iterator is not None:
            all_fetches += [self.iterator]
        values = run_graph(all_fetches, feed_dict=feed_dict)

        entry = {}
        start_index = 0
        end_index = len(self.data_generators)
        entry['generators'] = dict([(node.name, value) for node, value in zip(self.data_generators, values[start_index:end_index])])
        start_index += end_index

        # optionally set datasources and transformations
        if self.transformations is not None:
            end_index = start_index + len(self.transformations)
            entry['transformations'] = dict([(node.name, value) for node, value in zip(self.transformations, values[start_index:end_index])])
            start_index = end_index
        if self.data_sources is not None:
            end_index = start_index + len(self.data_sources)
            entry['datasources'] = dict([(node.name, value) for node, value in zip(self.data_sources, values[start_index:end_index])])
            start_index = end_index
        if self.iterator is not None:
            entry['id'] = values[start_index]
            # save debug images can only be called, if entry contains a value for 'id'
            self.save_debug_images(entry)

        return entry

    def get(self, id_dict):
        """
        Generates datasource_values and generated_values for a given id_dict.
        :param id_dict: dict of id for datasources
        :return: dict of datasource_values, generated_values and transformations for a given id_dict
            {'id', id_dict,
             'datasources': generated datasource values
             'generators': output of generators as np arrays
             'transformations': output of transformations as sitk transformations}
        """
        assert self.iterator is not None, 'iterator is not set'
        return self.get_graph_entries({self.iterator: id_dict})

    def get_next(self):
        """
        Returns the dict of id, datasources and datagenerators for the next id of the iterator.
        :return: see get(id_dict)
        """
        return self.get_graph_entries()
