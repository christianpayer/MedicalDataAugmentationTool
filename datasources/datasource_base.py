
from graph.node import Node


class DataSourceBase(Node):
    """
    Datasource Base class. Used for loading data (e.g., images, labels, landmarks) for on a given id_dict.
    The loaded data will be used as input for the data generators.
    """
    def __init__(self, id_dict_preprocessing=None, *args, **kwargs):
        """
        Init function that sets member variables.
        :param id_dict_preprocessing: Function that will be called for id_dict preprocessing, i.e., actual_id_dict = id_dict_preprocessing(id_dict)
        :param args: Arguments passed to super init.
        :param kwargs: Keyword arguments passed to super init.
        """
        super(DataSourceBase, self).__init__(*args, **kwargs)
        self.id_dict_preprocessing = id_dict_preprocessing

    def get(self, id_dict):
        """
        Function that returns the corresponding data for a given id_dict.
        :param id_dict: The id_dict for the data to load.
        :return: The loaded data.
        """
        raise NotImplementedError

    def preprocess_id_dict(self, id_dict):
        """
        Function that preprocesses an id_dict. Calls self.id_dict_preprocessing(id_dict).
        :param id_dict: The id_dict to preprocess.
        :return: The preprocessed id_dict.
        """
        if self.id_dict_preprocessing is not None:
            return self.id_dict_preprocessing(id_dict)
        return id_dict