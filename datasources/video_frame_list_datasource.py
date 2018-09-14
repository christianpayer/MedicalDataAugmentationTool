
from datasources.datasource_base import DataSourceBase

class VideoFrameListDatasource(DataSourceBase):
    """
    Datasource used for loading a data for a list of frames (i.e., id_dicts). Loads data from a wrapped datasource for a given
    list of id_dicts. Returns data as a list for every entry of id_dicts.
    """
    def __init__(self,
                 wrapped_datasource,
                 id_dict_preprocessing=None):
        """
        Initializer.
        :param wrapped_datasource: The wrapped datasource. get(id_dicts) calls wrapped_datasource.get(id_dict) for every entry in the id_dicts list.
        :param id_dict_preprocessing: Function that will be called for preprocessing a given id_dict.
        """
        super(VideoFrameListDatasource, self).__init__(id_dict_preprocessing=id_dict_preprocessing)
        self.wrapped_datasource = wrapped_datasource

    def get(self, id_dicts):
        """
        Returns as list of results of the wrapped_datasource for the list of given id_dicts.
        :param id_dicts: List of id_dicts.
        :return: List of data of the wrapped_datasource.
        """
        id_dicts = self.preprocess_id_dict(id_dicts)
        return [self.wrapped_datasource.get(id_dict) for id_dict in id_dicts]
