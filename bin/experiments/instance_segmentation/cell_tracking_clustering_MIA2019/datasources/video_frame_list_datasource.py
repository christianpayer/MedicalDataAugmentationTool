
from datasources.datasource_base import DataSourceBase


class VideoFrameListDatasource(DataSourceBase):
    """
    Datasource used for loading a data for a list of frames (i.e., id_dicts). Loads data from a wrapped datasource for a given
    list of id_dicts. Returns data as a list for every entry of id_dicts.
    """
    def __init__(self,
                 wrapped_datasource,
                 postprocessing=None,
                 *args, **kwargs):
        """
        Initializer.
        :param wrapped_datasource: The wrapped datasource. get(id_dicts) calls wrapped_datasource.get(id_dict) for every entry in the id_dicts list.
        :param postprocessing: Postprocessing function that will be called on the resulting list of data.
        :param args: Arguments passed to super init.
        :param kwargs: Keyword arguments passed to super init.
        """
        super(VideoFrameListDatasource, self).__init__(*args, **kwargs)
        self.wrapped_datasource = wrapped_datasource
        self.postprocessing = postprocessing

    def get(self, id_dicts):
        """
        Returns as list of results of the wrapped_datasource for the list of given id_dicts.
        :param id_dicts: List of id_dicts.
        :return: List of data of the wrapped_datasource or result of the defined postprocessing function.
        """
        id_dicts = self.preprocess_id_dict(id_dicts)
        data_list = [self.wrapped_datasource.get(id_dict) for id_dict in id_dicts]
        if self.postprocessing is not None:
            return self.postprocessing(data_list)
        return data_list
