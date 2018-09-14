
import utils.io.text
from datasources.datasource_base import DataSourceBase

class LabelDatasource(DataSourceBase):
    """
    Datasource used for loading labels. Uses id_dict['image_id'] as the key for loading the label from a given .csv file.
    The structure of the csv file is as follows:
    'key_0', 'label_0'
    'key_1', 'label_1'
    ...
    """
    def __init__(self,
                 label_list_file_name,
                 id_dict_preprocessing=None):
        """
        Initializer.
        :param label_list_file_name: The .csv file that will be loaded.
        :param id_dict_preprocessing: Function that will be called for preprocessing a given id_dict.
        """
        super(LabelDatasource, self).__init__(id_dict_preprocessing=id_dict_preprocessing)
        self.label_list_file_name = label_list_file_name
        self.load()

    def load(self):
        """
        Loads the .csv file and sets the internal label dict.
        """
        self.label_list = utils.io.text.load_dict_csv(self.label_list_file_name)

    def get(self, id_dict):
        """
        Returns the label for the given id_dict.
        :param id_dict: The id_dict. id_dict['image_id'] will be used as the key in the loaded label dict.
        """
        id_dict = self.preprocess_id_dict(id_dict)
        image_id = id_dict['image_id']
        return self.label_list[image_id]