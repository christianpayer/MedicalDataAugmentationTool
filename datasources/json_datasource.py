
import json
from datasources.datasource_base import DataSourceBase


class JsonDatasource(DataSourceBase):
    """
    Datasource used for loading json files. Uses id_dict['image_id'] as the key for loading the json entries from a given .json file.
    """
    def __init__(self,
                 json_file_name,
                 key='image_id',
                 *args, **kwargs):
        """
        Initializer.
        :param json_file_name: The .json file that will be loaded.
        :param key: The key in the id_dict to use.
        :param args: Arguments passed to super init.
        :param kwargs: Keyword arguments passed to super init.
        """
        super(JsonDatasource, self).__init__(*args, **kwargs)
        self.label_list_file_name = json_file_name
        self.key = key
        self.load()

    def load(self):
        """
        Loads the .json file and sets the internal dict.
        """
        with open(self.label_list_file_name, 'r') as f:
            self.json_file = json.load(f)

    def get(self, id_dict):
        """
        Returns the label for the given id_dict.
        :param id_dict: The id_dict. id_dict['image_id'] will be used as the key in the loaded dict.
        """
        id_dict = self.preprocess_id_dict(id_dict)
        image_id = id_dict[self.key]
        return self.json_file[image_id]
