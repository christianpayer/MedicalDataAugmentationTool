
import utils.io.landmark
from utils.landmark.common import Landmark
import os
from datasources.datasource_base import DataSourceBase

class LandmarkDataSource(DataSourceBase):
    """
    Datasource used for loading landmarks. Uses id_dict['image_id'] as the landmark file key and returns a list of landmarks.
    """
    def __init__(self,
                 point_list_file_name,
                 num_points,
                 dim,
                 silent_not_found=False,
                 id_dict_preprocessing=None):
        """
        Initializer.
        :param point_list_file_name: File that contains all the landmarks. Either a .csv file or a .idl file.
        :param num_points: Number of landmarks in the landmarks file.
        :param dim: Dimension of the landmarks.
        :param silent_not_found: If true, will return a list of invalid landmarks, in case of a not used key.
        :param id_dict_preprocessing: Function that will be called for preprocessing a given id_dict.
        """
        super(LandmarkDataSource, self).__init__(id_dict_preprocessing=id_dict_preprocessing)
        self.point_list_file_name = point_list_file_name
        self.num_points = num_points
        self.dim = dim
        self.silent_not_found = silent_not_found
        self.load()

    def load(self):
        """
        Loads the landmarks file. Either .idl or .csv.
        """
        ext = os.path.splitext(self.point_list_file_name)[1]
        if ext == '.csv':
            self.point_list = utils.io.landmark.load_csv(self.point_list_file_name, self.num_points, self.dim)
        if ext == '.idl':
            self.point_list = utils.io.landmark.load_idl(self.point_list_file_name, self.num_points, self.dim)

    def get_landmarks(self, image_id):
        """
        Returns the list of landmarks for a given image_id.
        :param image_id: The image_id.
        """
        try:
            return self.point_list[image_id]
        except KeyError:
            if self.silent_not_found:
                return [Landmark() for _ in range(self.num_points)]
            else:
                raise

    def get(self, id_dict):
        """
        Returns the list of landmarks for a given id_dict.
        :param id_dict: The id_dict. id_dict['image_id'] will be used as key for the landmarks file.
        :return: List of Landmarks().
        """
        id_dict = self.preprocess_id_dict(id_dict)
        image_id = id_dict['image_id']
        return self.get_landmarks(image_id)


class LandmarkDataSourceMultiple(DataSourceBase):
    """
    Datasource used for loading landmarks for images with possible multiple instances. Uses id_dict['image_id'] as the landmark file key and returns a list of landmarks.
    If multiple is True, all landmarks of all instances are returned. Otherwise, only the landmarks with the instance_id == id_dict['instance_id'] are returned.
    """
    def __init__(self,
                 multiple_point_list_file_name,
                 num_points,
                 dim,
                 multiple,
                 silent_not_found=False,
                 id_dict_preprocessing=None):
        """
        Initializer.
        :param multiple_point_list_file_name: File that contains all the landmarks for all instances. Must be a .csv file.
        :param num_points: Number of landmarks in the landmarks file.
        :param dim: Dimension of the landmarks.
        :param multiple: If true, all landmarks of all instances will be returned.
        :param silent_not_found: If true, will return a list of invalid landmarks, in case of a not used key.
        :param id_dict_preprocessing: Function that will be called for preprocessing a given id_dict.
        """
        super(LandmarkDataSourceMultiple, self).__init__(id_dict_preprocessing=id_dict_preprocessing)
        self.multiple_point_list_file_name = multiple_point_list_file_name
        self.num_points = num_points
        self.dim = dim
        self.multiple = multiple
        self.silent_not_found = silent_not_found
        self.load()

    def load(self):
        """
        Loads the landmarks file. Must be a .csv file.
        """
        ext = os.path.splitext(self.multiple_point_list_file_name)[1]
        if ext == '.csv':
            self.point_list = utils.io.landmark.load_multi_csv(self.multiple_point_list_file_name, self.num_points, self.dim)

    def get_landmarks(self, image_id, instance_id):
        """
        Returns the landmarks for a given image_id and optionally instance_id.
        :param image_id: Used as key for the landmarks file.
        :param instance_id: Used as key for the instance_id.
        :return: List of list of Landmarks(), if multiple is True. List of Landmarks(), otherwise.
        """
        try:
            if self.multiple:
                return list(self.point_list[image_id].values())
            else:
                #instance_id = kwargs.get('instance_id')
                return self.point_list[image_id][instance_id]
        except KeyError:
            if self.silent_not_found:
                if self.multiple:
                    return [[Landmark() for _ in range(self.num_points)]]
                else:
                    return [Landmark() for _ in range(self.num_points)]
            else:
                raise

    def get(self, id_dict):
        """
        Returns the landmarks for a given id_dict.
        :param id_dict: The id_dict. id_dict['image_id'] will be used as key for the landmarks file. id_dict['instance_id'] will be used as key for the instance_id.
        :return: List of list of Landmarks(), if multiple is True. List of Landmarks(), otherwise.
        """
        id_dict = self.preprocess_id_dict(id_dict)
        image_id = id_dict['image_id']
        instance_id = id_dict['instance_id']
        return self.get_landmarks(image_id, instance_id)
