
from iterators.iterator_base import IteratorBase
import utils.io.text
import os
from collections import OrderedDict


class IdListIteratorBase(IteratorBase):
    """
    Iterator over a list of ids that can be loaded either as a .txt or .csv file.
    """
    def __init__(self,
                 id_list_file_name,
                 keys=None,
                 postprocessing=None,
                 whole_list_postprocessing=None,
                 *args, **kwargs):
        """
        Initializer. Loads entries from the id_list_file_name (either .txt or .csv file). Each entry (entries) of a line of the file
        will be set to the entries of keys.
        Example:
          csv file line: 'i01,p02\n'
          keys: ['image', 'person']
          will result in the id dictionary: {'image': 'i01', 'person': 'p02'}
        :param id_list_file_name: The filename from which the id list is loaded. Either .txt or .csv file.
        :param random: If true, the id list will be shuffled before iterating.
        :param keys: The keys of the resulting id dictionary.
        :param postprocessing: Postprocessing function on the id dictionary that will be called after the id
                               dictionary is generated and before it is returned, i.e., return self.postprocessing(current_dict)
        :param whole_list_postprocessing: Postprocessing function on the loaded internal id_list id, i.e., return self.whole_list_postprocessing(self.id_list)
        :param use_shuffle: If True, shuffle id list and then iterate. Otherwise, randomly sample from the list.
        :param args: Arguments passed to super init.
        :param kwargs: Keyword arguments passed to super init.
        """
        super(IdListIteratorBase, self).__init__(*args, **kwargs)
        self.id_list_file_name = id_list_file_name
        self.keys = keys
        if self.keys is None:
            self.keys = ['image_id']
        self.postprocessing = postprocessing
        self.whole_list_postprocessing = whole_list_postprocessing
        self.id_list = []
        self.load()

    def load(self):
        """
        Loads the id_list_filename. Called internally when initializing.
        """
        ext = os.path.splitext(self.id_list_file_name)[1]
        if ext in ['.csv', '.txt']:
            self.id_list = utils.io.text.load_list_csv(self.id_list_file_name)
        assert [] not in self.id_list and '' not in self.id_list, 'Empty id in id list. Check the file ' + self.id_list_file_name
        if self.whole_list_postprocessing is not None:
            self.id_list = self.whole_list_postprocessing(self.id_list)
        print('loaded %i ids' % len(self.id_list))

    def num_entries(self):
        """
        Return the number of entries of the id_list.
        :return: The number of entries of the id_list.
        """
        return len(self.id_list)

    def current_dict_for_id_list(self, current_id_list):
        """
        Return current id dict for given id list.
        :param current_id_list: The entries of the current id.
        :return: The current id dict.
        """
        current_dict = OrderedDict(zip(self.keys, current_id_list))
        current_dict['unique_id'] = '_'.join(map(str, current_id_list))
        if self.postprocessing is not None:
            current_dict = self.postprocessing(current_dict)
        return current_dict

    def get_id_for_index(self, index):
        """
        Return the next id dictionary from an index.
        :param index: The index of the id_list.
        :return: The id dictionary.
        """
        current_id_list = self.id_list[index]
        return self.current_dict_for_id_list(current_id_list)

    def get_next_id(self):
        """
        Return the next id dictionary. Needs to be implemented in subclasses
        :return: The id dictionary.
        """
        raise NotImplementedError
