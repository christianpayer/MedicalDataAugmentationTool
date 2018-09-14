
from iterators.iterator_base import IteratorBase
import utils.io.text
import random
import multiprocessing
import os
from collections import OrderedDict


class IdListIterator(IteratorBase):
    """
    Iterator over a list of ids that can be loaded either as a .txt or .csv file.
    """
    def __init__(self,
                 id_list_file_name,
                 random=False,
                 keys=None,
                 postprocessing=None):
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
        """
        self.id_list_file_name = id_list_file_name
        self.random = random
        self.keys = keys
        if self.keys is None:
            self.keys = ['image_id']
        self.lock = multiprocessing.Lock()
        self.load()
        self.reset()
        self.postprocessing = postprocessing

    def load(self):
        """
        Loads the id_list_filename. Called internally when initializing.
        """
        ext = os.path.splitext(self.id_list_file_name)[1]
        if ext in ['.csv', '.txt']:
            self.id_list = utils.io.text.load_list_csv(self.id_list_file_name)
        print('loaded %i ids' % len(self.id_list))

    def reset(self):
        """
        Resets the current index and shuffles the id list if self.random is True.
        Called internally when initializing or when the internal iterator is at the end of the id_list.
        """
        self.index_list = list(range(len(self.id_list)))
        if self.random:
            random.shuffle(self.index_list)
        self.current_index = 0

    def num_entries(self):
        """
        Returns the number of entries of the id_list.
        :return: The number of entries of the id_list.
        """
        return len(self.id_list)

    def get_next_id(self):
        """
        Returns the next id dictionary. The dictionary will contain all entries as defined by self.keys, as well as
        the entry 'unique_id' which joins all current entries.
        :return: The id dictionary.
        """
        with self.lock:
            if self.current_index >= len(self.id_list):
                self.reset()
            current_id_list = self.id_list[self.index_list[self.current_index]]
            self.current_index += 1
            current_dict = OrderedDict(zip(self.keys, current_id_list))
            current_dict['unique_id'] = '_'.join(map(str, current_id_list))
            if self.postprocessing is not None:
                current_dict = self.postprocessing(current_dict)
            return current_dict
