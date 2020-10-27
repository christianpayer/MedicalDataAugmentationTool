
from iterators.id_list_iterator_base import IdListIteratorBase
import random
import multiprocessing


class IdListIterator(IdListIteratorBase):
    """
    Iterator over a list of ids that can be loaded either as a .txt or .csv file.
    """
    def __init__(self,
                 id_list_file_name,
                 random=False,
                 keys=None,
                 postprocessing=None,
                 whole_list_postprocessing=None,
                 use_shuffle=True,
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
        super(IdListIterator, self).__init__(id_list_file_name=id_list_file_name,
                                             keys=keys,
                                             postprocessing=postprocessing,
                                             whole_list_postprocessing=whole_list_postprocessing,
                                             *args, **kwargs)
        self.random = random
        self.use_shuffle = use_shuffle
        self.index_list = None
        self.lock = multiprocessing.Lock()
        self.id_list = []
        self.load()
        self.current_index = len(self.id_list)  # set to length of id_list to enforce shuffling at first call of get_next_id()

    def reset(self):
        """
        Resets the current index and shuffles the id list if self.random is True.
        Called internally when the internal iterator is at the end of the id_list.
        """
        self.index_list = list(range(len(self.id_list)))
        if self.random and self.use_shuffle:
            random.shuffle(self.index_list)
        self.current_index = 0

    def get_next_id(self):
        """
        Return the next id dictionary. The dictionary will contain all entries as defined by self.keys, as well as
        the entry 'unique_id' which joins all current entries.
        :return: The id dictionary.
        """
        with self.lock:
            if self.random and not self.use_shuffle:
                current_id_list = random.choice(self.id_list)
            else:
                if self.current_index >= len(self.id_list):
                    self.reset()
                current_id_list = self.id_list[self.index_list[self.current_index]]
                self.current_index += 1
            return self.current_dict_for_id_list(current_id_list)
