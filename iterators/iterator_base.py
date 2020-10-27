
from graph.node import Node


class IteratorBase(Node):
    """
    Iterator Base class. Used for iterating over ids.
    """
    def num_entries(self):
        """
        Returns the number of entries for the iterator.
        :return: The number of entries.
        """
        raise NotImplementedError

    def get_id_for_index(self, index):
        """
        Return the id entry dictionary for the given index
        :param index: The index in the dataset.
        :return: The id dictionary.
        """
        raise NotImplementedError

    def get_next_id(self):
        """
        Returns the next id entry dictionary. The dictionary may contain any entries.
        However, most of the datasources expect 'image_id' for loading and 'unique_id' for saving debug images.
        :return: The id dictionary.
        """
        raise NotImplementedError

    def get(self):
        """
        Calls get_next_id().
        :return: The result of get_next_id().
        """
        return self.get_next_id()
