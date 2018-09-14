
class DatasetBase(object):
    """
    Dataset Base class. Used as interface for generating entries.
    """
    def num_entries(self):
        """
        Returns the number of dataset entries, i.e. the number of individual images/volumes.
        :return The number of dataset entries.
        """
        raise NotImplementedError

    def get(self, id_dict):
        """
        Returns the generated entries for the given entry id.
        Most methods rely on the dataset generating a dictionary with the following entries:
        {'id', id_dict,
         'datasources': generated datasource values
         'generators': output of generators as np arrays}
        :param id_dict: The id dictionary that is used for generating data.
        :return: The dictionary of the generated data.
        """
        raise NotImplementedError

    def get_next(self):
        """
        Returns the next generated entries, based on the internal iterator.
        Calls get(id_dict) with the next id_dict from the internal iterator.
        :return: The dictionary of the generated data.
        """
        raise NotImplementedError
