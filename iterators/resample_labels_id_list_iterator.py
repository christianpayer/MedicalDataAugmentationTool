
from iterators.id_list_iterator_base import IdListIteratorBase
import utils.io.text
import numpy as np
import multiprocessing
import os
from collections import OrderedDict


class ResampleLabelsIdListIterator(IdListIteratorBase):
    """
    Iterator over a list of ids that can be loaded either as a .txt or .csv file. Also uses a label list to resample
    the observed labels to the the given label probabilities.
    """
    def __init__(self,
                 id_list_file_name,
                 labels_file_name,
                 labels,
                 label_probabilities=None,
                 label_postprocessing=None,
                 keys=None,
                 postprocessing=None,
                 whole_list_postprocessing=None,
                 id_to_label_function=None,
                 *args, **kwargs):
        """
        Initializer. Loads entries from the id_list_file_name (either .txt or .csv file). Each entry (entries) of a line of the file
        will be set to the entries of keys.
        Example:
          csv file line: 'i01,p02\n'
          keys: ['image', 'person']
          will result in the id dictionary: {'image': 'i01', 'person': 'p02'}
        :param id_list_file_name: The id list filename.
        :param labels_file_name: The labels filename.
        :param labels: The possible labels of the labels file.
        :param label_probabilities: Dictionary of probabilities of how often a label should be taken in average. If None, use uniform probabilities.
        :param label_postprocessing: Postprocessing function for the loaded labels.
        :param keys: The keys of the resulting id dictionary.
        :param postprocessing: Postprocessing function on the id dictionary that will be called after the id
                               dictionary is generated and before it is returned, i.e., return self.postprocessing(current_dict)
        :param whole_list_postprocessing: Postprocessing function on the loaded internal id_list id, i.e., return self.whole_list_postprocessing(self.id_list)
        :param id_to_label_function: If not None, use this function to return the label of the current id_list entry.
        :param args: Arguments passed to super init.
        :param kwargs: Keyword arguments passed to super init.
        """
        super(ResampleLabelsIdListIterator, self).__init__(id_list_file_name=id_list_file_name,
                                                           keys=keys,
                                                           postprocessing=postprocessing,
                                                           whole_list_postprocessing=whole_list_postprocessing,
                                                           *args, **kwargs)
        self.labels_file_name = labels_file_name
        self.labels = labels
        self.label_probabilities = label_probabilities
        if self.label_probabilities is None:
            # create uniform distribution
            self.label_probabilities = dict([(label, 1.0 / len(self.labels)) for label in self.labels])
        self.label_postprocessing = label_postprocessing
        self.id_to_label_function = id_to_label_function or (lambda curr_id: self.id_labels[curr_id[0]])
        self.id_labels = {}
        self.load_labels_file()
        self.id_probabilities = []
        self.create_id_probabilities()

    def load_labels_file(self):
        """
        Loads the id_list and id_labels. Called internally when initializing.
        """
        if self.labels_file_name is not None:
            self.id_labels = utils.io.text.load_dict_csv(self.labels_file_name)

    def create_id_probabilities(self):
        """
        Calculate the probabilities for each id based on the labels and label_probabilities.
        """
        # calculate the number of entries per class in the id list
        entries_per_label = dict([(label, 0) for label in self.labels])
        for curr_id in self.id_list:
            curr_label = self.id_to_label_function(curr_id)
            if self.label_postprocessing is not None:
                curr_label = self.label_postprocessing(curr_label)
            entries_per_label[curr_label] += 1

        # calculate the probabilities
        for curr_id in self.id_list:
            curr_label = self.id_to_label_function(curr_id)
            if self.label_postprocessing is not None:
                curr_label = self.label_postprocessing(curr_label)
            curr_weight = self.label_probabilities[curr_label] / entries_per_label[curr_label]
            self.id_probabilities.append(curr_weight)

    def get_next_id(self):
        """
        Returns the next id dictionary. The dictionary will contain all entries as defined by self.keys, as well as
        the entry 'unique_id' which joins all current entries.
        :return: The id dictionary.
        """
        current_index = np.random.choice(list(range(len(self.id_list))), p=self.id_probabilities)
        return self.current_dict_for_id_list(self.id_list[current_index])
