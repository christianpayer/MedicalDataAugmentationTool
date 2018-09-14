
import numpy as np
from generators.generator_base import GeneratorBase


class LabelGenerator(GeneratorBase):
    """
    Generator that returns the label as a np array.
    """
    def __init__(self, postprocessing=None):
        """
        Initializer.
        :param postprocessing: Function that will be called on each output.
                               Takes a np array as intput and returns a np array.
        """
        self.postprocessing = postprocessing

    def get(self, label, **kwargs):
        """
        Converts a label to a np array.
        :param label: The label.
        :param kwargs: Not used.
        :return: The np array.
        """
        float_labels = np.array([float(label)], np.float32)
        if self.postprocessing is not None:
            float_labels = self.postprocessing(float_labels)
        return float_labels
