
class GeneratorBase(object):
    """
    Generator Base class. Used as interface for generating np arrays from datasources.
    """
    def get(self, *args, **kwargs):
        """
        Generates a np array for the given parameters.
        :param args: See derived classes for possible parameters.
        :param kwargs: See derived classes for possible parameters.
        :return: A np array.
        """
        raise NotImplementedError

    def get_transformation(self, **kwargs):
        """
        Returns a sitk transformation for the given parameters.
        :param kwargs: See derived classes for possible parameters.
        :return: An sitk transformation.
        """
        return None
