
class Node(object):
    """
    Node object of a computation graph that has a name, parents and implements get().
    """
    def __init__(self, name=None, parents=None, kwparents=None):
        """
        Initializer.
        :param name: The name.
        :param parents: A list of parents.
        :param kwparents: A keyword dictionary of parents.
        """
        self.name = name
        self.parents = parents or []
        self.kwparents = kwparents or dict()

    def set_parents(self, *parents, **kwparents):
        """
        Sets the parents.
        :param parents: Argument list of parents.
        :param kwparents: Keyword argument parents.
        """
        self.parents = parents
        self.kwparents = kwparents

    def get(self, *args, **kwargs):
        """
        Get function that returns the current object.
        :param args: Arguments.
        :param kwargs: Keyword arguments.
        :return: The current object.
        """
        raise NotImplementedError


class LambdaNode(Node):
    """
    Node object that calls a given function f when get is being called. Useful for postprocessing other Node's outputs.
    """
    def __init__(self, f, *args, **kwargs):
        """
        Initializer.
        :param f: The function object that will be called by get.
        :param args: Arguments passed to super init.
        :param kwargs: Keyword arguments passed to super init.
        """
        super(LambdaNode, self).__init__(*args, **kwargs)
        self.f = f

    def get(self, *args, **kwargs):
        """
        Calls f with the given arguments.
        :param args: Arguments passed to f.
        :param kwargs: Keyword arguments passed to f.
        :return: The returned object of f.
        """
        return self.f(*args, **kwargs)
