
class BaseGenerator(object):
    def get(self, *args, **kwargs):
        raise NotImplementedError

    def get_transformation(self, **kwargs):
        return None
