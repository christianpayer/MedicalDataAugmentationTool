
class IntensityTransformBase(object):
    def get(self, image):
        raise NotImplementedError

    def update(self, **kwargs):
        raise NotImplementedError