
class SpatialTransformBase(object):
    """
    A generic spatial transform that can be applied to 2D and 3D images.
    """
    def __init__(self, dim):
        self.dim = dim

    def get(self, **kwargs):
        raise NotImplementedError

    def get_image_size_spacing(self, **kwargs):
        if 'image' in kwargs:
            assert not ('input_size' in kwargs or 'input_spacing' in kwargs), 'specify either image or input_size and input_spacing, but not both'
            input_image = kwargs.get('image')
            input_size = input_image.GetSize()
            input_spacing = input_image.GetSpacing()
            return input_size, input_spacing
        elif 'input_size' in kwargs and 'input_spacing' in kwargs:
            assert 'image' not in kwargs, 'specify either image or input_size and input_spacing, but not both'
            input_size = kwargs.get('input_size')
            input_spacing = kwargs.get('input_spacing')
            return input_size, input_spacing
        else:
            raise Exception('specify either image or input_size and input_spacing, but not both')