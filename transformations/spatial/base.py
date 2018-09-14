
class SpatialTransformBase(object):
    """
    A generic spatial transform that can be applied to 2D and 3D images.
    """
    def __init__(self, dim):
        """
        Initializer.
        :param dim: The dimension of the transformation.
        """
        self.dim = dim

    def get(self, **kwargs):
        """
        Returns the actual sitk transfrom object with the current parameters.
        :param kwargs: Various arguments that may be used by the transformation, e.g., 'image', 'input_size, 'landmarks', etc.
        :return: sitk transform.
        """
        raise NotImplementedError

    def get_image_size_spacing(self, **kwargs):
        """
        Returns a tuple of (input_size, input_spacing) that is defined by the current kwargs.
        :param kwargs: The current image arguments. Either 'image', or 'input_size' and 'input_spacing'
                       'image': sitk image from which the size and spacing will be read.
                       'input_size': Input size tuple.
                       'input_spacing': Input spacing tuple.
        :return: (input_size, input_spacing) tuple.
        """
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
            raise RuntimeError('specify either image or input_size and input_spacing')