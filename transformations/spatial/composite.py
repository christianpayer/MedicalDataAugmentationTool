
from transformations.spatial.base import SpatialTransformBase
from transformations.spatial.common import create_composite


class Composite(SpatialTransformBase):
    """
    A composite transformation consisting of multiple other consecutive transformations.
    """
    def __init__(self, dim, transformations, *args, **kwargs):
        """
        Initializer.
        :param dim: The dimension of the transform.
        :param transformations: List of other transformations.
        :param args: Arguments passed to super init.
        :param kwargs: Keyword arguments passed to super init.
        """
        super(Composite, self).__init__(dim, *args, **kwargs)
        self.transformations = transformations

    def get(self, **kwargs):
        """
        Returns the composite sitk transform.
        :param kwargs: Optional parameters sent to the other transformations.
        :return: The composite sitk transform.
        """
        transformations_list = [transformation.get(**kwargs) for transformation in self.transformations]
        return create_composite(self.dim, transformations_list)

