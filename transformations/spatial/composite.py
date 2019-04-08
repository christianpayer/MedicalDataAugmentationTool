import SimpleITK as sitk
from transformations.spatial.base import SpatialTransformBase


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
        compos = sitk.Transform(self.dim, sitk.sitkIdentity)
        for i in range(len(self.transformations)):
            compos.AddTransform(self.transformations[i].get(**kwargs))
        return compos
