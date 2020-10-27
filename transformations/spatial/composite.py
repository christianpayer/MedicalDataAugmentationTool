
from transformations.spatial.base import SpatialTransformBase
from transformations.spatial.common import create_composite


class Composite(SpatialTransformBase):
    """
    A composite transformation consisting of multiple other consecutive transformations.
    """
    def __init__(self, dim, transformations, use_deprecated_behaviour=False, *args, **kwargs):
        """
        Initializer.
        :param dim: The dimension of the transform.
        :param transformations: List of other transformations.
        :param use_deprecated_behaviour: If true, args and kwargs in get are considered as being transformation parents,
                                         otherwise they are considered as being transformations themselves.
        :param args: Arguments passed to super init.
        :param kwargs: Keyword arguments passed to super init.
        """
        super(Composite, self).__init__(dim, *args, **kwargs)
        self.transformations = transformations
        for t in self.transformations:
            t.parents.extend(self.parents)
            t.kwparents.update(self.kwparents)
        self.parents.extend(self.transformations)
        self.use_deprecated_behaviour = use_deprecated_behaviour

    def get(self, *args, **kwargs):
        """
        Returns the composite sitk transform.
        :param kwargs: Optional parameters sent to the other transformations.
        :return: The composite sitk transform.
        """
        if self.use_deprecated_behaviour:
            transformations_list = [transformation.get(**kwargs) for transformation in self.transformations]
            return create_composite(self.dim, transformations_list)
        return create_composite(self.dim, args)
