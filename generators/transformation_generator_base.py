
from generators.generator_base import GeneratorBase
import transformations.spatial.common


class TransformationGeneratorBase(GeneratorBase):
    """
    Base class for a Generator that uses a main transformation. Used for
    image to image generators.
    """
    def __init__(self,
                 dim,
                 pre_transformation=None,
                 post_transformation=None,
                 *args, **kwargs):
        """
        Initializer.
        :param dim: The dimension.
        :param pre_transformation: A spatial transformation that will be applied before the main transformation.
        :param post_transformation: A spatial transformation that will be applied after the main transformation.
        :param args: Arguments passed to super init.
        :param kwargs: Keyword arguments passed to super init.
        """
        super(TransformationGeneratorBase, self).__init__(*args, **kwargs)
        self.dim = dim
        self.pre_transformation = pre_transformation
        self.post_transformation = post_transformation

    def get_transformation(self, base_transformation, **kwargs):
        """
        Returns the final composite sitk transformation for the given base_transformation and parameters.
        :param base_transformation: The sitk base transformation.
        :param kwargs: These arguments are given to the pre_transformation and post_transformation.
        :return: A composite sitk transformation.
        """
        if self.pre_transformation is None and self.post_transformation is None:
            return base_transformation
        transformation_list = []
        if self.pre_transformation is not None:
            transformation_list.append(self.pre_transformation.get(**kwargs))
        transformation_list.append(base_transformation)
        if self.post_transformation is not None:
            transformation_list.append(self.post_transformation.get(**kwargs))
        return transformations.spatial.common.create_composite(self.dim, transformation_list)
