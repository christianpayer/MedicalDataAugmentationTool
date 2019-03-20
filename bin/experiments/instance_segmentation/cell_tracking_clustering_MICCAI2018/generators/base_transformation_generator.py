
from generators.base_generator import BaseGenerator
import transformations.spatial.common

class BaseTransformationGenerator(BaseGenerator):
    def __init__(self,
                 dim,
                 pre_transformation,
                 post_transformation):
        self.dim = dim
        self.pre_transformation = pre_transformation
        self.post_transformation = post_transformation

    def get_transformation(self, base_transformation, **kwargs):
        if self.pre_transformation is None and self.post_transformation is None:
            return base_transformation
        transformation_list = []
        if self.pre_transformation is not None:
            transformation_list.append(self.pre_transformation.get(**kwargs))
        transformation_list.append(base_transformation)
        if self.post_transformation is not None:
            transformation_list.append(self.post_transformation.get(**kwargs))
        return transformations.spatial.common.create_composite(self.dim, transformation_list)

