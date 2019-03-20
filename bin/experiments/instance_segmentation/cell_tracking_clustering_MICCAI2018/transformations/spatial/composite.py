import SimpleITK as sitk
from transformations.spatial.base import SpatialTransformBase


class Composite(SpatialTransformBase):
    def __init__(self, dim, transformations):
        super(Composite, self).__init__(dim)
        self.transformations = transformations

    def get_flipped(self):
        flipped = [False] * self.dim
        for i in range(len(self.transformations)):
            current_flipped = self.transformations[i].get_flipped()
            for j in range(len(current_flipped)):
                if current_flipped[i]:
                    flipped[i] = not flipped[i]
        return flipped

    def get(self, **kwargs):
        compos = sitk.Transform(self.dim, sitk.sitkIdentity)
        for i in range(len(self.transformations)):
            compos.AddTransform(self.transformations[i].get(**kwargs))
        return compos