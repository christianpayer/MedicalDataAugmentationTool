import SimpleITK as sitk


def create_composite(dim, transformations):
    compos = sitk.Transform(dim, sitk.sitkIdentity)
    for transformation in transformations:
        compos.AddTransform(transformation)
    return compos


def flipped_dimensions(transformation, size):
    dim = len(size)
    start = [0.0] * dim
    transformed_start = transformation.TransformPoint(start)
    flipped = [False] * dim
    for i in range(dim):
        end = [0.0] * dim
        end[i] = size[i]
        transformed_end = transformation.TransformPoint(end)
        flipped[i] = transformed_start[i] > transformed_end[i]
    return flipped
