
import SimpleITK as sitk
import math

from graph.node import Node


class DisplacementField(Node):
    """
    Node that converts any sitk transformation to a displacement field transform for faster computation.
    """
    def __init__(self, output_size, output_spacing=None, sampling_factor=1, keep_transformation_size=False, *args, **kwargs):
        """
        Initializer.
        :param output_size: The output image size in pixels.
        :param output_spacing: The output image spacing in mm.
        :param sampling_factor: The sampling factor of the transform. If 1, every output pixel will be calculated.
                                If 2, every second pixel will be calculated, while the intermediate ones will be interpolated.
        :param keep_transformation_size: If True, remove last value from each dimension of the displacement field.
        :param args: Arguments passed to super init.
        :param kwargs: Keyword arguments passed to super init.
        """
        super(DisplacementField, self).__init__(*args, **kwargs)
        self.output_size = output_size
        self.output_spacing = output_spacing
        self.sampling_factor = sampling_factor
        self.keep_transformation_size = keep_transformation_size
        if self.output_spacing is None:
            self.output_spacing = [1] * len(self.output_size)

    def get(self, transformation, **kwargs):
        """
        Returns the sitk displacement field transform from the given transform.
        :param transformation: The sitk transformation from which to calculate the displacment field transform.
        :param kwargs: These parameters are given to self.get_output_center().
        :return: The sitk.DisplacementFieldTransform().
        """
        output_size = kwargs.get('output_size', self.output_size)
        displacement_field_size = [int(math.ceil((s + int(self.sampling_factor > 1)) / self.sampling_factor)) for s in output_size]
        displacement_field_spacing = [s * self.sampling_factor for s in self.output_spacing]
        displacement_field = sitk.TransformToDisplacementField(transformation, sitk.sitkVectorFloat64, size=displacement_field_size, outputSpacing=displacement_field_spacing)

        #Note: hacky implementation to counteract `+ int(self.sampling_factor > 1)`
        # `+ int(self.sampling_factor > 1)` is necessary to prevent artifacts at the borders
        # the following code removes the last entry to ensure same size as output_size
        if self.keep_transformation_size and self.sampling_factor > 1:
            displacement_field_np = sitk.GetArrayFromImage(displacement_field)
            if len(output_size) == 3:
                displacement_field_np = displacement_field_np[0:-1,0:-1,0:-1]
            elif len(output_size) == 2:
                displacement_field_np = displacement_field_np[0:-1,0:-1]
            else:
                raise NotImplementedError
            displacement_field = sitk.GetImageFromArray(displacement_field_np)
            # print('displacement_field', displacement_field_np.shape)

        displacement_field_transform = sitk.DisplacementFieldTransform(displacement_field)
        return displacement_field_transform
