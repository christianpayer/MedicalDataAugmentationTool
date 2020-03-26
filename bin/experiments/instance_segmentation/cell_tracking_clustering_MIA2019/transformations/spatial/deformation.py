
import SimpleITK as sitk
import numpy as np

from transformations.spatial.base import SpatialTransformBase
from utils.random import float_uniform


class Deformation(SpatialTransformBase):
    """
    The deformation spatial transformation base class. Randomly transforms points on an image grid and interpolates with splines.
    """
    @staticmethod
    def get_deformation_transform(dim,
                                  grid_nodes,
                                  origin,
                                  physical_dimensions,
                                  spline_order,
                                  deformation_value):
        """
        Returns the sitk transform based on the given parameters.
        :param dim: The dimension.
        :param grid_nodes: The number of grid nodes in each dimension.
        :param origin: The domain origin.
        :param physical_dimensions: The domain physical size.
        :param spline_order: The spline order.
        :param deformation_value: The maximum deformation value.
        :return: The sitk.BSplineTransform() with the specified parameters.
        """
        mesh_size = [grid_node - spline_order for grid_node in grid_nodes]

        t = sitk.BSplineTransform(dim, spline_order)
        t.SetTransformDomainOrigin(origin)
        t.SetTransformDomainMeshSize(mesh_size)
        t.SetTransformDomainPhysicalDimensions(physical_dimensions)
        t.SetTransformDomainDirection(np.eye(dim).flatten())

        if isinstance(deformation_value, list) or isinstance(deformation_value, tuple):
            deform_params = []
            for v in deformation_value:
                for i in range(int(np.prod(grid_nodes))):
                    deform_params.append(float_uniform(-v, v))
        else:
            deform_params = [float_uniform(-deformation_value, deformation_value)
                             for _ in t.GetParameters()]
        t.SetParameters(deform_params)

        return t


class CenteredInput(Deformation):
    """
    A deformation transformation in the input image physical domain. Randomly transforms points on an image grid and interpolates with splines.
    Before this transformation, the image must be centered at the origin.
    """
    def __init__(self,
                 dim,
                 grid_nodes,
                 deformation_value,
                 spline_order=3,
                 *args, **kwargs):
        """
        Initializer.
        :param dim: The dimension.
        :param grid_nodes: A list of grid nodes per dimension.
        :param deformation_value: The maximum deformation value.
        :param spline_order: The spline order.
        :param args: Arguments passed to super init.
        :param kwargs: Keyword arguments passed to super init.
        """
        super(CenteredInput, self).__init__(dim, *args, **kwargs)
        self.grid_nodes = grid_nodes
        self.deformation_value = deformation_value
        self.spline_order = spline_order

    def get(self, **kwargs):
        """
        Returns the sitk transform based on the given parameters.
        :param kwargs: Must contain either 'image', or 'input_size' and 'input_spacing', which define the input image physical space.
        :return: The sitk.BSplineTransform().
        """
        input_size, input_spacing = self.get_image_size_spacing(**kwargs)

        origin = [-input_size[i] * input_spacing[i] * 0.5 for i in range(self.dim)]
        physical_dimensions = [input_size[i] * input_spacing[i] for i in range(self.dim)]

        current_transformation = self.get_deformation_transform(self.dim,
                                                                self.grid_nodes,
                                                                origin,
                                                                physical_dimensions,
                                                                self.spline_order,
                                                                self.deformation_value)

        return current_transformation


class Input(Deformation):
    """
    A deformation transformation in the input image physical domain. Randomly transforms points on an image grid and interpolates with splines.
    Before this transformation, the image origin must be at the physical origin.
    """
    def __init__(self,
                 dim,
                 grid_nodes,
                 deformation_value,
                 spline_order=3,
                 *args, **kwargs):
        """
        Initializer.
        :param dim: The dimension.
        :param grid_nodes: A list of grid nodes per dimension.
        :param deformation_value: The maximum deformation value.
        :param spline_order: The spline order.
        :param args: Arguments passed to super init.
        :param kwargs: Keyword arguments passed to super init.
        """
        super(Input, self).__init__(dim, *args, **kwargs)
        self.grid_nodes = grid_nodes
        self.deformation_value = deformation_value
        self.spline_order = spline_order

    def get(self, **kwargs):
        """
        Returns the sitk transform based on the given parameters.
        :param kwargs: Must contain either 'image', or 'input_size' and 'input_spacing', which define the input image physical space.
        :return: The sitk.BSplineTransform().
        """
        input_size, input_spacing = self.get_image_size_spacing(**kwargs)

        origin = [0] * self.dim
        physical_dimensions = [input_size[i] * input_spacing[i] for i in range(self.dim)]

        current_transformation = self.get_deformation_transform(self.dim,
                                                                self.grid_nodes,
                                                                origin,
                                                                physical_dimensions,
                                                                self.spline_order,
                                                                self.deformation_value)

        return current_transformation


class Output(Deformation):
    """
    A deformation transformation in the output image physical domain. Randomly transforms points on an image grid and interpolates with splines.
    Before this transformation, the image origin must be at the physical origin.
    """
    def __init__(self,
                 dim,
                 grid_nodes,
                 deformation_value,
                 output_size,
                 output_spacing=None,
                 spline_order=3,
                 *args, **kwargs):
        """
        Initializer.
        :param dim: The dimension.
        :param grid_nodes: A list of grid nodes per dimension.
        :param deformation_value: The maximum deformation value.
        :param output_size: The output image size in pixels.
        :param output_spacing: The output image spacing in mm.
        :param spline_order: The spline order.
        :param args: Arguments passed to super init.
        :param kwargs: Keyword arguments passed to super init.
        """
        super(Output, self).__init__(dim, *args, **kwargs)
        self.grid_nodes = grid_nodes
        self.deformation_value = deformation_value
        self.output_size = output_size
        if output_spacing is not None:
            self.output_spacing = output_spacing
        else:
            self.output_spacing = [1] * self.dim
        self.spline_order = spline_order

    def get(self, **kwargs):
        """
        Returns the sitk transform based on the given parameters.
        :param kwargs: Must contain either 'image', or 'input_size' and 'input_spacing', which define the input image physical space.
        :return: The sitk.BSplineTransform().
        """
        origin = [0] * self.dim
        physical_dimensions = [self.output_size[i] * self.output_spacing[i] for i in range(self.dim)]

        current_transformation = self.get_deformation_transform(self.dim,
                                                                self.grid_nodes,
                                                                origin,
                                                                physical_dimensions,
                                                                self.spline_order,
                                                                self.deformation_value)

        return current_transformation
