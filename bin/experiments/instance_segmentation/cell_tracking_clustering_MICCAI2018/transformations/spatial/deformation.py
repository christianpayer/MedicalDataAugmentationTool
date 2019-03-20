
import SimpleITK as sitk
import numpy as np

from transformations.spatial.base import SpatialTransformBase
from utils.random import float_uniform

class Deformation(SpatialTransformBase):
    def __init__(self, dim):
        super(Deformation, self).__init__(dim)

    def get_deformation_transform(self,
                                  grid_nodes,
                                  origin,
                                  physical_dimensions,
                                  spline_order):
        """
        Prepare the deformation transform without executing it on an image.

        :param output_size: list of int
            the output size, one int for each dimension
        :param grid_nodes: int
            the number of grid nodes equally spaced on the image
        :param dimensions: int
            the number of spatial dimensions (3 for 3D, 2 for 2D)
        :param direction: list
            the direction (transform matrix) of the output space, defaults to the identity matrix
        :param spline_order: int
            the spline order, if None it will be equal to the input dimensions
        :return: a configured BSplineTransform for the specified output space
        """

        # set enough context around the volume you want to sample
        #transform_domain_origin = [-i for i in output_size]
        #transform_domain_fixed_physical_dimensions = [(i * 3) for i in output_size]

        mesh_size = [grid_node - spline_order for grid_node in grid_nodes]

        # set up the transform
        t = sitk.BSplineTransform(self.dim, spline_order)

        t.SetTransformDomainOrigin(origin)
        t.SetTransformDomainMeshSize(mesh_size)
        t.SetTransformDomainPhysicalDimensions(physical_dimensions)
        t.SetTransformDomainDirection(np.eye(self.dim).flatten())

        # return the transform
        return t


class CenteredInput(Deformation):
    def __init__(self,
                 dim,
                 grid_nodes,
                 deformation_value,
                 spline_order=3):
        super(CenteredInput, self).__init__(dim)
        self.grid_nodes = grid_nodes
        self.deformation_value = deformation_value
        self.spline_order = spline_order

    def get(self, **kwargs):
        """
        Deform the image randomly with a given set of parameters using ITK's BSplineTransform.

        :param input_image: ITK image
            the input image
        :param grid_nodes: list of ints
            the number of nodes per dimension on the output space
        :param deform_range: float, list of floats
            random deformation ranges (in mm)
        :param spline_order: int, default is same as image dimension
            the order of the b-spline
        :param output_size: list
            the output size of the image (required to compute transformations)
        :param direction: list
            the output direction of the image (transform matrix)
        :param kwargs:
            chain: if True returns the transform instead of the output image (default=False)
        :return:
        """
        input_image = kwargs.get('image')
        input_size = input_image.GetSize()
        input_spacing = input_image.GetSpacing()

        origin = [-input_size[i] * input_spacing[i] * 0.5 for i in range(self.dim)]
        physical_dimensions = [input_size[i] * input_spacing[i] for i in range(self.dim)]

        # get the transform params
        current_transformation = self.get_deformation_transform(self.grid_nodes,
                                                                     origin,
                                                                     physical_dimensions,
                                                                     self.spline_order)

        # define the displacement range in mm per control point
        # modify the parameters
        deform_params = [float_uniform(-self.deformation_value, self.deformation_value)
                         for _ in current_transformation.GetParameters()]
        # set them back to the transform
        current_transformation.SetParameters(deform_params)
        # output spacing is input spacing: --> assumed (1, 1[, 1])!

        return current_transformation

class Input(Deformation):
    def __init__(self,
                 dim,
                 grid_nodes,
                 deformation_value,
                 spline_order=3):
        super(Input, self).__init__(dim)
        self.grid_nodes = grid_nodes
        self.deformation_value = deformation_value
        self.spline_order = spline_order

    def get(self, **kwargs):
        """
        Deform the image randomly with a given set of parameters using ITK's BSplineTransform.

        :param input_image: ITK image
            the input image
        :param grid_nodes: list of ints
            the number of nodes per dimension on the output space
        :param deform_range: float, list of floats
            random deformation ranges (in mm)
        :param spline_order: int, default is same as image dimension
            the order of the b-spline
        :param output_size: list
            the output size of the image (required to compute transformations)
        :param direction: list
            the output direction of the image (transform matrix)
        :param kwargs:
            chain: if True returns the transform instead of the output image (default=False)
        :return:
        """
        #input_image = kwargs.get('image')
        #input_size = input_image.GetSize()
        #input_spacing = input_image.GetSpacing()

        input_size, input_spacing = self.get_image_size_spacing(**kwargs)

        origin = [0] * self.dim
        physical_dimensions = [input_size[i] * input_spacing[i] for i in range(self.dim)]

        # get the transform params
        current_transformation = self.get_deformation_transform(self.grid_nodes,
                                                                     origin,
                                                                     physical_dimensions,
                                                                     self.spline_order)

        # define the displacement range in mm per control point
        # modify the parameters
        deform_params = [float_uniform(-self.deformation_value, self.deformation_value)
                         for _ in current_transformation.GetParameters()]
        # set them back to the transform
        current_transformation.SetParameters(deform_params)
        # output spacing is input spacing: --> assumed (1, 1[, 1])!

        return current_transformation


class Output(Deformation):
    def __init__(self,
                 dim,
                 grid_nodes,
                 deformation_value,
                 output_size,
                 output_spacing=None,
                 spline_order=3):
        super(Output, self).__init__(dim)
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
        Deform the image randomly with a given set of parameters using ITK's BSplineTransform.

        :param input_image: ITK image
            the input image
        :param grid_nodes: list of ints
            the number of nodes per dimension on the output space
        :param deform_range: float, list of floats
            random deformation ranges (in mm)
        :param spline_order: int, default is same as image dimension
            the order of the b-spline
        :param output_size: list
            the output size of the image (required to compute transformations)
        :param direction: list
            the output direction of the image (transform matrix)
        :param kwargs:
            chain: if True returns the transform instead of the output image (default=False)
        :return:
        """
        origin = [0] * self.dim
        physical_dimensions = [self.output_size[i] * self.output_spacing[i] for i in range(self.dim)]

        # get the transform params
        current_transformation = self.get_deformation_transform(self.grid_nodes,
                                                                origin,
                                                                physical_dimensions,
                                                                self.spline_order)

        # define the displacement range in mm per control point
        # modify the parameters
        deform_params = [float_uniform(-self.deformation_value, self.deformation_value)
                         for _ in current_transformation.GetParameters()]
        # set them back to the transform
        current_transformation.SetParameters(deform_params)
        # output spacing is input spacing: --> assumed (1, 1[, 1])!

        return current_transformation

