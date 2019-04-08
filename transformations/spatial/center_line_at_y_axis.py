
import numpy as np
import SimpleITK as sitk

from transformations.spatial.base import SpatialTransformBase


class CenterLineAtYAxis(SpatialTransformBase):
    """
    A composite transformation that centers a given line at the y axis. Used in the bone generators.
    """
    def __init__(self, dim, output_size, output_spacing, *args, **kwargs):
        """
        Initializer.
        :param dim: The dimension.
        :param output_size: The output image size in pixels.
        :param output_spacing: The output image spacing in mm.
        :param args: Arguments passed to super init.
        :param kwargs: Keyword arguments passed to super init.
        """
        super(CenterLineAtYAxis, self).__init__(dim, *args, **kwargs)
        self.output_size = output_size
        self.output_spacing = output_spacing

    def get(self, **kwargs):
        """
        Returns the sitk transform based on the given parameters.
        :param kwargs: Must contain 'image', which defines the input image.
                       Must contain 'line', which defines the lin in the input image to center and scale.
                       Must contain 'output_size' and 'output_spacing', which define the output image physical space.
        :return: The sitk.Transform().
        """
        if self.dim == 2:
            return self.get_2d(**kwargs)
        elif self.dim == 3:
            return self.get_3d(**kwargs)

    def get_2d(self, **kwargs):
        """
        Returns the sitk transform based on the given parameters. 2D implementation.
        :param kwargs: Must contain 'image', which defines the input image.
                       Must contain 'line', which defines the lin in the input image to center and scale.
                       Must contain 'output_size' and 'output_spacing', which define the output image physical space.
        :return: The sitk.Transform().
        """
        input_image = kwargs.get('image')
        line = kwargs.get('line')
        output_size = kwargs.get('output_size', self.output_size)
        output_spacing = kwargs.get('output_spacing', self.output_spacing)

        input_spacing = np.asarray(input_image.GetSpacing())
        # print("\nSpacing: %s" % spacing)

        # set points and direction vector
        start = line[0]
        end = line[1]
        vec = end - start

        # ###############################
        # compute the transformations
        # ###############################

        # 1. TRANSLATION
        # translate start point to origin
        # IMPORTANT: use spacing!
        translate_1 = sitk.AffineTransform(2)
        offset_vector_1 = list(start * input_spacing)
        # print("Offset vector 1 %s" % offset_vector_1)
        translate_1.Translate(offset=offset_vector_1)

        # 2. ROTATION
        # rotate such that the input line is parallel to the y axis
        rotate = sitk.AffineTransform(2)

        # rotate about the z axis (i.e. change the coordinates of the xy plane)
        angle = np.arctan2(vec[0], vec[1])
        # print("z-angle: %s" % angle001)
        rotate.Rotate(0, 1, float(angle))

        # 3. SCALE
        # scale by the length of the input line such that it fits into the y region size
        scale = sitk.AffineTransform(2)
        vec_length = np.linalg.norm(vec * input_spacing)
        scale_factor = vec_length / self.output_size[1]
        scale.Scale(scale_factor)

        # 4. TRANSLATE
        translate_2 = sitk.AffineTransform(2)

        # compute offset vector
        offset_vector_2 = [-output_size[0] * output_spacing[0] * 0.5, 0]
        translate_2.Translate(offset=offset_vector_2)

        # 6. COMPOSITE TRANSFORMATION
        compos = sitk.Transform(self.dim, sitk.sitkIdentity)
        compos.AddTransform(translate_1)
        #compos.AddTransform(scale)
        compos.AddTransform(rotate)
        compos.AddTransform(translate_2)
        #if do_deform is True:
        #    compos.AddTransform(deform)

        return compos

    def get_3d(self, **kwargs):
        """
        Returns the sitk transform based on the given parameters. 3D implementation.
        :param kwargs: Must contain 'image', which defines the input image.
                       Must contain 'line', which defines the lin in the input image to center and scale.
                       Must contain 'output_size' and 'output_spacing', which define the output image physical space.
        :return: The sitk.Transform().
        """
        input_image = kwargs.get('image')
        line = kwargs.get('line')
        output_size = kwargs.get('output_size', self.output_size)
        output_spacing = kwargs.get('output_spacing', self.output_spacing)

        input_spacing = np.asarray(input_image.GetSpacing())
        # print("\nSpacing: %s" % spacing)

        # set points and direction vector
        start = line[0]
        end = line[1]
        vec = end - start

        # ###############################
        # compute the transformations
        # ###############################

        # 1. TRANSLATION
        # translate start point to origin
        # IMPORTANT: use spacing!
        translate_1 = sitk.AffineTransform(3)
        offset_vector_1 = list(start * input_spacing)
        # print("Offset vector 1 %s" % offset_vector_1)
        translate_1.Translate(offset=offset_vector_1)

        # 2. ROTATION
        # rotate such that the input line is parallel to the y axis
        rotate_z = sitk.AffineTransform(3)
        rotate_x = sitk.AffineTransform(3)

        # rotate about the z axis (i.e. change the coordinates of the xy plane)
        angle001 = np.arctan2(vec[0],
                              vec[1])
        # print("z-angle: %s" % angle001)
        rotate_z.Rotate(0, 1, float(angle001))

        # transform start point of dir vector for correct second rotation angle
        dir_transformed = rotate_z.GetInverse().TransformPoint(vec.tolist())

        # rotate about the x axis (i.e. change the coordinates of the yz plane)
        # NB: rotations in yz plane are 0 for right clavicle, and -pi for left clavicle
        angle100 = np.arctan2(dir_transformed[2],
                              dir_transformed[1])
        # print("x-angle: %s" % angle100)
        rotate_x.Rotate(1, 2, angle100)

        # 3. SCALE
        # scale by the length of the input line such that it fits into the y region size
        #scale = sitk.AffineTransform(3)
        #vec_length = np.linalg.norm(vec * input_spacing)
        #scale_factor = vec_length / output_size[1]
        #scale.Scale(scale_factor)

        # 4. TRANSLATE
        translate_2 = sitk.AffineTransform(3)

        # compute offset vector
        offset_vector_2 = [
            - output_size[0] * output_spacing[0] * 0.5,  # x
            0,  # y
            - output_size[2] * output_spacing[2] * 0.5  # z
        ]
        translate_2.Translate(offset=offset_vector_2)

        # 6. COMPOSITE TRANSFORMATION
        compos = sitk.Transform(self.dim, sitk.sitkIdentity)
        compos.AddTransform(translate_1)
        #compos.AddTransform(scale)
        compos.AddTransform(rotate_z)
        compos.AddTransform(rotate_x)
        compos.AddTransform(translate_2)
        #if do_deform is True:
        #    compos.AddTransform(deform)

        return compos
