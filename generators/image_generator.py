
import numpy as np
import utils.sitk_np
from generators.transformation_generator_base import TransformationGeneratorBase
from utils.sitk_image import resample


class ImageGenerator(TransformationGeneratorBase):
    """
    Generator that uses an sitk image (or a list of sitk images) and an sitk transformation as an input to generate a resampled np array.
    """
    def __init__(self,
                 dim,
                 output_size,
                 output_spacing=None,
                 post_processing_sitk=None,
                 post_processing_np=None,
                 interpolator='linear',
                 resample_sitk_pixel_type=None,
                 resample_default_pixel_value=None,
                 return_zeros_if_not_found=False,
                 data_format='channels_first',
                 np_pixel_type=np.float32,
                 valid_output_sizes=None,
                 *args, **kwargs):
        """
        Initializer.
        :param dim: The dimension.
        :param output_size: The resampled output image size in sitk format ([x, y] or [x, y, z]). May contain entries
                            that are None. In this case, the corresponding dimension will either take the smallest value
                            out of valid_output_sizes in which the resampled image fits. If valid_output_sizes is not
                            defined, the output size will be calculated, such that resampled image fits exactly in the
                            output image.
        :param output_spacing: The resampled output spacing.
        :param post_processing_sitk: A function that will be called after resampling the sitk image. This function
                                     must take a list of sitk images as input and return a list of sitk images.
        :param post_processing_np: A function that will be called after resampling the sitk image and converting it
                                   to a np array. This function must take a np array as input and return a np array.
        :param interpolator: The sitk interpolator string that will be used. See utils.sitk_image.get_sitk_interpolator
                             for possible values.
        :param resample_sitk_pixel_type: The sitk output pixel type of the resampling operation.
        :param resample_default_pixel_value: The default pixel value of pixel values that are outside the image region.
        :param return_zeros_if_not_found: If the sitk image is None and if return_zeros_if_not_found is True, a zero
                                          image will be returned. Otherwise if the sitk image is None, an exception
                                          will be thrown.
        :param data_format: The data format of the numpy array. If 'channels_first', the output np array will have
                            the shape [n_channels, (depth,) height, width]. If 'channels_last', the output np array
                            will have the shape [(depth,) height, width, n_channels].
        :param np_pixel_type: The output np pixel type.
        :param valid_output_sizes: A list of valid output sizes per dimension (a list of lists). See output_size
                                   parameter for usage.
        :param args: Arguments passed to super init.
        :param kwargs: Keyword arguments passed to super init.
        """
        super(ImageGenerator, self).__init__(dim=dim,
                                             *args, **kwargs)
        self.output_size = output_size
        self.output_spacing = output_spacing
        if self.output_spacing is None:
            self.output_spacing = [1] * dim
        self.interpolator = interpolator
        assert data_format == 'channels_first' or data_format == 'channels_last', 'unsupported data format'
        self.data_format = data_format
        self.post_processing_sitk = post_processing_sitk
        self.post_processing_np = post_processing_np
        self.resample_sitk_pixel_type = resample_sitk_pixel_type
        self.resample_default_pixel_value = resample_default_pixel_value
        self.return_zeros_if_not_found = return_zeros_if_not_found
        self.np_pixel_type = np_pixel_type
        self.valid_output_sizes = valid_output_sizes

    def get_output_size(self, image):
        """
        Calculates the output size for the given sitk image based on the parameters.
        :param image: The sitk image.
        :return: The output size as a list.
        """
        output_size = []
        for i in range(self.dim):
            if self.output_size[i] is not None:
                # if the output size is fixed for the current dimension, use it.
                output_size.append(self.output_size[i])
            elif self.valid_output_sizes is not None and self.valid_output_sizes[i] is not None:
                # if the output size is None, but valid_output_sizes is not None,
                # use minimal valid_output_sizes such that the resampled image fits.
                size = int(np.ceil(image.GetSize()[i] * image.GetSpacing()[i] / self.output_spacing[i]))
                for valid_size in sorted(self.valid_output_sizes[i]):
                    if size < valid_size:
                        # break as soon as the image fits into the current size
                        break
                output_size.append(valid_size)
            else:
                # otherwise (output_size is None and valid_output_sizes is None), calculate the
                # output size such that the resampled image fits exactly
                size = int(np.ceil(image.GetSize()[i] * image.GetSpacing()[i] / self.output_spacing[i]))
                output_size.append(size)
        return output_size

    def get_resampled_images(self, images, transformation):
        """
        Transforms the given sitk image (or list of sitk images) with the given transformation.
        :param images: The sitk image (or list of sitk images).
        :param transformation: The sitk transformation.
        :return: The resampled sitk image (or list of sitk images).
        """
        if isinstance(images, list) or isinstance(images, tuple):
            return [self.get_resampled_image(image, transformation) for image in images]
        else:
            return self.get_resampled_image(images, transformation)

    def get_resampled_image(self, image, transformation):
        """
        Transforms the given sitk image with the given transformation.
        :param images: The sitk image.
        :param transformation: The sitk transformation.
        :return: The resampled sitk image.
        """
        output_size = self.get_output_size(image)
        output_image = resample(image,
                                transformation,
                                output_size,
                                self.output_spacing,
                                interpolator=self.interpolator,
                                output_pixel_type=self.resample_sitk_pixel_type,
                                default_pixel_value=self.resample_default_pixel_value)
        return output_image

    def get_np_image_list(self, output_image_sitk):
        """
        Converts an sitk image to a list of np arrays, depending on the number of
        pixel components (RGB or grayscale).
        :param output_image_sitk: The sitk image to convert.
        :return: A list of np array.
        """
        output_image_list_np = []
        output_image_np = utils.sitk_np.sitk_to_np(output_image_sitk, self.np_pixel_type)
        pixel_components = output_image_sitk.GetNumberOfComponentsPerPixel()
        if pixel_components > 1:
            for i in range(pixel_components):
                output_image_list_np.append(output_image_np[..., i])
        else:
            output_image_list_np.append(output_image_np)
        return output_image_list_np

    def get_np_image(self, output_image_sitk):
        """
        Converts an sitk image (or a list of sitk images) to a np array, where the entries of
        output_image_sitk are used as channels.
        :param output_image_sitk: The sitk image (or list of sitk images).
        :return: A single np array that contains all entries of output_image_sitk as channels.
        """
        output_image_list_np = []
        if isinstance(output_image_sitk, list):
            for current_output_image_sitk in output_image_sitk:
                output_image_list_np.extend(self.get_np_image_list(current_output_image_sitk))
        else:
            output_image_list_np = self.get_np_image_list(output_image_sitk)

        if self.data_format == 'channels_first':
            output_image_np = np.stack(output_image_list_np, axis=0)
        elif self.data_format == 'channels_last':
            output_image_np = np.stack(output_image_list_np, axis=self.dim)

        return output_image_np

    def get(self, image, transformation, **kwargs):
        """
        Uses the sitk image and transformation to generate a resampled np array.
        :param image: The sitk image.
        :param transformation: The sitk transformation.
        :param kwargs: Not used.
        :return: The resampled np array.
        """
        if image is None and self.return_zeros_if_not_found:
            if self.data_format == 'channels_first':
                output_image_np = np.zeros([1] + list(reversed(self.output_size)), self.np_pixel_type)
            else: # if self.data_format == 'channels_last':
                output_image_np = np.zeros(list(reversed(self.output_size)) + [1], self.np_pixel_type)
        else:
            output_image_sitk = self.get_resampled_images(image, transformation)

            if self.post_processing_sitk is not None:
                output_image_sitk = self.post_processing_sitk(output_image_sitk)

            # convert to np array
            output_image_np = self.get_np_image(output_image_sitk)

        if self.post_processing_np is not None:
            output_image_np = self.post_processing_np(output_image_np)

        return output_image_np
