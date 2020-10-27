
import numpy as np
from generators.generator_base import GeneratorBase


class ImageSizeGenerator(GeneratorBase):
    """
    Generator that uses an sitk image (or a list of sitk images) and an sitk transformation as an input and calculates an output size.
    """
    def __init__(self,
                 dim,
                 output_size,
                 output_spacing=None,
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
        :param valid_output_sizes: A list of valid output sizes per dimension (a list of lists). See output_size
                                   parameter for usage.
        :param args: Arguments passed to super init.
        :param kwargs: Keyword arguments passed to super init.
        """
        super(ImageSizeGenerator, self).__init__(*args, **kwargs)
        self.dim = dim
        self.output_size = output_size
        self.output_spacing = output_spacing or [1] * dim
        self.valid_output_sizes = valid_output_sizes

    def get(self, image=None, extent=None):
        """
        Calculates the output size for the given sitk image or an extent based on the parameters.
        :param image: The sitk image.
        :param extent: The extent.
        :return: The output size as a list.
        """
        output_size = []
        assert (image is not None) != (extent is not None) , 'Either image or extent must be set.'
        if extent is None:
            extent = [image.GetSize()[i] * image.GetSpacing()[i] for i in range(self.dim)]
        for i in range(self.dim):
            if self.output_size[i] is not None:
                # if the output size is fixed for the current dimension, use it.
                output_size.append(self.output_size[i])
            elif self.valid_output_sizes is not None and self.valid_output_sizes[i] is not None:
                # if the output size is None, but valid_output_sizes is not None,
                # use minimal valid_output_sizes such that the resampled image fits.
                size = int(np.ceil(extent[i] / self.output_spacing[i]))
                valid_size = None
                for valid_size in sorted(self.valid_output_sizes[i]):
                    if size < valid_size:
                        # break as soon as the image fits into the current size
                        break
                output_size.append(valid_size)
            else:
                # otherwise (output_size is None and valid_output_sizes is None), calculate the
                # output size such that the resampled image fits exactly
                size = int(np.ceil(extent[i] / self.output_spacing[i]))
                output_size.append(size)
        return output_size