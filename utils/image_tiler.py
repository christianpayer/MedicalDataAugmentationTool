
import numpy as np


class TilerBase(object):
    """
    Tiler base object that splits shapes into smaller (possibly overlapping) tiles. The class provides
    an interface for iterating over all tiles, as well as setting and getting the object on the current position.
    """
    def __init__(self, full_size, tiled_size, step_size):
        """
        Initializer.
        :param full_size: The full size of the object to iterate over.
        :param tiled_size: The cropped size of the object.
        :param step_size: The step size for each iteration.
        """
        assert len(full_size) == len(tiled_size), 'sizes must have the same dimension, are ' + str(full_size) + ', ' + str(tiled_size) + ', ' + str(step_size)
        assert len(full_size) == len(step_size), 'sizes must have the same dimension, are ' + str(full_size) + ', ' + str(tiled_size) + ', ' + str(step_size)
        self.dim = len(full_size)
        self.full_size = full_size
        self.cropped_size = tiled_size
        self.step_size = step_size
        self.current_tile = [0] * self.dim
        self.current_offset = None

    def reset(self):
        """
        Resets the Tiler. Must be called from within subclasses, when they overwrite reset().
        """
        self.current_offset = None

    def reset_current_offset(self):
        """
        Resets the current offset.
        """
        self.current_offset = []
        for i in range(self.dim):
            if self.cropped_size[i] > self.full_size[i]:
                self.current_offset.append(-(self.cropped_size[i] - self.full_size[i]) // 2)
            else:
                self.current_offset.append(0)

    def is_at_end(self):
        """
        Returns true, if the iterator is at the end position, false otherwise.
        :return: True, if the iterator is at the end position, false otherwise.
        """
        if self.current_offset is None:
            return False
        current_inc_dim = 0
        while True:
            if current_inc_dim >= self.dim:
                return True
            if self.cropped_size[current_inc_dim] > self.full_size[current_inc_dim]:
                # current dimension size is larger than the full size -> increment
                current_inc_dim += 1
                continue
            if self.current_offset[current_inc_dim] + self.cropped_size[current_inc_dim] == self.full_size[current_inc_dim]:
                # current dimension offset + cropped size size is equal to the full size -> increment
                current_inc_dim += 1
                continue
            break
        return False

    def increment(self):
        """
        Increments the current offset.
        """
        assert not self.is_at_end(), 'The tiler is already at the final position. Call reset() first, or use the __iter__ interface.'
        assert self.current_offset is not None, 'The tiler is not initialized. Call reset() first, or use the __iter__ interface.'
        current_inc_dim = 0
        while True:
            if current_inc_dim >= self.dim:
                raise RuntimeError('The tiler is already at end position.')
            if self.cropped_size[current_inc_dim] > self.full_size[current_inc_dim]:
                # current dimension size is larger than the full size -> increment
                current_inc_dim += 1
                continue
            if self.current_offset[current_inc_dim] + self.cropped_size[current_inc_dim] == self.full_size[current_inc_dim]:
                # current dimension offset + cropped size size is equal to the full size
                # -> set current dimension offset to 0 and increment dimension
                self.current_offset[current_inc_dim] = 0
                current_inc_dim += 1
                continue
            self.current_offset[current_inc_dim] += self.step_size[current_inc_dim]
            if self.current_offset[current_inc_dim] + self.cropped_size[current_inc_dim] > self.full_size[current_inc_dim]:
                # current dimension offset + cropped size size is larger to the full size -> change it  such that it is equal to full size
                self.current_offset[current_inc_dim] = self.full_size[current_inc_dim] - self.cropped_size[current_inc_dim]
            break

    def __iter__(self):
        """
        Return an iteratable object, i.e., calls reset() and returns self.
        :return: self
        """
        self.reset()
        return self

    def __next__(self):
        """
        Reset current offset (right after call to __iter__) or increment the current iteratable object.
        Raise StopIteration when self.is_at_end() == True.
        :return: self
        """
        if self.current_offset is None:
            self.reset_current_offset()
            return self
        if self.is_at_end():
            raise StopIteration
        self.increment()
        return self

    def get_current_data(self, **kwargs):
        """
        Abstract method for returning the data on the current offset.
        :param kwargs: Keyword arguments.
        :return: The data on the current offset.
        """
        raise NotImplementedError

    def set_current_data(self, **kwargs):
        """
        Abstract method for setting the data on the current offset.
        :param kwargs: Keyword arguments.
        """
        raise NotImplementedError


class ImageTiler(TilerBase):
    """
    Image tiler that allows to iterate over a larger image and implements get_current_data as well as set_current_data.
    """
    def __init__(self, full_size, tiled_size, step_size, create_output_image=False, default_pixel_value=0, output_image_dtype=np.float32):
        """
        Initializer.
        :param full_size: The full size of the object to iterate over.
        :param tiled_size: The cropped size of the object.
        :param step_size: The step size for each iteration.
        :param create_output_image: If true, also create an output image that will be used in set_current_data.
        :param default_pixel_value: The default pixel value of the output image.
        :param output_image_dtype: The dtype of the output image.
        """
        super(ImageTiler, self).__init__(full_size, tiled_size, step_size)
        self.create_output_image = create_output_image
        self.default_pixel_value = default_pixel_value
        self.output_image_dtype = output_image_dtype
        self.output_image = None

    def reset(self):
        """
        Resets the Tiler. Clears the output image.
        """
        super(ImageTiler, self).reset()
        if self.create_output_image:
            self.output_image = np.ones(self.full_size, dtype=self.output_image_dtype) * self.default_pixel_value

    def get_current_slices(self):
        """
        Return the current slices for the full image and the cropped image.
        :return: A tuple of the full_slice and a tuple of the tiled_slice
        """
        full_slice = []
        tiled_slice = []
        for i in range(self.dim):
            if self.cropped_size[i] > self.full_size[i]:
                full_slice.append(slice(None))
                tiled_slice.append(slice(-self.current_offset[i], -self.current_offset[i] + self.full_size[i]))
            else:
                full_slice.append(slice(self.current_offset[i], self.current_offset[i] + self.cropped_size[i]))
                tiled_slice.append(slice(None))
        return tuple(full_slice), tuple(tiled_slice)

    def get_current_data(self, image):
        """
        Return the current image data on the current offset.
        :param image: The image to crop the data from.
        :return: The image data on the current offset.
        """
        full_slice, tiled_slice = self.get_current_slices()
        output_image = np.ones(self.cropped_size) * self.default_pixel_value
        output_image[tiled_slice] = image[full_slice]
        return output_image

    def set_current_data(self, image, merge=np.maximum, merge_whole_image=False):
        """
        Set the image data on the current offset.
        :param image: The image that will be set on the current offset.
        :param merge: The merging function.
                      If merge_whole_image is true, the function has the following signature:
                          merge(self.output_image, image, full_slice, tiled_slice)
                      else:
                          merge(self.output_image[full_slice], image[tiled_slice])
        :param merge_whole_image: If true, calls the merging function on the whole image, otherwise only on the current tile.
        """
        assert self.create_output_image, 'create_output_image must be set to True, when calling set_current_data()'
        full_slice, tiled_slice = self.get_current_slices()
        if merge_whole_image:
            self.output_image = merge(self.output_image, image, full_slice, tiled_slice)
        else:
            self.output_image[full_slice] = merge(self.output_image[full_slice], image[tiled_slice])


class LandmarkTiler(TilerBase):
    """
    Tiler for landmarks. Subtracts the current_offset from the landmarks. Only allows get_current_data.
    """
    def get_current_data(self, landmarks):
        """
        Return the current landmark data on the current offset.
        :param landmarks: The landmarks.
        :return: The landmarks shifted by the current offset.
        """
        return landmarks[:, ...] - np.array(self.current_offset, np.float32)

    def set_current_data(self, **kwargs):
        """
        Abstract method for setting the data on the current offset.
        :param kwargs: Keyword arguments.
        """
        raise NotImplementedError
