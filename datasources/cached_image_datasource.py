
import SimpleITK as sitk
from datasources.image_datasource import ImageDataSource
from cachetools import LRUCache
import re
from threading import Lock

class CachedImageDataSource(ImageDataSource):
    """
    DataSource used for loading sitk images. Uses id_dict['image_id'] as image path and returns the sitk_image at the given path.
    Supports caching for holding the images in memory.
    Preprocesses the path as follows: file_path_to_load = os.path.join(root_location, file_prefix + id_dict['image_id'] + file_suffix + file_ext)
    FIXME: has some problems when doing cross validation, i.e., memory is sometimes not freed.
    """
    def __init__(self,
                 root_location,
                 file_prefix='',
                 file_suffix='',
                 file_ext='.mha',
                 id_dict_preprocessing=None,
                 set_identity_spacing=False,
                 set_zero_origin=True,
                 set_identity_direction=True,
                 round_spacing_precision=None,
                 preprocessing=None,
                 sitk_pixel_type=sitk.sitkInt16,
                 return_none_if_not_found=False,
                 cache_maxsize=8192):
        """
        Initializer.
        :param root_location: Root path, where the images will be loaded from.
        :param file_prefix: Prefix of the file path.
        :param file_suffix: Suffix of the file path.
        :param file_ext: Extension of the file path.
        :param id_dict_preprocessing: Function that will be called for preprocessing a given id_dict.
        :param set_identity_spacing: If true, the spacing of the sitk image will be set to 1 for every dimension.
        :param set_zero_origin: If true, the origin of the sitk image will be set to 0 for every dimension.
        :param set_identity_direction: If true, the direction of the sitk image will be set to 1 for every dimension.
        :param round_spacing_precision: If > 0, spacing will be rounded to this precision (as in round(x, round_spacing_origin_direction))
        :param preprocessing: Function that will be called for preprocessing a loaded sitk image, i.e., sitk_image = preprocessing(sitk_image)
        :param sitk_pixel_type: sitk pixel type to which the loaded image will be converted to.
        :param return_none_if_not_found: If true, instead of raising an exception, None will be returned, if the image at the given path could not be loaded.
        :param cache_maxsize: Max size of cache in MB.
        """
        super(CachedImageDataSource, self).__init__(root_location=root_location,
                                                    file_prefix=file_prefix,
                                                    file_suffix=file_suffix,
                                                    file_ext=file_ext,
                                                    id_dict_preprocessing=id_dict_preprocessing,
                                                    set_identity_spacing=set_identity_spacing,
                                                    set_zero_origin=set_zero_origin,
                                                    set_identity_direction=set_identity_direction,
                                                    round_spacing_precision=round_spacing_precision,
                                                    preprocessing=preprocessing,
                                                    sitk_pixel_type=sitk_pixel_type,
                                                    return_none_if_not_found=return_none_if_not_found)
        self.cache = LRUCache(cache_maxsize, getsizeof=self.image_size)
        self.cache.__missing__ = self.load_and_preprocess
        self.lock = Lock()

    def image_size(self, image):
        """
        Returns the image size in MB. Used for calculating the current cache size.
        :param image: The sitk image or a list of sitk images.
        :return: The size of the image in MB.
        """
        # it could be the case that the image is None (if self.return_none_if_not_found == True)
        if image is None:
            return 0
        reference_image = image
        scale_factor = 1
        if isinstance(image, list) or isinstance(image, tuple):
            reference_image = image[0]
            scale_factor = len(image)
        try:
            # ugly silent catch, but the calculated size is only an estimate and we do not care
            num_bits_per_pixel = int(re.search('\d+', reference_image.GetPixelIDTypeAsString())[0])
        except:
            # fallback, if something went wrong (regular expression, unknown pixel id string)
            num_bits_per_pixel = 8
        total_num_bits = reference_image.GetNumberOfPixels() * reference_image.GetNumberOfComponentsPerPixel() * num_bits_per_pixel * scale_factor
        return total_num_bits / 8 / 1024 / 1024

    def get(self, id_dict):
        """
        Returns the cached image for a given id_dict. If the image is not in the cache, loads and processes the image for the given id_dict. Returns the sitk image.
        :param id_dict: The id_dict. id_dict['image_id'] will be used as the path for loading the sitk image.
        :return: The loaded and processed sitk image.
        """
        id_dict = self.preprocess_id_dict(id_dict)
        image_id = id_dict['image_id']
        with self.lock:
            return self.cache[image_id]
