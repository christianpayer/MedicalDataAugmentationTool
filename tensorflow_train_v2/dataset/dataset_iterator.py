import tensorflow as tf

from tensorflow_train_v2.dataset.data_generator_base import DataGeneratorBase


class DatasetIterator(DataGeneratorBase):
    """
    DataGenerator that is based on the new tf.data input pipeline. Should be faster as other DataGenerators.
    """

    def __init__(self,
                 n_threads=None,
                 prefetch_to_device='/gpu:0',
                 *args, **kwargs):
        """
        Initializer.
        :param n_threads: The number of background threads. If None, then calculate n_threads based on CPU cores.
        :param prefetch_to_device: The device to copy the data to. '/gpu:0' should be save to use. If None, the prefetched
                                   data will be copied to the device not until being accessed.
        :param args: Arguments, see DataGeneratorBase.
        :param kwargs: Keyword arguments, see DataGeneratorBase.
        """
        super(DatasetIterator, self).__init__(*args, **kwargs)
        self.n_threads = n_threads or tf.data.experimental.AUTOTUNE
        self.prefetch_to_device = prefetch_to_device
        # NOTE: although the variable 'running' is accessed by multiple threads (inside get_dummy()),
        # it is not secured by a mutex, as it is only set at the initialization and unset at closing.
        self.running = True
        self.iterator_next = None
        self.init_pipeline()

    def init_pipeline(self):
        """
        Init tf.data pipeline.
        """
        # TODO: very hacky code. Dataset.from_generator only creates dummy objects ('dummy'), and map then 'maps' the dummy objects (which are never used)
        #  to the next data entries. Hence, map does the heavy calculations in order to support multithreading. As the python generator interface only supports
        #  generating one entry after the other, Dataset.from_generator will possibly never support better multithreading.
        #  See for more details https://stackoverflow.com/questions/47086599/parallelising-tf-data-dataset-from-generator
        #  Check if this can be changed to another interface in newer versions of tf.
        data_pipeline = tf.data.Dataset.from_generator(self.get_dummy, (tf.string,))
        data_pipeline = data_pipeline.map(self.get_next_pyfunc, num_parallel_calls=self.n_threads)
        if self.prefetch_to_device:
            data_pipeline = data_pipeline.apply(tf.data.experimental.copy_to_device(self.prefetch_to_device))
        data_pipeline = data_pipeline.prefetch(buffer_size=self.queue_size)
        self.iterator_next = iter(data_pipeline.batch(batch_size=self.batch_size, drop_remainder=True))

    def get_next_pyfunc(self, *args, **kwargs):
        """
        Returns a py_func tensor that generates and returns the next dataset entry.
        :param args: Not used.
        :param kwargs: Not used.
        :return: Tensor tuple with next dataset entry.
        """
        queue_types, queue_shapes = self.get_queue_types_and_shapes_tuples()
        entries_wo_shape = tf.py_function(self.get_next_data_entry, [], queue_types)
        entries = [tf.ensure_shape(entry, shape) for entry, shape in zip(entries_wo_shape, queue_shapes)]
        return tuple(entries)

    def get_dummy(self):
        """
        Return the dummy string tuple as the python generator interface.
        :return: The tuple ('dummy',) as generator.
        """
        while self.running:
            yield 'dummy',

    def get_next_data_entry(self):
        """
        Return the next dataset entry.
        :return: Next dataset entry.
        """
        current_dict = self.dataset.get_next()
        data_generators = current_dict['generators']
        np_data_dict = self.get_np_data(data_generators)
        return tuple(np_data_dict.values())

    def get_next(self):
        """
        Return the next entry of the iterable object.
        :return: Tuple of the next entry.
        """
        return next(self.iterator_next)

    def close(self):
        """
        Stop the iterator by ending the data generation loop and clearing the queue.
        """
        if not self.running:
            return

        self.running = False
        try:
            while True:
                self.get_next()
        except StopIteration:
            pass
