import multiprocessing
import threading
import time

import numpy as np
import tensorflow as tf

from tensorflow_train_v2.dataset.data_generator_base import DataGeneratorBase


class DatasetIteratorMultiprocessing(DataGeneratorBase):
    """
    DataGenerator that is based on the new tf.data input pipeline. Should be faster as other DataGenerators.
    """

    def __init__(self,
                 n_threads=None,
                 prefetch_to_device='/gpu:0',
                 use_multiprocessing=True,
                 *args, **kwargs):
        """
        Initializer.
        :param n_threads: The number of background threads. If None, then calculate n_threads based on CPU cores.
        :param prefetch_to_device: The device to copy the data to. '/gpu:0' should be save to use. If None, the prefetched
                                   data will be copied to the device not until being accessed.
        :param use_multiprocessing: If True, use multiprocessing for workers, otherwise use threads.
                                    Multiprocessing should be much faster, but will duplicate memory.
        :param args: Arguments, see DataGeneratorBase.
        :param kwargs: Keyword arguments, see DataGeneratorBase.
        """
        super(DatasetIteratorMultiprocessing, self).__init__(*args, **kwargs)
        self.n_threads = n_threads or tf.data.experimental.AUTOTUNE
        self.prefetch_to_device = prefetch_to_device
        # NOTE: although the variable 'running' is accessed by multiple threads (inside get_dummy()),
        # it is not secured by a mutex, as it is only set at the initialization and unset at closing.
        self.running = True
        self.iterator_next = None
        self.use_multiprocessing = use_multiprocessing
        # variables for managing background processes/threads
        self.manager = multiprocessing.Manager()
        self.queue = self.manager.Queue(maxsize=self.queue_size)
        self.should_stop = self.manager.Value('b', False)
        self.threads = []
        self.init_pipeline()
        self.start_workers()

    def init_pipeline(self):
        """
        Init tf.data pipeline.
        """
        queue_types, queue_shapes = self.get_queue_types_and_shapes_tuples()
        data_pipeline = tf.data.Dataset.from_generator(self.get_next_generator, queue_types, queue_shapes)
        if self.prefetch_to_device:
            data_pipeline = data_pipeline.apply(tf.data.experimental.copy_to_device(self.prefetch_to_device))
        data_pipeline = data_pipeline.prefetch(buffer_size=self.queue_size)
        self.iterator_next = iter(data_pipeline.batch(batch_size=self.batch_size, drop_remainder=True))

    def get_next_generator(self):
        """
        Return the next data entry from the queue as the python generator interface.
        :return: The data entry tuple as generator.
        """
        while self.running:
            error, data_entry_tuple = self.queue.get()
            if error:
                self.running = False
                raise error
            yield data_entry_tuple

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

        self.stop_workers()

    def worker_main(self):
        """
        Main function of the prefetching processes/threads.
        """
        while not self.should_stop.value:
            try:
                dataset_entry = self.dataset.get_next()
                if self.should_stop.value:
                    break
                data_generators = dataset_entry['generators']
                np_data_dict = self.get_np_data(data_generators)
                self.queue.put((None, tuple(np_data_dict.values())))
            except BaseException as e:
                self.queue.put((e, None))
                self.should_stop.value = True
                break

    def start_workers(self):
        """
        Start the prefetching threads.
        """
        self.should_stop.value = False
        for i in range(self.n_threads):
            if self.use_multiprocessing:
                thread = multiprocessing.Process(target=self.worker_main)
                np.random.seed(int(time.time() + i))
            else:
                thread = threading.Thread(target=self.worker_main)
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        self.running = True

    def stop_workers(self):
        """
        Stop and joins the prefetching threads.
        """
        self.should_stop.value = True
        while not self.queue.empty():
            self.queue.get_nowait()
        for thread in self.threads:
            thread.join()
        self.threads = []
        # clear queue once again, as threads may have added new entries before they stopped
        while not self.queue.empty():
            self.queue.get_nowait()
