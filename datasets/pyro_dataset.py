
import datetime
import multiprocessing
import threading
import time

import Pyro4
import numpy as np

from datasets.dataset_base import DatasetBase

try:
    import lz4.frame
except ModuleNotFoundError as e:
    print('lz4 module not found, lz4 compression not supported.')
    pass
try:
    import zfpy
except ModuleNotFoundError as e:
    print('zfpy module not found, zfpy compression not supported.')
    pass

Pyro4.config.COMPRESSION = False
Pyro4.config.SERIALIZER = 'pickle'
Pyro4.config.SERIALIZERS_ACCEPTED = {'pickle'}


@Pyro4.expose
class PyroServerDataset(DatasetBase):
    """
    Pyro server dataset that prefetches entry dicts for an encapsulated dataset (self.dataset).
    Needs to be derived in order to set the dataset and to set it as a Pyro object.
    """
    def __init__(self, queue_size=128, refill_queue_factor=0.0, n_threads=8, use_multiprocessing=False, compression_type=None):
        """
        Initializer.
        :param queue_size: The number of entries in the queue used for caching.
        :param refill_queue_factor: If the number of entries in the queue is less than queue_size*refill_queue_factor, the entry
                                    that is returned with get() will be again put into the end of the queue.
        :param n_threads: The number of prefetching threads.
        :param use_multiprocessing: If true, use processes instead of threads. May be faster, as it circumvents the GIL, but may also use much more memory, as less memory is shared.
        :param compression_type: The used compression type. None, 'lz4', or 'zfpy'.
        """
        self.queue_size = queue_size
        self.refill_queue_factor = refill_queue_factor
        self.n_threads = n_threads
        self.use_multiprocessing = use_multiprocessing
        self.compression_type = compression_type
        self.manager = multiprocessing.Manager()
        self.queue = self.manager.Queue(maxsize=self.queue_size)
        self.should_stop = self.manager.Value('b', False)
        self.threads = []
        self.dataset = None
        self.args = []
        self.kwargs = {}

    def __del__(self):
        self.stop_threads()

    def initialized_with_same_parameters(self, args, kwargs):
        """
        Return true, if the previous call of the PyroDataset was initialized with the same dataset args and kwargs.
        :param args: args that were used for initializing the internal dataset.
        :param kwargs: kwargs that were used for initializing the internal dataset.
        :return: True, if same args and kwargs, False otherwise.
        """
        return list(self.args) == list(args) and dict(self.kwargs) == dict(kwargs)

    def init_with_parameters(self, *args, **kwargs):
        """
        Method that gets called, after getting the Proxy object from the server.
        Overwrite, if needed.
        """
        pass

    def numpy_data_to_queue_data(self, numpy_data):
        """
        Convert numpy data to queue data that will be serialized through pyro.
        :param numpy_data: numpy array.
        :return: Tuple of bytes, shape, and dtype.
        """
        if self.compression_type == 'lz4':
            return lz4.frame.compress(numpy_data.tobytes()), numpy_data.shape, numpy_data.dtype
        elif self.compression_type == 'zfp':
            return zfpy.compress_numpy(numpy_data)
        return numpy_data.tobytes(), numpy_data.shape, numpy_data.dtype

    def dataset_entry_to_queue_entry(self, dataset_entry):
        """
        Convert all generators of the dataset entry to the queue entries that get transmitted through pyro.
        :param dataset_entry: The dataset entry.
        :return: A dict that contains the converted entries of dataset_entry['generators'].
        """
        queue_entry = {}
        for key, value in dataset_entry['generators'].items():
            queue_entry[key] = self.numpy_data_to_queue_data(value)
        return queue_entry

    def thread_main(self):
        """
        Main function of the prefetching threads.
        """
        print('PyroServerDataset thread start')
        while not self.should_stop.value:
            dataset_entry = self.dataset.get_next()
            queue_entry = self.dataset_entry_to_queue_entry(dataset_entry)
            if self.should_stop.value:
                break
            self.queue.put(queue_entry)
            print('put', self.queue.qsize(), datetime.datetime.now())
        print('PyroServerDataset thread stop')

    def start_threads(self):
        """
        Starts the prefetching threads.
        """
        self.should_stop.value = False
        for i in range(self.n_threads):
            if self.use_multiprocessing:
                thread = multiprocessing.Process(target=self.thread_main)
                np.random.seed(int(time.time() + i))
            else:
                thread = threading.Thread(target=self.thread_main)
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)

    def stop_threads(self, clear_queue=True):
        """
        Stops and joins the prefetching threads.
        """
        self.should_stop.value = True
        # clear queue to wake up sleeping threads
        if clear_queue:
            print('clearing queue')
            while not self.queue.empty():
                self.queue.get_nowait()
        else:
            for i in range(2 * len(self.threads)):
                self.queue.get_nowait()
        for thread in self.threads:
            thread.join()
        # clear queue once again, as threads may have added new entries before they stopped
        if clear_queue:
            print('clearing queue')
            while not self.queue.empty():
                self.queue.get_nowait()

    def get_next(self):
        """
        Returns the next entry dict. This function is called by PyroClientDataset().get_next()
        :return: The next entry dict of the internal queue.
        """
        print('get', self.queue.qsize(), datetime.datetime.now())
        return self.queue.get()


    def num_entries(self):
        """
        Not supported.
        """
        raise RuntimeError('num_entries() is not supported for PyroServerDataset.')

    def get(self, id_dict):
        """
        Not supported.
        """
        raise RuntimeError('get(id_dict) is not supported for PyroServerDataset. Use get_next() instead.')


class PyroClientDataset(DatasetBase):
    """
    Pyro client dataset that encapsulate a Pyro server dataset at the given uri.
    """
    def __init__(self, uri, compression_type=None, *args, **kwargs):
        """
        Gets the server dataset at the given URI and stops and starts its threads.
        :param uri: URI to connect to.
        :param compression_type: The used compression type. None, 'lz4', or 'zfpy'.
        :param args: Arguments passed to init_with_parameters.
        :param kwargs: Keyword arguments passed to init_with_parameters.
        """
        self.uri = uri
        self.server_dataset = Pyro4.Proxy(self.uri)
        self.compression_type = compression_type
        same_parameters = self.server_dataset.initialized_with_same_parameters(args, kwargs)
        self.server_dataset.stop_threads(clear_queue=not same_parameters)
        if not same_parameters:
            self.server_dataset.init_with_parameters(*args, **kwargs)
        self.server_dataset.start_threads()

    def queue_data_to_numpy_data(self, queue_data):
        """
        Convert queue data that was serialized through pyro to numpy data.
        :param queue_data: Queue data.
        :return: Numpy array.
        """
        if self.compression_type == 'lz4':
            return np.frombuffer(lz4.frame.decompress(queue_data[0]), dtype=queue_data[2]).reshape(queue_data[1])
        elif self.compression_type == 'zfp':
            return zfpy.decompress_numpy(queue_data)
        return np.frombuffer(queue_data[0], dtype=queue_data[2]).reshape(queue_data[1])

    def queue_entry_to_dataset_entry(self, queue_entry):
        """
        Convert the queue entries that were transmitted through pyro to the dataset entries.
        :param queue_entry: The queue entry.
        :return: A dict that contains the converted entries as dataset_entry['generators'].
        """
        dataset_entry = {'generators': {}}
        for key, value in queue_entry.items():
            dataset_entry['generators'][key] = self.queue_data_to_numpy_data(value)
        return dataset_entry

    def get_next(self):
        """
        Returns the next entry dict of the server_dataset.
        :return: The next entry dict of the server_dataset.
        """
        queue_entry = self.server_dataset.get_next()
        return self.queue_entry_to_dataset_entry(queue_entry)

    def num_entries(self):
        """
        Not supported.
        """
        raise RuntimeError('num_entries() is not supported for PyroClientDataset.')

    def get(self, id_dict):
        """
        Not supported.
        """
        raise RuntimeError('get(id_dict) is not supported for PyroClientDataset. Use get_next() instead.')
