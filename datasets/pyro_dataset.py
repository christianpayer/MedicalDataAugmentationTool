
from datasets.dataset_base import DatasetBase
import Pyro4
Pyro4.config.COMPRESSION = False
Pyro4.config.SERIALIZER = 'pickle'
Pyro4.config.SERIALIZERS_ACCEPTED = {'pickle'}
import queue
import threading
import multiprocessing
import datetime


@Pyro4.expose
class PyroServerDataset(DatasetBase):
    """
    Pyro server dataset that prefetches entry dicts for an encapsulated dataset (self.dataset).
    Needs to be derived in order to set the dataset and to set it as a Pyro object.
    """
    def __init__(self, queue_size=128, refill_queue_factor=0.5, n_threads=8, use_multiprocessing=False):
        """
        Initializer.
        :param queue_size: The number of entries in the queue used for caching.
        :param refill_queue_factor: If the number of entries in the queue is less than queue_size*refill_queue_factor, the entry
                                    that is returned with get() will be again put into the end of the queue.
        :param n_threads: The number of prefetching threads.
        :param use_multiprocessing: If true, use processes instead of threads. May be faster, as it circumvents the GIL, but may also use much more memory, as less memory is shared.
        """
        self.queue_size = queue_size
        self.refill_queue_factor = refill_queue_factor
        self.n_threads = n_threads
        self.use_multiprocessing = use_multiprocessing
        self.manager = multiprocessing.Manager()
        self.queue = self.manager.Queue(maxsize=self.queue_size)
        self.should_stop = self.manager.Value('b', False)
        self.threads = []
        self.dataset = None
        self.args = None
        self.kwargs = None

    def __del__(self):
        self.stop_threads()

    def initialized_with_same_parameters(self, *args, **kwargs):
        return self.args == args and self.kwargs == kwargs

    def init_with_parameters(self, *args, **kwargs):
        """
        Method that gets called, after getting the Proxy object from the server.
        Overwrite, if needed.
        """
        pass

    def thread_main(self):
        """
        Main function of the prefetching threads.
        """
        print('PyroServerDataset thread start')
        while not self.should_stop.value:
            entry = self.dataset.get_next()
            if 'datasources' in entry:
                del entry['datasources']
            if 'transformations' in entry:
                del entry['transformations']
            self.queue.put(entry)
            print('put', self.queue.qsize(), datetime.datetime.now())
        print('PyroServerDataset thread stop')

    def start_threads(self):
        """
        Starts the prefetching threads.
        """
        self.should_stop.value = False
        for _ in range(self.n_threads):
            if self.use_multiprocessing:
                thread = multiprocessing.Process(target=self.thread_main)
            else:
                thread = threading.Thread(target=self.thread_main)
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)

    def stop_threads(self):
        """
        Stops and joins the prefetching threads.
        """
        self.should_stop.value = True
        # clear queue to wake up sleeping threads
        while not self.queue.empty():
            self.queue.get_nowait()
        for thread in self.threads:
            thread.join()
        # clear queue once again, as threads may have added new entries before they stopped
        while not self.queue.empty():
            self.queue.get_nowait()

    def get_next(self):
        """
        Returns the next entry dict. This function is called by PyroClientDataset().get_next()
        :return: The next entry dict of the internal queue.
        """
        print('get', self.queue.qsize(), datetime.datetime.now())
        next = self.queue.get()
        if self.queue.qsize() < self.queue_size * self.refill_queue_factor:
            self.queue.put(next)
        return next

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
    def __init__(self, uri, *args, **kwargs):
        """
        Gets the server dataset at the given URI and stops and starts its threads.
        :param uri: URI to connect to.
        :param args: Arguments passed to init_with_parameters.
        :param kwargs: Keyword arguments passed to init_with_parameters.
        """
        self.uri = uri
        self.server_dataset = Pyro4.Proxy(self.uri)
        restart_threads = True  # not self.server_dataset.initialized_with_same_parameters(args, kwargs)
        if restart_threads:
            self.server_dataset.stop_threads()
        self.server_dataset.init_with_parameters(*args, **kwargs)
        if restart_threads:
            self.server_dataset.start_threads()

    def get_next(self):
        """
        Returns the next entry dict of the server_dataset.
        :return: The next entry dict of the server_dataset.
        """
        return self.server_dataset.get_next()

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
