import threading
import numpy as np
import tensorflow as tf
from collections import OrderedDict


class DataGeneratorBase(object):
    def __init__(self,
                 dataset,
                 coord,
                 data_names_and_shapes,
                 batch_size,
                 data_types=None,
                 queue_size=32,
                 n_threads=8):
        assert isinstance(data_names_and_shapes, OrderedDict) or isinstance(data_names_and_shapes, list), \
            'only OrderedDict and list are allowed for data_names_and_shapes'
        self.dataset = dataset
        if isinstance(data_names_and_shapes, OrderedDict):
            self.data_names_and_shapes = list(data_names_and_shapes.items())
        elif isinstance(data_names_and_shapes, list):
            self.data_names_and_shapes = data_names_and_shapes
        self.data_types = data_types
        if self.data_types is None:
            self.data_types = {}
        self.batch_size = batch_size
        self.queue_size = queue_size
        self.n_threads = n_threads
        self.coord = coord
        self.threads = []
        self.placeholders = None
        self.queue = None
        self.enqueue = None
        self.init_queue()

    def init_queue(self):
        raise NotImplementedError()

    def get_feed_dict(self):
        return NotImplementedError()

    def num_entries(self):
        return self.dataset.num_entries()

    def size(self):
        return self.queue.size()

    def dequeue(self):
        return self.queue.dequeue()

    def update(self):
        return None  # do nothing

    def get_feed_dict_batch(self):
        np_dicts = {}
        for i in range(len(self.data_names_and_shapes)):
            name = self.data_names_and_shapes[i][0]
            np_dicts[name] = []

        for batch_index in range(self.batch_size):
            dict = self.dataset.get_next()
            data_generators = dict['generators']
            for i in range(len(self.data_names_and_shapes)):
                name = self.data_names_and_shapes[i][0]
                np_dicts[name].append(data_generators[name])

        feed_dict = {}
        for i in range(len(self.data_names_and_shapes)):
            placeholder = self.placeholders[i]
            name = self.data_names_and_shapes[i][0]
            max_shape = np.max([a.shape for a in np_dicts[name]], axis=0)
            padded_values = []
            for a in np_dicts[name]:
                shape = a.shape
                padding = list(zip([0] * len(shape), (max_shape - shape)))
                padded_values.append(np.pad(a, padding, 'constant'))
            feed_dict[placeholder] = np.stack(padded_values)

        return feed_dict

    def get_feed_dict_single(self):
        dict = self.dataset.get_next()
        data_generators = dict['generators']
        feed_dict = {}
        for i in range(len(self.data_names_and_shapes)):
            placeholder = self.placeholders[i]
            name = self.data_names_and_shapes[i][0]
            feed_dict[placeholder] = data_generators[name]
            #print(data_generators[name].shape)

        return feed_dict

    def thread_main(self, sess):
        print('Data generator thread start')
        while not self.coord.should_stop():
            try:
                feed_dict = self.get_feed_dict()
                sess.run(self.enqueue, feed_dict=feed_dict)
            except Exception as e:
                # request stop, when there was an exception, but the threads should keep running
                if not self.coord.should_stop():
                    self.coord.request_stop(e)
                    self.close(sess)
                # otherwise, continue loop
                continue
        print('Data generator thread stop')

    def start_threads(self, sess):
        for _ in range(self.n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            self.coord.register_thread(thread)
            thread.start()
            self.threads.append(thread)
        return self.threads

    def close(self, sess):
        sess.run(self.queue.close(True))
