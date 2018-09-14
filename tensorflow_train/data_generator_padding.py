
import tensorflow as tf
from tensorflow_train.data_generator_base import DataGeneratorBase


class DataGeneratorPadding(DataGeneratorBase):
    """
    DataGenerator that supports shapes where some entries are None.
    """
    # def init_queue(self):
    #     self.queue = tf.PaddingFIFOQueue(self.queue_size,
    #                                      [tf.float32] * len(self.data_names_and_shapes),
    #                                      [shape for (name, shape) in self.data_names_and_shapes])
    #     self.placeholders = [tf.placeholder(tf.float32, shape, name='placeholder_' + name) for (name, shape) in self.data_names_and_shapes]
    #     self.enqueue = self.queue.enqueue(self.placeholders)

    def init_queue(self):
        queue_types = []
        queue_shapes = []
        self.placeholders = []
        for (name, shape) in self.data_names_and_shapes:
            if self.data_types is not None and name in self.data_types:
                types = self.data_types[name]
            else:
                types = tf.float32
            queue_shapes.append(shape)
            queue_types.append(types)
            self.placeholders.append(tf.placeholder(types, shape, name='placeholder_' + name))

        self.queue = tf.PaddingFIFOQueue(self.queue_size, queue_types, queue_shapes)
        self.enqueue = self.queue.enqueue(self.placeholders)

    def get_feed_dict(self):
        return self.get_feed_dict_single()

    def dequeue(self):
        return self.queue.dequeue_many(self.batch_size)
