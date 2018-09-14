
import tensorflow as tf
from tensorflow_train.data_generator_base import DataGeneratorBase

class DataGenerator(DataGeneratorBase):
    """
    Basic DataGenerator.
    """
    def init_queue(self):
        queue_types = []
        queue_shapes = []
        self.placeholders = []
        for (name, shape) in self.data_names_and_shapes:
            if name in self.data_types:
                types = self.data_types[name]
            else:
                types = tf.float32
            queue_shapes.append([self.batch_size] + shape)
            queue_types.append(types)
            self.placeholders.append(tf.placeholder(types, [self.batch_size] + shape, name='placeholder_' + name))

        self.queue = tf.FIFOQueue(self.queue_size, queue_types, queue_shapes)
        self.enqueue = self.queue.enqueue(self.placeholders)

    def get_feed_dict(self):
        return self.get_feed_dict_batch()
