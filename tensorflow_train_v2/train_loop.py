import numpy as np
import tensorflow as tf

from utils.io.image import write_multichannel_np
from utils.io.text import save_string_txt
import os


class MainLoopBase(object):
    """
    Main loop class that handles initialization, training and testing loops.
    """
    def __init__(self):
        self.init_tf()
        self.batch_size = None
        self.learning_rate = None
        self.current_iter = 0
        self.first_iteration = True
        self.disp_iter = 1
        self.max_iter = None
        self.snapshot_iter = None
        self.test_iter = None
        self.test_initialization = True
        self.load_model_filename = None

        # TODO: put raise_on_nan_loss into loss_metric_logger_train
        self.raise_on_nan_loss = True
        self.loss_name_for_nan_loss_check = 'loss'

        # the following objects should be initialized in the corresponding init functions.
        self.model = None
        self.optimizer = None
        self.output_folder_handler = None
        self.checkpoint = None
        self.checkpoint_manager = None
        self.dataset_train = None
        self.dataset_train_iter = None
        self.dataset_val = None
        self.loss_metric_logger_train = None
        self.loss_metric_logger_val = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def init_tf(self):
        """
        Init tensorflow and set tf.config. This method is called before everything else in __init__.
        """
        # set memory growth
        physical_devices = tf.config.list_physical_devices('GPU')
        for physical_device in physical_devices:
            tf.config.experimental.set_memory_growth(physical_device, enable=True)

    def init_all(self):
        """
        Init all objects. Calls abstract init_*() functions.
        """
        self.init_model()
        self.init_optimizer()
        self.init_output_folder_handler()
        self.init_checkpoint()
        self.init_checkpoint_manager()
        self.init_datasets()
        self.init_loggers()

    def close_all(self):
        """
        Close all objects.
        """
        # clear the session, graph, and all variables, in case a previous train loop was already defined (e.g. in cross validation).
        tf.keras.backend.clear_session()
        # close data augmentation loop
        if self.dataset_train_iter is not None:
            self.dataset_train_iter.close()
        # undo redirection of stdout
        if self.output_folder_handler is not None:
            self.output_folder_handler.close()

    def load_model(self, model_filename=None, assert_consumed=False):
        """
        Load the model.
        """
        model_filename = model_filename or self.load_model_filename
        print('Restoring model ' + model_filename)
        status = self.checkpoint.restore(model_filename)
        if assert_consumed:
            status.assert_consumed()
        else:
            status.expect_partial()

    def save_model(self):
        """
        Save the model.
        """
        print('Creating snapshot...')
        save_path = self.checkpoint_manager.save(self.current_iter)
        print('Model saved in file ' + save_path)

    def print_training_parameters(self):
        """
        Print training parameters.
        """
        print('Training parameters:')
        if self.optimizer is not None:
            print('Optimizer: ', self.optimizer)
        if self.batch_size is not None:
            print('Batch size:', self.batch_size)
        if self.learning_rate is not None:
            print('Learning rate:', self.learning_rate)
        if self.max_iter is not None:
            print('Max iterations:', self.max_iter)
        if self.output_folder_handler is not None:
            print('Output folder:', self.output_folder_handler.folder_base())

    def run(self):
        """
        Init all and run train loop.
        """
        try:
            self.init_all()
            if self.load_model_filename is not None:
                self.load_model()
            print('Starting main loop')
            self.print_training_parameters()
            self.train()
        finally:
            self.close_all()

    def train(self):
        """
        Run the train loop.
        """
        while self.current_iter <= self.max_iter:
            # snapshot
            if (self.current_iter % self.snapshot_iter) == 0 and not self.first_iteration:
                self.save_model()
            # test
            if (self.current_iter % self.test_iter) == 0 and (self.test_initialization or not self.first_iteration):
                self.test()
            # do not train in last iteration
            if self.current_iter < self.max_iter:
                self.train_step()
                # display loss and save summary
                if self.loss_metric_logger_train is not None and (self.current_iter % self.disp_iter) == 0:
                    summary_values = self.loss_metric_logger_train.finalize(self.current_iter)
                    # check if current loss is nan and if training should be stopped
                    if self.raise_on_nan_loss and self.loss_name_for_nan_loss_check in summary_values:
                        if np.isnan(summary_values[self.loss_name_for_nan_loss_check]):
                            raise RuntimeError('\'{}\' is nan'.format(self.loss_name_for_nan_loss_check))
            self.current_iter += 1
            self.first_iteration = False

    def run_test(self):
        """
        Init all, load model and run test loop.
        """
        try:
            self.init_all()
            self.load_model()
            print('Starting main test loop')
            self.test()
        finally:
            self.close_all()

    def run_generate_training_images(self, num_images):
        """
        Init all, load model and run test loop.
        """
        try:
            self.init_datasets()
            for i in range(num_images):
                entry = self.dataset_train.get_next()
                image_id = entry['id']['image_id']
                for key, value in entry['generators'].items():
                    if len(value.shape) in [3, 4]:
                        write_multichannel_np(value, os.path.join('train_images', f'{i}_{image_id}_{key}.nii.gz'), data_format=self.data_format, image_type=np.float32)
                    elif len(value.shape) in [0, 1]:
                        save_string_txt(str(value), os.path.join('train_images', f'{i}_{image_id}_{key}.txt'))
        finally:
            self.close_all()


    def init_model(self):
        """
        Init self.model.
        """
        pass

    def init_optimizer(self):
        """
        Init self.optimizer.
        """
        pass

    def init_checkpoint(self):
        """
        Init self.checkpoint.
        """
        pass

    def init_checkpoint_manager(self):
        """
        Init self.checkpoint_manager.
        """
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.output_folder_handler.path('weights'), max_to_keep=None)

    def init_datasets(self):
        """
        Init self.dataset_train, self.dataset_train_iter, self.dataset_val.
        """
        pass

    def init_output_folder_handler(self):
        """
        Init self.output_folder_handler.
        """
        pass

    def init_loggers(self):
        """
        Init self.loss_metric_logger_train, self.loss_metric_logger_val.
        """
        pass

    def train_step(self):
        """
        Perform a training step.
        """
        raise NotImplementedError()

    def test(self):
        """
        Perform the testing loop.
        """
        raise NotImplementedError()
