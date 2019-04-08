
from generators.generator_base import GeneratorBase
import numpy as np


class VideoFrameListGenerator(GeneratorBase):
    """
    Generator that calls get() of a wrapped generator with individual given list entries. Used for videos.
    """
    def __init__(self,
                 wrapped_generator,
                 stack_axis=1,
                 post_processing_np=None,
                 *args, **kwargs):
        """
        Initializer.
        :param wrapped_generator: The wrapped data generator.
        :param stack_axis: Axis where to stack the generated np arrays.
        :param post_processing_np: Postprocessing function on the generated np arrays
        :param args: Arguments passed to super init.
        :param kwargs: Keyword arguments passed to super init.
        """
        super(VideoFrameListGenerator, self).__init__(*args, **kwargs)
        self.wrapped_generator = wrapped_generator
        self.stack_axis = stack_axis
        self.post_processing_np = post_processing_np

    def get_transformation(self, **kwargs):
        """
        Returns the transformation of the wrapped generator.
        :param kwargs: Given to self.wrapped_generator.get_transformation()
        :return: The transformation.
        """
        return self.wrapped_generator.get_transformation(**kwargs)

    def get(self, *args, **kwargs):
        """
        Calls get() of the wrapped generator for each entry of the given lists and returns the stacked np array.
        Each call of get() of the wrapped generator will receive corresponding entries of the given list parameters,
        e.g., self.wrapped_generator.get(args_0[i], args_1[i], kwargs_0[i], kwargs_1[i]) for i in range(len(args_0))
        :param args: Arguments that will be given to the wrapped generator's get() function.
                     Either single entries or lists of same length.
        :param kwargs: Keyword arguments that will be given to the wrapped generator's get() function.
                       Either single entries or lists of same length.
        :return: The stacked np array of the individual outputs of get() of the wrapped generator.
        """
        # at first, determine the list length
        list_length = None
        for arg in args:
            if isinstance(arg, list):
                list_length = len(arg)
        for kwarg_value in kwargs.values():
            if isinstance(kwarg_value, list):
                list_length = len(kwarg_value)
        assert list_length is not None, 'no list argument is given'

        # collect individual list outputs
        outputs = []
        for i in range(list_length):
            # determine the current args
            current_args = []
            for arg in args:
                if isinstance(arg, list):
                    # if current argument is a list, use the value of the current index i
                    assert len(arg) == list_length
                    current_args.append(arg[i])
                else:
                    # otherwise, just use the value
                    current_args.append(arg)
            # determine the current kwargs
            current_kwargs = {}
            for kwarg_key, kwarg_value in kwargs.items():
                if isinstance(kwarg_value, list):
                    # if current argument is a list, use the value of the current index i
                    assert len(kwarg_value) == list_length
                    current_kwargs[kwarg_key] = kwarg_value[i]
                else:
                    # otherwise, just use the value
                    current_kwargs[kwarg_key] = kwarg_value
            # call wrapped generator's get with the current arguments
            current_output = self.wrapped_generator.get(*current_args, **current_kwargs)
            outputs.append(current_output)

        # stack all outputs
        stacked_outputs = np.stack(outputs, axis=self.stack_axis)

        if self.post_processing_np is not None:
            stacked_outputs = self.post_processing_np(stacked_outputs)

        return stacked_outputs
