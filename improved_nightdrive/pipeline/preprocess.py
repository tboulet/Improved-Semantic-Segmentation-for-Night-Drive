from typing import Tuple, TYPE_CHECKING

import tensorflow as tf



class Preprocess():
    """
    Attributes:
        func: outputs = f(inputs)

    Methods:
        __call__: Applies preprocess
    """

    def __init__(self, func):
        self.func = func

    def __call__(self, *args):
        return self.func(*args)


class Normalization(Preprocess):
    """Normalization preprocess
    """

    def __init__(self):

        def normalize(inputs: tf.Tensor) -> tf.Tensor:
            """Normalizes a batch of inputs

            Args:
                inputs: A batch of inputs

            Returns:
                outputs: A batch of normalized inputs
            """
            means = tf.reduce_mean(inputs, inputs.shape[1:], keepdims=True)
            stds = tf.math.reduce_std(inputs, inputs.shape[1:], keepdims=True)

            outputs = (inputs - means) / stds

            return outputs
            
        super().__init__(normalize)


class Crop(Preprocess):

    def __init__(self, shape: Tuple[int]):

        def crop(batch_image: tf.Tensor) -> tf.Tensor:
            # def _crop(image: tf.Tensor):
            h, w, c = shape    
            return tf.image.resize_with_crop_or_pad(batch_image, h, w)
            
            # return tf.vectorized_map(_crop, batch_image)

        super().__init__(crop)

    def __call__(self, *args):
        return self.func(*args)