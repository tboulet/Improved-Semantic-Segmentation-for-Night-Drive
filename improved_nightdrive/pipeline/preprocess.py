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

    def __init__(self):
        super().__init__(normalize)


def normalize(inputs: tf.Tensor) -> tf.Tensor:
    """Normalize a batch of inputs

    Args:
        inputs: A batch of inputs

    Returns:
        outputs: A batch of normalized inputs
    """

    means = tf.reduce_mean(inputs, inputs.shape[1:], keepdims=True)
    stds = tf.math.reduce_std(inputs, inputs.shape[1:], keepdims=True)

    outputs = (inputs - means) / stds

    return outputs
