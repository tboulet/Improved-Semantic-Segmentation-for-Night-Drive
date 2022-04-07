
from functools import partial
from typing import Tuple

import tensorflow as tf


class Preprocess():
    """Root class for all processes

    Attributes:
        func: f(x_inputs, y_inputs) -> modif_x_inputs, modif_y_inputs

    Methods:
        __call__: Applies preprocess
    """
    def __init__(self, func):
        self.func = func


    def __call__(self, *args) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.func(*args)


class AddNoise(Preprocess):
    """Adds noise to first input"""
    def __init__(self):

        def add_noise(
            x_inputs: tf.Tensor, 
            y_inputs: tf.Tensor=None
        ) -> Tuple[tf.Tensor, tf.Tensor]:

            noise_map = tf.random.normal(shape=x_inputs)

            return (x_inputs + noise_map, y_inputs)

        super().__init__(add_noise)
     

class RandomCrop(Preprocess):
    """Crops randomly both inputs"""
    def __init__(self, size: int):

        def crop(
            x_inputs: tf.Tensor, 
            y_inputs: tf.Tensor
        ) -> tf.Tensor:

            c = x_inputs.shape[-1]

            random_crop_fn = partial(one_crop_random, size=size)
            outputs = tf.map_fn(random_crop_fn, tf.concat((x_inputs, y_inputs), axis=-1))  

            x_inputs, y_inputs = outputs[..., :c], outputs[..., c:]  

            return (x_inputs, y_inputs)

        super().__init__(crop)


class Normalize(Preprocess):
    """Normalizes first input"""
    def __init__(self):

        def normalize(
            x_inputs: tf.Tensor,
            y_inputs: tf.Tensor=None
        ) -> Tuple[tf.Tensor, tf.Tensor]:

            means = tf.reduce_mean(x_inputs, x_inputs.shape[1:], keepdims=True)
            stds = tf.math.reduce_std(x_inputs, x_inputs.shape[1:], keepdims=True)

            x_inputs = (x_inputs - means) / stds

            return (x_inputs, y_inputs)
            
        super().__init__(normalize)


def one_crop_random(
    input: tf.Tensor, 
    size: int
) -> tf.Tensor:

    h, w, c = input.shape

    target_height = max(h, size)
    target_width = max(w, size)
    input = tf.image.resize_with_pad(input, target_height, target_width)

    random_upper_left_x = tf.random.uniform(shape=(), minval=0, maxval=(w-size), dtype=tf.int32)
    random_upper_left_y = tf.random.uniform(shape=(), minval=0, maxval=(h-size), dtype=tf.int32)

    cropped_input = input[random_upper_left_y: random_upper_left_y+size, random_upper_left_x: random_upper_left_x+size]

    return cropped_input