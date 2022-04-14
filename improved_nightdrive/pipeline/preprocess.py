
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
    def __init__(self, func) -> None:
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

            noise_map = tf.random.normal(shape=x_inputs.shape)

            return (x_inputs + noise_map, y_inputs)

        super().__init__(add_noise)
     

class RandomCrop(Preprocess):
    """Crops randomly both inputs"""
    def __init__(self, size: int):

        def crop(
            x_inputs: tf.Tensor, 
            y_inputs: tf.Tensor
        ) -> Tuple[tf.Tensor]:

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


class Resize(Preprocess):
    """Resizes both input"""
    def __init__(self, intermediate_size: Tuple[int] = (240, 320)):

        def resize(
            x_inputs: tf.Tensor,
            y_inputs: tf.Tensor
        ) -> Tuple[tf.Tensor]:

            x_inputs = tf.image.resize(x_inputs, intermediate_size)
            y_inputs = tf.image.resize(y_inputs, intermediate_size, 'nearest')

            return (x_inputs, y_inputs)

        super().__init__(resize)


class RandomFlip(Preprocess):
    """Randomly flips both input"""
    def __init__(self):

        def flip(
            x_inputs: tf.Tensor,
            y_inputs: tf.Tensor
        ) -> Tuple[tf.Tensor]:
            
            c = x_inputs.shape[-1]
            outputs = tf.map_fn(one_random_flip, tf.concat((x_inputs, y_inputs), axis=-1))

            x_inputs, y_inputs = outputs[..., :c], outputs[..., c:]

            return (x_inputs, y_inputs)

        super().__init__(flip)


def one_crop_random(
    input: tf.Tensor, 
    size: int
) -> tf.Tensor:
    """Crops an input with a random upper left corner"""
    h, w, c = input.shape
    
    target_height = max(h, size)
    target_width = max(w, size)
    input = tf.image.resize_with_pad(input, target_height, target_width)
    
    random_upper_left_x = tf.random.uniform(shape=(), minval=0, maxval=(w-size), dtype=tf.int32)
    random_upper_left_y = tf.random.uniform(shape=(), minval=0, maxval=(h-size), dtype=tf.int32)

    cropped_input = input[
        random_upper_left_y: random_upper_left_y+size, 
        random_upper_left_x: random_upper_left_x+size]

    return cropped_input


def one_random_flip(input: tf.Tensor) -> tf.Tensor:
    """Randomly flips the input"""
    if tf.random.uniform(shape=()) > 0.5:
        input = tf.image.flip_left_right(input)

    return input