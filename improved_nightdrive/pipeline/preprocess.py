"""Contains all the preprocesses"""
from functools import partial
from math import gamma
from typing import List, Tuple
import yaml

import numpy as np
import tensorflow as tf
from tensorflow_addons.image import equalize, gaussian_filter2d
import tf_clahe


correspondance = yaml.safe_load(
    open("./improved_nightdrive/pipeline/correspondance.yml", "r")
)
_NEW_CLASSES = len(np.unique(list(correspondance.values())))


class Preprocess:
    """Base class for all processes

    Attributes:
        func: f(x_inputs, y_inputs) -> modif_x_inputs, modif_y_inputs

    Methods:
        __call__: Apply preprocess
    """

    def __init__(self, func) -> None:
        self.func = func

    def __call__(self, *args) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.func(*args)


class GaussianBlur(Preprocess):
    """Blur first input"""

    def __init__(self, filter_shape=3, sigma=1):
        self.filter_shape = filter_shape
        self.sigma = sigma

        super().__init__(lambda x, y: (self.blur(x), y))

    def blur(self, x: tf.Tensor) -> tf.Tensor:
        """Blur image"""
        return gaussian_filter2d(x, self.filter_shape, self.sigma)


class AddNoise(Preprocess):
    """Add noise to first input"""

    def __init__(self, noise_factor):

        self.noise_factor = noise_factor

        super().__init__(lambda x, y: (self.add_noise(x), y))

    def add_noise(self, x: tf.Tensor) -> tf.Tensor:
        """Add noise"""
        noise_map = tf.random.normal(shape=x.shape) * self.noise_factor
        x = tf.clip_by_value(x + noise_map, 0.0, 1.0)
        return x


class EqualizeHistogram(Preprocess):
    """Apply log process to first input"""

    def __init__(self) -> None:

        super().__init__(lambda x, y: (self.apply_eq_histogram(x), y))

    def apply_eq_histogram(self, x: tf.Tensor) -> tf.Tensor:
        """Equalize histogram"""
        return tf.map_fn(hist_eq, x)


class FastClahe(Preprocess):
    """Apply fast clahe to first input"""

    def __init__(self) -> None:

        super().__init__(lambda x, y: (self.apply_fast_clahe(x), y))

    def apply_fast_clahe(self, x: tf.Tensor) -> tf.Tensor:
        "Apply fast clahe"
        return tf.map_fn(fast_clahe, x)


class GammaProcess(Preprocess):
    """Apply gamma process to first input"""

    def __init__(self, p: List[float] = [1.0, 1.0, 1.0]) -> None:
        self.p = p

        super().__init__(lambda x, y: (self.apply_gamma(x, self.p), y))

    def apply_gamma(self, x: tf.Tensor, p: List[float]) -> tf.Tensor:
        "Apply gamma"
        func = partial(gamma_process, p=p)
        return tf.map_fn(func, x)


class Normalize(Preprocess):
    """Normalize first input"""

    def __init__(self):

        super().__init__(lambda x, y: (self.normalize(x), y))

    def normalize(self, x: tf.Tensor) -> tf.Tensor:
        "Normalize image"
        means = tf.reduce_mean(x, x.shape[1:], keepdims=True)
        stds = tf.math.reduce_std(x, x.shape[1:], keepdims=True)
        return (x - means) / stds


class LogProcess(Preprocess):
    """Apply log process to first input"""

    def __init__(self, c: float = 0.5) -> None:
        self.c = c

        super().__init__(lambda x, y: (self.apply_log(x, self.c), y))

    def apply_log(self, x: tf.Tensor, c: float) -> tf.Tensor:
        func = partial(log_process, c=c)
        return tf.map_fn(func, x)


class RandomCrop(Preprocess):
    """Crops randomly both inputs"""

    def __init__(self, size: int):
        self.size = size

        super().__init__(partial(self.crop, size=self.size))

    def crop(
        self, x_inputs: tf.Tensor, y_inputs: tf.Tensor, size: int
    ) -> Tuple[tf.Tensor]:

        c = x_inputs.shape[-1]

        random_crop_fn = partial(one_crop_random, size=size)
        outputs = tf.map_fn(random_crop_fn, tf.concat((x_inputs, y_inputs), axis=-1))

        x_inputs, y_inputs = outputs[..., :c], outputs[..., c:]

        return (x_inputs, y_inputs)


class RandomFlip(Preprocess):
    """Randomly flips both input"""

    def __init__(self):

        super().__init__(self.flip)

    def flip(self, x_inputs: tf.Tensor, y_inputs: tf.Tensor) -> Tuple[tf.Tensor]:

        c = x_inputs.shape[-1]
        outputs = tf.map_fn(one_random_flip, tf.concat((x_inputs, y_inputs), axis=-1))

        x_inputs, y_inputs = outputs[..., :c], outputs[..., c:]

        return (x_inputs, y_inputs)


class ReClass(Preprocess):
    """Reclasses labels according to given correspondance"""

    def __init__(self, num_classes: int = 19):

        self.num_classes = num_classes
        self.new_classes = _NEW_CLASSES

        self.correspondance = correspondance

        super().__init__(lambda x, y: (x, self.change_class(y)))

    def change_class(
        self,
        y_inputs: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:

        _y = tf.argmax(y_inputs, axis=-1)

        for i in range(self.num_classes):
            _y = tf.where(_y == i, self.correspondance[i], _y)
        _y = tf.where(_y == 255, self.correspondance[255], _y)

        _y = tf.one_hot(_y, self.new_classes, axis=-1, dtype=tf.float32)

        return _y


class Resize(Preprocess):
    """Resizes both input"""

    def __init__(self, intermediate_size: Tuple[int] = (240, 320)):

        super().__init__(partial(self.resize, intermediate_size=intermediate_size))

    def resize(
        self, x_inputs: tf.Tensor, y_inputs: tf.Tensor, intermediate_size: Tuple[int]
    ) -> Tuple[tf.Tensor]:

        x_inputs = tf.image.resize(x_inputs, intermediate_size)
        y_inputs = tf.image.resize(y_inputs, intermediate_size, "nearest")

        return (x_inputs, y_inputs)


def one_crop_random(input: tf.Tensor, size: int) -> tf.Tensor:
    """Crops an input with a random upper left corner"""
    h, w, c = input.shape

    target_height = max(h, size)
    target_width = max(w, size)
    input = tf.image.resize_with_pad(input, target_height, target_width)

    random_upper_left_x = tf.random.uniform(
        shape=(), minval=0, maxval=(w - size), dtype=tf.int32
    )
    random_upper_left_y = tf.random.uniform(
        shape=(), minval=0, maxval=(h - size), dtype=tf.int32
    )

    cropped_input = input[
        random_upper_left_y : random_upper_left_y + size,
        random_upper_left_x : random_upper_left_x + size,
    ]

    return cropped_input


def one_random_flip(input: tf.Tensor) -> tf.Tensor:
    """Randomly flips the input"""
    if tf.random.uniform(shape=()) > 0.5:
        input = tf.image.flip_left_right(input)

    return input


# == Image processing == #
@tf.function(experimental_compile=True)  # Enable XLA
def fast_clahe(img: tf.Tensor) -> tf.Tensor:
    return tf_clahe.clahe(img, gpu_optimized=True)


def gamma_process(img: tf.Tensor, p: List[int] = [1, 1, 1]) -> tf.Tensor:
    if type(p) != list:
        p = [p] * 3

    r = img[..., 0] ** p[0]
    g = img[..., 1] ** p[1]
    b = img[..., 2] ** p[2]

    output = tf.stack([r, g, b], axis=-1)

    return output


def hist_eq(img: tf.Tensor) -> tf.Tensor:
    return equalize(img * 255) / 255


def log_process(img: tf.Tensor, c: float = 1.0) -> tf.Tensor:
    return c * tf.math.log(img + 1)
