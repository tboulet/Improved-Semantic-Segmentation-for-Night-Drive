"""Contains all the preprocesses"""
from functools import partial
from typing import List, Tuple
import yaml


import numpy as np
import tensorflow as tf
from tensorflow_addons.image import equalize, gaussian_filter2d
import tf_clahe


correspondance = yaml.safe_load(
    open("./improved_nightdrive/pipeline/correspondance.yml", "r")
)
new_classes = len(np.unique(list(correspondance.values())))


class Preprocess:
    """Base class for all processes

    Attributes:
        func: f(x_inputs, y_inputs) -> modif_x_inputs, modif_y_inputs

    Methods:
        __call__: Applies preprocess
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

        def func(x, y):
            return self.blur(x), y

        super().__init__(func)

    def blur(self, x: tf.Tensor) -> tf.Tensor:
        """Blur image"""
        return gaussian_filter2d(x, self.filter_shape, self.sigma)


class AddNoise(Preprocess):
    """Adds noise to first input"""

    def __init__(self, noise_factor):

        self.noise_factor = noise_factor

        def add_noise(
            x_inputs: tf.Tensor, y_inputs: tf.Tensor = None
        ) -> Tuple[tf.Tensor, tf.Tensor]:

            noise_map = tf.random.normal(shape=x_inputs.shape) * self.noise_factor
            x_inputs = tf.clip_by_value(x_inputs + noise_map, 0.0, 1.0)

            return (x_inputs, y_inputs)

        super().__init__(add_noise)


class EqualizeHistogram(Preprocess):
    """Apply log process to first input"""

    def __init__(self) -> None:
        def func(
            x_inputs: tf.Tensor,
            y_inputs: tf.Tensor = None,
        ) -> Tuple[tf.Tensor, tf.Tensor]:

            x_inputs = tf.map_fn(hist_eq, x_inputs)

            return (x_inputs, y_inputs)

        super().__init__(func)


class FastClahe(Preprocess):
    """Apply fast clahe to first input"""

    def __init__(self) -> None:
        def func(
            x_inputs: tf.Tensor,
            y_inputs: tf.Tensor = None,
        ) -> Tuple[tf.Tensor, tf.Tensor]:

            x_inputs = tf.map_fn(fast_clahe, x_inputs)

            return (x_inputs, y_inputs)

        super().__init__(func)


class GammaProcess(Preprocess):
    """Apply gamma process to first input"""

    def __init__(self, p: List[float] = [1.0, 1.0, 1.0]) -> None:
        def func(
            x_inputs: tf.Tensor,
            y_inputs: tf.Tensor = None,
        ) -> Tuple[tf.Tensor, tf.Tensor]:

            _gamma_process = partial(gamma_process, p=p)
            x_inputs = tf.map_fn(_gamma_process, x_inputs)

            return (x_inputs, y_inputs)

        super().__init__(func)

    def masked_func(
        self, x_inputs: tf.Tensor, p: List[List[float]], y_preds: tf.Tensor
    ):
        """
        Args:
            x_inputs (tf.Tensor): [B, H, W, 3]
            p (List[float]): [n_class, 3]
            y_preds (tf.Tensor): [B, H, W, n_class]
        Returns:
            A list of `n_class` modified images with process applied per mask
        """
        x_outputs = []
        for i, _p in enumerate(p):
            x_out = gamma_process(x_inputs, _p)
            x_out = tf.where(y_preds[..., i, None] > 0.0, x_out, x_inputs)
            x_outputs.append(x_out)
        return x_outputs


class Normalize(Preprocess):
    """Normalizes first input"""

    def __init__(self):
        def normalize(
            x_inputs: tf.Tensor, y_inputs: tf.Tensor = None
        ) -> Tuple[tf.Tensor, tf.Tensor]:

            means = tf.reduce_mean(x_inputs, x_inputs.shape[1:], keepdims=True)
            stds = tf.math.reduce_std(x_inputs, x_inputs.shape[1:], keepdims=True)

            x_inputs = (x_inputs - means) / stds

            return (x_inputs, y_inputs)

        super().__init__(normalize)


class LogProcess(Preprocess):
    """Apply log process to first input"""

    def __init__(self, c: float) -> None:
        def func(
            x_inputs: tf.Tensor,
            y_inputs: tf.Tensor = None,
        ) -> Tuple[tf.Tensor, tf.Tensor]:

            _log_process = partial(log_process, c=c)
            x_inputs = tf.map_fn(_log_process, x_inputs)

            return (x_inputs, y_inputs)

        super().__init__(func)


class RandomCrop(Preprocess):
    """Crops randomly both inputs"""

    def __init__(self, size: int):
        def crop(x_inputs: tf.Tensor, y_inputs: tf.Tensor) -> Tuple[tf.Tensor]:

            c = x_inputs.shape[-1]

            random_crop_fn = partial(one_crop_random, size=size)
            outputs = tf.map_fn(
                random_crop_fn, tf.concat((x_inputs, y_inputs), axis=-1)
            )

            x_inputs, y_inputs = outputs[..., :c], outputs[..., c:]

            return (x_inputs, y_inputs)

        super().__init__(crop)


class RandomFlip(Preprocess):
    """Randomly flips both input"""

    def __init__(self):
        def flip(x_inputs: tf.Tensor, y_inputs: tf.Tensor) -> Tuple[tf.Tensor]:

            c = x_inputs.shape[-1]
            outputs = tf.map_fn(
                one_random_flip, tf.concat((x_inputs, y_inputs), axis=-1)
            )

            x_inputs, y_inputs = outputs[..., :c], outputs[..., c:]

            return (x_inputs, y_inputs)

        super().__init__(flip)


class ReClass(Preprocess):
    """Reclasses labels according to given correspondance"""

    def __init__(self, num_classes: int = 19):

        self.num_classes = num_classes
        self.new_classes = new_classes

        self.correspondance = correspondance

        def change_class(
            x_inputs: tf.Tensor,
            y_inputs: tf.Tensor,
        ) -> Tuple[tf.Tensor, tf.Tensor]:

            _y = tf.argmax(y_inputs, axis=-1)

            for i in range(self.num_classes):
                _y = tf.where(_y == i, self.correspondance[i], _y)
            _y = tf.where(_y == 255, self.correspondance[255], _y)

            _y = tf.one_hot(_y, self.new_classes, axis=-1, dtype=tf.float32)

            return (x_inputs, _y)

        super().__init__(change_class)


class Resize(Preprocess):
    """Resizes both input"""

    def __init__(self, intermediate_size: Tuple[int] = (240, 320)):
        def resize(x_inputs: tf.Tensor, y_inputs: tf.Tensor) -> Tuple[tf.Tensor]:

            x_inputs = tf.image.resize(x_inputs, intermediate_size)
            y_inputs = tf.image.resize(y_inputs, intermediate_size, "nearest")

            return (x_inputs, y_inputs)

        super().__init__(resize)


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
        random_upper_left_y: random_upper_left_y + size,
        random_upper_left_x: random_upper_left_x + size,
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
