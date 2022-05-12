"""All image preprocesses"""
from typing import List

import tensorflow as tf
from tensorflow_addons.image import equalize
import tf_clahe


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
