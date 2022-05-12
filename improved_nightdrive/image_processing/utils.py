import tensorflow as tf
import tf_clahe
import numpy as np


@tf.function(experimental_compile=True)  # Enable XLA
def fast_clahe(img):
    return tf_clahe.clahe(img, gpu_optimized=True)


def log_process(img, c=1):
    return c * tf.math.log(img + 1)


def gamma_process(img, p=[1, 1, 1]):
    if type(p) != list:
        p = [p]*3

    r = img[..., 0]**p[0]
    g = img[..., 1]**p[1]
    b = img[..., 2]**p[2]

    output = tf.stack([r, g, b], axis=-1)

    return output


def lab_gamma_process(img, p=1, c=100):
    
    L = c * (img[:, :, 0]/c)**p
    a = img[:, :, 1]
    b = img[:, :, 2]
    return tf.stack((L, a, b), axis=-1)


def tf_equalize_histogram(image):
    image = image * 255
    values_range = tf.constant([0., 255.], dtype = tf.float32)
    histogram = tf.histogram_fixed_width(image, values_range, 256)
    cdf = tf.cumsum(histogram)
    cdf_min = cdf[tf.reduce_min(tf.where(tf.greater(cdf, 0)))]

    img_shape = tf.shape(image)
    pix_cnt = img_shape[-3] * img_shape[-2]
    px_map = tf.round(tf.cast(cdf - cdf_min, dtype=tf.float32) * 255. / tf.cast(pix_cnt - 1, dtype=tf.float32))
    px_map = tf.cast(px_map, tf.uint8)
    image = tf.clip_by_value(image,0, 255)

    eq_hist = tf.expand_dims(tf.gather_nd(px_map, tf.cast(image, tf.int32)), 2)
    return tf.squeeze(eq_hist)


def lab_hist(image):
    values_range = tf.constant([0., 100.], dtype = tf.float32)
    histogram = tf.histogram_fixed_width(image, values_range, 100)
    cdf = tf.cumsum(histogram)
    cdf_min = cdf[tf.reduce_min(tf.where(tf.greater(cdf, 0)))]

    img_shape = tf.shape(image)
    pix_cnt = img_shape[-3] * img_shape[-2]
    px_map = tf.round(tf.cast(cdf - cdf_min, dtype=tf.float32) * 100. / tf.cast(pix_cnt - 1, dtype=tf.float32))
    px_map = tf.cast(px_map, tf.uint8)
    image = tf.clip_by_value(image,0, 99)

    eq_hist = tf.expand_dims(tf.gather_nd(px_map, tf.cast(image, tf.int32)), 2)
    return tf.squeeze(eq_hist)
    