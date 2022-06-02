from functools import partial
from typing import List

import numpy as np
import tensorflow as tf


class Metric:
    """Base class for all metrics

    Attributes:
        func: f(y, ypred) -> m

    Methods:
        __call__: Evaluate the metric
    """

    def __init__(self, func, name="metric") -> None:

        self.func = func
        self.name = name

    def __call__(self, labels, predictions) -> float:
        return self.func(labels, predictions)


class MeanIOU(Metric):
    """Mean IOU metric"""

    def __init__(self, num_classes) -> None:
        super().__init__(name="miou", func=partial(mean_iou, num_classes=num_classes))


class ClassMeanIOU(Metric):
    """Mean IOU metric per class"""

    def __init__(self, num_classes) -> None:
        super().__init__(
            name="cmiou", func=partial(class_mean_iou, num_classes=num_classes)
        )


class ConfMatrix(Metric):
    """Confusion matrix"""

    def __init__(self, num_classes) -> None:
        super().__init__(
            name="confmat", func=partial(conf_matrix, num_classes=num_classes)
        )


def mean_iou(
    labels: tf.Tensor,
    predictions: tf.Tensor,
    num_classes: int,
) -> float:
    """Compute the mean IOU for a batch

    Args:
        labels (tf.Tensor): The segmentation map label,
            as tensor of shape [batch_size, height, width, num_classes]
            or as tensor of shape [batch_size, height, width] (will be one-hotted)
        predictions (tf.Tensor): The segmentation map prediction,
            as softmax [batch_size, height, width, num_classes]
        num_classes (int): Number of classes

    Returns:
        mean_iou_score (float): The Mean IoU Score
    """
    if labels.shape[-1] != num_classes:
        labels = tf.one_hot(labels, num_classes)

    _labels = tf.cast(labels, dtype=tf.int32)
    _predictions = tf.one_hot(
        tf.argmax(predictions, axis=-1), num_classes, dtype=tf.int32
    )
    intersection = _labels * _predictions
    union = _labels + _predictions - intersection
    iou_score = tf.reduce_sum(intersection, (1, 2, 3)) / tf.reduce_sum(union, (1, 2, 3))

    return tf.reduce_mean(iou_score).numpy()


def class_mean_iou(
    labels: tf.Tensor,
    predictions: tf.Tensor,
    num_classes: int,
) -> List[float]:
    """Compute the mean IOU for a batch for each class

    Args:
        labels (tf.Tensor): The segmentation map label,
            as tensor of shape [batch_size, height, width, num_classes]
            or as tensor of shape [batch_size, height, width] (will be one-hotted)
        predictions (tf.Tensor): The segmentation map prediction,
            as softmax [batch_size, height, width, num_classes]
        num_classes (int): Number of classes

    Returns:
        mean_iou_score (List[float]): The Mean IoU Score for each class
    """
    if labels.shape[-1] != num_classes:
        labels = tf.one_hot(labels, num_classes)

    _labels = tf.cast(labels, dtype=tf.int32)
    _predictions = tf.one_hot(
        tf.argmax(predictions, axis=-1), num_classes, dtype=tf.int32
    )
    intersection = _labels * _predictions
    union = _labels + _predictions - intersection
    iou_score = tf.math.divide_no_nan(
        tf.cast(tf.reduce_sum(intersection, (1, 2)), tf.float32),
        tf.cast(tf.reduce_sum(union, (1, 2)), tf.float32),
    )

    return tf.reduce_mean(iou_score, axis=0).numpy()


def conf_matrix(
    labels: tf.Tensor,
    predictions: tf.Tensor,
    num_classes: int,
) -> List[float]:
    """Computes the confusion matrix for the current batch

    Args:
        labels (tf.Tensor): The segmentation map label,
            as tensor of shape [batch_size, height, width, num_classes] (will be flattened along last axis)
            or as tensor of shape [batch_size, height, width]
        predictions (tf.Tensor): The segmentation map prediction,
            as softmax [batch_size, height, width, num_classes] (will be flattened along last axis)
        num_classes (int): Number of classes

    Returns:
        confusion matrix (np.ndarray): The confusion matrix
    """
    if labels.shape[-1] == num_classes:
        labels = tf.argmax(labels, axis=-1)

    _labels = tf.cast(labels, dtype=tf.int32)
    _predictions = tf.argmax(predictions, axis=-1)

    shape = tf.math.reduce_prod(_labels.shape)

    _labels = tf.reshape(_labels, shape)
    _predictions = tf.reshape(_predictions, shape)

    norm = np.array(
        [[tf.reduce_sum(tf.where(_labels == i, 1.0, 0.0))] for i in range(num_classes)]
    )

    matrix = (
        tf.math.confusion_matrix(
            _labels,
            _predictions,
            num_classes=num_classes,
            weights=None,
            dtype=tf.dtypes.int32,
            name=None,
        ).numpy()
        / norm
    )

    return matrix
