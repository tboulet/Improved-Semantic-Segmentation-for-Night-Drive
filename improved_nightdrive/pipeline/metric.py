
from functools import partial

import tensorflow as tf


class Metric():
    """Root class for all metrics

    Attributes:
        func: f(y, ypred) -> m

    Methods:
        __call__: Evaluates the metric
    """
    def __init__(self, func) -> None:
        self.func = func


    def __call__(self, *args) -> float:
        return self.func(*args)


class MeanIOU(Metric):
    """Mean IOU metric"""
    def __init__(self, num_classes) -> None:
        super().__init__(partial(mean_iou, num_classes))


def mean_iou(
    num_classes: int, 
    labels: tf.Tensor, 
    predictions: tf.Tensor
) -> float:
    """Computes the mean IOU for a batch

    Args:
        num_classes (int): Number of classes
        labels (tf.Tensor): The segmentation map label, 
            as tensor of shape (batch, h, w, num_classes)
            or as tensor of shape (batch, h, w) (will be one hot-ed)
        predictions (tf.Tensor): The segmentation map prediction, 
            as softmax of shape(batch, h, w, num_classes)

    Returns:
        mean_iou_score: The Mean IoU Score
    """
    if labels.shape[-1] != num_classes:
        labels = tf.one_hot(labels, num_classes)

    _labels = tf.cast(labels, dtype=tf.int32)
    _predictions = tf.one_hot(
        tf.argmax(predictions, axis=-1), num_classes, dtype=tf.int32)
    intersection = _labels * _predictions
    union = _labels + _predictions - intersection
    iou_score = tf.reduce_sum(intersection, (1, 2, 3)) / \
        tf.reduce_sum(union, (1, 2, 3))
        
    return tf.reduce_mean(iou_score)
