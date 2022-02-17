import tensorflow as tf
from metrics import Metric
from preprocess import Preprocess
from typing import List


class Evaluation():
    """Evaluation pipeline of a trained model

    Attributes:
        model: A trained keras model
        preprocess: A list of preprocesses to apply IN ORDER
        metrics: A list of Metric instances
    """

    def __init__(self,
                 model: tf.keras.models.Model,
                 preprocesses: List[Preprocess],
                 metrics: List[Metric]) -> None:

        self.model = model
        self.preprocesses = preprocesses
        self.metrics = metrics

    def evaluate(self, x, y):
        """Evaluate the model

        Args:
            x: Inputs
            y: Labels

        Returns:
            _metrics: A list of evaluated metrics
        """
        _x = x
        for p in self.preprocesses:
            _x = p(_x)

        pred = self.model(_x)

        _metrics = []
        for m in self.metrics:
            _m = m(y, pred)
            _metrics.append(_m)

        return _metrics
