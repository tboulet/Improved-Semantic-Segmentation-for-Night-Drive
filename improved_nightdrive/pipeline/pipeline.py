from typing import List, Optional, TYPE_CHECKING

import tensorflow as tf

from tensorflow.keras.models import Model

if TYPE_CHECKING:
    from improved_nightdrive.pipeline import Metric, Preprocess


class Evaluation():
    """Evaluation pipeline of a trained model

    Attributes:
        model: A trained keras model
        metrics: A list of Metric instances
        preprocesses: A list of preprocesses to apply IN ORDER
    """

    def __init__(
        self, 
        model: Model,
        metrics: List['Metric'],
        preprocesses: Optional[List['Preprocess']] = None
     ) -> None:

        self.model = model
        self.metrics = metrics
        self.preprocesses = preprocesses

    def evaluate(self, x: tf.Tensor, y: tf.Tensor) -> List[float]:
        """Evaluates the model

        Args:
            x: Inputs
            y: Labels

        Returns:
            _metrics: A list of evaluated metrics
        """
        _x = x
        if self.preprocesses is not None:
            for p in self.preprocesses:
                _x = p(_x)

        pred = self.model(_x)

        _metrics = []
        for m in self.metrics:
            _m = m(y, pred)
            _metrics.append(_m)

        return _metrics

class Training():
    """Training pipeline of a model

    Attributes:
        model: A keras model
        loss: A loss function that will be used for training
            available losses : ['cce', 'mse']
        metrics: A list of Metric to follow progress
        preprocesses: A list of preprocesses to apply IN ORDER
    """

    def __init__(
        self, 
        model: Model,
        loss: str,
        metrics: List['Metric'],
        preprocesses: Optional[List['Preprocess']] = None
     ) -> None:

        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.preprocesses = preprocesses
    
    def train(
        self, 
        x: tf.Tensor, 
        y: tf.Tensor,
        num_epochs: int = 10
    ) -> None:
        """Trains the model

        Args:
            x: Inputs
            y: Labels
        """