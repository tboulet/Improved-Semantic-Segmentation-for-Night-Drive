import os
import random
from typing import List, Optional, Tuple, TYPE_CHECKING
from warnings import warn
import yaml

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model, save_model
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError
from tqdm import tqdm

from improved_nightdrive.pipeline.metric import Metric

if TYPE_CHECKING:
    from improved_nightdrive.pipeline.callback import Callback
    from improved_nightdrive.pipeline.preprocess import Preprocess

correspondance = yaml.safe_load(
    open("./improved_nightdrive/pipeline/correspondance.yml", "r")
)

# TODO@raffaelbdl: Possibly inusable
class Evaluation:
    """Evaluation pipeline of a trained model

    Attributes:
        model (Model): A Keras model for segmentation
        num_classes (int): Number of classes
        loss (str): A loss function used for training
            -> available losses : ['cce', 'mse']

        x_dir_path (str): Path to image dataset
        y_dir_path (str): Path to label dataset

        metrics (List[Metric]): A list of metrics to follow progress
        preprocesses (List[Preprocess]): A list of preprocesses to apply IN ORDER
    """

    def __init__(
        self,
        model: Model,
        num_classes: int,
        loss: str = "cce",
        x_dir_path: str = "./dataset/rgb/",
        y_dir_path: str = "./dataset/seg/",
        metrics: List["Metric"] = [],
        preprocesses: Optional[List["Preprocess"]] = [],
        batch_size=16,
    ) -> None:

        assert len(metrics) > 0, "Metrics must be specified"
        for metric in metrics:
            assert isinstance(metric, Metric), "Metrics must be of type Metric"

        self.model = model

        self.metrics = metrics
        self.preprocesses = preprocesses

        self.num_classes = num_classes

        if loss == "cce":
            self.loss = CategoricalCrossentropy()
        elif loss == "mse":
            self.loss = MeanSquaredError()
        else:
            warn(loss + " is not implemented, cce used by default")
            self.loss = CategoricalCrossentropy()

        self.batch_size = batch_size

        self.x_dir_path = x_dir_path
        self.y_dir_path = y_dir_path

    def sample_evaluate(self, x: tf.Tensor, y: tf.Tensor) -> List[float]:
        """Evaluates the model one one sample

        Args:
            x_input (tf.Tensor): Input
            y_input (tf.Tensor): Label

        Returns:
            metrics (List[float]): A list of evaluated metrics
        """
        if len(x.shape) == 3:
            x = tf.expand_dims(x, axis=0)
            y = tf.expand_dims(y, axis=0)

        for preprocess in self.preprocesses:
            x, y = preprocess.func(x, y)

        ypred = self.model(x)

        evaluated_metrics = []
        for metric in self.metrics:
            evaluated_metrics.append(metric.func(y, ypred))

        return evaluated_metrics

    def evaluate(self):
        dataset_metrics = [0 for _ in range(len(self.metrics))]
        sorted_x_dir = sorted(os.listdir(self.x_dir_path))
        sorted_y_dir = sorted(os.listdir(self.y_dir_path))
        num_elem = len(sorted_x_dir)

        idx = list(range(num_elem))
        random.shuffle(idx)

        num_batch = len(idx) // self.batch_size
        for i in tqdm(range(num_batch), desc="Evaluation"):
            idx_batch = idx[i * self.batch_size : (i + 1) * self.batch_size]
            train_xbatch, train_ybatch = load_batch(
                idx_batch,
                self.num_classes,
                self.x_dir_path,
                self.y_dir_path,
                sorted_x_dir,
                sorted_y_dir,
            )
            sample_metrics = self.sample_evaluate(train_xbatch, train_ybatch)
            for j in range(len(dataset_metrics)):
                dataset_metrics[j] += sample_metrics[j] / num_batch

        return dataset_metrics


class Training:
    """Training pipeline of a model

    Attributes:
        model (Model): A Keras model for segmentation
        num_classes (int): Number of classes
        loss (str): A loss function used for training
            -> available losses : ['cce', 'mse']

        x_dir_path (str): Path to images dataset
        y_dir_path (str): Path to labels dataset

        metrics (List[Metric]): A list of metrics to follow progress
        preprocesses (List[Preprocess]): A list of preprocesses to apply IN ORDER
        callbacks (List[Callback]): A list of callbacks
    """

    def __init__(
        self,
        model: Model,
        num_classes: int,
        loss: str = "cce",
        x_dir_path: str = "./dataset/rgb/",
        y_dir_path: str = "./dataset/seg/",
        metrics: List["Metric"] = [],
        preprocesses: Optional[List["Preprocess"]] = [],
        callbacks: Optional[List["Callback"]] = [],
    ) -> None:

        assert len(metrics) > 0, "Metrics must be specified"
        for metric in metrics:
            assert isinstance(metric, Metric), "Metrics must be of type Metric"

        self.model = model

        self.metrics = metrics
        self.preprocesses = preprocesses
        self.callbacks = callbacks

        self.num_classes = num_classes

        if loss == "cce":
            self.loss = CategoricalCrossentropy()
        elif loss == "mse":
            self.loss = MeanSquaredError()
        else:
            warn(loss + " is not implemented, cce used by default")
            self.loss = CategoricalCrossentropy()

        self.x_dir_path = x_dir_path
        self.y_dir_path = y_dir_path

    def train(
        self,
        num_epochs: int = 10,
        batch_size: int = 12,
        save_model_bool: bool = False,
        save_name: str = "",
        lr: float = 2e-4,
    ) -> None:
        """Trains the model"""
        sorted_x_dir = sorted(os.listdir(self.x_dir_path))
        sorted_y_dir = sorted(os.listdir(self.y_dir_path))
        num_elem = len(sorted_x_dir)

        opt = tf.keras.optimizers.Adam(lr)

        idx = list(range(num_elem))
        # random.shuffle(idx)
        self.train_idx = train_idx = idx[: int(len(idx) * 0.8)]
        self.valid_idx = valid_idx = idx[int(len(idx) * 0.8) : int(len(idx) * 0.9)]
        self.test_idx = test_idx = idx[int(len(idx) * 0.9) :]

        for epoch in tqdm(range(num_epochs), desc="Epoch training loop"):

            logs = {}
            prev_valid = 0

            # Train
            random.shuffle(train_idx)
            num_batch = len(train_idx) // batch_size
            for i in tqdm(range(num_batch), desc="Inner training loop"):
                idx_batch = train_idx[i * batch_size : (i + 1) * batch_size]
                train_xbatch, train_ybatch = load_batch(
                    idx_batch,
                    self.num_classes,
                    self.x_dir_path,
                    self.y_dir_path,
                    sorted_x_dir,
                    sorted_y_dir,
                )

                for preprocess in self.preprocesses:
                    train_xbatch, train_ybatch = preprocess.func(
                        train_xbatch, train_ybatch
                    )

                train_ybatch = train_ybatch - tf.where(
                    train_ybatch == 255.0, 255.0, 0.0
                )

                with tf.GradientTape() as tape:
                    train_ypred = self.model(train_xbatch)
                    _y = tf.reshape(train_ybatch, shape=(-1, train_ybatch.shape[-1]))
                    _ypred = tf.reshape(train_ypred, shape=(-1, train_ypred.shape[-1]))
                    train_loss = self.loss(_y, _ypred)

                grads = tape.gradient(train_loss, self.model.trainable_weights)
                opt.apply_gradients(zip(grads, self.model.trainable_weights))

            logs["train_loss"] = train_loss.numpy()
            for metric in self.metrics:
                logs["train_" + metric.name] = metric(train_ybatch, train_ypred)

            # Valid
            random.shuffle(valid_idx)
            num_batch = len(valid_idx) // batch_size
            for i in range(num_batch):
                idx_batch = valid_idx[i * batch_size : (i + 1) * batch_size]
                valid_xbatch, valid_ybatch = load_batch(
                    idx_batch,
                    self.num_classes,
                    self.x_dir_path,
                    self.y_dir_path,
                    sorted_x_dir,
                    sorted_y_dir,
                )

                for preprocess in self.preprocesses:
                    valid_xbatch, valid_ybatch = preprocess.func(
                        valid_xbatch, valid_ybatch
                    )
                valid_ybatch = valid_ybatch - tf.where(
                    valid_ybatch == 255.0, 255.0, 0.0
                )

                valid_ypred = self.model(valid_xbatch)
                _y = tf.reshape(valid_ybatch, shape=(-1, valid_ybatch.shape[-1]))
                _ypred = tf.reshape(valid_ypred, shape=(-1, valid_ypred.shape[-1]))
                valid_loss = self.loss(_y, _ypred)

            logs["valid_loss"] = valid_loss.numpy()
            for metric in self.metrics:
                logs["valid_" + metric.name] = metric(valid_ybatch, valid_ypred)

            for callback in self.callbacks:
                callback.at_epoch_end(logs=logs, model=self.model, epoch=epoch)

            if save_model_bool:
                if logs["valid_miou"] > prev_valid:
                    save_model(self.model, save_name + f"_at_best_vmiou")
                    prev_valid = logs["valid_miou"]
            print(f"Epoch {epoch+1}: ", logs)


def load_batch(
    idx: List[int],
    num_classes: int,
    x_dir_path: str,
    y_dir_path: str,
    sorted_x_dir: List[str],
    sorted_y_dir: List[str],
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Loads a batch of images from disk"""
    xs = []
    ys = []

    for i in idx:
        ix = np.array(Image.open(os.path.join(x_dir_path, sorted_x_dir[i])))
        iy = np.array(Image.open(os.path.join(y_dir_path, sorted_y_dir[i])))

        xs.append(ix)
        ys.append(iy)

    x = tf.constant(np.array(xs), dtype=tf.float32) / 255.0
    y = tf.one_hot(tf.constant(np.array(ys)), num_classes, axis=-1, dtype=tf.float32)

    return x, y
