import os
from typing import List, Optional, TYPE_CHECKING
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import Model

from improved_nightdrive.pipeline.preprocess import Crop
from improved_nightdrive.pipeline.metrics import MeanIOU

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
        x_dir_path: str = "./dataset/rgb/",
        y_dir_path: str = "./dataset/seg/",
        num_epochs: int = 10,
        batch_size: int = 12
    ) -> None:
        """Trains the model
        """

        x_dir = sorted(os.listdir(x_dir_path))
        y_dir = sorted(os.listdir(y_dir_path))

        def load_batch(idx):

            xs = []
            ys = []
            for i in idx:
                ix = Image.open(x_dir_path+x_dir[i])
                iy = Image.open(y_dir_path+y_dir[i])
                xs.append(np.array(ix))
                ys.append(np.array(iy))
            x = tf.constant(np.array(xs), dtype=tf.float32)
            y = tf.one_hot(tf.constant(np.array(ys)), num_classes, axis=-1, dtype=tf.float32)

            return x, y

        num_classes = 12
        batch_size = batch_size
        num_elem = len(os.listdir(x_dir_path))

        pre = Crop(shape=(224,224,3))
        mo = self.model
        me = MeanIOU(num_classes)

        opt = tf.keras.optimizers.Adam(2e-4)
        cce = tf.keras.losses.CategoricalCrossentropy()
        # weights = tf.constant([0.1, 0.5, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1], dtype=tf.float32)

        idx = list(range(num_elem))
        random.shuffle(idx)
        self.train_idx = train_idx = idx[:int(len(idx)*0.8)]
        self.valid_idx = valid_idx = idx[int(len(idx)*0.8):int(len(idx)*0.9)]
        self.test_idx = test_idx = idx[int(len(idx)*0.9):]
        for ep in tqdm(range(num_epochs)):

            # Train 
            random.shuffle(train_idx)
            num_batch = len(train_idx) // batch_size
            for i in tqdm(range(num_batch), desc="inner loop"):
                idx_batch = train_idx[i*batch_size: (i+1)*batch_size]
                xbatch, ybatch = load_batch(idx_batch)
                xbatch = pre.func(xbatch)
                ybatch = pre.func(ybatch)
                with tf.GradientTape() as tape:
                    ypred = mo(xbatch)
                    _y = tf.reshape(ybatch, shape=(-1, ybatch.shape[-1]))
                    _ypred = tf.reshape(ypred, shape=(-1, ypred.shape[-1]))
                    loss = cce(_y, _ypred)
                    # loss = weighted_cce(_y, _ypred, weights)
                grads = tape.gradient(loss, mo.trainable_weights)
                opt.apply_gradients(zip(grads, mo.trainable_weights))

            # Valid
            random.shuffle(valid_idx)
            num_batch = len(valid_idx) // batch_size
            for i in range(num_batch):
                idx_batch = train_idx[i*batch_size: (i+1)*batch_size]
                vxbatch, vybatch = load_batch(idx_batch)
                vxbatch = pre.func(vxbatch)
                vybatch = pre.func(vybatch)
                vypred = mo(vxbatch)
                v_y = tf.reshape(vybatch, shape=(-1, vybatch.shape[-1]))
                v_ypred = tf.reshape(vypred, shape=(-1, vypred.shape[-1]))
                v_loss = cce(v_y, v_ypred)

            self.model = mo

            print("ep ", ep, " | loss ", loss.numpy(), " | iou ", me.func(ybatch, ypred).numpy(), " | v_loss ", v_loss.numpy(), " | v_iou ", me.func(vybatch, vypred).numpy())
