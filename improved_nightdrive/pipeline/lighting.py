"""Scripts to apply local image process"""
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow_addons.image import gaussian_filter2d

from improved_nightdrive.pipeline.preprocess import GammaProcess


def apply_lighting(
    original_input: np.ndarray,
    prediction: np.ndarray,
    gamma_list: List[float],
) -> np.ndarray:

    gamma_process = GammaProcess()
    gamma_lists = [[g, g, g] for g in gamma_list]
    modified_inputs = gamma_process.masked_func(
        original_input,
        gamma_lists,
        prediction,
    )

    return modified_inputs


def plot_mask(
    prediction: tf.Tensor, mask_indice: int, original: Optional[tf.Tensor] = None
) -> None:

    assert len(prediction.shape) == 3
    masked_input = original * prediction[..., mask_indice, None]
    if original is not None:
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(prediction[..., mask_indice], vmin=0, vmax=1, cmap="gray")
        axs[1].imshow(
            tf.where(
                masked_input > 0.0,
                masked_input,
                1.0,
            )
        )
        axs[2].imshow(original)
    else:
        plt.imshow(prediction[..., mask_indice], vmin=0, vmax=1, cmap="gray")
    plt.show()


def apply_threshold(prediction: tf.Tensor, threshold: float) -> tf.Tensor:
    return tf.where(prediction > threshold, 1.0, 0.0)


def apply_gaussian_blur(
    prediction: tf.Tensor, kernel: int, sigma: float = 1.0
) -> tf.Tensor:
    return gaussian_filter2d(prediction, kernel, sigma)


def apply_gamma_map(
    input: tf.Tensor,
    mask: tf.Tensor,
    mask_indice: int,
    max_gamma: float = 0.5,
    min_gamma: float = 0.1,
):
    gamma_mask = mask * (min_gamma - max_gamma) + max_gamma
    gamma_mask = gamma_mask[..., mask_indice, None]
    input = input**gamma_mask
    return input
