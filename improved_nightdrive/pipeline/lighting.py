"""Scripts to apply local image process"""
from typing import Optional

import matplotlib.pyplot as plt
import tensorflow as tf


def plot_mask(
    prediction: tf.Tensor, mask_indice: int, original: Optional[tf.Tensor] = None
) -> None:
    """Plot a prediction mask"""
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
    """Apply a threshold on image"""
    return tf.where(prediction > threshold, 1.0, 0.0)


def apply_gamma_map(
    input: tf.Tensor,
    mask: tf.Tensor,
    mask_indice: int,
    max_gamma: float = 0.5,
    min_gamma: float = 0.1,
):
    """Apply gamma process to all pixels"""
    gamma_mask = mask * (min_gamma - max_gamma) + max_gamma
    gamma_mask = gamma_mask[..., mask_indice, None]
    input = input**gamma_mask
    return input
