import os
import numpy as np
from PIL import Image
import tensorflow as tf

from improved_nightdrive.pipeline.preprocess import ReClass


def class_proportion(label_path: str):
    reclass = ReClass()

    class_count = np.zeros((5,))
    pixel_count = 0
    for label_name in os.listdir(label_path):
        label = np.expand_dims(
            np.array(Image.open(os.path.join(label_path, label_name))), axis=0
        )
        label = tf.one_hot(label, 19, axis=-1, dtype=tf.float32)
        label = reclass.change_class(label)
        for i in range(len(class_count)):
            class_count[i] += np.sum(label[..., i])
        pixel_count += np.prod(label.shape[:-1])

    return class_count / pixel_count