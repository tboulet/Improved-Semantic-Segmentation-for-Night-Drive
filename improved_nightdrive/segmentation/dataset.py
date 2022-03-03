from typing import Tuple
import tensorflow as tf
import numpy as np
from PIL import Image

def get_label_shape(path: str) -> Tuple[int]:

    img = np.array(Image.open(path))
    print(img.shape)
    print(img)
    print(np.unique(img))
    one_hot = tf.one_hot(img, 11)
    print(one_hot.shape)

