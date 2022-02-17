import tensorflow as tf
from tensorflow.keras import applications as ka

ka.MobileNetV2(
    input_shape=None,
    alpha=1.0,
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)

print(ka.summary)