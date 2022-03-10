import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random 

import tensorflow as tf

from improved_nightdrive.pipeline.pipeline import Training
from improved_nightdrive.pipeline.metrics import MeanIOU
from improved_nightdrive.pipeline.preprocess import Crop
from improved_nightdrive.segmentation.models import u_net_pretrained

num_classes = 12

model = u_net_pretrained(num_classes)
metric = MeanIOU(num_classes)
preprocess = Crop(shape=(224,224,3))
T = Training(
    model=model,
    loss='cce',
    metrics=[metric],
    preprocesses=[preprocess],
)

T.train(
    num_epochs=20,
    batch_size=12,
)

### VISUALIZE RESULTS ###

x_dir = sorted(os.listdir("./dataset/rgb/"))
y_dir = sorted(os.listdir("./dataset/seg/"))

test_id = random.choice(T.test_idx)

x = np.expand_dims(np.array(Image.open("./dataset/rgb/"+x_dir[test_id]), dtype=np.float32), axis=0)
y = np.expand_dims(np.expand_dims(np.array(Image.open("./dataset/seg/"+y_dir[test_id]), dtype=np.float32), axis=-1), axis=0)
x = preprocess.func(x)
y = np.squeeze(preprocess.func(y), axis=-1)
ypred = T.model(x)
f, ax = plt.subplots(1, 3)
ypred = np.argmax(ypred, axis=-1)[0].astype(np.uint8)
y = tf.keras.utils.to_categorical(y, num_classes)
y = np.argmax(y, axis=-1)[0].astype(np.uint8)
ax[0].imshow(ypred)
ax[1].imshow(y)
ax[2].imshow(x.numpy()[0].astype(np.uint8))
plt.show()