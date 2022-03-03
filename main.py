from curses.ascii import CR
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import os
from PIL import Image
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

from improved_nightdrive.pipeline.preprocess import Crop
from improved_nightdrive.pipeline import Evaluation, MeanIOU
from improved_nightdrive.segmentation.models import MobileUNET_S, PreTrainedMobileUnetv2

# def weighted_cce(y, ypred, weights):
#     assert len(weights) == ypred.shape[-1]

#     _y = tf.reshape(y, shape=(-1, y.shape[-1]))
#     _ypred = tf.reshape(ypred, shape=(-1, y.shape[-1]))
#     _weights = tf.expand_dims(weights, axis=0)

#     loss = - tf.math.reduce_sum(_y * tf.math.log_softmax(_ypred, axis=-1) * _weights)

#     return tf.reduce_mean(loss)
def load_batch(idx):

    x_dir = os.listdir("./dataset/rgb/")
    y_dir = os.listdir("./dataset/seg/")
    xs = []
    ys = []
    for i in idx:
        ix = Image.open("./dataset/rgb/"+x_dir[i])
        iy = Image.open("./dataset/seg/"+y_dir[i])
        xs.append(np.array(ix))
        ys.append(np.array(iy))
    x = tf.constant(np.array(xs), dtype=tf.float32)
    y = tf.one_hot(tf.constant(np.array(ys)), num_classes, axis=-1, dtype=tf.float32)

    return x, y

num_classes = 12
batch_size = 4
num_elem = len(os.listdir("./dataset/rgb/"))

# mo = MobileUNET_S(num_classes, False)
pre = Crop(shape=(224,224,3))
mo = PreTrainedMobileUnetv2(num_classes)
me = MeanIOU(num_classes)
e = Evaluation(mo, [me], None)

# x = np.expand_dims(np.array(Image.open("0.png"), dtype=np.float32), axis=0)
# y = np.expand_dims(np.array(Image.open("0GT.png"), dtype=np.float32), axis=0)

# r = e.evaluate(x, y)
# print(r)

# xs = []
# ys = []
# for i in range(len(x_dir)):
#     ix = Image.open("./dataset/rgb/"+ x_dir[i])
#     iy = Image.open("./dataset/seg/"+y_dir[i])
#     xs.append(np.array(ix))
#     ys.append(np.array(iy))

# x = tf.constant(np.array(xs))
# y = tf.constant(np.array(ys))
# print(y.shape)
# # y = tf.reshape(y, shape=(y.shape[0], -1))
# # y = tf.keras.utils.to_categorical(y, num_classes)
# y = tf.constant(tf.one_hot(y, num_classes, axis=-1, dtype=tf.float32))

# print(y)

# opt = tfa.optimizers.AdamW(0.99, learning_rate=0.0002)
opt = tf.keras.optimizers.Adam(2e-4)
cce = tf.keras.losses.CategoricalCrossentropy()
# # weights = tf.constant([0.1, 0.5, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1], dtype=tf.float32)

# mo.compile(optimizer='adam', loss='categorical_crossentropy')
# mo.fit(x, y, epochs=100)

idx = list(range(num_elem))
for ep in tqdm(range(10)):
    random.shuffle(idx)
    num_batch = num_elem // batch_size
    for i in range(num_batch):
        idx_batch = idx[i*batch_size: (i+1)*batch_size]
        xbatch, ybatch = load_batch(idx_batch)
        xbatch = pre.func(xbatch)
        ybatch = pre.func(ybatch)
        with tf.GradientTape() as tape:
            ypred = mo.predict(xbatch)
            _y = tf.reshape(ybatch, shape=(-1, ybatch.shape[-1]))
            _ypred = tf.reshape(ypred, shape=(-1, ypred.shape[-1]))
            loss = cce(_y, _ypred)
            # loss = weighted_cce(y, ypred, weights)
        grads = tape.gradient(loss, mo.trainable_weights)
        # print(grads)
        opt.apply_gradients(zip(grads, mo.trainable_weights))
    print("ep ", ep, " | loss ", loss.numpy(), " | iou ", me.func(ybatch, ypred).numpy())

# print(me.func(y, ypred))

x_dir = os.listdir("./dataset/rgb/")
y_dir = os.listdir("./dataset/seg/")

x = np.expand_dims(np.array(Image.open("./dataset/rgb/"+x_dir[0]), dtype=np.float32), axis=0)
y = np.expand_dims(np.array(Image.open("./dataset/seg/"+y_dir[0]), dtype=np.float32), axis=0)
ypred = mo(x)
f, ax = plt.subplots(1, 2)
ypred = np.argmax(ypred, axis=-1)[0].astype(np.uint8)
y = tf.keras.utils.to_categorical(y, num_classes)
y = np.argmax(y, axis=-1)[0].astype(np.uint8)
ax[0].imshow(ypred)
ax[1].imshow(y)
plt.show()