import tensorflow as tf
import matplotlib.pyplot as plt
# import tensorflow_io as tfio


from PIL import Image

from utils import *

img = tf.convert_to_tensor(Image.open('ressources/night_images/night1.jpg'), dtype=tf.float32) / 255

images = {'original' : img}
plt.imshow(img)
plt.show()

## log processing

modified_img = tf.identity(img)
modified_img = log_process(modified_img)

modified_img /= tf.reduce_max(modified_img)

images['log process'] = modified_img


## gamma processing

modified_img = tf.identity(img)
modified_img = gamma_process(modified_img, 2)

modified_img /= tf.reduce_max(modified_img)

images['gamma process'] = modified_img
plt.imshow(modified_img)
plt.show()



## both processing

modified_img = tf.identity(img)
modified_img = log_process(gamma_process(modified_img, .4))

modified_img /= tf.reduce_max(modified_img)

images['log/gamma process'] = modified_img
plt.imshow(modified_img)
plt.show()


## histogram equalizing

modified_img = tf.identity(img)

modified_img = tf.stack([tf_equalize_histogram(tf.expand_dims(modified_img[:, :, i], axis=-1)) for i in range(3)], axis=-1)

plt.imshow(modified_img)
plt.show()

images['3 channel'] = modified_img


# ## lab histogram equalizing

# modified_img = tf.identity(img)

# modified_img = tfio.experimental.color.rgb_to_lab(modified_img)


# lum = tf.expand_dims(lab_hist(tf.expand_dims(modified_img[:, :, 0], axis=-1)), axis=-1)
# modified_img = tf.concat((tf.cast(lum, dtype=tf.float32), modified_img[:, :, 1:]), axis=-1)

# modified_img = tfio.experimental.color.lab_to_rgb(modified_img)

# modified_img = tf.cast(modified_img, tf.float32)

# modified_img /= tf.reduce_max(modified_img)
# plt.imshow(modified_img)
# plt.show()

# images['lab hist process'] = modified_img


# ## lab gamma equalizing

# modified_img = tf.identity(img)

# modified_img = tfio.experimental.color.rgb_to_lab(modified_img)

# modified_img = lab_gamma_process(modified_img, p=2)

# modified_img = tfio.experimental.color.lab_to_rgb(modified_img)

# modified_img = tf.cast(modified_img, tf.float32)

# modified_img /= tf.reduce_max(modified_img)
# plt.imshow(modified_img)
# plt.show()

# images['lab gamma process'] = modified_img


## adaptative

modified_img = tf.cast(img*255, dtype=tf.uint8)
print(modified_img.dtype)

modified_img = fast_clahe(modified_img)

plt.imshow(modified_img, cmap='gray')
plt.show()

images['adaptative hist process'] = modified_img


# ## lab adaptative histogram equalizing

# modified_img = tf.identity(img*255)

# modified_img = tfio.experimental.color.rgb_to_lab(modified_img)

# lum = tf.expand_dims(fast_clahe(modified_img[:, :, 0] * 255), axis=-1)
# modified_img = tf.concat((tf.cast(lum * 100 / 255, dtype=tf.float32), modified_img[:, :, 1:]), axis=-1)

# modified_img = tfio.experimental.color.lab_to_rgb(modified_img)

# modified_img = tf.cast(modified_img, tf.float32)

# modified_img /= tf.reduce_max(modified_img)
# plt.imshow(lum, cmap='gray')
# plt.show()

# images['lab ada hist process'] = modified_img


## plotting

w = int((len(images.keys())-1)**.5) + 1
h = len(images.keys()) // w + ((len(images.keys()) % w) > 0)
_, axes = plt.subplots(h, w)

for i, key in enumerate(images.keys()):
    axes[i//w, i%w].set_title(key)
    axes[i//w, i%w].axis('off')
    axes[i//w, i%w].imshow(images[key], cmap='gray')

for i in range(len(images.keys())-1, w*h):
    axes[i//w, i%w].axis('off')

plt.show()