from improved_nightdrive.pipeline import Evaluation, MeanIOU, Normalization

import tensorflow as tf

m = tf.keras.models.Sequential()
me = MeanIOU(3)
e = Evaluation(m, me)
