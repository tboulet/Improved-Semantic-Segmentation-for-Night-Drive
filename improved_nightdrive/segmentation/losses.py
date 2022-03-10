import tensorflow as tf

def weighted_cce(y: tf.Tensor, ypred: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    assert len(weights) == ypred.shape[-1]

    _y = tf.reshape(y, shape=(-1, y.shape[-1]))
    _ypred = tf.reshape(ypred, shape=(-1, y.shape[-1]))
    _weights = tf.expand_dims(weights, axis=0)

    loss = - tf.math.reduce_sum(_y * tf.math.log(_ypred) * _weights, axis=-1)

    return tf.reduce_mean(loss)