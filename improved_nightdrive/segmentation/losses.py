
import tensorflow as tf


def softmax_weighted_cce(
    targets: tf.Tensor, 
    predictions_logits: tf.Tensor, 
    weights: tf.Tensor
) -> tf.Tensor:
    """Computes the weight categorical crossentropy
    
    Arguments:
        targets (tf.Tensor), [batch_size, num_classes]
        predictions_logits (tf.Tensor), [batch_size, num_classes]
        weights (tf.Tensor), [batch_size,]
    
    Returns:
        loss (float)
    """
    _targets = tf.reshape(targets, shape=(-1, targets.shape[-1]))
    _predictions = tf.reshape(predictions_logits, shape=(-1, predictions_logits.shape[-1]))
    _weights = tf.expand_dims(weights, axis=0)

    loss = - tf.math.reduce_sum(_targets * tf.math.log_softmax(_predictions) * _weights, axis=-1)

    return tf.reduce_mean(loss)