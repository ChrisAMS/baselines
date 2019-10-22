import tensorflow as tf


def l_norm(W, ord_):
    return tf.reduce_sum(tf.norm(W, axis=1, ord=ord_))


def l21_norm(v):
    # Computes a group regularization loss from a list of weight matrices corresponding
    # to the different layers (see line 93 for its use).
    const_coeff = lambda W: tf.sqrt(tf.cast(W.get_shape().as_list()[1], tf.float32))
    return tf.reduce_sum([tf.multiply(const_coeff(W), l_norm(W, 2)) for W in v if 'bias' not in W.name])
