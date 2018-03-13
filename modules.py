# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/vq-vae
'''

from __future__ import print_function, division

import tensorflow as tf


def conv1d(inputs,
           filters=None,
           size=1,
           rate=1,
           strides=1,
           padding="SAME",
           dropout_rate=0,
           use_bias=True,
           activation_fn=None,
           bn=False,
           training=True,
           scope="conv1d",
           reuse=None):
    '''
    Args:
      inputs: A 3-D tensor with shape of [batch, time, depth].
      filters: An int. Number of outputs (=activation maps)
      size: An int. Filter size.
      rate: An int. Dilation rate.
      padding: Either `same` or `valid` or `causal` (case-insensitive).
      dropout_rate: A float of [0, 1].
      use_bias: A boolean.
      activation_fn: A string.
      training: A boolean. If True, dropout is applied.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
    Returns:
      A masked tensor of the same shape and dtypes as `inputs`.
    '''
    with tf.variable_scope(scope):
        if padding.lower() == "causal":
            # pre-padding for causality
            pad_len = (size - 1) * rate  # padding size
            inputs = tf.pad(inputs, [[0, 0], [pad_len, 0], [0, 0]])
            padding = "valid"

        if filters is None:
            filters = inputs.get_shape().as_list()[-1]

        params = {"inputs": inputs, "filters": filters, "kernel_size": size,
                  "strides": strides, "dilation_rate": rate, "padding": padding,
                  "use_bias": use_bias, "reuse": reuse}

        tensor = tf.layers.conv1d(**params)
        if bn:
            tensor = tf.layers.batch_normalization(tensor, training=training, renorm=True)

        if activation_fn is not None:
            tensor = activation_fn(tensor)

        tensor = tf.layers.dropout(tensor, rate=dropout_rate, training=training)

    return tensor

def residual_block(inputs, size, rate, speaker_emb, z_q, scope="res_block", reuse=None):
    '''
    Args:
      inputs: (B, T, H)
      size: Kernel size.
      rate: Dilation rate.
      speaker_emb: [B, len(speakers)]. One-hot speaker embedding. Global condition.
      z_q: nearest embeddings. [B, t, D], where t is compressed temporal dimensionality.
      scope:
      reuse:

    Returns
      residual connected: outputs + inputs. (B, T, H)
      outputs: (B, T, H)
    '''
    _, T, H = inputs.get_shape().as_list() # T: original temporal dimensionality
    B, t, D = z_q.get_shape().as_list() # t: compressed temporal dimensionality
    with tf.variable_scope(scope, reuse=reuse):
        # convolution
        conv = conv1d(inputs, filters=H, size=size, rate=rate, bn=True, padding="causal", scope="conv") # (B, T, H)

        # conditions
        speaker_emb = tf.tile(tf.expand_dims(speaker_emb, 1), [1, t, 1]) # (B, t, L)
        cond = conv1d(inputs=tf.concat((speaker_emb, z_q), -1), filters=H, padding="causal", bn=True) # (B, t, H)

        # Merge
        cond = tf.expand_dims(cond, -2) # (B, t, 1, H)
        # print(cond, conv)
        conv = tf.reshape(conv, (B, t, -1, H)) # (B, t, ?, H)

        conv += cond # (B, t, ?, H)
        conv = tf.reshape(conv, (B, T, H))  # (B, T, H)

        # Gated Activation
        conv, gate = tf.split(conv, 2, -1) # 2 * (B, T, H/2)
        conv = tf.tanh(conv)
        gate = tf.nn.sigmoid(gate)
        conv *= gate # (B, T, H/2)

        # Channel Restoration
        outputs = conv1d(conv, filters=H, size=1, scope="one_by_one")

    return outputs + inputs, outputs
