# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/vq-vae
'''

from __future__ import print_function

from hparams import Hyperparams as hp
from modules import conv1d, residual_block
import tensorflow as tf


def encoder(x):
    '''
    Args:
      x: waveform. [B, T, 1]

    Returns:
      z_e: encoded variable. [B, T', D]
    '''
    with tf.variable_scope("encoder"):
        for i in range(hp.encoder_layers):
            x = tf.pad(x, [[0, 0], [1, 1], [0, 0]])
            x = conv1d(x,
                       filters=hp.D,
                       size=hp.winsize,
                       strides=hp.stride,
                       padding="valid",
                       bn=True,
                       activation_fn=tf.nn.relu if i < hp.encoder_layers-1 else None,
                       scope="conv1d_{}".format(i))
    z_e = x
    return z_e

def vq(z_e):
    '''Vector Quantization.

    Args:
      z_e: encoded variable. [B, t, D].

    Returns:
      z_q: nearest embeddings. [B, t, D].
    '''
    with tf.variable_scope("vq"):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[hp.K, hp.D],
                                       initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
        z = tf.expand_dims(z_e, -2) # (B, t, 1, D)
        lookup_table_ = tf.reshape(lookup_table, [1, 1, hp.K, hp.D]) # (1, 1, K, D)
        dist = tf.norm(z - lookup_table_, axis=-1) # Broadcasting -> (B, T', K)
        k = tf.argmin(dist, axis=-1) # (B, t)
        z_q = tf.gather(lookup_table, k) # (B, t, D)

    return z_q

def decoder(decoder_inputs, speaker_emb, z_q):
    '''Wavenet decoder.
    Args:
      decoder_inputs: [B, T, 1].
      speaker_emb: [B, len(speakser)]. One-hot. Global condition.
      z_q: [B, T', D]. Local condition.

    '''
    with tf.variable_scope("decoder"):
        # Prenet
        z = conv1d(decoder_inputs, hp.num_units, activation_fn=tf.tanh, padding="causal", bn=True, scope='conv_in') # (B, T, H)

        # Residual blocks
        skip = 0  # skip connections
        for i in range(hp.num_blocks):
            for r in hp.dilations:
                z, s = residual_block(z, size=hp.size, rate=r, speaker_emb=speaker_emb, z_q=z_q, scope="res_block_{}_{}".format(i, r))
                skip += s

        # Postnet
        skip = tf.nn.relu(skip)
        skip = conv1d(skip, padding="causal", activation_fn=tf.nn.relu, bn=True, scope="one_by_one_1") # (B, T, H)
        y = conv1d(skip, filters=hp.Q, padding="causal", scope="one_by_one_2") # (B, T, Q) wave logits.

    return y
