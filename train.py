# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/vq-vae
'''

from __future__ import print_function

import tensorflow as tf
from tqdm import tqdm
from data_load import get_batch, load_data
from hparams import Hyperparams as hp
from networks import encoder, vq, decoder
from utils import mu_law_decode, get_wav

class Graph:
    def __init__(self, mode="train"):
        '''
        Args:
          mode: Either "train" or "eval".
        '''
        # Set flag
        training = True if mode=="train" else False

        # Graph
        # Data Feeding
        ## x: Quantized wav. (B, T, 1) int32
        ## wavs: Raw wav. (B, length) float32
        ## speakers: Speaker ids. (B,). [0, 108]. int32.
        if mode=="train":
            self.x, self.wavs, self.speaker_ids, self.num_batch = get_batch()
            self.y = self.x
        else:  # test
            self.x = tf.placeholder(tf.int32, shape=(2, 63488, 1))
            self.y = tf.placeholder(tf.int32, shape=(2, 63488, 1))
            self.speaker_ids = tf.placeholder(tf.int32, shape=(2,))

        # inputs:
        self.encoder_inputs = tf.to_float(self.x)
        self.decoder_inputs = tf.to_float(self.y)
        self.decoder_inputs = tf.concat((tf.zeros_like(self.decoder_inputs[:, :1, :]), self.decoder_inputs[:, :-1, :]), 1)

        # speaker embedding
        self.speakers = tf.one_hot(self.speaker_ids, len(hp.speakers)) # (B, len(speakers))

        # encoder
        self.z_e = encoder(self.encoder_inputs) # (B, T', D)

        # vq
        self.z_q = vq(self.z_e) # (B, T', D)

        # decoder: y -> reconstructed logits.
        self.y_logits = decoder(self.decoder_inputs, self.speakers, self.z_q) # (B, T, Q)
        self.y_hat = tf.argmax(self.y_logits, -1) # (B, T)

        # monitor
        self.sample0 = tf.py_func(mu_law_decode, [self.y_hat[0]], tf.float32)
        self.sample1 = tf.py_func(mu_law_decode, [self.y_hat[1]], tf.float32)

        # speech samples
        # tf.summary.audio('{}/original1'.format(mode), self.wavs[:1], hp.sr, 1)
        # tf.summary.audio('{}/original2'.format(mode), self.wavs[1:], hp.sr, 1)
        tf.summary.audio('{}/sample0'.format(mode), tf.expand_dims(self.sample0, 0), hp.sr, 1)
        tf.summary.audio('{}/sample1'.format(mode), tf.expand_dims(self.sample1, 0), hp.sr, 1)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if training:
            self.dec_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.y_logits, labels=tf.squeeze(self.y)))
            self.vq_loss = tf.reduce_mean(tf.squared_difference(tf.stop_gradient(self.z_e), self.z_q))
            self.enc_loss = hp.beta * tf.reduce_mean(tf.squared_difference(self.z_e, tf.stop_gradient(self.z_q)))

            # decoder grads
            decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "decoder")
            decoder_grads = tf.gradients(self.dec_loss, decoder_vars)
            decoder_grads_vars = list(zip(decoder_grads, decoder_vars))

            # embedding variables grads
            embed_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "vq")
            embed_grads = tf.gradients(self.dec_loss + self.vq_loss, embed_vars)
            embed_grads_vars = list(zip(embed_grads, embed_vars))

            # encoder grads
            encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "encoder")
            transferred_grads = tf.gradients(self.dec_loss, self.z_q)
            encoder_grads = [tf.gradients(self.z_e, var, transferred_grads)[0] + tf.gradients(self.enc_loss, var)[0]
                                 for var in encoder_vars]
            encoder_grads_vars = list(zip(encoder_grads, encoder_vars))

            # total grads
            self.grads_vars = decoder_grads_vars + embed_grads_vars + encoder_grads_vars

            # Training Scheme
            self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr)

            # Summary
            tf.summary.scalar('train/dec_loss', self.dec_loss)
            tf.summary.scalar('train/vq_loss', self.vq_loss)
            tf.summary.scalar('train/enc_loss', self.enc_loss)

            # tf.summary.scalar("lr", self.lr)

            # gradient clipping
            self.clipped = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in self.grads_vars]

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.train_op = self.optimizer.apply_gradients(self.clipped, global_step=self.global_step)

        # Summary
        self.merged = tf.summary.merge_all()

if __name__ == '__main__':
    g = Graph(); print("Training Graph loaded")

    sv = tf.train.Supervisor(logdir=hp.logdir, save_model_secs=0, global_step=g.global_step)
    with sv.managed_session() as sess:
        while 1:
            for _ in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                gs, _ = sess.run([g.global_step, g.train_op])

                # Write checkpoint files at every 1k steps
                if gs % 100 == 0:
                    sv.saver.save(sess, hp.logdir + '/model_gs_{}'.format(str(gs).zfill(5)))

    print("Done")