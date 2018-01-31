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
        ## wavs: Raw wav. (B, length, 1) float32
        ## speakers: Speaker ids. (B,). [0, 108]. int32.
        if mode=="train":
            self.x, self.wavs, self.speakers, self.num_batch = get_batch()
        else:  # eval
            self.x = tf.placeholder(tf.int32, shape=(2, 3*hp.T))
            self.wavs = tf.placeholder(tf.float32, shape=(2, 3*hp.T))
            self.speakers = tf.placeholder(tf.int32, shape=(2,))

        # inputs:
        self.encoder_inputs = tf.to_float(self.x)
        self.decoder_inputs = tf.concat((tf.zeros_like(self.encoder_inputs[:, :1, :]), self.encoder_inputs[:, :-1, :]), 1)

        # speaker embedding
        self.speakers = tf.one_hot(self.speakers, len(hp.speakers)) # (B, len(speakers))

        # encoder
        self.z_e = encoder(self.encoder_inputs) # (B, T', D)

        # vq
        self.z_q = vq(self.z_e) # (B, T', D)

        # decoder: y -> reconstructed logits.
        self.y = decoder(self.decoder_inputs, self.speakers, self.z_q) # (B, T, Q)
        self.y_hat = tf.argmax(self.y, -1) # (B, T)
        tf.gradients
        # monitor
        self.sample0 = tf.py_func(mu_law_decode, [self.y_hat[0]], tf.float32)
        self.sample1 = tf.py_func(mu_law_decode, [self.y_hat[1]], tf.float32)

        # speech samples
        tf.summary.audio('{}/original1'.format(mode), self.wavs[0], hp.sr, 1)
        tf.summary.audio('{}/original2'.format(mode), self.wavs[1], hp.sr, 1)
        tf.summary.audio('{}/sample0'.format(mode), tf.expand_dims(self.sample0, 0), hp.sr, 1)
        tf.summary.audio('{}/sample1'.format(mode), tf.expand_dims(self.sample1, 0), hp.sr, 1)

        if training:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            self.rec_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.y, labels=tf.squeeze(self.x)))
            self.vq_loss = tf.reduce_mean(tf.squared_difference(tf.stop_gradient(self.z_e), self.z_q))
            self.enc_loss = hp.beta * tf.reduce_mean(tf.squared_difference(self.z_e, tf.stop_gradient(self.z_q)))

            # decoder grads
            decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "decoder")
            decoder_grads = tf.gradients(self.rec_loss, decoder_vars)

            # embedding variables grads
            embed_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "vq")
            embed_grads = tf.gradients(self.rec_loss + self.vq_loss, embed_vars)

            # encoder grads
            encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "encoder")
            transferred_grads = tf.gradients(self.rec_loss, self.z_q)
            encoder_grads = tf.gradients(self.enc_loss, encoder_vars, transferred_grads)

            # total grads
            self.grads = decoder_grads + embed_grads + encoder_grads

            # Training Scheme
            self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr)

            # Summary
            tf.summary.scalar('train/rec_loss', self.rec_loss)
            tf.summary.scalar('train/vq_loss', self.vq_loss)
            tf.summary.scalar('train/enc_loss', self.enc_loss)
            # tf.summary.scalar('train/LOSS', self.loss)

            # tf.summary.scalar("lr", self.lr)

            ## gradient clipping
            # self.gvs = self.optimizer.compute_gradients(self.loss)
            # self.clipped = []
            # for grad, var in self.gvs:
            #     grad = tf.clip_by_value(grad, -1., 1.)
            #     self.clipped.append((grad, var))

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.train_op = self.optimizer.apply_gradients(self.grads, global_step=self.global_step)

        # Summary
        self.merged = tf.summary.merge_all()

def eval():
    # Load data: two samples
    files, speaker_ids = load_data(mode="eval")
    speaker_ids = speaker_ids[::-1] # swap

    # Parse
    wavs, x = [], []
    for f in files:
        wav, qt = get_wav(f, 3*hp.T)
        wavs.append(wav)
        x.append(qt)

    # Graph
    g = Graph("eval"); print("Evaluation Graph loaded")
    with tf.Session() as sess:
        saver = tf.train.Saver()
        # Restore saved variables
        ckpt = tf.train.latest_checkpoint(hp.logdir)
        if ckpt is not None: saver.restore(sess, ckpt)

        # Writer
        writer = tf.summary.FileWriter(hp.logdir, sess.graph)

        # Evaluation
        merged, gs = sess.run([g.merged, g.global_step], {g.wavs: wavs, g.x: x, g.speakers: speaker_ids})

        #  Write summaries
        writer.add_summary(merged, global_step=gs)
        writer.close()

if __name__ == '__main__':
    g = Graph(); print("Training Graph loaded")

    sv = tf.train.Supervisor(logdir=hp.logdir, save_model_secs=0, global_step=g.global_step)
    with sv.managed_session() as sess:
        while 1:
            print(g.num_batch)
            for _ in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                gs, _ = sess.run([g.global_step, g.train_op])

                # Write checkpoint files at every 1k steps
                if gs % 10 == 0:
                    sv.saver.save(sess, hp.logdir + '/model_gs_{}'.format(str(gs // 1000).zfill(3) + "k"))

                    # evaluation
                    y = sess.run(g.y)


                # break
                # if gs > hp.num_iterations: break

print("Done")