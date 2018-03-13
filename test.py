# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/vq-vae
'''

from __future__ import print_function

import tensorflow as tf
from data_load import load_data
from hparams import Hyperparams as hp
from utils import get_wav, mu_law_decode
from train import Graph
import librosa
import numpy as np
from scipy.io.wavfile import write
import os
from tqdm import tqdm

def test():
    # Load data: two samples
    files, speaker_ids = load_data(mode="test")
    speaker_ids = speaker_ids[::-1] # swap

    # Parse
    x = np.zeros((2, 63488, 1), np.int32)
    for i, f in enumerate(files):
        f = np.load(f)
        length = min(63488, len(f))
        x[i, :length, :] = f[:length]

    # Graph
    g = Graph("test"); print("Test Graph loaded")
    with tf.Session() as sess:
        saver = tf.train.Saver()

        # Restore saved variables
        ckpt = tf.train.latest_checkpoint(hp.logdir)
        if ckpt is not None: saver.restore(sess, ckpt)

        # Feed Forward
        y_hat = np.zeros((2, 63488, 1), np.int32)
        for j in tqdm(range(63488)):
            _y_hat = sess.run(g.y_hat, {g.x: x, g.y: y_hat, g.speaker_ids: speaker_ids})
            _y_hat = np.expand_dims(_y_hat, -1)
            y_hat[:, j, :] = _y_hat[:, j, :]

        for i, y in tqdm(enumerate(y_hat)):
            audio = mu_law_decode(y)
            write(os.path.join(hp.sampledir, '{}.wav'.format(i + 1)), hp.sr, audio)


if __name__ == '__main__':
    test()
    print("Done")