# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/vq-vae
'''

from __future__ import print_function

from hparams import Hyperparams as hp
import tensorflow as tf
from utils import get_wav
import os
import glob
import numpy as np

def speaker2id(speaker):
    func = {speaker:id for id, speaker in enumerate(hp.speakers)}
    return func.get(speaker, None)

def id2speaker(id):
    func = {id:speaker for id, speaker in enumerate(hp.speakers)}
    return func.get(id, None)

def load_data(mode="train"):
    '''Loads data
    Args:
      mode: "train" or "eval".

    Returns:
      files: A list of sound file paths.
      speaker_ids: A list of speaker ids.
    '''
    if mode=="train":
        wavs = glob.glob('vctk/wavs/*.npy')
        qts = [wav.replace("wavs", "qts") for wav in wavs]
        speakers = np.array([speaker2id(os.path.basename(wav)[:4]) for wav in wavs], np.int32)

        return wavs, qts, speakers
        # speaker_ids = [speaker2id(os.path.basename(f)[:4]) for f in files]
    else: # evaluation. two samples.
        files = [line.split("|")[0] for line in hp.test_data.splitlines()]
        speaker_ids = [int(line.split("|")[1]) for line in hp.test_data.splitlines()]
        return files, speaker_ids

# load_data()
def get_batch():
    """Loads training data and put them in queues"""
    with tf.device('/cpu:0'):
        # Load data
        # files, speakers = load_data() # list
        wavs, qts, speakers = load_data() # list


        # Calc total batch count
        # num_batch = len(files) // hp.batch_size
        num_batch = len(wavs) // hp.batch_size

        # to tensor
        # files = tf.convert_to_tensor(files, tf.string)
        wavs = tf.convert_to_tensor(wavs, tf.string)
        qts = tf.convert_to_tensor(qts, tf.string)
        speakers = tf.convert_to_tensor(speakers, tf.int32)

        # Create Queues
        # f, speaker = tf.train.slice_input_producer([files, speakers], shuffle=True)
        wav, qt, speaker = tf.train.slice_input_producer([wavs, qts, speakers], shuffle=True)

        print("A")

        # Parse
        def _parse(wav):
            speaker_id = speaker2id(os.path.basename(wav)[:4])
            return np.array(speaker_id, np.int32)
        # wav, qt = tf.py_func(get_wav, [f], [tf.float32, tf.int32])  # (T,), (T, 1)
        wav, = tf.py_func(lambda x: np.load(x), [wav], [tf.float32])  # (T,), (T, 1)
        qt, = tf.py_func(lambda x: np.load(x), [qt], [tf.int32])  # (T,), (T, 1)
        # speaker, = tf.py_func(_parse, [wav], [tf.int32])  # (T,), (T, 1)

        print("A")
        # Add shape information
        wav.set_shape((None,))
        qt.set_shape((hp.T, 1))
        speaker.set_shape(())
        print("A")

        # Batching
        qts, wavs, speakers = tf.train.batch(tensors=[qt, wav, speaker],
                                             batch_size=hp.batch_size,
                                             shapes=([hp.T, 1], [None,], []),
                                             num_threads=32,
                                             dynamic_pad=True)
        print("A")
        return qts, wavs, speakers, num_batch