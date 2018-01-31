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
        files = glob.glob(hp.data)
        for f in files:
            p=os.path.basename(f)[:4]
            s = speaker2id(p)
            if s is None:
                print(p)
        speaker_ids = [speaker2id(os.path.basename(f)[:4]) for f in files]
    else: # evaluation. two samples.
        files = [line.split("|")[0] for line in hp.test_data.splitlines()]
        speaker_ids = [int(line.split("|")[1]) for line in hp.test_data.splitlines()]
    return files, speaker_ids

def get_batch():
    """Loads training data and put them in queues"""
    with tf.device('/cpu:0'):
        # Load data
        files, speakers = load_data() # list

        # Calc total batch count
        num_batch = len(files) // hp.batch_size

        # to tensor
        files = tf.convert_to_tensor(files, tf.string)
        speakers = tf.convert_to_tensor(speakers, tf.int32)

        # Create Queues
        f, speaker = tf.train.slice_input_producer([files, speakers], shuffle=True)

        # Parse
        wav, qt = tf.py_func(get_wav, [f], [tf.float32, tf.int32])  # (T,), (T, 1)

        # Add shape information
        wav.set_shape((None,))
        qt.set_shape((hp.T, 1))

        # Batching
        qts, wavs, speakers = tf.train.batch(tensors=[qt, wav, speaker],
                                             batch_size=hp.batch_size,
                                             shapes=([hp.T, 1], [None,], []),
                                             num_threads=32,
                                             dynamic_pad=True)

    return qts, wavs, speakers, num_batch