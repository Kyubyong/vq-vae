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
from data_load import load_data
import numpy as np

# Load data
files, speakers = load_data() # list

for f, s in zip(files, speakers):
    w, q = get_wav(f)
    fname = "vctk/" + os.path.basename(f).replace('wav', 'npz')
    np.savez(fname, wav=w, qt=q, speaker=s)

    # print(w.shape, q.shape)
    # print(f, s)
