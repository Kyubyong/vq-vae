# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/vq-vae
'''

from __future__ import print_function

from hparams import Hyperparams as hp
from utils import get_wav
import os
import glob
import numpy as np

for f in glob.glob(hp.data):
    w, q = get_wav(f)
    fname = os.path.basename(f).replace('wav', 'npy')
    np.save("vctk/wavs/{}".format(fname), w)
    np.save("vctk/qts/{}".format(fname), q)

