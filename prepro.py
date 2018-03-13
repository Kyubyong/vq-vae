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
from multiprocessing import Pool
from tqdm import tqdm

# Creates pool
p = Pool(4)

def f(fpath):
    w, q = get_wav(fpath)
    fname = os.path.basename(fpath).replace('wav', 'npy')
    if not os.path.exists("/data/private/speech/vctk/wavs"): os.makedirs("/data/private/speech/vctk/wavs")
    if not os.path.exists("/data/private/speech/vctk/qts"): os.makedirs("/data/private/speech/vctk/qts")
    np.save("/data/private/speech/vctk/wavs/{}".format(fname), w)
    np.save("/data/private/speech/vctk/qts/{}".format(fname), q)

fpaths = glob.glob(hp.data)
total_files = len(fpaths)
with tqdm(total=total_files) as pbar:
    for i, _ in tqdm(enumerate(p.imap_unordered(f, fpaths))):
        pbar.update()




