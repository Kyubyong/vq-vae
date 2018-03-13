# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/vq-vae
'''

from hparams import Hyperparams as hp
import librosa
import numpy as np

def mu_law_encode(audio):
    '''Quantizes waveform amplitudes.
    Mostly adaped from
    https://github.com/ibab/tensorflow-wavenet/blob/master/wavenet/ops.py#L64-L75

    Args:
      audio: Raw wave signal. float32.
    '''
    mu = float(hp.Q - 1)
    # Perform mu-law companding transformation (ITU-T, 1988).
    # Minimum operation is here to deal with rare large amplitudes caused
    # by resampling.
    magnitude = np.log1p(mu * np.abs(audio)) / np.log1p(mu)
    signal = np.sign(audio) * magnitude
    # Quantize signal to the specified number of levels.
    return ((signal + 1) / 2 * mu + 0.5).astype(np.int32)


def mu_law_decode(output):
    '''Recovers waveform from quantized values.
    Mostly adapted from
    https://github.com/ibab/tensorflow-wavenet/blob/master/wavenet/ops.py#L64-L75
    '''
    mu = hp.Q - 1.
    # Map values back to [-1, 1].
    signal = 2 * (output.astype(np.float32) / mu) - 1
    # Perform inverse of mu-law transformation.
    magnitude = (1 / mu) * ((1 + mu)**np.abs(signal) - 1)
    return np.sign(signal) * magnitude

def get_wav(fpath):
    '''Loads waveform from `fpath` and
    quantize it.

    Args:
      fpath: A string. Sound file path.

    Returns:
      wav: A float32 array of raw waveform. Shape is [None,].
      qt: A float32 array of quantized waveform. Same shape as `wav`.
    '''
    wav, sr = librosa.load(fpath, sr=hp.sr)
    wav, _ = librosa.effects.trim(wav)
    wav_ = wav / np.abs(wav).max()
    qt = mu_law_encode(wav_)

    # # Padding
    # qt = np.pad(qt, ([0, maxlen]), mode="constant")[:maxlen]

    # Dimension expansion
    qt = np.expand_dims(qt, -1) # (None, 1)

    return wav.astype(np.float32), qt.astype(np.int32)
