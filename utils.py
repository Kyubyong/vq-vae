
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
    mu = float(hp.quantization_channels - 1)
    # Perform mu-law companding transformation (ITU-T, 1988).
    # Minimum operation is here to deal with rare large amplitudes caused
    # by resampling.
    safe_audio_abs = min(np.abs(audio), 1.0)
    magnitude = np.log1p(mu * safe_audio_abs) / np.log1p(mu)
    signal = np.sign(audio) * magnitude
    # Quantize signal to the specified number of levels.
    return ((signal + 1) / 2 * mu + 0.5).astype(np.int32)


def mu_law_decode(output):
    '''Recovers waveform from quantized values.
    Mostly adapted from
    https://github.com/ibab/tensorflow-wavenet/blob/master/wavenet/ops.py#L64-L75
    '''
    mu = hp.quantization_channels - 1
    # Map values back to [-1, 1].
    signal = 2 * (output.astype(np.float32) / mu) - 1
    # Perform inverse of mu-law transformation.
    magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
    return np.sign(signal) * magnitude



y, sr = librosa.load(fpath, sr=hp.sr)
qt = mu_law(y)

# length
if len(raw) <= self.length:
            # padding
            pad = self.length-len(raw)
            raw = np.concatenate(
                (raw, np.zeros(pad, dtype=np.float32)))
            qt = np.concatenate(
                (qt, self.mu // 2 * np.ones(pad, dtype=np.int32)))
        else:
            # triming
            if self.random:
                start = random.randint(0, len(raw) - self.length-1)
                raw = raw[start:start + self.length]
                qt = qt[start:start + self.length]
            else:
                raw = raw[:self.length]
qt = qt[:self.length]


# expand dimension
        raw = raw.reshape((1, -1, 1))
        y = np.identity(self.mu)[qt].astype(np.float32) #mu=256
        y = np.expand_dims(y.T, 2)
        t = np.expand_dims(qt.astype(np.int32), 1)
        if self.speaker_dic is None:
            return raw[:, :-1, :], y[:, :-1, :], t[1:, :]
        else:
return raw[:, :-1, :], y[:, :-1, :], np.int32(speaker), t[1:, :]