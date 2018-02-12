# import tensorflow as tf
from modules import *
import numpy as np
import glob
import os
from data_load import speaker2id
files = glob.glob('vctk/*.npz')
for f in files:
    fname = os.path.basename(f)
    f_ = np.load(f)
    wav, qt = f_["wav"], f_["qt"]
    np.save("vctk/wavs/{}".format(fname.replace("npz", "npy")), wav)
    np.save("vctk/qts/{}".format(fname.replace("npz", "npy")), qt)
    speaker_id = speaker2id(os.path.basename(wav)[:4])
    _parse



# from hparams import Hyperparams as hp
# def encoder(inputs):
#     for i in range(hp.encoder_layers):
#         # inputs = tf.pad(inputs, [[0, 0], [1, 1], [0, 0]])
#         inputs = conv1d(inputs,
#                        filters=hp.D,
#                        size=hp.winsize,
#                        strides=hp.stride,
#                        padding="causal",
#                        activation_fn=tf.nn.relu if i < hp.encoder_layers-1 else None,
#                        scope="conv1d_{}".format(i))
#         print(i, inputs)
#     z = inputs
#     return z
# # tf.norm
inputs = tf.ones((1, 16000, 1))
z = encoder(inputs)
# print(16000//64)
# print(z)
# a = tf.ones((1, 10, 20, 30))
# b = tf.ones((1, 1, 20, 30))
# c = tf.norm(a-b, axis=-1)
# k = tf.argmin(c, axis=-1)
#
# embed = tf.ones((10, 40), tf.float32)
# k = tf.ones((2, 10), tf.int32)
# out = tf.gather(embed, k)
# print(out)
# a=tf.ones((1, 3, 4))
# b=tf.tile(tf.ones((1, 1, 2)), [1, 3, 1])
# c=tf.concat((a, b), -1)
# print(c)

# a=("p225", "p226", "p227", "p228", "p229", "p230", "p231", "p232", "p233", "p234", "p236", "p237", "p238", "p239", "p240", "p241", "p243", "p244", "p245", "p246", "p247", "p248", "p249", "p250", "p251", "p252", "p253", "p254", "p255", "p256", "p257", "p258", "p259", "p260", "p261", "p262", "p263", "p264", "p265", "p266", "p267", "p268", "p269", "p270", "p271", "p272", "p273", "p274", "p275", "p276", "p277", "p278", "p279", "p280", "p281", "p282", "p283", "p284", "p285", "p286", "p287", "p288", "p292", "p293", "p294", "p295", "p297", "p298", "p299", "p300", "p301", "p302", "p303", "p304", "p305", "p306", "p307", "p308", "p310", "p311", "p312", "p313", "p314", "p315", "p316", "p317", "p318", "p323", "p326", "p329", "p330", "p333", "p334", "p335", "p336", "p339", "p340", "p341", "p343", "p345", "p347", "p351", "p360", "p361", "p362", "p363", "p364", "p374", "p376")
# print(len(a))