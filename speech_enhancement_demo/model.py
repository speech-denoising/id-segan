import numpy as np
import os

from generator import *
from data_loader import pre_emph


def de_emph(y, coeff=0.95):
    if coeff <= 0:
        return y
    x = np.zeros(y.shape[0], dtype=np.float32)
    x[0] = y[0]
    for n in range(1, y.shape[0], 1):
        x[n] = coeff * x[n - 1] + y[n]
    return x


class Model(object):
    def __init__(self, name='BaseModel'):
        self.name = name

    def load(self, meta_path):
        if not os.path.exists(meta_path):
            raise ValueError("The meta path doen't exist")
        self.saver = tf.train.import_meta_graph(meta_path)
        model_path = ''.join(meta_path.split('.')[:-1])
        self.saver.restore(self.sess, model_path)


class SEGAN(Model):
    """ Speech Enhancement Generative Adversarial Network """

    def __init__(self, sess, noisybatch, devices, g_type, name='SEGAN'):
        super(SEGAN, self).__init__(name)
        self.sess = sess
        self.devices = devices
        self.noisybatch = noisybatch

        # cleaning args
        self.canvas_size = 2**14

        self.batch_size = 1
        self.gtruth_noisy = []

        # args for generator
        self.deconv_type = 'deconv' # type of deconv
        self.bias_downconv = True
        self.bias_deconv = True
        self.g_dilated_blocks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512] # dilation factors per layer
        self.g_enc_depths = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024] # num fmaps for AutoEncoder SEGAN (v1)

        if g_type == 'ae':
            self.generator = AEGenerator(self)
        elif g_type == 'dwave':
            self.generator = Generator(self)
        else:
            raise ValueError('Unrecognized G type {}'.format(g_type))

        noisybatch = tf.expand_dims(noisybatch, -1)
        self.Gs = self.generator(noisybatch, is_ref=False)

    def clean(self, x, preemph):
        """ clean a utterance x
            x: numpy array containing the normalized noisy waveform
        """
        c_res = None
        for beg_i in range(0, x.shape[0], self.canvas_size):
            if x.shape[0] - beg_i < self.canvas_size:
                length = x.shape[0] - beg_i
                pad = (self.canvas_size) - length
            else:
                length = self.canvas_size
                pad = 0
            x_ = np.zeros((1, self.canvas_size))
            if pad > 0:
                x_[0] = np.concatenate((x[beg_i:beg_i + length], np.zeros(pad)))
            else:
                x_[0] = x[beg_i:beg_i + length]

            fdict = {self.noisybatch: x_}
            canvas_w = self.sess.run(self.Gs, feed_dict=fdict)[0]
            canvas_w = canvas_w.reshape((self.canvas_size))
            if pad > 0:
                canvas_w = canvas_w[:-pad]
            if c_res is None:
                c_res = canvas_w
            else:
                c_res = np.concatenate((c_res, canvas_w))
        # deemphasize
        c_res = de_emph(c_res, preemph)
        return c_res
