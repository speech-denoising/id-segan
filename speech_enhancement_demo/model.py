import numpy as np
import os

from generator import *


def de_emph(y, coeff=0.95):
    if coeff <= 0:
        return y
    x = np.zeros(y.shape[0], dtype=np.float32)
    x[0] = y[0]
    for n in range(1, y.shape[0], 1):
        x[n] = coeff * x[n - 1] + y[n]
    return x


class SEGAN:
    """ Speech Enhancement Generative Adversarial Network """

    def __init__(self, sess, model_type, noisybatch, devices, depth=1, iterations=1):
        self.sess = sess
        self.devices = devices
        self.noisybatch = noisybatch

        # cleaning args
        self.canvas_size = 2**14
        self.batch_size = 1

        self.depth = depth

        # args for generator
        self.deconv_type = 'deconv'
        self.bias_downconv = True
        self.bias_deconv = True

        self.generator = Generator(self)

        noisybatch = tf.expand_dims(noisybatch, -1)

        if model_type in ['segan', 'dsegan']:
            self.Gs = self.generator(noisybatch)
        else:
            input = noisybatch
            for i in range(iteration):
                G = self.generator(input)
                input = G
            self.Gs = G

    def load(self, meta_path):
        if not os.path.exists(meta_path):
            raise ValueError("The meta path does not exist")
        self.saver = tf.train.import_meta_graph(meta_path)
        model_path = ''.join(meta_path.split('.')[:-1])
        self.saver.restore(self.sess, model_path)

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
            x_ = np.zeros((self.batch_size, self.canvas_size))
            if pad > 0:
                x_[0] = np.concatenate((x[beg_i:beg_i + length], np.zeros(pad)))
            else:
                x_[0] = x[beg_i:beg_i + length]

            fdict = {self.noisybatch: x_}
            canvas_w = self.sess.run(self.Gs, feed_dict=fdict)
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
