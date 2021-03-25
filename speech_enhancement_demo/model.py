import numpy as np
import tensorflow.compat.v1 as tf

from generator import Generator
from utils import *

class SEGAN:
    """ Speech Enhancement Generative Adversarial Network """

    def __init__(self, sess, dataset, model_type, udevices, depth=1, iterations=1):
        self.sess = sess
        self.batch_size = 1
        self.window_size = 2**14
        
        self.at = model_type
        self.depth = depth
        self.iterations = iterations

        self.kwidth = 31
        self.g_enc_depths = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
        self.bias_downconv = True
        self.bias_deconv = True
        self.deconv_type = 'deconv'
        if self.deconv_type not in ('deconv', 'nn_deconv'):
            raise ValueError(f'Unknown deconv type {self.deconv_type}')

        self.init = False
        self.generator = Generator(self)


    def load(self, meta_path):
        saver = tf.train.import_meta_graph(meta_path)
        model_path = ''.join(meta_path.split('.')[:-1])
        saver.restore(self.sess, model_path)


    def clean(self, x):
        """ clean a utterance x
            x: numpy array containing the normalized noisy waveform
        """
        c_res = None

        for beg_i in range(0, x.shape[0], self.window_size):
            if x.shape[0] - beg_i < self.window_size:
                length = x.shape[0] - beg_i
                pad = (self.window_size) - length
            else:
                length = self.window_size
                pad = 0
            chunk = np.zeros((1, self.window_size))
            if pad > 0:
                chunk[0] = np.concatenate((x[beg_i:beg_i + length], np.zeros(pad)))
            else:
                chunk[0] = x[beg_i:beg_i + length]

            chunk_tf = tf.convert_to_tensor(chunk, dtype=tf.float32)
            fdict = {chunk_tf: chunk}

            clean_chunk = self.generator(chunk_tf)
            res = tf.reshape(clean_chunk, [1, self.window_size])
            
            if not self.init:
                self.init = True
                self.sess.run(tf.global_variables_initializer())

            clean_chunk = self.sess.run(res, feed_dict=fdict)
            clean_chunk = clean_chunk.reshape((self.window_size))
            if pad > 0:
                clean_chunk = clean_chunk[:-pad]
            if c_res is None:
                c_res = clean_chunk
            else:
                c_res = np.concatenate((c_res, clean_chunk))

        return np.array(c_res)
