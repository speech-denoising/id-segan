from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm, fully_connected, flatten
from tensorflow.contrib.layers import xavier_initializer
from ops import *
import numpy as np


class Generator(object):
    def __init__(self, segan):
        self.g_enc_depths = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
        self.segan = segan

    def __call__(self, wave, do_prelu=False):
        """ Build the graph propagating (wave) --> x
        On first pass will make variables.
        """
        segan = self.segan
        if hasattr(segan, 'generator_built'):
            tf.get_variable_scope().reuse_variables()

        kwidth = 31
        enc_layers = 7
        skips = []
        for nr in range(segan.depth):
            skips.append([])

        waves = []
        #=================generator========================#
        with tf.variable_scope('g_ae'):
            #AE to be built is shaped:
            # enc ~ [16384x1, 8192x16, 4096x32, 2048x32, 1024x64, 512x64, 256x128, 128x128, 64x256, 32x256, 16x512, 8x1024]
            # dec ~ [8x2048, 16x1024, 32x512, 64x512, 8x256, 256x256, 512x128, 1024x128, 2048x64, 4096x64, 8192x32, 16384x1]
            # create chained generators of DSEGAN here
            for nr in range(segan.depth):
                # ENCODER
                in_dims = wave.get_shape().as_list()
                if len(in_dims) == 2:
                    wave = tf.expand_dims(wave, -1) # expand channel dimension
                elif len(in_dims) < 2 or len(in_dims) > 3:
                    raise ValueError('Generator input must be 2-D or 3-D')

                for layer_idx, layer_depth in enumerate(self.g_enc_depths):
                    bias_init = None
                    if segan.bias_downconv:
                        bias_init = tf.constant_initializer(0.)
                    wave = downconv(wave, layer_depth, kwidth=kwidth,
                                    init=tf.truncated_normal_initializer(stddev=0.02),
                                    bias_init=bias_init,
                                    name='enc_{}_{}'.format(nr, layer_idx))
                    if layer_idx < len(self.g_enc_depths) - 1:
                        # store skip connection
                        # last one is not stored cause it's the code
                        skips[nr].append(wave)
                    if do_prelu:
                        wave = prelu(wave, ref=False, name='enc_prelu_{}_{}'.format(nr, layer_idx))
                    else:
                        wave = leakyrelu(wave)
                # DECODER (reverse order)
                g_dec_depths = self.g_enc_depths[:-1][::-1] + [1]
                for layer_idx, layer_depth in enumerate(g_dec_depths):
                    h_i_dim = wave.get_shape().as_list()
                    out_shape = [h_i_dim[0], h_i_dim[1] * 2, layer_depth]
                    bias_init = None
                    # deconv
                    if segan.deconv_type == 'deconv':
                        if segan.bias_deconv:
                            bias_init = tf.constant_initializer(0.)
                        wave = deconv(wave, out_shape, kwidth=kwidth, dilation=2,
                                      init=tf.truncated_normal_initializer(stddev=0.02),
                                      bias_init=bias_init,
                                      name='dec_{}_{}'.format(nr, layer_idx))
                    elif segan.deconv_type == 'nn_deconv':
                        if segan.bias_deconv:
                            bias_init = 0.
                        wave = nn_deconv(wave, kwidth=kwidth, dilation=2,
                                         init=tf.truncated_normal_initializer(stddev=0.02),
                                         bias_init=bias_init,
                                         name='dec_{}_{}'.format(nr, layer_idx))
                    else:
                        raise ValueError('Unknown deconv type {}'.format(segan.deconv_type))

                    if layer_idx < len(g_dec_depths) - 1:
                        if do_prelu:
                            wave = prelu(wave, ref=False, name='dec_prelu_{}_{}'.format(nr, layer_idx))
                        else:
                            wave = leakyrelu(wave)
                        # fuse skip connection
                        skip_ = skips[nr][-(layer_idx + 1)]
                        wave = tf.concat([wave, skip_], 2)
                    else: # last layer
                        wave = tf.tanh(wave)

                waves.append(wave)

            segan.generator_built = True
            return waves[-1] # return the wave from n-th depth
