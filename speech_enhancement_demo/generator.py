import tensorflow.compat.v1 as tf
from utils import downconv, leakyrelu, prelu, nn_deconv, deconv


class Generator:
    def __init__(self, segan):
        self.segan = segan        
        self.kwidth = 31
        self.g_enc_depths = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
        self.bias_downconv = True
        self.bias_deconv = True
        self.deconv_type = 'deconv'
        if self.deconv_type not in ('deconv', 'nn_deconv'):
            raise ValueError(f'Unknown deconv type {self.deconv_type}')

    def __call__(self, noisy_w, z_on=True, do_prelu=False):
        """ Build the graph propagating (noisy_w) --> x
        On first pass will make variables.
        """
        segan = self.segan

        def make_z(shape, mean=0., stddev=1., name='z'):
            return tf.random_normal(shape, mean=mean, stddev=stddev, name=name, dtype=tf.float32)

        if hasattr(self, 'generator_built'):
            tf.get_variable_scope().reuse_variables()

        input_i = noisy_w

        skips = [[] for nr in range(segan.depth)]
        waves = []
        zs = []

        with tf.variable_scope('g_ae'):
            for nr in range(segan.depth):
                # ENCODER
                in_dims = input_i.get_shape().as_list()
                if len(in_dims) == 2:
                    input_i = tf.expand_dims(input_i, -1)
                elif len(in_dims) < 2 or len(in_dims) > 3:
                    raise ValueError('Generator input must be 2-D or 3-D')

                h_i = input_i
                for layer_idx, layer_depth in enumerate(self.g_enc_depths):
                    bias_init = None
                    if self.bias_downconv:
                        bias_init = tf.constant_initializer(0.)
                    h_i = downconv(h_i, tf.truncated_normal_initializer(stddev=0.02),
                                   layer_depth, kwidth=self.kwidth,
                                   bias_init=bias_init,
                                   name=f'enc_{layer_idx}')
                    if layer_idx < len(self.g_enc_depths) - 1:
                        skips[nr].append(h_i) # store skip connection
                    if do_prelu:
                        h_i = prelu(h_i, init=tf.constant_initializer(0.0),
                                    name=f'enc_prelu_{layer_idx}')
                    else:
                        h_i = leakyrelu(h_i)

                if z_on:
                    z = make_z([segan.batch_size, h_i.get_shape().as_list()[1], self.g_enc_depths[-1]],
                               name=f'z{nr}')
                    h_i = tf.concat([z, h_i], 2)
                # DECODER (reverse order)
                g_dec_depths = self.g_enc_depths[:-1][::-1] + [1]
                for layer_idx, layer_depth in enumerate(g_dec_depths):
                    h_i_dim = h_i.get_shape().as_list()
                    out_shape = [h_i_dim[0], h_i_dim[1] * 2, layer_depth]
                    bias_init = None
                    # deconv
                    if self.deconv_type == 'deconv':
                        if self.bias_deconv:
                            bias_init = tf.constant_initializer(0.)
                        h_i = deconv(h_i,
                                     out_shape, kwidth=self.kwidth, dilation=2,
                                     init=tf.truncated_normal_initializer(stddev=0.02),
                                     bias_init=bias_init, name=f'dec_{layer_idx}')
                    elif self.deconv_type == 'nn_deconv':
                        if self.bias_deconv:
                            bias_init = 0.
                        h_i = nn_deconv(h_i, w_init=tf.truncated_normal_initializer(stddev=0.02),
                                        kwidth=self.kwidth, dilation=2,
                                        bias_init=bias_init, name=f'dec_{layer_idx}')
                    if layer_idx < len(g_dec_depths) - 1:
                        if do_prelu:
                            h_i = prelu(h_i, init=tf.constant_initializer(0.0),
                                        name=f'dec_prelu_{layer_idx}')
                        else:
                            h_i = leakyrelu(h_i)
                        # fuse skip connection
                        skip_ = skips[nr][-(layer_idx + 1)]
                        h_i = tf.concat([h_i, skip_], 2)
                    else: # last layer
                        h_i = tf.tanh(h_i)

                wave = h_i      # checkpiont nr-th output
                input_i = h_i   # input to the next iternation
                waves.append(wave)
                zs.append(z)

            self.generator_built = True
            return waves[-1]
