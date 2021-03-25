import tensorflow.compat.v1 as tf
import numpy as np
from contextlib import contextmanager

def preemphasis(signal: np.ndarray, coeff=0.95):
    if not coeff or coeff <= 0.0:
        return signal
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def depreemphasis(signal: np.ndarray, coeff=0.95):
    if not coeff or coeff <= 0.0:
        return signal
    x = np.zeros(signal.shape[0], dtype=np.float32)
    x[0] = signal[0]
    for n in range(1, signal.shape[0], 1):
        x[n] = coeff * x[n - 1] + signal[n]
    return x


def downconv(x, w_init, output_dim, kwidth=5, pool=2,
             uniform=False, bias_init=None, name='downconv'):
    x2d = tf.expand_dims(x, 2)
    with tf.variable_scope(name):
        W = tf.get_variable('W', [kwidth, 1, x.get_shape()[-1], output_dim],
                            initializer=w_init)
        conv = tf.nn.conv2d(x2d, W, strides=[1, pool, 1, 1], padding='SAME')
        if bias_init is not None:
            b = tf.get_variable('b', [output_dim], initializer=bias_init)
            conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())
        else:
            conv = tf.reshape(conv, conv.get_shape())
        # reshape back to 1d
        conv = tf.reshape(
            conv,
            conv.get_shape().as_list()[:2] + [conv.get_shape().as_list()[-1]])
        return conv


def leakyrelu(x, alpha=0.3, name='lrelu'):
    return tf.maximum(x, alpha * x, name=name)


def prelu(x, init, name='prelu'):
    in_shape = x.get_shape().as_list()
    with tf.variable_scope(name):
        # make one alpha per feature
        alpha = tf.get_variable('alpha', in_shape[-1],
                                initializer=init,
                                dtype=tf.float32)
        pos = tf.nn.relu(x)
        neg = alpha * (x - tf.abs(x)) * .5
        return pos + neg


def conv1d(x, w_init, kwidth=5, num_kernels=1,
           uniform=False, bias_init=None, name='conv1d', padding='SAME'):
    input_shape = x.get_shape()
    in_channels = input_shape[-1]
    assert len(input_shape) >= 3
    with tf.variable_scope(name):
        W = tf.get_variable('W', [kwidth, in_channels, num_kernels],
                             initializer=w_init)
        conv = tf.nn.conv1d(x, W, stride=1, padding=padding)
        if bias_init is not None:
            b = tf.get_variable('b', [num_kernels],
                                initializer=tf.constant_initializer(bias_init))
            conv = conv + b
        return conv


def repeat_elements(x, rep, axis):
    x_shape = x.get_shape().as_list()
    # slices along the repeat axis
    splits = tf.split(split_dim=axis, num_split=x_shape[axis], value=x)
    # repeat each slice the given number of reps
    x_rep = [s for s in splits for _ in range(rep)]
    return tf.concat(axis, x_rep)


def nn_deconv(x, w_init, kwidth=5, dilation=2,
              uniform=False, bias_init=None, name='nn_deconv1d'):
    # first compute nearest neighbour interpolated x
    interp_x = repeat_elements(x, dilation, 1)
    # run a convolution over the interpolated fmap
    dec = conv1d(interp_x, w_init, kwidth=5, num_kernels=1, uniform=uniform,
                 bias_init=bias_init, name=name, padding='SAME')
    return dec


def deconv(x, output_shape, kwidth=5, dilation=2,
           uniform=False, init=None, bias_init=None, name='deconv1d'):
    w_init = init
    input_shape = x.get_shape()
    in_channels = input_shape[-1]
    out_channels = output_shape[-1]
    assert len(input_shape) >= 3
    # reshape the tensor to use 2d operators
    x2d = tf.expand_dims(x, 2)
    o2d = output_shape[:2] + [1] + [output_shape[-1]]
    with tf.variable_scope(name):
        W = tf.get_variable(
            'W', [kwidth, 1, out_channels, in_channels], initializer=w_init)
        deconv = tf.nn.conv2d_transpose(
            x2d, W, output_shape=o2d, strides=[1, dilation, 1, 1])
        if bias_init is not None:
            b = tf.get_variable('b', [out_channels],
                                initializer=tf.constant_initializer(0.))
            deconv = tf.reshape(tf.nn.bias_add(deconv, b), deconv.get_shape())
        else:
            deconv = tf.reshape(deconv, deconv.get_shape())
        # reshape back to 1d
        deconv = tf.reshape(deconv, output_shape)
        return deconv
