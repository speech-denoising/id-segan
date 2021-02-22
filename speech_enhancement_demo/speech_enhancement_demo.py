from tensorflow.python.client import device_lib
import tensorflow as tf
from argparse import ArgumentParser, SUPPRESS
import os

import numpy as np

from data_loader import *
from model import *

from scipy.io import wavfile


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('-i', '--input', type=str, required=True,
                      help="Required. Noisy audio file to process.")
    args.add_argument('-m', "--model", type=str, required=True,
                      help="Required. Path to a .meta file with a trained model.")
    args.add_argument('-at', type=str, required=True, choices=('segan', 'dsegan', 'isegan'),
                      help="Required. Type of the network, either 'dsegan' for "
                           "deep SEGAN, 'isegan' for iterated SEGAN or 'segan' for SEGAN.")
    args.add_argument('-o', '--output', type=str, required=True,
                      help="Required. The output file in .wav format, where the clean audio file will be stored.")
    args.add_argument('-depth', type=int, default=1,
                      help='Optional. The depth of DSEGAN.')
    args.add_argument('-iter', '--iterations', type=int, default=1,
                      help='Optional. The number of iterations of ISEGAN.')
    args.add_argument('-p', '--preemph', type=float, default=0.95,
                      help="Optional. The preemph coeff.")
    args.add_argument("-d", "--device", type=str, default="CPU",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is"
                           " acceptable. The sample will look for a suitable plugin for device specified. "
                           "Default value is CPU")
    return parser.parse_args()


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def pre_emph_test(coeff, canvas_size):
    x_ = tf.placeholder(tf.float32, shape=[canvas_size,])
    x_preemph = pre_emph(x_, coeff)
    return x_, x_preemph

def read_and_decode(filename, canvas_size, preemph=0.):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'noisy_raw': tf.FixedLenFeature([], tf.string),
        })
    noisy = tf.decode_raw(features['noisy_raw'], tf.int32)
    noisy.set_shape(canvas_size)
    noisy = (2. / 65535.) * tf.cast((noisy - 32767), tf.float32) + 1.

    if preemph > 0:
        noisy = tf.cast(pre_emph(noisy, preemph), tf.float32)

    return noisy

def slice_signal(signal, window_size, stride=0.5):
    """ Return windows of the given signal by sweeping in stride fractions
        of window
    """
    assert signal.ndim == 1, signal.ndim
    n_samples = signal.shape[0]
    offset = int(window_size * stride)
    slices = []
    for beg_i, end_i in zip(
            range(0, n_samples, offset),
            range(window_size, n_samples + offset, offset)):
        if end_i - beg_i < window_size:
            break
        slice_ = signal[beg_i:end_i]
        if slice_.shape[0] == window_size:
            slices.append(slice_)
    return np.array(slices, dtype=np.int32)

def read_and_slice(filename, wav_canvas_size, stride=0.5):
    fm, wav_data = wavfile.read(filename)
    if fm != 16000:
        raise ValueError('Sampling rate is expected to be 16kHz!')
    signals = slice_signal(wav_data, wav_canvas_size, stride)
    return signals


def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    devices = device_lib.list_local_devices()
    udevices = []
    for device in devices:
        if len(devices) > 1 and 'CPU' in device.name:
            # Use cpu only when we dont have gpus
            continue
        udevices.append(device.name)

    args = build_argparser()

    preemph = args.preemph
    noisy_filename = args.input
    canvas_size = 2**14

    # Execute the session
    with tf.Session(config=config) as sess:

        noisy_signals = read_and_slice(noisy_filename, canvas_size)
        out_file = tf.python_io.TFRecordWriter('segan.tfrecords')

        for noisy in noisy_signals:
            noisy_raw = noisy.tostring()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'noisy_raw': _bytes_feature(noisy_raw)
                        }))
            out_file.write(example.SerializeToString())
        out_file.close()

        filename_queue = tf.train.string_input_producer(['segan.tfrecords'])
        wave = read_and_decode(filename_queue, 2**14, preemph)
        noisybatch = tf.train.shuffle_batch([wave],
                                             batch_size=1, num_threads=2, capacity=1000 + 3 * 1,
                                             min_after_dequeue=1000,
                                             name='wav_and_noisy')
        if args.depth < 1:
            raise ValueError('The depth cannot be negative!')
            
        print('Loading model weights...')
        model = SEGAN(sess, args.at, noisybatch, args.depth, args.iterations, udevices)
        model.load(args.model)

        tf.initialize_all_variables().run()

        fm, wav_data = wavfile.read(args.input)
        if fm != 16000:
            # Здесь нужно вызвать препроцессинг
            raise ValueError('16kHz required! Test file is different')
        wave = (2. / 65535.) * (wav_data.astype(np.float32) - 32767) + 1.

        if preemph > 0:
            x_pholder, preemph_op = pre_emph_test(preemph, wave.shape[0])
            wave = sess.run(preemph_op, feed_dict={x_pholder: wave})

        clean_wave = model.clean(wave, preemph)
        wavfile.write(args.output, int(16e3), clean_wave)
        print("Done cleaning!")


if __name__ == '__main__':
    tf.app.run()
