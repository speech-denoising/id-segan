import os
from argparse import ArgumentParser, SUPPRESS

import numpy as np
import soundfile as sf
import tensorflow.compat.v1 as tf

from model import SEGAN
from utils import depreemphasis
from audio_source import AudioSource


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('-i', '--input', type=str, required=True,
                      help="Required. Input dir with noisy audio files to process.")
    args.add_argument('-m', "--model", type=str, required=True,
                      help="Required. Path to a .meta file with a trained model.")
    args.add_argument('-at', type=str, required=True, choices=('segan', 'dsegan', 'isegan'),
                      help="Required. Type of the network, either 'dsegan' for "
                           "deep SEGAN, 'isegan' for iterated SEGAN or 'segan' for SEGAN.")
    args.add_argument('-o', '--output', type=str, required=True,
                      help="Required. The output dir, where the clean audio files will be stored.")
    args.add_argument('-depth', type=int, default=1,
                      help='Optional. The depth of DSEGAN.')
    args.add_argument('-iter', '--iterations', type=int, default=1,
                      help='Optional. The number of iterations of ISEGAN.')
    return parser.parse_args()



def main(_):
    args = build_argparser()
    if not os.path.exists(args.model):
        raise ValueError(f'The meta path {args.model} does not exist.')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    tf.disable_v2_behavior()
    tf.enable_eager_execution()

    # Execute the session
    with tf.Session(config=config) as sess:

        noisy_dir = args.input
        clean_dir = args.output
        
        sample_rate = 16000
        source = AudioSource(noisy_dir)
        model = SEGAN(sess, source, args.at, args.depth, args.iterations)
        print('[*] Loading model weights...')
        model.load(args.model)
        print('Weights are loaded!')

        sess.run(tf.global_variables_initializer())

        data_loader = source.create()
        iterator = tf.data.make_one_shot_iterator(data_loader)
        next_element = iterator.get_next()

        while True:
            try:
                filename, noisy_wave = sess.run(next_element)
                signal = model.clean(noisy_wave)
                clean_wave = depreemphasis(signal)
                clean_path = os.path.join(clean_dir, filename.decode("utf-8"))
                sf.write(clean_path, clean_wave, sample_rate)
            except tf.errors.OutOfRangeError:
                break
        print('Done cleaning!')


if __name__ == '__main__':
    tf.app.run()
