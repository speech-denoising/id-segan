import os

import librosa
import numpy as np
import tensorflow.compat.v1 as tf

from utils import preemphasis


def read_raw_audio(audio, sample_rate=16000):
    if isinstance(audio, str):
        wave, _ = librosa.load(os.path.expanduser(audio), sr=sample_rate, mono=True)
    elif isinstance(audio, bytes):
        wave, sr = sf.read(io.BytesIO(audio))
        if wave.ndim > 1: wave = np.mean(wave, axis=-1)
        wave = np.asfortranarray(wave)
        if sr != sample_rate:
            wave = librosa.resample(wave, sr, sample_rate)
    elif isinstance(audio, np.ndarray):
        if audio.ndim > 1: ValueError("input audio must be single channel")
        return audio
    else:
        raise ValueError("input audio must be either a path or bytes")
    return wave


class AudioSource:
    def __init__(self, input_path):
        self.input_path = input_path
        self.filenames = os.listdir(input_path)

    def create(self):
        def _gen_data():
            for filename in self.filenames:
                path = os.path.join(self.input_path, filename)
                signal = read_raw_audio(path)
                noisy_w = preemphasis(signal)
                yield filename, signal

        dataset = tf.data.Dataset.from_generator(_gen_data, output_types=(tf.string, tf.float32))
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
