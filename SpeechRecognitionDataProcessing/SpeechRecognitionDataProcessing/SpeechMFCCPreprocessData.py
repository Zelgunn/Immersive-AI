from pydub import AudioSegment

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank

import numpy as np

class SpeechMFCCPreprocessData(object):
    def __init__(self, filename : str, times : list):
        self.filename = filename
        self.times = times

        # Opening and reading of the audio file
        if filename.endswith(".mp3"):
          audio_file = AudioSegment.from_mp3(filename)
        else:
          audio_file = AudioSegment.from_wav(filename)

        channels_count = audio_file.channels
        samples = audio_file.get_array_of_samples()
        samples_count = int(len(samples) / channels_count)

        samples = np.reshape(samples, (samples_count, channels_count))
        rate = audio_file.frame_rate

        # Features extracting
        features = []
        for [start, end] in times:
            sentence = samples[start:end]

            mfcc_feat = mfcc(sentence, rate)
            d_mfcc_feat = delta(mfcc_feat, 2)
            fbank_feat = logfbank(sentence, rate, nfilt = 40)

            features.append(fbank_feat)

        self.features = np.array(features)

    def save_fbank_as_binary(self, filename=None):
        if filename is None:
            filename = SpeechDataPreprocess.get_npy_filename(self.filename)

        np.save(filename, self.features)

    @staticmethod
    def get_npy_filename(filename : str):
        if filename.endswith(".mp3") :
            filename = filename.replace(".mp3", ".npy")
        else:
            filename += ".sents.npy"

        return filename