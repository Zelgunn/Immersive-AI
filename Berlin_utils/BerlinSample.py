from pydub import AudioSegment

from python_speech_features import mfcc, fbank, logfbank

import numpy as np
import soundfile as sf

"""
Emotions are labeled in the 6th position (index 5) of the .wav file name, as follow :

W - Anger
L - Boredom
E - digust
A - Fear/Anxiety
F - Happiness
T - Sadness
N - Neutral
"""

class BerlinSample(object):
    #emotion_t2l_dictionary = {
    #    0 : 'W',
    #    1 : 'L',
    #    2 : 'E',
    #    3 : 'A',
    #    4 : 'F',
    #    5 : 'T',
    #    6 : 'N'
    #}

    emotion_l2t_dictionary = {
        'W' : 0,
        'L' : None,
        'E' : None,
        'A' : 1,
        'F' : 2,
        'T' : 3,
        'N' : 4
    }

    def __init__(self, sample_name : str, timit_dataset_path : str):
        self.sample_name = sample_name
        self.audio_file_name = sample_name + ".wav"
        self.mfcc_file_name = sample_name + ".npy"

        self.timit_dataset_path = timit_dataset_path

        self.audio_data = None
        short_sample_name = sample_name.split('\\')[-1]
        self.emotion_label = short_sample_name[5]
        self.emotion_token = BerlinSample.emotion_l2t_dictionary[self.emotion_label]
        self.mfcc = None
        self.mfcc_length = None

    def preprocess_wav_file_to_mfcc(self, features_count = 40, winstep = 0.01, winlen = 0.025):
        audio_file_path = self.timit_dataset_path + self.audio_file_name

        with sf.SoundFile(audio_file_path, 'r') as audio_file:
            channels_count = audio_file.channels
            sample_rate = audio_file.samplerate
            data = audio_file.read()

        mfcc_feat = mfcc(data, sample_rate, nfilt = features_count, numcep = features_count, winstep = winstep, winlen = winlen)
        #mfcc_feat, _ = fbank(data, sample_rate, nfilt = features_count, winstep = winstep, winlen = winlen)
        #mfcc_feat = logfbank(data, sample_rate, nfilt = features_count, winstep = winstep, winlen = winlen)

        #mfcc_feat = BerlinSample.keep_raw(data, features_count, 50)

        mfcc_file_path = self.timit_dataset_path + self.mfcc_file_name
        np.save(mfcc_file_path, mfcc_feat.astype(np.float32))

    def keep_raw(raw_wav_data, features_count, reduction_factor):
        raw = raw_wav_data

        pre_size = features_count * reduction_factor
        split_count = int((len(raw) - 1) / pre_size)
        size = split_count * pre_size

        raw = np.diff(np.array(raw))
        raw = np.array(np.split(raw[:size], split_count))
        raw = np.array(np.split(raw, reduction_factor, axis = 1))
        raw = np.transpose(raw, [1, 0, 2])
        raw = np.mean(raw, axis = 1)

        return raw

    def load(self,
           load_mfcc = False, load_wav = False):
        if load_mfcc:
            self._load_mfcc()
        if load_wav:
            self._load_wav()

    def _load_mfcc(self):
        data = np.load(self.timit_dataset_path + self.mfcc_file_name)
        self.mfcc = data
        self.mfcc_length = np.shape(data)[0]

    def _load_wav(self):
        with sf.SoundFile(self.timit_dataset_path + self.audio_file_name, 'r') as audio_file:
            channels_count = audio_file.channels
            sample_rate = audio_file.samplerate
            data = audio_file.read()
            self.audio_data = (data, sample_rate, channels_count)