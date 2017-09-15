from pydub import AudioSegment

from python_speech_features import mfcc

import numpy as np
import soundfile as sf

class TimitSample(object):
  def __init__(self, sample_name : str, timit_dataset_path : str):
    self.sample_name = sample_name
    self.audio_file_name = sample_name + ".WAV"
    self.phonemes_file_name = sample_name + ".PHN"
    self.sentence_file_name = sample_name + ".TXT"
    self.words_file_name = sample_name + ".WRD"
    self.mfcc_file_name = sample_name + ".npy"
    self.mfcc_byword_file_name = sample_name + "_byword.npy"

    self.timit_dataset_path = timit_dataset_path

    self.audio_data = None
    self.phonemes = None
    self.sentence = None
    self.words = None
    self.mfcc = None
    self.mfcc_length = None
    self.mfcc_byword = None
    self.mfcc_byword_length = None

  def preprocess_wav_file_to_mfcc(self, features_count = 40, by_word = False, winstep = 0.01, winlen = 0.025):
    audio_file_path = self.timit_dataset_path + self.audio_file_name

    with sf.SoundFile(audio_file_path, 'r') as audio_file:
      channels_count = audio_file.channels
      sample_rate = audio_file.samplerate
      data = audio_file.read()

    if not by_word:
      mfcc_feat = mfcc(data, sample_rate, nfilt = features_count, numcep = features_count, winstep = winstep, winlen = winlen)
      mfcc_file_path = self.timit_dataset_path + self.mfcc_file_name
    else:
      if self.words is None:
        self._load_words()
      mfcc_feat = []
      for start, end, _ in self.words:
        if start == end:
          continue
        word_mfcc_feat = mfcc(data[start:end], sample_rate, nfilt = features_count, numcep = features_count, winstep = winstep, winlen = winlen)
        mfcc_feat.append(word_mfcc_feat)
      mfcc_feat = np.array(mfcc_feat)
      mfcc_file_path = self.timit_dataset_path + self.mfcc_byword_file_name
    np.save(mfcc_file_path, mfcc_feat)

  def load(self,
           load_phonemes = False, load_sentence = False, load_words = False,
           load_mfcc = False, load_mfcc_byword = False, load_wav = False):
    if load_phonemes:
      self._load_phonemes()
    if load_sentence:
      self._load_sentence()
    if load_words:
      self._load_words()
    if load_mfcc:
      self._load_mfcc()
    if load_mfcc_byword:
      self._load_mfcc_byword()
    if load_wav:
      self._load_wav()

  def _load_phonemes(self):
    data_file = open(self.timit_dataset_path + self.phonemes_file_name)
    data = data_file.readlines()
    self.phonemes = []
    for line in data:
      start, end, phoneme = line.split(' ')
      if phoneme.endswith('\n'):
        phoneme = phoneme[:-1]
      self.phonemes.append((int(start), int(end), phoneme))

  def _load_sentence(self):
    data_file = open(self.timit_dataset_path + self.sentence_file_name)
    data = data_file.read()
    index_of_first_space = data.index(' ')
    index_of_second_space = data.index(' ', index_of_first_space + 1)
    sentence = data[index_of_second_space + 1:]
    sentence = sentence.replace('\n', '')
    sentence = sentence.replace('.', '')
    sentence = sentence.replace('?', '')
    sentence = sentence.replace(',', '')
    self.sentence = sentence

  def _load_words(self):
    data_file = open(self.timit_dataset_path + self.words_file_name)
    data = data_file.readlines()
    self.words = []
    for line in data:
      start, end, word = line.split(' ')
      if word.endswith('\n'):
        word = word[:-1]
      self.words.append((int(start), int(end), word))

  def _load_mfcc(self):
    data = np.load(self.timit_dataset_path + self.mfcc_file_name)
    self.mfcc = data
    self.mfcc_length = np.shape(data)[0]

  def _load_mfcc_byword(self):
    data = np.load(self.timit_dataset_path + self.mfcc_byword_file_name)
    self.mfcc_byword = data
    self.mfcc_byword_length = []
    for i in range(np.shape(data)[0]):
      self.mfcc_byword_length.append(np.shape(data[i])[0])

  def _load_wav(self):
    with sf.SoundFile(self.timit_dataset_path + self.audio_file_name, 'r') as audio_file:
      channels_count = audio_file.channels
      sample_rate = audio_file.samplerate
      data = audio_file.read()
      self.audio_data = (data, sample_rate, channels_count)