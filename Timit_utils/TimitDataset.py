import os
import numpy as np

from Timit_utils.TimitSample import TimitSample

class TimitDataset(object):
  def __init__(self, dataset_path : str, name : str, mfcc_features_count = 40):
    self.dataset_path = dataset_path
    self.name = name
    self.mfcc_features_count = mfcc_features_count

    self.samples_list_path = os.path.join(dataset_path, "samples_list.txt")
    self.samples_names = None

    self.mfcc_lengths_path = os.path.join(dataset_path, "mfcc_lengths_batch.npy")
    self.mfcc_features_path = os.path.join(dataset_path, "mfcc_features_batch.npy")

    self.mfcc_lengths_byword_path = os.path.join(dataset_path, "mfcc_lengths_byword_batch.npy")
    self.mfcc_features_byword_path = os.path.join(dataset_path, "mfcc_features_byword_batch.npy")

    self.phonemes_ids_path = os.path.join(dataset_path, "phonemes_ids_batch.npy")
    self.phonemes_lengths_path = os.path.join(dataset_path, "phonemes_lengths_batch.npy")

    self.words_ids_path = os.path.join(dataset_path, "words_ids_path.npy")
    self.words_lenghts_path = os.path.join(dataset_path, "words_lenghts_path.npy")

    self.lengths_reordering_list = None
    self.lengths_byword_reordering_list = None

    self.mfcc_features = None
    self.mfcc_features_lengths = None
    self.phonemes_ids = None
    self.phonemes_lengths = None
    self.words_ids = None
    self.batch_index = 0

    self.phonemes_dictionary_size = 0
    self.words_dictionary_size = 0

  def load_samples_list(self):
    # If the list file doesn't exist, we create it
    if not os.path.exists(self.samples_list_path):
      self.samples_names = self.build_samples_list()
      return self.samples_names
    # Then we load it inside a python list
    samples_names = []
    with open(self.samples_list_path, 'r') as samples_list_file:
      lines = samples_list_file.readlines()
      for line in lines:
        samples_names.append(line.replace('\n', ''))
    # Keeping the list inside the object
    self.samples_names = samples_names
    return self.samples_names

  def load_samples(self,
                    load_phonemes = False, load_sentence = False, load_words = False,
                    load_mfcc = False, load_mfcc_byword = False, load_wav = False):
    samples = []
    for sample_name in self.samples_names:
      sample = TimitSample(sample_name, self.dataset_path)
      sample.load(load_phonemes = load_phonemes,
                  load_sentence = load_sentence,
                  load_words = load_words,
                  load_mfcc = load_mfcc,
                  load_mfcc_byword = load_mfcc_byword,
                  load_wav = load_wav)
      samples.append(sample)
    return samples


  def build_samples_list(self):
    samples_names = []
    base_path_len = len(self.dataset_path)
    # Walking inside dataset
    for root, dirs, files in os.walk(self.dataset_path):
      if len(files) == 0:
        continue
      sample_root = root[base_path_len:]
      for file in files:
        if not file.endswith(".TXT"):
          continue
        samples_names.append(sample_root + '\\' + file[:-4])
    # Building of final string to write
    samples_names_string = ""
    for sample_name in samples_names:
      samples_names_string += sample_name + '\n'
    # Writing to file
    with open(self.samples_list_path, 'w') as samples_list_file:
      samples_list_file.write(samples_names_string)
    return samples_names

  def build_samples_mfcc_features(self, by_word, winstep = 0.01, winlen = 0.025):
    for sample_name in self.samples_names:
      sample = TimitSample(sample_name, self.dataset_path)
      sample.preprocess_wav_file_to_mfcc(self.mfcc_features_count, by_word, winstep, winlen)

  def build_mfcc_lengths_batch(self):
    lengths = []
    samples = self.load_samples(load_mfcc = True)
    for sample in samples:
      lengths.append(sample.mfcc_length)
    lengths = sorted(lengths)
    np.save(self.mfcc_lengths_path, lengths)

  def build_mfcc_lengths_batch_byword(self):
    lengths = []
    samples = self.load_samples(load_mfcc_byword = True)
    for sample in samples:
      lengths += sample.mfcc_byword_length
    lengths = sorted(lengths)
    np.save(self.mfcc_lengths_byword_path, lengths)

  def get_lengths_reordering_list(self):
    if self.lengths_reordering_list is not None:
      return self.lengths_reordering_list
    lengths = []
    samples = self.load_samples(load_mfcc = True)
    for sample in samples:
      lengths.append(sample.mfcc_length)
    reorder = np.arange(len(lengths))
    lengths, reorder = zip(*sorted(zip(lengths, reorder), key=lambda pair: pair[0]))
    self.lengths_reordering_list = reorder
    return reorder

  def reorder_by_mfcc_lenghts(self, data):
    reorder = self.get_lengths_reordering_list()
    reorder = np.array(reorder)
    data = np.array(data)
    data = data[reorder]
    return data

  def build_mfcc_batch(self, max_mfcc_length):
    samples = self.load_samples(load_mfcc = True)
    lengths = []
    for sample in samples:
      lengths.append(sample.mfcc_length)
    # Sorting
    lengths, samples = zip(*sorted(zip(lengths, samples), key=lambda pair: pair[0]))
    # Padding
    batch_size = len(samples)
    mfcc_batch = np.zeros([batch_size, max_mfcc_length, self.mfcc_features_count])
    print("build_mfcc batch dimensions: ", [batch_size, max_mfcc_length, self.mfcc_features_count])
    for i in range(batch_size):
      for j in range(lengths[i]):
        for k in range(self.mfcc_features_count):
          mfcc_batch[i, j, k] = samples[i].mfcc[j, k]
    # Saving
    np.save(self.mfcc_features_path, mfcc_batch)

  def get_lengths_byword_reordering_list(self):
    if self.lengths_byword_reordering_list is not None:
      return self.lengths_byword_reordering_list
    samples = self.load_samples(load_mfcc_byword = True)
    lengths = []
    for sample in samples:
      lengths += sample.mfcc_byword_length
    reorder = np.arange(len(lengths))
    lengths, reorder = zip(*sorted(zip(lengths, reorder), key=lambda pair: pair[0]))
    self.lengths_byword_reordering_list = reorder
    return reorder

  def reorder_by_mfcc_lenghts_byword(self, data):
    reorder = self.get_lengths_byword_reordering_list()
    reorder = np.array(reorder)
    data = np.array(data)
    data = data[reorder]
    return data

  def build_mfcc_batch_byword(self, max_mfcc_byword_length):
    samples = self.load_samples(load_mfcc_byword = True)
    lengths = []
    mfcc_features_byword = []
    for sample in samples:
      lengths += sample.mfcc_byword_length
      mfcc_features_byword += sample.mfcc_byword.tolist()
    # Sorting
    lengths, mfcc_features_byword = zip(*sorted(zip(lengths, mfcc_features_byword), key=lambda pair: pair[0]))
    # Padding
    batch_size = len(lengths)
    mfcc_batch = np.zeros([batch_size, max_mfcc_byword_length, self.mfcc_features_count])
    print("build_mfcc_by_word batch dimensions: ", [batch_size, max_mfcc_byword_length, self.mfcc_features_count])
    index = 0
    for i in range(batch_size):
      for j in range(lengths[i]):
        for k in range(self.mfcc_features_count):
          mfcc_batch[i, j, k] = mfcc_features_byword[i][j][k]
    # Saving
    np.save(self.mfcc_features_byword_path, mfcc_batch)

  def build_phonemes_ids_batch(self, phoneme_to_id_dictionary : dict, max_phonemes_length: int):
    samples = self.load_samples(load_phonemes = True)
    phonemes_ids = []
    lengths = []
    for sample in samples:
      phonemes_infos = sample.phonemes
      phonemes_ids_for_sample = []
      for _, _, phoneme in phonemes_infos:
        phoneme_id = phoneme_to_id_dictionary[phoneme]
        phonemes_ids_for_sample.append(phoneme_id)
      lengths.append(len(sample.phonemes))
      phonemes_ids.append(phonemes_ids_for_sample)

    batch_size = len(samples)
    phonemes_ids_padded = np.zeros([batch_size, max_phonemes_length], dtype = int)

    for i in range(batch_size):
      for j in range(lengths[i]):
        phonemes_ids_padded[i, j] = phonemes_ids[i][j]
      for j in range(lengths[i], max_phonemes_length):
        phonemes_ids_padded[i, j] = len(phoneme_to_id_dictionary) - 1

    phonemes_ids = self.reorder_by_mfcc_lenghts(phonemes_ids_padded)

    np.save(self.phonemes_ids_path, phonemes_ids)

  def build_phonemes_lengths_batch(self):
    samples = self.load_samples(load_phonemes = True)
    phonemes_lengths = []
    for sample in samples:
      phonemes_lengths.append(len(sample.phonemes))
    phonemes_lengths = self.reorder_by_mfcc_lenghts(phonemes_lengths)
    np.save(self.phonemes_lengths_path, phonemes_lengths)

  #def build_words_ids_batch(self):
  #  samples = self.load_samples(load_words = True)
  #  words_ids = []
  #  for sample in samples:
  #    words_infos = sample.words

  def load_mfcc_lengths(self):
    self.mfcc_features_lengths = np.load(self.mfcc_lengths_path)
    
  def load_mfcc_lengths_byword(self):
    self.mfcc_features_lengths_byword = np.load(self.mfcc_lengths_byword_path)

  def load_phonemes_lengths(self):
    self.phonemes_lengths = np.load(self.phonemes_lengths_path)

  def load_batch(self):
    self.mfcc_features = np.load(self.mfcc_features_path)
    self.mfcc_features_lengths = np.load(self.mfcc_lengths_path)
    self.phonemes_ids = np.load(self.phonemes_ids_path)
    self.phonemes_lengths = np.load(self.phonemes_lengths_path)

    return self.mfcc_features, self.mfcc_features_lengths, self.phonemes_ids, self.phonemes_lengths

  #def load_batch_byword(self):
  #  self.mfcc_features = np.load(self.mfcc_features_byword_path)
  #  self.mfcc_features_lengths = np.load(self.mfcc_lengths_byword_path)
  #  self.words_ids = np.load(self.words_ids_path)

  def shuffle_batch(self):
    order = np.arange(len(self.mfcc_features))
    np.random.shuffle(order)
    self.mfcc_features = self.mfcc_features[order]
    self.mfcc_features_lengths = self.mfcc_features_lengths[order]

    if self.phonemes_ids is not None:
      self.phonemes_ids = self.phonemes_ids[order]
      self.phonemes_lengths = self.phonemes_lengths[order]
    if self.words_ids is not None:
      self.words_ids = self.words_ids[order]

  @staticmethod
  def to_one_hot(ids : list, ids_class_count : int):
    batch_size = len(ids)
    max_ids_length = len(ids[0])
    ids_to_one_hot = np.zeros([batch_size, max_ids_length, ids_class_count])
    for i in range(batch_size):
      for j in range(max_ids_length):
        id = int(ids[i, j])
        try:
          ids_to_one_hot[i, j, id] = 1
        except:
          print(i, j, id)
          input()
          ids_to_one_hot[i, j, id] = 1
    return ids_to_one_hot

  def next_batch(self, batch_size : int, one_hot = True):
    if self.mfcc_features is None:
      self.load_batch()
      self.shuffle_batch()
    batch_limit_size = len(self.mfcc_features)
    start = self.batch_index
    end = min(self.batch_index + batch_size, batch_limit_size - 1)
    mfcc_features_batch         = self.mfcc_features[start : end]
    mfcc_features_lengths_batch = self.mfcc_features_lengths[start : end]
    phonemes_ids_batch          = self.phonemes_ids[start : end]
    phonemes_lengths_batch      = self.phonemes_lengths[start : end]
    self.batch_index += batch_size
    if self.batch_index >= batch_limit_size:
      self.batch_index -= batch_limit_size
      self.shuffle_batch()
      #if self.batch_index > 0:
      #  mfcc_features_batch         += self.mfcc_features[0 : self.batch_index]
      #  mfcc_features_lengths_batch += self.mfcc_features_lengths[0 : self.batch_index]
      #  phonemes_ids_batch          += self.phonemes_ids[0 : self.batch_index]
      #  phonemes_lengths_batch      += self.phonemes_lengths[0 : self.batch_index]
    if(one_hot):
      phonemes_ids_batch = TimitDataset.to_one_hot(phonemes_ids_batch, self.phonemes_dictionary_size)
    return (mfcc_features_batch, mfcc_features_lengths_batch, phonemes_lengths_batch, phonemes_ids_batch)