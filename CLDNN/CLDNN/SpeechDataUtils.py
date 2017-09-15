import os
import numpy as np
from tqdm import tqdm
from dictionary_utils import get_words_dictionary

import time

class SpeechDataSet(object):
  def __init__(self, batches_infos : list, batch_file_size : int, dictionary_size : int, allow_autorewind : bool):
    self.batches_infos = batches_infos
    self.dictionary_size = dictionary_size

    self.batch_file_size = batch_file_size
    self.batch_files_count = len(self.batches_infos)
    self.batch_total_count = batch_file_size * self.batch_files_count

    self.current_batch_index = 0
    self.current_input_index = 0

    self.current_mfcc_batch = None
    self.current_input_lengths_batch = None
    self.current_tokenized_transcripts_batch = None
    self.current_output_lengths_batch = None

    self.allow_autorewind = allow_autorewind
    self.load_batch()

  def load_batch(self):
    selected_file = self.batches_infos[self.current_batch_index]
    mfcc_batch_file_path, mfcc_batch_lengths_file_path, \
      _, \
      transcripts_lengths_batch_file_path, tokenized_transcripts_batch_file_path = selected_file

    if self.current_mfcc_batch is not None :
      self.current_batch_index += 1

    if self.current_batch_index >= self.batch_files_count :
      if self.allow_autorewind:
        self.current_batch_index = 0
      else:
        return False

    self.current_mfcc_batch = np.load(mfcc_batch_file_path)
    self.current_input_lengths_batch = np.load(mfcc_batch_lengths_file_path)
    self.current_tokenized_transcripts_batch = np.load(tokenized_transcripts_batch_file_path)
    self.current_output_lengths_batch = np.load(transcripts_lengths_batch_file_path)

    index_shuffle = np.arange(len(self.current_mfcc_batch))
    np.random.shuffle(index_shuffle)

    self.current_mfcc_batch = self.current_mfcc_batch[index_shuffle]
    self.current_input_lengths_batch = self.current_input_lengths_batch[index_shuffle]
    self.current_tokenized_transcripts_batch = self.current_tokenized_transcripts_batch[index_shuffle]
    self.current_output_lengths_batch = self.current_output_lengths_batch[index_shuffle]

    return True

  def next_batch(self, batch_size : int, one_hot = True):
    index_in_batch = self.current_input_index

    inputs = self.current_mfcc_batch[index_in_batch : index_in_batch + batch_size]
    input_lengths = self.current_input_lengths_batch[index_in_batch : index_in_batch + batch_size]
    outputs = self.current_tokenized_transcripts_batch[index_in_batch : index_in_batch + batch_size]
    output_lengths = self.current_output_lengths_batch[index_in_batch : index_in_batch + batch_size]

    self.current_input_index += batch_size

    # Selection d'un nouveau paquet de données quand on arrive au bout de l'actuel
    if self.current_input_index > self.batch_file_size:
      self.current_input_index -= self.batch_file_size
      successfully_loaded = self.load_batch()
      #if successfully_loaded:
      #  inputs += self.current_mfcc_batch[:self.current_input_index]
      #  input_lengths += self.current_input_lengths_batch[:self.current_input_index]
      #  outputs += self.current_tokenized_transcripts_batch[:self.current_input_index]
      #  output_lengths += self.current_output_lengths_batch[:self.current_input_index]
    elif self.current_input_index == self.batch_file_size:
      self.current_input_index -= self.batch_file_size
      self.load_batch()

    # Si l'option OneHot est activée, transforme les tokens en vecteurs one-hot dans les outputs (les phrases)
    try:
      if one_hot:
        outputs = SpeechDataSet.token_to_onehot(outputs, batch_size, self.dictionary_size)
      else :
        outputs = SpeechDataSet.tokens_for_sparse(outputs)
    except:
      return self.next_batch(batch_size, one_hot)

    batch = (inputs, input_lengths, outputs, output_lengths)
    return batch

  @staticmethod
  def token_to_onehot(tokens, batch_size : int, dictionary_size : int):
    max_sequence_length = len(tokens[0])

    outputs = np.zeros((batch_size, max_sequence_length, dictionary_size))
    
    for entry in range(batch_size):
      for token in range(max_sequence_length):
        token_class = tokens[entry][token]
        outputs[entry][token][token_class] = 1
    return outputs

  @staticmethod
  def tokens_for_sparse(sequences):
    eos_value = 9632
    tmp = []
    for seq_idx in range(len(sequences)):
      seq = sequences[seq_idx]
      for i in range(len(seq)):
        end_idx = i
        if seq[i] == eos_value:
          break
      tmp.append(seq[:end_idx])
      
    sequences = tmp

    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=np.int32)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape

class SpeechDataUtils(object):
  def __init__(self, librispeech_path = r"C:\tmp\LibriSpeech", bucket_size = 150, eval_batch_files_count = 2, allow_autorewind = True):
    # Chemin vers le dossier Librispeech
    self.librispeech_path = librispeech_path
    # Données contenant l'ordre utilisé lors du preprocessus
    self.preprocess_order = get_preprocess_order(librispeech_path)

    # Nombre de features calculées pour le MFCC
    self.features_count = 40
    # Taille maximale en nombre de frames des MFCC
    self.bucket_size = bucket_size

    # Dictionnaire des mots trouvés lors du preprocess
    self.dictionary, self.reverse_dictionary = get_words_dictionary(librispeech_path, reduced_dictionary = True)
    # Taille du dictionnaire
    self.dictionary_size = len(self.dictionary)
    # Informations sur les paquets de données ayant été préprocess
    # Contient une liste de N listes d'informations, les informations sont des chemins vers les paquets (N : Nombre de paquets)
    # La liste d'information d'un fichier est de la forme :
    # [Paquet de MFCC, Paquet des longueurs des MFCC, Paquet de transcripts, Paquet de longueurs des transcripts, Paquet de transcripts sous forme de tokens (int)]
    batches_info = get_batches_info(librispeech_path)[self.bucket_size]

    index_shuffle = np.arange(len(batches_info))
    np.random.shuffle(index_shuffle)

    self.batches_info = []
    for i in index_shuffle:
      self.batches_info.append(batches_info[i])

    # Nombre d'entrées/sorties dans un paquet
    self.batch_file_size = 5000

    train_batch_files_count = len(self.batches_info) - eval_batch_files_count
    self.sets = dict()
    self.sets["eval"] = SpeechDataSet(self.batches_info[-eval_batch_files_count:], self.batch_file_size, self.dictionary_size, allow_autorewind)
    self.sets["train"] = SpeechDataSet(self.batches_info[:train_batch_files_count], self.batch_file_size, self.dictionary_size, allow_autorewind)

  @property
  def train(self) -> SpeechDataSet:
    return self.sets["train"]

  @property
  def eval(self) -> SpeechDataSet:
    return self.sets["eval"]


def get_preprocess_order(librispeech_path : str):
  preprocess_order_file = open(os.path.join(librispeech_path, "process_order.txt"), 'r')

  preprocess_order = preprocess_order_file.readlines()
  for i in range(len(preprocess_order)):
    # Removing '\n' at the end with :   [:-1]
    tmp = preprocess_order[i]
    if tmp.endswith('\n'):
      tmp = tmp[:-1]
    # and splitting into [index, npy_file_path, sentence_file_path] with :   split(',')
    preprocess_order[i] = tmp.split(',')

  preprocess_order_file.close()

  return preprocess_order

def get_mfcc_lengths(librispeech_path : str, save_to_file = True, sort_lengths = True, force_process = False):
  # Retrieve existing file
  mfcc_lengths_file = os.path.join(librispeech_path, "mfcc_lengths.npy")
  mfcc_lengths = None

  if not force_process and os.path.exists(mfcc_lengths_file):
    mfcc_lengths = np.load(mfcc_lengths_file)
    print("Found existing list of mfcc lengths with", len(mfcc_lengths), "sentences.")
    return mfcc_lengths

  preprocess_order = get_preprocess_order(librispeech_path)
  # If force to (re)process or if file is not found, process
  for file_index in tqdm(range(len(preprocess_order))):
    entry = preprocess_order[file_index]
    full_path = librispeech_path + entry[1]
    mfcc = np.load(full_path)

    sentence_count = len(mfcc)
    for sentence_index in range(sentence_count):
      mfcc_length = len(mfcc[sentence_index])
      sentence_info = [file_index, sentence_index, mfcc_length]

      if mfcc_lengths is None:
        mfcc_lengths = np.array(sentence_info)
      else:
        mfcc_lengths = np.vstack([mfcc_lengths, sentence_info])
  
  mfcc_lengths = mfcc_lengths[mfcc_lengths[:,2].argsort()]

  if save_to_file:
    np.save(mfcc_lengths_file, mfcc_lengths)

  return mfcc_lengths

def get_batches_info(librispeech_path : str):
  batches_info_file = open(os.path.join(librispeech_path, "batches_infos.txt"), 'r')
  batches_info_astext = batches_info_file.readlines()
  batches_count = len(batches_info_astext)
  batch_folder = os.path.join(librispeech_path, "batches")

  batches_info = dict()
  for i in range(batches_count):
    splitted_infos = batches_info_astext[i].split(',')
    mfcc_padded_length, \
      batch_file_path, mfcc_batch_lengths_file_path, \
      transcripts_batch_file_path, transcripts_batch_lengths_file_path, \
      tokenized_transcripts_batch_file_path = splitted_infos

    mfcc_padded_length = int(mfcc_padded_length)
    splitted_infos = splitted_infos[1:]

    for j in range(len(splitted_infos)):
      if splitted_infos[j].endswith('\n'):
        splitted_infos[j] = splitted_infos[j][:-1]
      splitted_infos[j] = os.path.join(batch_folder, splitted_infos[j])

    if mfcc_padded_length not in batches_info:
      batches_info[mfcc_padded_length] = [splitted_infos]
    else:
      tmp = batches_info[mfcc_padded_length]
      tmp += [splitted_infos]
      batches_info[mfcc_padded_length] = tmp
  return batches_info


  #str(mfcc_padded_length) + ' ' + batch_file_path + ' ' + mfcc_batch_lengths_file_path + ' ' + transcripts_batch_file_path + ' ' + transcripts_batch_lengths_file_path + ' ' + tokenized_transcripts_batch_file_path

def get_transcript_from_file_and_index(transcript_file_path : str, sentence_index : int):
    transcript_file = open(transcript_file_path, 'r')
    transcript_lines = transcript_file.readlines()
    transcript_line = transcript_lines[sentence_index]
    transcript_file.close()

    # Parsing
    split_index = transcript_line.find(' ') + 1

    words_in_line = transcript_line[split_index:]
    words_in_line = words_in_line.split(' ')
    words_count_in_line = len(words_in_line)

    transcript = ' '

    for word_index in range(words_count_in_line):
      word = words_in_line[word_index]

      if word.endswith("\'S"):
        word = word[:-2]
      if word.endswith("\'"):
        word = word[:-1]

      transcript += word
      if (word_index + 1) != words_count_in_line:
        transcript += ' '

    return transcript, words_count_in_line

def get_mfcc_from_file_and_index(sentences_mfcc_file_path : str, sentence_index : int):
  sentences_mfcc_in_file = np.load(sentences_mfcc_file_path)
  sentence_mfcc = sentences_mfcc_in_file[sentence_index]

  return sentence_mfcc

def get_best_bucket(length : int, buckets : list):
  for bucket in buckets:
    if(length <= bucket[0]):
      length = bucket
      break
  if(length > buckets[-1][0]):
    length = buckets[-1]
  return length

def save_batch(librispeech_path : str, batch_id : int, mfcc_padded_length : int, mfcc_batch, 
               mfcc_batch_lengths : list, transcripts_batch : list, transcripts_batch_lengths : list, tokenized_transcripts_batch : list):
  batch_directory = librispeech_path + r"\batches"
  ### MFCC
  mfcc_batch = np.array(mfcc_batch)
  batch_file_path = "mfcc_batch" + str(batch_id) + "_l" + str(mfcc_padded_length) + ".npy"
  batch_file_path = os.path.join(batch_directory, batch_file_path)
  np.save(batch_file_path, mfcc_batch)

  ### Lengths
  mfcc_batch_lengths = np.array(mfcc_batch_lengths)
  mfcc_batch_lengths_file_path = "mfcc_batch_lengths_" + str(batch_id) + "_l" + str(mfcc_padded_length) + ".npy"
  mfcc_batch_lengths_file_path = os.path.join(batch_directory, mfcc_batch_lengths_file_path)
  np.save(mfcc_batch_lengths_file_path, mfcc_batch_lengths)

  ### Transcript
  transcripts_batch_file_path = "transcripts_batch_" + str(batch_id) + "_l" + str(mfcc_padded_length) + ".txt"
  transcripts_batch_file_path = os.path.join(batch_directory, transcripts_batch_file_path)
  transcripts_batch_output_file = open(transcripts_batch_file_path, 'w')
  transcripts_batch_output_file.writelines(transcripts_batch)
  transcripts_batch_output_file.close()

  transcripts_batch_lengths = np.array(transcripts_batch_lengths)
  transcripts_batch_lengths_file_path = "transcripts_batch_lengths_" + str(batch_id) + "_l" + str(mfcc_padded_length) + ".npy"
  transcripts_batch_lengths_file_path = os.path.join(batch_directory, transcripts_batch_lengths_file_path)
  np.save(transcripts_batch_lengths_file_path, transcripts_batch_lengths)

  tokenized_transcripts_batch = np.array(tokenized_transcripts_batch)
  tokenized_transcripts_batch_file_path = "tokenized_transcripts_batch_" + str(batch_id) + "_l" + str(mfcc_padded_length) + ".npy"
  tokenized_transcripts_batch_file_path = os.path.join(batch_directory, tokenized_transcripts_batch_file_path)
  np.save(tokenized_transcripts_batch_file_path, tokenized_transcripts_batch)

  return batch_file_path, mfcc_batch_lengths_file_path, transcripts_batch_file_path, transcripts_batch_lengths_file_path, tokenized_transcripts_batch_file_path

def tokenize_transcript(dictionary : dict, transcript : str, split_char = ' '):
  words = transcript.split(split_char)
  
  transcript_length = len(words)
  result = []
  for i in range(transcript_length):
    word = words[i]
    if word.endswith('\n'):
      word = word[:-1]
    if word in dictionary:
      result.append(dictionary[word])
  return result

def create_batches_of_sequences(librispeech_path : str, batch_size = 5000, buckets = ((150, 22), (250, 40), (500, 60), (1000, 80), (1500, 100), (2000, 120))):
  preprocess_order = get_preprocess_order(librispeech_path)
  mfcc_lengths = get_mfcc_lengths(librispeech_path)
  word_dictionary, _ = get_words_dictionary(librispeech_path)

  sentence_count = len(mfcc_lengths)
  i = 0
  batch_id = 0

  batches_infos = ""

  while i < sentence_count:
    batch_of_mfcc_infos = mfcc_lengths[i:min(i + batch_size, sentence_count - 1)]

    # MFCC
    mfcc_batch = []
    mfcc_batch_lengths = []

    # Transcripts
    transcripts_batch = []
    transcripts_batch_lengths = []
    tokenized_transcripts_batch = []

    max_mfcc_length_in_batch = 0

    test_ratio = 0

    # Main Loop : Gathering data + determination of max length
    print("Starting batch n°" + str(batch_id) + "...")

    for j in tqdm(range(len(batch_of_mfcc_infos))):
      [file_index, sentence_index, mfcc_length] = batch_of_mfcc_infos[j]

      ### MFCC
      sentences_mfcc_file_path = librispeech_path + preprocess_order[file_index][1]
      sentence_mfcc = get_mfcc_from_file_and_index(sentences_mfcc_file_path, sentence_index)
      mfcc_batch.append(sentence_mfcc)

      ### Length
      max_mfcc_length_in_batch = max(max_mfcc_length_in_batch, mfcc_length)
      mfcc_batch_lengths.append(mfcc_length)

      ### Text
      transcript_file_path = librispeech_path + preprocess_order[file_index][2]
      transcript_file_path = transcript_file_path.replace("seg", "trans")
      transcript, transcript_length = get_transcript_from_file_and_index(transcript_file_path, sentence_index)
      transcripts_batch += [transcript]
      transcripts_batch_lengths += [transcript_length]

      tokenized_transcripts_batch += [tokenize_transcript(word_dictionary, transcript)]

      test_ratio += mfcc_length / transcript_length

    print("test ratio", batch_id, "=", test_ratio / len(batch_of_mfcc_infos))

    # Choice of bucket (and padding size)
    mfcc_padded_length, transcript_padded_length = get_best_bucket(max_mfcc_length_in_batch, buckets)

    # Padding
    for j in tqdm(range(len(batch_of_mfcc_infos))):
      # Padding MFCC
      sentence_mfcc = mfcc_batch[j]
      pad_length = mfcc_padded_length - len(sentence_mfcc)
      mfcc_batch[j] = np.lib.pad(sentence_mfcc, ((0, pad_length), (0,0)), 'constant', constant_values=0)

      # Padding Transcript tokens
      tokenized_transcript = tokenized_transcripts_batch[j]
      pad_length = transcript_padded_length - len(tokenized_transcript)
      tokenized_transcripts_batch[j] = np.lib.pad(tokenized_transcript, ((pad_length), (0)), 'constant', constant_values=-1)

    # Saving (data and metadata)
    batch_file_path, mfcc_batch_lengths_file_path, transcripts_batch_file_path, transcripts_batch_lengths_file_path, tokenized_transcripts_batch_file_path = save_batch(librispeech_path, batch_id, mfcc_padded_length, mfcc_batch, mfcc_batch_lengths, transcripts_batch, transcripts_batch_lengths, tokenized_transcripts_batch)
    batches_infos += str(mfcc_padded_length) + ',' + \
      batch_file_path + ',' + mfcc_batch_lengths_file_path + ',' + \
      transcripts_batch_file_path + ',' + transcripts_batch_lengths_file_path + ',' + \
      tokenized_transcripts_batch_file_path + '\n'

    # Iteration
    i += batch_size
    batch_id += 1

  batches_infos_output_file = open(os.path.join(librispeech_path, "batches_infos.txt"), 'w')
  batches_infos_output_file.write(batches_infos)        
  batches_infos_output_file.close()

def main():
  librispeech_path = r"E:\LibriSpeech"
  #create_batches_of_sequences(librispeech_path, batch_size = 5000)

if __name__ == '__main__':
  main()
