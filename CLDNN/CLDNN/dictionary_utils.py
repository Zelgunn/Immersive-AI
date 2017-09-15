import os
from tqdm import tqdm
import numpy as np


def get_words_dictionary(librispeech_path : str, reduced_dictionary = False):
  # Retrieve existing file
  if not reduced_dictionary:
    dictionary_file_name = "dictionary.txt"
  else:
    dictionary_file_name = "reduced_dictionary.txt"
  dictionary_file_path = os.path.join(librispeech_path, dictionary_file_name)
  dictionary = dict()
  reverse_dictionary = dict()

  dictionary_file = open(dictionary_file_path, 'r')
  lines = dictionary_file.readlines()

  for i in range(len(lines)):
    word = lines[i]
    if word.endswith('\n'):
      word = word[:-1]
    dictionary[word] = i
    reverse_dictionary[i] = word

  dictionary_file.close()

  return dictionary, reverse_dictionary

    #  transcript += word
    #  if (word_index + 1) != words_count_in_line:
    #    transcript += ' '

    #return transcript, words_count_in_line

def get_words_utterance_counts(librispeech_path : str, descending_order = True):
  words_utterance_counts = dict()
  import SpeechDataUtils as SDU
  preprocess_order = SDU.get_preprocess_order(librispeech_path)
  transcript_files = []
  for preprocess_info in preprocess_order:
    transcript_file_path = librispeech_path + preprocess_info[2]
    transcript_file_path = transcript_file_path.replace("seg", "trans")
    transcript_files.append(transcript_file_path)

  for i in tqdm(range(len(transcript_files))):
    transcript_file_path = transcript_files[i]
    transcript_file = open(transcript_file_path, 'r')
    lines = transcript_file.readlines()
    transcript_file.close()

    for line in lines:
      split_index = line.find(' ') + 1
      words_in_line = line[split_index:]
      words_in_line = words_in_line.split(' ')

      for word_index in range(len(words_in_line)):
        word = words_in_line[word_index]
        if word.endswith('\n'):
          word = word[:-1]
        if word.endswith("\'S"):
          word = word[:-2]
        if word.endswith("\'"):
          word = word[:-1]

        if word in words_utterance_counts:
          words_utterance_counts[word] += 1
        else:
          words_utterance_counts[word] = 1

  words = words_utterance_counts.keys()
  counts = words_utterance_counts.values()

  words_counts_zip = zip(counts, words)
  words_counts_zip = sorted(words_counts_zip, reverse = descending_order)
  
  return words_counts_zip

def reduced_dictionary(librispeech_path : str, min_utterence_count : int, add_symbols = True):
  words_utterance_counts = get_words_utterance_counts(librispeech_path, False)
  words_count = len(words_utterance_counts)

  index = 0
  for count, word in words_utterance_counts:
    if count > min_utterence_count:
      break
    index += 1

  result = [word for _, word in words_utterance_counts[index:]]
  if add_symbols:
    result.append('<UNK>')
    result.append('<EOS>')
  return result

def create_reduced_dictionary(librispeech_path : str, min_utterence_count : int, add_symbols = True, dictionary_name = "reduced_dictionary.txt"):
  dictionary = reduced_dictionary(librispeech_path, min_utterence_count, add_symbols)
  dictionary_path = os.path.join(librispeech_path, dictionary_name)

  dictionary_to_string = ''
  for word_index in range(len(dictionary)):
    dictionary_to_string += dictionary[word_index]
    if (word_index + 1) != len(dictionary):
      dictionary_to_string += '\n'

  with open(dictionary_path, 'w') as dictionary_file:
    dictionary_file.write(dictionary_to_string)

def reduce_tokenized_transcripts(librispeech_path : str):
  def reduce_dict_size_of(fullpath : str):
    data = np.load(fullpath)
    w = data.shape[0]
    h = data.shape[1]

    for i in range(w):
        for j in range(h):
          data[i][j] = conv_dict[data[i][j]]
    np.save(fullpath, data)

  def repad(fullpath : str):
    data = np.load(fullpath)
    w = data.shape[0]
    h = data.shape[1]

    for i in range(w):
      pad_index = h
      offset = -1
      for j in range(h):
        token = data[i][j]
        pad_index -= 1
        offset += 1
        if token != 9632:
          break
      line_copy = data[i].copy()
      for j in range(h):
        if j <= pad_index:
          data[i][j] = line_copy[j + offset]
        else:
          data[i][j] = 9632
    np.save(fullpath, data)


  r_dict, _ = get_words_dictionary(librispeech_path, True)
  f_dict, _ = get_words_dictionary(librispeech_path, False)

  print(r_dict["<EOS>"])

  conv_dict = dict()
  for word in f_dict:
    f_id = f_dict[word]
    if word in r_dict:
      conv_dict[f_id] = r_dict[word]
    else:
      conv_dict[f_id] = r_dict["<UNK>"]
  conv_dict[77963] = r_dict["<EOS>"]

  batches_path = os.path.join(librispeech_path, "batches")
  filename_1 = "tokenized_transcripts_batch_"

  batches = ((27, "_l150.npy"), (53, "_l250.npy"), (81, "_l500.npy"), (88, "_l1000.npy"))
  prev_size = 0
  for batch in batches:
    size = batch[0]
    filename_2 = batch[1]
    for i in range(prev_size, size):
      filename = filename_1 + str(i) + filename_2
      fullpath = os.path.join(batches_path, filename)
      print(i, fullpath)
      #reduce_dict_size_of(fullpath)
      repad(fullpath)
    prev_size = size
