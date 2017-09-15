import tensorflow as tf
import numpy as np
from tqdm import tqdm
from Timit_utils.TimitDatabase import TimitDatabase
from SpeechDataUtils import SpeechDataSet

timit_database = TimitDatabase(r"G:\TIMIT")

##### PARAMETERS #######
max_input_sequence_length = timit_database.get_max_mfcc_features_length()
max_output_sequence_length = timit_database.get_max_phonemes_length()
dictionary_size = timit_database.phonemes_dictionary_size

train_data = timit_database.train_dataset
eval_data = timit_database.test_dataset

##### PLACEHOLDERS #######
input_placeholder = tf.placeholder(tf.float32, shape = [None, max_input_sequence_length, 40])
input_lengths_placeholder = tf.placeholder(tf.int32, shape = [None])
output_placeholder = tf.sparse_placeholder(tf.int32)
output_lengths_placeholder = tf.placeholder(tf.int32, shape = [None])

cell = tf.nn.rnn_cell.LSTMCell(128)
cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 2)
rnn, _ = tf.nn.dynamic_rnn(cell, input_placeholder, input_lengths_placeholder, dtype = tf.float32)

rnn = tf.reshape(rnn, shape = [-1, max_input_sequence_length * 128])

weights = tf.Variable(tf.random_normal(shape = [max_input_sequence_length * 128, 256], stddev = 0.1))
biases = tf.Variable(tf.zeros(shape = [256]))

layer = tf.matmul(rnn, weights) + biases
layer = tf.nn.relu(layer)

weights = tf.Variable(tf.random_normal(shape = [256, max_output_sequence_length * dictionary_size], stddev = 0.1))
biases = tf.Variable(tf.zeros(shape = [max_output_sequence_length * dictionary_size]))

layer = tf.matmul(layer, weights) + biases
layer = tf.nn.relu(layer)
##### LOGITS ######
logits = tf.reshape(layer, shape = [-1, max_output_sequence_length, dictionary_size])

##### LOSS #####
logits_timemajor = tf.transpose(logits, [1, 0, 2])
loss = tf.nn.ctc_loss(logits_timemajor, output_placeholder, output_lengths_placeholder)
loss = tf.reduce_mean(loss)

##### TRAIN #####
optimizer = tf.train.AdamOptimizer()
train = optimizer.minimize(loss)

session_config = tf.ConfigProto()
session_config.gpu_options.allow_growth = True
##### SESSION #####
with tf.Session(config = session_config) as session:

  init = tf.global_variables_initializer()
  session.run(init)

  for i in tqdm(range(1000)):
    batch_inputs, batch_input_lengths, batch_outputs, batch_output_lengths = train_data.next_batch(11, False)

    tmp = []
    for j in range(len(batch_outputs)):
      tmp.append(batch_outputs[j][:batch_output_lengths[j]])

    batch_outputs = np.array(tmp)
    batch_outputs = SpeechDataSet.tokens_for_sparse(batch_outputs)

    feed_dict = \
    {
      input_placeholder : batch_inputs,
      input_lengths_placeholder : batch_input_lengths,
      output_placeholder : batch_outputs,
      output_lengths_placeholder : batch_output_lengths
    }

    session.run(train, feed_dict = feed_dict)

    if(i%100 == 0):
      print('######')
      print(session.run(loss, feed_dict = feed_dict))
      print('######')

  ##### TEST #####
  batch_inputs, batch_input_lengths, batch_outputs, batch_output_lengths = train_data.next_batch(1, False)
  feed_dict = \
  {
    input_placeholder : batch_inputs,
    input_lengths_placeholder : batch_input_lengths,
  }

  test = session.run(logits, feed_dict = feed_dict)
  test = session.run(tf.transpose(test, [1, 0, 2]))
  decoded, prob = tf.nn.ctc_beam_search_decoder(test, batch_output_lengths, 100, 5)

  decoded = session.run(decoded)
  prob = session.run(prob)

  results = []
  for decoded_path in decoded:
    phonemes_ids = decoded_path.values
    result_words = ""
    for idx in phonemes_ids:
      word = timit_database.id_to_phoneme_dictionary[idx]
      if word != "<EOS>":
        result_words += word + ' '
    results += [result_words]

  target = batch_outputs[0]
  target_words = ""
  for idx in target:
    word = timit_database.id_to_phoneme_dictionary[idx]
    if word != "<EOS>":
      target_words += word + ' '

  print("--------------")
  print("--------------")
  print("--------------")
  print("Target : \n", target_words, '\n')
  print("Results")
  for result_sentence in results:
    print(result_sentence)
  #print(prob)
  print("--------------")
  print("--------------")
  print("--------------")
