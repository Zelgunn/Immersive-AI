import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm

from SpeechDataUtils import SpeechDataUtils, SpeechDataSet
from DNN.DNNModel import DNNModel
from Timit_utils.TimitDatabase import TimitDatabase

class DNNRun():
  def __init__(self, librispeech_path : str, timit_database_path : str, summary_base_path : str):
    self.librispeech_path = librispeech_path
    self.timit_database_path = timit_database_path
    self.summary_base_path = summary_base_path
    self._init_config()
    self._init_data()
    self._init_graph()
    self._init_session()

  def _init_config(self):
    self.features_count = 40
    self.timit_database = TimitDatabase(self.timit_database_path)

    self.max_input_sequence_length = self.timit_database.get_max_mfcc_features_length()
    self.max_output_sequence_length = self.timit_database.get_max_phonemes_length()

    self.summary_logs_path = self.summary_base_path + r"\logs"
    self.checkpoints_path = self.summary_base_path + r"\checkpoints"
    self.log_every_x_steps = 100
    self.save_every_x_steps = 1000

    if not os.path.exists(self.summary_base_path):
      os.mkdir(self.summary_base_path)
    if not os.path.exists(self.summary_logs_path):
      os.mkdir(self.summary_logs_path)
    if not os.path.exists(self.checkpoints_path):
      os.mkdir(self.checkpoints_path)

  def _init_data(self):
    #self.data = SpeechDataUtils(librispeech_path = self.librispeech_path, bucket_size = self.max_input_sequence_length)
    #self.train_data = self.data.train
    #self.eval_data = self.data.eval
    #self.dictionary_size = self.data.dictionary_size

    self.train_data = self.timit_database.train_dataset
    self.eval_data = self.timit_database.test_dataset
    self.dictionary_size = self.timit_database.phonemes_dictionary_size

  def _init_graph(self):
    self.graph = tf.Graph()
    self.graph.as_default()

    self.dnn_model = DNNModel(self.max_input_sequence_length, self.features_count, self.max_output_sequence_length, self.dictionary_size)

    self.inference_op = self.dnn_model.inference
    self.loss_op = self.dnn_model.loss
    self.train_op = self.dnn_model.training
    self.eval_op = self.dnn_model.evaluation

    self.summary = tf.summary.merge_all()
    self.saver = tf.train.Saver()

  def _init_session(self):
    self.session_config = tf.ConfigProto()
    self.session_config.gpu_options.allow_growth = True

    self.session = tf.Session(config = self.session_config)

    self.summary_writer = tf.summary.FileWriter(self.summary_logs_path, self.session.graph)

    checkpoint = tf.train.get_checkpoint_state(self.checkpoints_path)
    if checkpoint and checkpoint.model_checkpoint_path:
      print("Found existing checkpoint in", checkpoint.model_checkpoint_path)
      self.saver.restore(self.session, checkpoint.model_checkpoint_path)
      print("Loaded checkpoints from", checkpoint.model_checkpoint_path)
    else:
      print("No checkpoint found. Starting from scratch.")
      init_op = tf.global_variables_initializer()
      print("Initializing variables...")
      self.session.run(init_op)
      print("Variables initialized !")

  def train(self, training_iteration_count : int, batch_size : int):
    for i in tqdm(range(training_iteration_count)):
      batch_inputs, batch_input_lengths, batch_outputs, batch_output_lengths = self.train_data.next_batch(batch_size)

      tmp = []
      for sample_output in batch_outputs:
        tmp.append(sample_output[:self.max_output_sequence_length])
      batch_outputs = tmp

      feed_dict = {
          self.dnn_model.input_placeholder : batch_inputs,
          self.dnn_model.output_placeholder : batch_outputs,
          self.dnn_model.dropout_placeholder : 0.5,
          self.dnn_model.learning_rate_placeholder : 1e-4
          }

      _, loss_value = self.session.run([self.train_op, self.loss_op], feed_dict = feed_dict)

      if i%self.log_every_x_steps == 0:
        summary_str = self.session.run(self.summary, feed_dict = feed_dict)
        self.summary_writer.add_summary(summary_str, i)
        self.summary_writer.flush()

      if (i%self.save_every_x_steps == 0) or ((i + 1) == training_iteration_count):
        checkpoint_file = self.checkpoints_path + "\model.ckpt"
        print("\nAt step", i, "loss = ", loss_value, '\n')
        self.saver.save(self.session, checkpoint_file, global_step = i)

        batch_inputs, batch_input_lengths, batch_outputs, batch_output_lengths = self.train_data.next_batch(1)
        tmp = []
        for sample_output in batch_outputs:
          tmp.append(sample_output[:self.max_output_sequence_length])
        batch_outputs = tmp

        feed_dict = {
            self.dnn_model.input_placeholder : batch_inputs,
            self.dnn_model.dropout_placeholder : 1
            }

        result = self.session.run(self.inference_op, feed_dict = feed_dict)
        result = np.argmax(result[0], axis=1)
        target = np.argmax(batch_outputs[0], axis=1)

        result_words = ""
        for idx in result:
          word = self.timit_database.id_to_phoneme_dictionary[idx]
          if word != "<EOS>":
            result_words += word + ' '

        target_words = ""
        for idx in target:
          word = self.timit_database.id_to_phoneme_dictionary[idx]
          if word != "<EOS>":
            target_words += word + ' '

        save_example_path = os.path.join(self.checkpoints_path, "save_example_" + str(i) + ".txt")
        save_example_string = result_words + '\n' + target_words
        save_example_file = open(save_example_path, 'w')
        save_example_file.write(save_example_string)
        save_example_file.close()

  def evaluate(self, evaluation_iteration_count : int):
    total_eval = np.zeros(self.max_output_sequence_length)
    print("Testing model on", evaluation_iteration_count, "samples")
    for i in tqdm(range(evaluation_iteration_count)):
      batch_inputs, batch_input_lengths, batch_outputs, batch_output_lengths = self.eval_data.next_batch(1)

      feed_dict = {
          self.dnn_model.input_placeholder : batch_inputs,
          self.dnn_model.output_placeholder : batch_outputs,
          self.dnn_model.dropout_placeholder : 1
          }

      session_eval = self.session.run(self.eval_op, feed_dict = feed_dict)
      total_eval += session_eval.reshape(self.max_output_sequence_length,)
    total_eval /= evaluation_iteration_count
    print("\nAccuracy = " + str(total_eval * 100) + "%\n")

    batch_inputs, batch_input_lengths, batch_outputs, batch_output_lengths = self.eval_data.next_batch(1, one_hot = True)

    feed_dict = {
        self.dnn_model.input_placeholder : batch_inputs,
        self.dnn_model.dropout_placeholder : 1
        }

    result = self.session.run(self.inference_op, feed_dict = feed_dict)
    result = np.argmax(result[0], axis=1)
    target = np.argmax(batch_outputs[0], axis=1)
    print("Result :\n", result)
    print("\nTarget:\n", target)

    result_words = ""
    for idx in result:
      word = self.data.reverse_dictionary[idx]
      if word != "<EOS>":
        result_words += word + ' '

    target_words = ""
    for idx in target:
      word = self.data.reverse_dictionary[idx]
      if word != "<EOS>":
        target_words += word + ' '

    print("\n\nResult words :\n", result_words)
    print("\nTarget words:\n", target_words)