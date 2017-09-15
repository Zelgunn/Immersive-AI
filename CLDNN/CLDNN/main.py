import os
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import math

from DNN.DNNRun import DNNRun
from CLDNNModel import CLDNNModel, CLDNNConfig
from SpeechDataUtils import SpeechDataUtils, SpeechDataSet
import dictionary_utils

class CLDNN():
  def __init__(self, librispeech_path : str, summary_base_path : str):
    self.librispeech_path = librispeech_path
    self.summary_base_path = summary_base_path
    self._init_config()
    self._init_data()
    self._init_graph()
    self._init_session()

  def _init_config(self):
    self.features_count = 40
    self.max_input_sequence_length = 250
    self.max_output_sequence_length = 40

    self.model_config_path = "ModelConfig.ini"
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
    self.data = SpeechDataUtils(librispeech_path = self.librispeech_path, bucket_size = self.max_input_sequence_length)
    self.train_data = self.data.train
    self.eval_data = self.data.eval

    self.dictionary_size = self.data.dictionary_size

  def _init_graph(self):
    self.graph = tf.Graph()
    self.graph.as_default()

    self.cldnn_config = CLDNNConfig(self.max_input_sequence_length, self.features_count, 
                                     self.max_output_sequence_length, self.dictionary_size,
                                     self.model_config_path)
    self.cldnn_model = CLDNNModel(self.cldnn_config)

    self.inference_op = self.cldnn_model.inference
    self.loss_op = self.cldnn_model.loss
    self.train_op = self.cldnn_model.training
    self.eval_op = self.cldnn_model.evaluation

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
      learning_rate = max(1e-5, self.cldnn_model.config.learning_rate / math.log(math.e * (i + 1)))
      batch_inputs, batch_input_lengths, batch_outputs, batch_output_lengths = self.train_data.next_batch(batch_size, one_hot = True)

      #tmp = []
      #for sample_output in batch_outputs:
      #  tmp.append(sample_output[:self.max_output_sequence_length])
      #batch_outputs = tmp

      feed_dict = {
          self.cldnn_model.input_placeholder : batch_inputs,
          self.cldnn_model.input_lengths_placeholder : batch_input_lengths,
          self.cldnn_model.output_placeholder : batch_outputs,
          self.cldnn_model.output_lengths_placeholder : batch_output_lengths,
          self.cldnn_model.learning_rate_placeholder : learning_rate
          }

      _, loss_value = self.session.run([self.train_op, self.loss_op], feed_dict = feed_dict)

      if i%self.log_every_x_steps == 0:
        summary_str = self.session.run(self.summary, feed_dict = feed_dict)
        self.summary_writer.add_summary(summary_str, i)
        self.summary_writer.flush()

      if (i%self.save_every_x_steps == 0) or ((i + 1) == training_iteration_count):
        checkpoint_file = self.checkpoints_path + "\model.ckpt"
        print("\nAt step", i, "loss = ", loss_value,"\tLearning rate :", learning_rate, '\n')
        self.saver.save(self.session, checkpoint_file, global_step = i)


    batch_inputs, batch_input_lengths, batch_outputs, batch_output_lengths = self.train_data.next_batch(batch_size, one_hot = True)

    #tmp = []
    #for sample_output in batch_outputs:
    #  tmp.append(sample_output[:self.max_output_sequence_length])
    #batch_outputs = tmp

    feed_dict = {
        self.cldnn_model.input_placeholder : batch_inputs,
        self.cldnn_model.input_lengths_placeholder : batch_input_lengths,
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

  def evaluate(self, evaluation_iteration_count : int):
    total_eval = np.zeros(self.max_output_sequence_length)
    print("Testing model on", evaluation_iteration_count, "samples")
    for i in tqdm(range(evaluation_iteration_count)):
      batch_inputs, batch_input_lengths, batch_outputs, batch_output_lengths = self.eval_data.next_batch(1, one_hot = True)

      feed_dict = {
          self.cldnn_model.input_placeholder : batch_inputs,
          self.cldnn_model.input_lengths_placeholder : batch_input_lengths,
          self.cldnn_model.output_placeholder : batch_outputs,
          self.cldnn_model.output_lengths_placeholder : batch_output_lengths
          }

      session_eval = self.session.run(self.eval_op, feed_dict = feed_dict)
      total_eval += session_eval.reshape(self.max_output_sequence_length,)
    total_eval /= evaluation_iteration_count
    print("\nAccuracy = " +str(total_eval * 100) + "%\n")

    batch_inputs, batch_input_lengths, batch_outputs, batch_output_lengths = self.eval_data.next_batch(1, one_hot = True)

    feed_dict = {
        self.cldnn_model.input_placeholder : batch_inputs,
        self.cldnn_model.input_lengths_placeholder : batch_input_lengths,
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

def main():
  timit_database_path = r"C:\tmp\TIMIT"
  librispeech_path = r"C:\tmp\Librispeech"
  #cldnn = CLDNN(librispeech_path, r"C:\tmp\custom\CLDNN_CTC")
  #cldnn.train(125000, 10)
  #cldnn.evaluate(1000) # not working :(
  #dictionary_utils.reduce_tokenized_transcripts(librispeech_path)
  dnn = DNNRun(librispeech_path, timit_database_path, r"C:\tmp\custom\DNN")
  dnn.train(125000, 10)

if __name__ == '__main__':
  main()
