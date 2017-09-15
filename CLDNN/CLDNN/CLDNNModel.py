import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm
import functools
from SpeechDataUtils import SpeechDataUtils
from SpeechDataUtils import SpeechDataSet
from CLDNNConfig import CLDNNConfig

def define_scope(function):
  attribute = '_cache_' + function.__name__

  @property
  @functools.wraps(function)
  def decorator(self):
    if not hasattr(self, attribute):
      with tf.variable_scope(function.__name__):
        setattr(self, attribute, function(self))
    return getattr(self, attribute)

  return decorator

class CLDNNModel:
  def __init__(self, config : CLDNNConfig):
    self.config = config

    self.init_placeholders()
        
    self.input_size = int(self.config.max_timesteps)
    self.output_size = int(self.config.max_output_length)

    print("Inference : ", self.inference)
    print("Loss : ", self.loss)
    print("Training : ", self.training)
    print("Evaluation : ", self.evaluation)

  def init_placeholders(self):
    self.input_placeholder = tf.placeholder(tf.float32, [None, self.config.max_timesteps, self.config.mfcc_features_count], name="input__placeholder")
    self.input_lengths_placeholder = tf.placeholder(tf.int32, [None], name="input_lengths_placeholder")

    self.sparse_output_placeholder = tf.sparse_placeholder(tf.int32, name="sparse_true_output_placeholder")
    self.output_placeholder = tf.placeholder(tf.int32, shape=[None, self.config.max_output_length, self.config.dictionary_size], name="true_output_placeholder")
    self.output_lengths_placeholder = tf.placeholder(tf.int32, [None], name="output_lengths_placeholder")

    self.learning_rate_placeholder = tf.placeholder(tf.float32, [], name="learning_rate")

  @define_scope
  def inference(self):
    input_as_2d_tensor = tf.reshape(self.input_placeholder, [-1, self.config.max_timesteps, self.config.mfcc_features_count, 1])

    # 1st Layer (Convolution)
    ## Weights & Bias
    with tf.name_scope("Convolution"):
      weights_conv_layer = CLDNNModel.weight_variable([self.config.conv_kernel_size, self.config.conv_kernel_size, 1, self.config.conv_features_count])
      bias_conv_layer = CLDNNModel.bias_variable([self.config.conv_features_count])
      ## Result
      conv_layer = CLDNNModel.conv2d(input_as_2d_tensor, weights_conv_layer) + bias_conv_layer
      relu_conv_layer = tf.nn.relu(conv_layer)

    # 2nd Layer (Max Pooling)
    with tf.name_scope("Max_pooling"):
      max_pool_layer = CLDNNModel.max_pool_1xN(relu_conv_layer, self.config.max_pooling_size)

    # 3rd Layer (Dimension reduction)
    ## Flattening (from 2D to 1D)
    with tf.name_scope("Dim_reduction"):
      convoluted_size = int(self.config.max_timesteps) * int(self.config.mfcc_features_count / self.config.max_pooling_size)
      flatten_size = convoluted_size * self.config.conv_features_count
      #flatten_size = int(convoluted_size * self.conv_features_count / self.config.max_timesteps)
      max_pool_layer_flatten = tf.reshape(max_pool_layer, [-1, flatten_size], name="Flatten_maxpool")
      ## Weights and Bias
      time_red_size = int(self.config.max_timesteps / self.config.time_reduction_factor)
      dim_red_size = time_red_size * self.config.dimension_reduction_output_size
      weights_dim_red_layer = CLDNNModel.weight_variable([flatten_size, dim_red_size])
      bias_dim_red_layer = CLDNNModel.bias_variable([dim_red_size])
      ## Result
      dim_red_layer = tf.matmul(max_pool_layer_flatten, weights_dim_red_layer) + bias_dim_red_layer

    # Input reduction (for memory issues :( )
    with tf.name_scope("Input_reduction"):
      flatten_input_size = self.config.max_timesteps * self.config.mfcc_features_count
      flatten_input_size_red = int(flatten_input_size / self.config.time_reduction_factor)
      flatten_input = tf.reshape(self.input_placeholder, [-1, flatten_input_size], name="flatten_input")
      
      weights = CLDNNModel.weight_variable([flatten_input_size, flatten_input_size_red])
      biaises = CLDNNModel.bias_variable([flatten_input_size_red])

      red_input = tf.matmul(flatten_input, weights) + biaises
      red_time = tf.cast(tf.ceil(self.input_lengths_placeholder / self.config.time_reduction_factor), tf.int32)

    # 4th Layer (Concatenation)
    with tf.name_scope("Concatenation"):
      concatenation_layer = tf.concat(1, [dim_red_layer, red_input])
      concatenation_layer_reshaped = tf.reshape(concatenation_layer, (-1, time_red_size, self.config.dimension_reduction_output_size + self.config.mfcc_features_count), name="reshape_timesteps_concat")

    # 5th Layer (LSTM 1)
    with tf.name_scope("LSTM1"):
      with tf.variable_scope("LSTMCell1"):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(self.config.lstm1_hidden_units_count)
        lstm1_output, lstm_state = tf.nn.dynamic_rnn(lstm_cell, concatenation_layer_reshaped, dtype=tf.float32, sequence_length = red_time)

    # 6th Layer (LSTM 2)
    with tf.name_scope("LSTM2"):
      with tf.variable_scope("LSTMCell2"):
        lstm_cell = tf.nn.rnn_cell.LSTMCell(self.config.lstm2_hidden_units_count)
        lstm2_output, lstm_state = tf.nn.dynamic_rnn(lstm_cell, lstm1_output, dtype=tf.float32)

    lstm2_output_shape = lstm2_output.get_shape()
    lstm2_output_shape = [-1, int(lstm2_output_shape[1] * lstm2_output_shape[2])]
    lstm2_output_reshaped = tf.reshape(lstm2_output, lstm2_output_shape)

    # 7th Layer (Fully connected 1)
    with tf.name_scope("Fully_connected1"):
      weights = CLDNNModel.weight_variable([lstm2_output_shape[1], self.config.fully_connected1_size])
      biases = CLDNNModel.bias_variable([self.config.fully_connected1_size])

      fully_connected_layer1 = tf.matmul(lstm2_output_reshaped, weights) + biases

    # 7th Layer (Fully connected 2)
    with tf.name_scope("Fully_connected2"):
      weights = CLDNNModel.weight_variable([self.config.fully_connected1_size, self.output_size * self.config.dictionary_size])
      biases = CLDNNModel.bias_variable([self.output_size * self.config.dictionary_size])

      fully_connected_layer2 = tf.matmul(fully_connected_layer1, weights) + biases

    logits = tf.reshape(fully_connected_layer2, [-1, self.output_size , self.config.dictionary_size])
        
    return logits # Should be the 7th layer's ouput

  @define_scope
  def loss(self):
    if self.config.use_ctc_loss:
      inference_time_major = tf.transpose(self.inference, [1, 0, 2])
      ctc = tf.nn.ctc_loss(inference_time_major, self.sparse_output_placeholder, self.output_lengths_placeholder, time_major = True, ctc_merge_repeated=False)
      return tf.reduce_mean(ctc)
    else:
      cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.inference, self.output_placeholder)
      return tf.reduce_mean(cross_entropy)

  @define_scope
  def training(self):
    tf.summary.scalar('loss', self.loss)

    #optimizer = tf.train.GradientDescentOptimizer(self.config.learning_rate)
    optimizer = tf.train.MomentumOptimizer(self.learning_rate_placeholder, 0.9)

    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_op = optimizer.minimize(self.loss, global_step = global_step)

    return train_op
    
  @define_scope
  def evaluation(self):
    correct_prediction = tf.equal(tf.argmax(self.inference, 2), tf.argmax(self.output_placeholder, 2))
    return tf.cast(correct_prediction, tf.float32)
    
  @staticmethod
  def conv2d(inputTensor, weights):
    return tf.nn.conv2d(inputTensor, weights, strides=[1, 1, 1, 1], padding='SAME')

  @staticmethod
  def max_pool_1xN(inputTensor, max_pooling_size):
    return tf.nn.max_pool(inputTensor, ksize=[1, 1, max_pooling_size, 1], strides=[1, 1, max_pooling_size, 1], padding='SAME')

  @staticmethod
  def init_variable(shape, init_method='uniform', xavier_params = (None, None)):
    if init_method == 'zeros':
      return tf.Variable(tf.zeros(shape, dtype=tf.float32))
    elif init_method == 'uniform':
      return tf.Variable(tf.random_normal(shape, stddev=0.01, dtype=tf.float32))
    else: #xavier
      (fan_in, fan_out) = xavier_params
      low = -4*np.sqrt(6.0/(fan_in + fan_out)) # {sigmoid:4, tanh:1} 
      high = 4*np.sqrt(6.0/(fan_in + fan_out))
      return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))
    # Need for gaussian (for LSTM)

  @staticmethod
  def weight_variable(shape, init_method='uniform', xavier_params = (None, None)):
    return CLDNNModel.init_variable(shape, init_method, xavier_params)

  @staticmethod
  def bias_variable(shape, init_method='uniform', xavier_params = (None, None)):
    return CLDNNModel.init_variable(shape, init_method, xavier_params)