import tensorflow as tf

from Supervised.ModelSkeleton import ModelSkeleton, define_scope
from Supervised.SimpleDNN.SimpleDNNConfig import SimpleDNNConfig

class SimpleDNNModel(ModelSkeleton):
    def __init__(self, config : SimpleDNNConfig, input_size : int, output_size : int):
      self.config = config

      self.input_size = input_size
      self.output_size = output_size

      self.input_placeholder = tf.placeholder(tf.uint8, (input_size))
      self.output_placeholder = tf.placeholder(tf.float32, (output_size))

      self.inference
      self.loss
      self.training
      self.evaluation

    @define_scope
    def placeholders(self):
      return (self.input_placeholder, self.output_placeholder)

    @define_scope
    def inference(self):

      with tf.name_scope("DNN_start"):
        weights = tf.get_variable("weights_start", shape = (self.input_size, self.config.layers_size), dtype = tf.float32)
        biases = tf.get_variable("biases_start", shape = (self.config.layers_size), dtype = tf.float32)

        reshape_input = tf.reshape(self.input_placeholder, (-1, self.input_size))
        convert_input = tf.cast(reshape_input, dtype = tf.float32)
        layer = tf.matmul(convert_input, weights) + biases

      for i in range(1, self.config.layer_count):
        with tf.name_scope("DNN_" + str(i)):
          weights = tf.get_variable("weights_" + str(i), shape = (self.config.layers_size, self.config.layers_size), dtype = tf.float32)
          biases = tf.get_variable("biases_" + str(i), shape = (self.config.layers_size), dtype = tf.float32)

          layer = tf.matmul(layer, weights) + biases

      with tf.name_scope("DNN_last"):
        weights = tf.get_variable("weights_last", shape = (self.config.layers_size, self.output_size), dtype = tf.float32)
        biases = tf.get_variable("biases_last", shape = (self.output_size), dtype = tf.float32)

        layer = tf.matmul(layer, weights) + biases

      return layer

    @define_scope
    def loss(self):
      cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.inference, self.output_placeholder))
      return cross_entropy

    @define_scope
    def training(self):
      optimizer = tf.train.GradientDescentOptimizer(self.config.learning_rate)
      training_op = optimizer.minimize(self.loss)
      return training_op

    @define_scope
    def evaluation(self):
      correct_prediction = tf.equal(tf.argmax(self.inference, 1), tf.argmax(self.output_placeholder, 1))
      return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
