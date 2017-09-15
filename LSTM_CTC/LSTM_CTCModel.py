import tensorflow as tf
import numpy as np
from tqdm import tqdm

from Timit_utils.TimitDatabase import TimitDatabase
from ModelSkeletons.SupervisedModel import SupervisedModel, define_scope
from LSTM_CTCConfig import LSTM_CTCModelConfig

class LSTM_CTCModel(SupervisedModel):
    def __init__(self, config : LSTM_CTCModelConfig):
        self.config = config

        self.graph = tf.Graph()
        self.graph.as_default()

        self.build_placeholders()
        self.inference
        self.loss
        self.training
        self.decoded_inference
        self.evaluation

        self.session = None

    def build_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.float32, shape = [None, self.config.input_max_time, self.config.input_frame_size], name = "input_placeholder")
        self.input_lengths_placeholder = tf.placeholder(tf.int32, shape = [None], name = "input_lengths_placeholder")
        self.output_placeholder = tf.sparse_placeholder(tf.int32, name = "output_placeholder")
        self.output_lengths_placeholder = tf.placeholder(tf.int32, shape = [None], name = "output_placeholder")

    @define_scope
    def inference(self):
        # LSTM
        with tf.name_scope("LSTM"):
            cell = tf.nn.rnn_cell.LSTMCell(self.config.cell_size)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.config.cell_count)

            cell_initial = cell.zero_state(tf.shape(self.input_placeholder)[0], tf.float32)
            rnn, _ = tf.nn.dynamic_rnn(cell, self.input_placeholder, self.output_lengths_placeholder, cell_initial)

            flatten_rnn = tf.reshape(rnn, shape = [-1, self.config.input_max_time * self.config.cell_size])
            #flatten_rnn = tf.reshape(rnn, shape = [-1, self.config.fully_connected_size])

        # Hidden layers
        layer = flatten_rnn
        previous_layer_size = int(layer.get_shape()[1])
        for i in range(self.config.fully_connected_count):
            with tf.name_scope("FullyConnected_n" + str(i)):
                # WEIGHTS & BIASES #
                weights = tf.Variable(tf.random_normal(shape = [previous_layer_size, self.config.fully_connected_size], stddev = 0.1), name = "Weights_n" + str(i))
                biases = tf.Variable(tf.zeros(shape = [self.config.fully_connected_size]), name = "Biases_n" + str(i))
                # MATMUL + RELU #
                layer = tf.matmul(layer, weights) + biases
                layer = tf.nn.relu(layer)
                #layer = tf.nn.softplus(layer)
                #layer = tf.nn.softmax(layer)
                previous_layer_size = int(layer.get_shape()[1])

        with tf.name_scope("FullyConnected_Last"):
            # Last layer
            # WEIGHTS & BIASES #
            weights = tf.Variable(tf.random_normal(shape = [previous_layer_size, self.config.output_max_time * self.config.output_frame_size], stddev = 0.1))
            biases = tf.Variable(tf.zeros(shape = [self.config.output_max_time * self.config.output_frame_size]))
            # MATMUL + RELU #
            layer = tf.matmul(layer, weights) + biases
            #layer = tf.nn.relu(layer)
            #layer = tf.nn.softmax(layer)

        logits = tf.reshape(layer, shape = [-1, self.config.output_max_time, self.config.output_frame_size])
        return logits

    @define_scope
    def loss(self):
        #logits_timemajor = tf.transpose(self.inference, [1, 0, 2])
        ctc_loss = tf.nn.ctc_loss(self.inference, self.output_placeholder, self.output_lengths_placeholder, time_major = False)
        return tf.reduce_mean(ctc_loss, name = "cost")

    @define_scope
    def training(self):
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("error", self.evaluation)
        global_step = tf.Variable(0, name="global_step", trainable=False)

        optimizer = tf.train.AdamOptimizer(self.config.learning_rate, epsilon = 1e-03)
        return optimizer.minimize(self.loss, global_step = global_step)

    @define_scope
    def decoded_inference(self):
        inference_timemajor = tf.transpose(self.inference, [1, 0, 2])
        decoded, prob = tf.nn.ctc_greedy_decoder(inference_timemajor, self.output_lengths_placeholder)
        return decoded

    @define_scope
    def evaluation(self):
        average_error = []
        for result in self.decoded_inference:
            predictions = tf.cast(result, tf.int32)
            error = tf.edit_distance(predictions, self.output_placeholder, normalize = True)
            average_error.append(tf.reduce_mean(error, name='error'))
        return tf.reduce_mean(average_error)
 
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
            indices.extend(zip([n] * len(seq), range(len(seq))))
            values.extend(seq)

        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=np.int32)
        shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

        return indices, values, shape

    def train(self, database, iteration_count  : int, batch_size : int):
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True
        self.session = tf.Session(config = session_config)

        self.init_model()

        train_data = database.train_dataset
        for i in tqdm(range(iteration_count)):
            feed_dict = self.train_step(train_data, batch_size)
            if feed_dict is None:
                continue

            for key in feed_dict:
                if feed_dict[key] is None:
                    continue

            if len(feed_dict) == 0:
                continue

            if((i + 1) % 10 == 0):
                summary_str = self.session.run(self.summary, feed_dict = feed_dict)
                self.summary_writer.add_summary(summary_str, i)
                self.summary_writer.flush()

            if((i + 1) % 100 == 0):
                
                print('######')
                print("Loss :", self.session.run(self.loss, feed_dict = feed_dict))
                print('######')

            if((i + 1) % 1000 == 0):
                self.saver.save(self.session, self.config.checkpoints_path + r"\network", global_step=i)
                self.test(database)

    def init_model(self):
        self.summary = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(self.config.checkpoints_path + "/logs", self.session.graph)
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(self.config.checkpoints_path)

        if checkpoint and checkpoint.model_checkpoint_path:
            print("Found existing checkpoint, loading ...")
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
        else: 
            print("Didn't find existing checkpoint, creating one ...")
            self.session.run(tf.global_variables_initializer())
        print("Model initialized !")

    def get_feed_dict(self, dataset, batch_size, one_hot = False):
        batch_inputs, batch_input_lengths, batch_outputs, batch_output_lengths = dataset.next_batch(batch_size, one_hot = one_hot)

        tmp = []
        for j in range(len(batch_outputs)):
            tmp.append(batch_outputs[j][:batch_output_lengths[j]])

        batch_outputs_feed = np.array(tmp)
        try:
            batch_outputs_feed = LSTM_CTCModel.tokens_for_sparse(batch_outputs_feed)
        except:
            return None

        tmp = []
        for j in range(len(batch_inputs)):
            tmp.append(batch_inputs[j][:self.config.input_max_time])
        batch_inputs = np.array(tmp)

        feed_dict = \
        {
            self.input_placeholder : batch_inputs,
            self.input_lengths_placeholder : batch_input_lengths,
            self.output_placeholder : batch_outputs_feed,
            self.output_lengths_placeholder : batch_output_lengths
        }

        return feed_dict

    def train_step(self, train_data, batch_size : int):
        feed_dict = self.get_feed_dict(train_data, batch_size, one_hot = False)
        self.session.run(self.training, feed_dict = feed_dict)
        return feed_dict

    def test(self, database, batch_size = 1000):
        if self.session is None:
            session_config = tf.ConfigProto()
            session_config.gpu_options.allow_growth = True

            self.session = tf.Session(config = session_config)
            self.init_model()

        test_data = database.test_dataset
        feed_dict = self.get_feed_dict(test_data, batch_size, one_hot = False)

        print('#-----#')
        print("Evaluation :", (1 - self.session.run(self.evaluation, feed_dict = feed_dict)) * 100, '%')
        print('#-----#')
