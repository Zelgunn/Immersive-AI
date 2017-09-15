import tensorflow as tf
import numpy as np
import os

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators.model_fn import ModelFnOps

from Berlin_utils.BerlinDatabase import BerlinDatabase
from Berlin_utils.BerlinTFRecords import BerlinTFRecords
from Berlin_utils.BerlinTFRecords import input_fn as berlin_input_fn

FEATURES = ["mfcc", "mfcc_lengths"]

def exp_decay(learning_rate, global_step):
            return tf.train.natural_exp_decay(
                learning_rate=learning_rate, global_step=global_step,
                decay_steps=100, decay_rate=0.1)

def cldnn_model_fn(features, labels, mode, params):
    inputs = tf.cast(features["mfcc"], tf.float32)
    inputs_lengths = features["mfcc_lengths"]

    #with tf.name_scope("Convolution"):
    #    input_layer = tf.reshape(inputs, shape = [-1, params["max_timesteps"], params["mfcc_features_count"], 1])

    #    conv_layer1 = tf.layers.conv2d(
    #        inputs = input_layer,
    #        filters = params["conv1_features_count"],
    #        kernel_size = [params["conv1_kernel_size1"], params["conv1_kernel_size2"]],
    #        padding = "same",
    #        activation = tf.nn.relu)

    #    pool_layer = tf.layers.max_pooling2d(
    #        inputs = conv_layer1,
    #        pool_size = [1, params["max_pooling_size"]],
    #        strides = [1, params["max_pooling_size"]])

    #    conv_layer2 = tf.layers.conv2d(
    #        inputs = pool_layer,
    #        filters = params["conv2_features_count"],
    #        kernel_size = [params["conv2_kernel_size1"], params["conv2_kernel_size2"]],
    #        padding = "same",
    #        activation = tf.nn.relu)

    #    pool_output_freq_size = int(params["mfcc_features_count"] / params["max_pooling_size"])

    #    conv_layer2_flat = tf.reshape(
    #        conv_layer2, 
    #        shape = [-1, params["max_timesteps"] * pool_output_freq_size * params["conv2_features_count"]]
    #        )

    #with tf.name_scope("Dimension_reduction"):
    #    dim_reduction_layer = tf.layers.dense(
    #        inputs = conv_layer2_flat,
    #        units = params["max_timesteps"] *params["dimension_reduction_size"],
    #        activation = tf.nn.relu)

        #dim_reduction_layer = tf.reshape(
        #    tensor = dim_reduction_layer,
        #    shape = [-1, params["max_timesteps"], params["dimension_reduction_size"]])

    #with tf.name_scope("Concatenation"):
    #    inputs = tf.reshape(
    #        tensor = inputs,
    #        shape = [-1, params["max_timesteps"], params["mfcc_features_count"]])
    #    concatenation_layer = tf.concat(
    #        values = [inputs, dim_reduction_layer],
    #        axis = 2)

    with tf.name_scope("Recurrent"):
        inputs_reshape = tf.reshape(
            tensor = inputs,
            shape = [-1, params["max_timesteps"], params["mfcc_features_count"]])

        inputs_reshape = tf.transpose(
            inputs_reshape,
            [0, 2, 1])

        lstm_cell = tf.contrib.rnn.LSTMCell(
            num_units = params["lstm_units"],
            num_proj = params["lstm_projection"],
            activation = tf.tanh)

        lstm_cell = tf.contrib.rnn.MultiRNNCell(
            cells = [lstm_cell] * params["lstm_cell_count"])

        lstm_output, lstm_state = tf.nn.dynamic_rnn(
            cell = lstm_cell,
            inputs = inputs_reshape,
            sequence_length = inputs_lengths,
            dtype = tf.float32)

        lstm_output = tf.reshape(
            tensor = lstm_output,
            #shape = [-1, params["max_timesteps"] * params["lstm_projection"]])
            shape = [-1, params["lstm_projection"] * params["mfcc_features_count"]])

    with tf.name_scope("Fully_connected"):
        dense_layer = lstm_output

        if params["fully_connected_sizes"] is not None:
            for size in params["fully_connected_sizes"]:
                dense_layer = tf.layers.dense(
                    inputs = dense_layer,
                    units = size,
                    activation = tf.nn.relu,
                    use_bias = True,
                    kernel_initializer = tf.truncated_normal_initializer(stddev = 0.2, mean = 0),
                    bias_initializer = tf.constant_initializer(value = 0.01))
            

    dropout = tf.layers.dropout(
        inputs = dense_layer,
        rate = 0.4,
        training = mode==learn.ModeKeys.TRAIN)

    with tf.name_scope("Logits"):
        logits_flat = tf.layers.dense(
            inputs = dropout,
            units = params["labels_class_count"],
            use_bias = True,
            kernel_initializer = tf.truncated_normal_initializer(stddev = 0.2, mean = 0),
            bias_initializer = tf.constant_initializer(value = 0))

        logits = tf.reshape(
            logits_flat,
            shape = [-1, params["labels_class_count"]])

    predictions = \
    {
        "classes" : tf.argmax(input = logits, axis = 1),
        "probabilities" : tf.nn.softmax(logits, name = "softmax_tensor"),
    }
    
    loss = None
    train_op = None
    eval_metric_ops = None

    if mode != learn.ModeKeys.INFER:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = tf.one_hot(labels, params["labels_class_count"]))
        loss = tf.reduce_mean(cross_entropy)

    if mode == learn.ModeKeys.TRAIN:
        
        train_op = tf.contrib.layers.optimize_loss(
            loss = loss,
            global_step = tf.contrib.framework.get_global_step(),
            learning_rate = params["learning_rate"],
            optimizer = params["optimizer"])

    if mode == learn.ModeKeys.EVAL:
        batch_size = int(logits.get_shape()[0])
        eval_metric_ops = {
            "accuracy" : tf.metrics.accuracy(
            labels = labels,
            predictions = tf.argmax(input = logits, axis = 1))
        }

    return ModelFnOps(mode = mode, predictions = predictions, loss = loss, train_op = train_op, eval_metric_ops = eval_metric_ops)

#BERLIN_DATABASE = BerlinDatabase(r"C:\tmp\Berlin")
#BERLIN_TRAIN_DATASET = BERLIN_DATABASE.next_batch(435)
#BERLIN_TEST_DATASET = BERLIN_DATABASE.next_batch(100)
BERLIN_DATASET_PATH = r"C:\tmp\Berlin"
BERLIN_TRAIN_DATASET = os.path.join(BERLIN_DATASET_PATH, "train.tfrecords")
BERLIN_VALID_DATASET = os.path.join(BERLIN_DATASET_PATH, "valid.tfrecords")

def berlin_input_fn_train_dataset():
    return berlin_input_fn(BERLIN_TRAIN_DATASET, 32)

def berlin_input_fn_valid_dataset():
    return berlin_input_fn(BERLIN_VALID_DATASET, 50)

def main(unused_argv):
    parameters = \
    {
        "max_timesteps" : 181,
        "mfcc_features_count" : 20,

        "labels_class_count" : 5,

        "conv1_features_count" : 64,     # 256
        "conv1_kernel_size1" : 9,         # 9
        "conv1_kernel_size2" : 9,         # 9

        "max_pooling_size" : 3,           # 3

        "conv2_features_count" : 64,     # 256
        "conv2_kernel_size1" : 4,         # 4
        "conv2_kernel_size2" : 3,         # 3

        "dimension_reduction_size" : 64, # 256
    
        "lstm_units" : 128,               # 832
        "lstm_projection" : 128,          # 512
        "lstm_cell_count" : 4,            # 2

        "fully_connected_sizes" : [128, 64],   # 1024

        "learning_rate" : 1e-3,
        "optimizer" : "Adam" #tf.train.MomentumOptimizer(learning_rate = 1e-3, momentum = 0.5)
    }

    run_config = learn.RunConfig(
        gpu_memory_fraction = 0.8,
        save_checkpoints_secs = 60)

    cldnn_classifier = learn.Estimator(
        model_fn = cldnn_model_fn,
        model_dir = r"C:\tmp\berlin_lstm\25_MFCC",
        params = parameters,
        config = run_config)

    #tensors_to_log = {"probabilities": "softmax_tensor"}
    #logging_hook = tf.train.LoggingTensorHook(
    #    tensors = tensors_to_log,
    #    every_n_iter = 50)

    validation_monitor = learn.monitors.ValidationMonitor(
        input_fn = lambda:berlin_input_fn(BERLIN_VALID_DATASET, 100),
        eval_steps = 1,
        every_n_steps = 50)

    tf.logging.set_verbosity(tf.logging.INFO)

    cldnn_classifier.fit(
        input_fn = berlin_input_fn_train_dataset,
        steps = 15000,
        monitors=[validation_monitor])

    eval_results = cldnn_classifier.evaluate(
        input_fn = lambda:berlin_input_fn(BERLIN_VALID_DATASET, 100),
        steps = 1)
    print("############################")
    print("############################")
    print("#########      #############")
    print("######### %s #############" % (eval_results["accuracy"]))
    print("#########      #############")
    print("############################")
    print("############################")

if __name__ == "__main__":
    tf.app.run()
