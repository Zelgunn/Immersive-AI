import tensorflow as tf
import numpy as np
import os

import time

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators.model_fn import ModelFnOps
import Tone_utils

def embed_lestm_model_fn(features, labels, mode, params):
    inputs = features["inputs"]

    with tf.name_scope("Embedding"):
        embeddings = tf.get_variable(
            name = "embeddings",
            shape = [params["vocabulary_size"], params["embedding_size"]],
            initializer = tf.random_uniform_initializer(
                minval = -1,
                maxval = 1
                )
            )

        embedding_lookup = tf.nn.embedding_lookup(
            params = embeddings,
            ids = inputs)

    with tf.name_scope("LSTM"):
        if params["use_gru"]:
            cell = tf.contrib.rnn.GRUCell(num_units = params["lstm_units"])
        else:
            cell = tf.contrib.rnn.LSTMCell(
                num_units = params["lstm_units"],
                num_proj = params["lstm_projection"],
                activation = tf.tanh)

        cell = tf.contrib.rnn.MultiRNNCell(
            cells = [cell] * params["lstm_cell_count"])

        lstm_output, lstm_state = tf.nn.dynamic_rnn(
            cell = cell,
            inputs = embedding_lookup,
            dtype = tf.float32)

        lstm_output = tf.reshape(
            tensor = lstm_output,
            shape = [-1, params["lstm_projection"] * params["max_inputs_len"]])

    with tf.name_scope("Fully_connected"):
        fc_layer = lstm_output
        for fc_layer_size in params["fc_layers_size"]:
            fc_layer = tf.layers.dense(
                inputs = fc_layer,
                units = fc_layer_size,
                activation = tf.nn.relu,
                use_bias = True,
                kernel_initializer = tf.truncated_normal_initializer(
                    mean = 0,
                    stddev = 0.1),
                bias_initializer = tf.constant_initializer(
                    value = 0
                )
            )

    dropout = tf.layers.dropout(
        inputs = fc_layer,
        rate = params["dropout"],
        training = mode == learn.ModeKeys.TRAIN)

    logits = tf.layers.dense(
        inputs = dropout,
        units = params["nb_classes"],
        activation = tf.identity,
        use_bias = True,
        kernel_initializer = tf.truncated_normal_initializer(
            mean = 0,
            stddev = 0.1),
            bias_initializer = tf.constant_initializer(
                value = 0
            )
        )

    predictions = \
    {
        "classes" : tf.argmax(input = logits, axis = 1),
        "probabilities" : tf.nn.softmax(logits = logits)
    }

    loss = None
    train_op = None
    eval_metric_ops = None

    if mode != learn.ModeKeys.INFER:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = tf.one_hot(labels, params["nb_classes"]))
        loss = tf.reduce_mean(cross_entropy, name = "loss")

    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss = loss,
            global_step = tf.contrib.framework.get_global_step(),
            learning_rate = params["learning_rate"],
            optimizer = params["optimizer"]
        )

    if mode == learn.ModeKeys.EVAL:
        eval_metric_ops = {
            "accuracy" : tf.metrics.accuracy(
            labels = labels,
            predictions = predictions["classes"])
        }

    return ModelFnOps(mode = mode, predictions = predictions, loss = loss, train_op = train_op, eval_metric_ops = eval_metric_ops)



def run_version(i, dropout, use_gru, fc_layers_sizes, cell_units, cell_proj, cell_count):
    parameters = \
    {
        "vocabulary_size" : 10000,
        "max_inputs_len" : Tone_utils.MAXLEN,
        "embedding_size" : 128,

        "nb_classes" : Tone_utils.NB_CLASSES,
    
        "use_gru" : use_gru,
        "lstm_units" : cell_units,
        "lstm_projection" : cell_proj,
        "lstm_cell_count" : cell_count,

        "fc_layers_size" : fc_layers_sizes,

        "dropout" : dropout,
        "learning_rate" : 1e-4,
        "optimizer" : "Adam"
    }

    database_path = r"C:\tmp\imdb"
    def tone_input_fn_train():
        return Tone_utils.input_fn(os.path.join(database_path, "train.tfrecords"), 32)
    def tone_input_fn_valid():
        return Tone_utils.input_fn(os.path.join(database_path, "valid.tfrecords"), 32)
    def tone_input_fn_test():
        return Tone_utils.input_fn(os.path.join(database_path, "test.tfrecords"), 32)

    run_config = learn.RunConfig(
        gpu_memory_fraction = 0.8,
        save_checkpoints_secs = 60)

    version_path = r"C:\tmp\tone_classifier\version=%d dropout=%.2f fc_layers=%r cells lstm=%d units=%d proj=%d count=%d" % (i, dropout, fc_layers_sizes, use_gru, cell_units, cell_proj, cell_count)
    tone_classifier = learn.Estimator(
        model_fn = embed_lestm_model_fn,
        model_dir = version_path,
        params = parameters,
        config = run_config)

    tf.logging.set_verbosity(tf.logging.INFO)

    #### TRAIN ####
    tone_classifier.fit(
        input_fn = tone_input_fn_train,
        steps = 10000)

    #### VALIDATION ####
    eval_results = tone_classifier.evaluate(
        input_fn = tone_input_fn_valid,
        steps = 100)
    print("Validation set accuracy :", eval_results["accuracy"])

    #### TEST ####
    eval_results = tone_classifier.evaluate(
        input_fn = tone_input_fn_test,
        steps = 100)
    print("Test set accuracy :", eval_results["accuracy"])

def main(unused_argv):
    versions = []
    versions.append([0.8, False,    [64],   128, 128, 2])
    versions.append([0.9, False,    [64],   128, 128, 2])
    versions.append([0.8, True,     [64],   128, 128, 2])
    versions.append([0.9, True,     [64],   128, 128, 2])
    versions.append([0.8, False,    [],     128, 128, 2])
    versions.append([0.9, False,    [],     128, 128, 2])
    versions.append([0.8, True,     [],     128, 128, 2])
    versions.append([0.9, True,     [],     128, 128, 2])

    versions.append([0.8, False,    [64],   128, 128, 1])
    versions.append([0.9, False,    [64],   128, 128, 1])
    versions.append([0.8, True,     [64],   128, 128, 1])
    versions.append([0.9, True,     [64],   128, 128, 1])
    versions.append([0.8, False,    [],     128, 128, 1])
    versions.append([0.9, False,    [],     128, 128, 1])
    versions.append([0.8, True,     [],     128, 128, 1])
    versions.append([0.9, True,     [],     128, 128, 1])

    versions.append([0.8, False,    [128],   256, 256, 1])
    versions.append([0.9, False,    [128],   256, 256, 1])
    versions.append([0.8, True,     [128],   256, 256, 1])
    versions.append([0.9, True,     [128],   256, 256, 1])
    versions.append([0.8, False,    [],     256, 256, 1])
    versions.append([0.9, False,    [],     256, 256, 1])
    versions.append([0.8, True,     [],     256, 256, 1])
    versions.append([0.9, True,     [],     256, 256, 1])

    versions.append([0.8, False,    [128],   256, 256, 2])
    versions.append([0.9, False,    [128],   256, 256, 2])
    versions.append([0.8, True,     [128],   256, 256, 2])
    versions.append([0.9, True,     [128],   256, 256, 2])
    versions.append([0.8, False,    [],     256, 256, 2])
    versions.append([0.9, False,    [],     256, 256, 2])
    versions.append([0.8, True,     [],     256, 256, 2])
    versions.append([0.9, True,     [],     256, 256, 2])

    for i in range(len(versions)):
        dropout, use_gru, fc_layers_sizes, cell_units, cell_proj, cell_count = versions[i]
        run_version(i, dropout, use_gru, fc_layers_sizes, cell_units, cell_proj, cell_count)
        time.sleep(300)

if __name__ == "__main__":
    tf.app.run()