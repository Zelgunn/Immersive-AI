import tensorflow as tf
import numpy as np
import os

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb

NB_CLASSES = 2
PAD_VALUE = 0
MAXLEN = 100

def preprocess_dataset(dataset : tuple, maxlen : int, pad_value : int, nb_classes : int):
    inputs, labels = dataset
    inputs = pad_sequences(
        sequences = inputs,
        maxlen = maxlen,
        value = pad_value)
    labels = to_categorical(
        y = labels,
        nb_classes = nb_classes)
    return inputs, labels

def preprocess_database(database_path : str, maxlen : int, valid_portion : float, n_words):
    train_dataset, valid_dataset, test_dataset = imdb.load_data(
        path = os.path.join(database_path, "imdb.pkl"),
        n_words = n_words,
        valid_portion = valid_portion,
        maxlen = maxlen)

    train_dataset = preprocess_dataset(train_dataset, maxlen, PAD_VALUE, NB_CLASSES)
    valid_dataset = preprocess_dataset(valid_dataset, maxlen, PAD_VALUE, NB_CLASSES)
    test_dataset = preprocess_dataset(test_dataset, maxlen, PAD_VALUE, NB_CLASSES)

    convert_dataset_to_tfrecords(train_dataset, os.path.join(database_path, "train"))
    convert_dataset_to_tfrecords(valid_dataset, os.path.join(database_path, "valid"))
    convert_dataset_to_tfrecords(test_dataset, os.path.join(database_path, "test"))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_dataset_to_tfrecords(dataset, dataset_path : str):
    inputs, labels = dataset
    labels = np.argmax(labels, axis = 1)
    samples_count = len(inputs)
    if len(labels) != samples_count:
        raise ValueError("Inputs samples size (%d) does not match Labels samples size (%d)" % (samples_count, len(labels)))

    if not dataset_path.endswith(".tfrecords"):
        dataset_path += ".tfrecords"
    print("Writing dataset to %s)" % (dataset_path))
    writer = tf.python_io.TFRecordWriter(dataset_path)
    for i in range(samples_count):
        raw_input = inputs[i].tostring()
        example = tf.train.Example(
            features = tf.train.Features(
                feature = {
                    "input_raw" : _bytes_feature(raw_input),
                    "label" : _int64_feature(labels[i])
                    }
                )
            )
        writer.write(example.SerializeToString())
    writer.close()
    print("Wrote %d samples to %s" % (samples_count, dataset_path))

def read_and_decode_dataset(filename_queue):
        reader = tf.TFRecordReader()
        keys, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized = serialized_example,
            features = \
                {
                    "input_raw" : tf.FixedLenFeature([], tf.string),
                    "label" : tf.FixedLenFeature([], tf.int64)
                }
            )

        inputs = tf.decode_raw(features["input_raw"], tf.int32)
        inputs.set_shape([MAXLEN])

        labels = tf.cast(features["label"], tf.int32)

        return inputs, labels

def input_fn(dataset_path, batch_size, capacity = 1000):
    filename_queue = tf.train.string_input_producer(
        [dataset_path],
        num_epochs = None)

    inputs, labels = read_and_decode_dataset(filename_queue)
    inputs, labels = tf.train.shuffle_batch(
        tensors = [inputs, labels],
        batch_size = batch_size,
        num_threads = 2,
        capacity = capacity + 2 * batch_size,
        min_after_dequeue = capacity)

    feature_cols = {"inputs" : inputs}

    return feature_cols, labels

def main():
    preprocess_database(r"C:\tmp\imdb", MAXLEN, 0.1, 10000)

if __name__ == "__main__":
    main()