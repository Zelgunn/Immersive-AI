import tensorflow as tf
import numpy as np
import os

from Berlin_utils.BerlinDatabase import BerlinDatabase

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

class BerlinTFRecords(object):
    def __init__(self, database : BerlinDatabase):
        self.database = database
        self.database_path = database.database_path
        if self.database is None:
            raise Exception("No database was provided")

    def convert_to_tfrecords(self, validation_size = 50):
        self.database.load_batch()
        if self.database.samples_count <= validation_size:
            raise Exception("Validation size (%d) is bigger than database size (%d)" % (validation_size, self.database.samples_count))
        mfcc = self.database.mfcc_features.astype(np.float32)
        mfcc_lengths = self.database.mfcc_features_lengths
        labels = self.database.emotion_tokens

        train = (
            mfcc[:-validation_size],
            mfcc_lengths[:-validation_size],
            labels[:-validation_size])

        valid = (
            mfcc[-validation_size:],
            mfcc_lengths[-validation_size:],
            labels[-validation_size:])

        self.convert_dataset_to_tfrecords(train, "train")
        self.convert_dataset_to_tfrecords(valid, "valid")

    def convert_dataset_to_tfrecords(self, dataset, dataset_name):
        mfcc = dataset[0]
        mfcc_lengths = dataset[1]
        labels = dataset[2]
        samples_count = len(mfcc)
        if len(mfcc_lengths) != samples_count:
            raise ValueError("MFCCs samples size (%d) does not match MFCCs lengths samples size (%d)" % (samples_count, len(mfcc_lengths)))
        if len(labels) != samples_count:
            raise ValueError("MFCCs samples size (%d) does not match labels samples size (%d)" % (samples_count, len(labels)))

        filename = os.path.join(self.database_path, dataset_name + ".tfrecords")
        print("Converting dataset : %s (writing to %s)" % (dataset_name, filename))
        writer = tf.python_io.TFRecordWriter(filename)
        for i in range(samples_count):
            mfcc_raw = mfcc[i].tostring()
            example = tf.train.Example(
                features = tf.train.Features(
                    feature = {
                        "mfcc_raw" : _bytes_feature(mfcc_raw),
                        "mfcc_length" : _int64_feature(mfcc_lengths[i]),
                        "label" : _int64_feature(labels[i])
                        }
                    )
                )
            writer.write(example.SerializeToString())
        writer.close()

    def read_and_decode_dataset(filename_queue):
        reader = tf.TFRecordReader()
        keys, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized = serialized_example,
            features = \
                {
                    "mfcc_raw" : tf.FixedLenFeature([], tf.string),
                    "mfcc_length" : tf.FixedLenFeature([], tf.int64),
                    "label" : tf.FixedLenFeature([], tf.int64)
                }
            )

        mfcc = tf.decode_raw(features["mfcc_raw"], tf.float32)
        mfcc.set_shape([20 * 143])

        mfcc_length = tf.cast(features["mfcc_length"], tf.int32)

        label = tf.cast(features["label"], tf.int32)

        return mfcc, mfcc_length, label

def input_fn(dataset_path, batch_size, capacity = 1000):
    FEATURES = ["mfcc", "mfcc_lengths"]
    feat_count = len(FEATURES)

    filename_queue = tf.train.string_input_producer(
        [dataset_path],
        num_epochs = None)

    mfcc, lengths, labels = BerlinTFRecords.read_and_decode_dataset(filename_queue)
    dataset = tf.train.shuffle_batch(
        tensors = [mfcc, lengths, labels],
        batch_size = batch_size,
        num_threads = 3,
        capacity = capacity + 3 * batch_size,
        min_after_dequeue = capacity)

    feature_cols = {FEATURES[k] : dataset[k]
                    for k in range(feat_count)}

    labels = dataset[feat_count]

    return feature_cols, labels

if __name__ == "__main__":
    database_path = r"C:\tmp\Berlin"
    database = BerlinDatabase(database_path, 20)
    converter = BerlinTFRecords(database)
    converter.convert_to_tfrecords(50)