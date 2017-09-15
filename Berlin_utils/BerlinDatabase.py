import os
import numpy as np

from Berlin_utils.BerlinSample import BerlinSample

"""
Emotions are labeled in the 6th position (index 5) of the .wav file name, as follow :

W - Anger
L - Boredom
E - digust
A - Fear/Anxiety
F - Happiness
T - Sadness
N - Neutral
"""

class BerlinDatabase(object):
    def __init__(self, database_path : str, mfcc_features_count = 40):
        self.database_path = database_path
        if not os.path.exists(self.database_path):
            raise FileNotFoundError("Database directory does not exists")
        self.mfcc_features_count = mfcc_features_count

        self.samples_list_path = os.path.join(database_path, "samples_list.txt")
        self.mfcc_lengths_path = os.path.join(database_path, "mfcc_lengths_batch.npy")
        self.mfcc_features_path = os.path.join(database_path, "mfcc_features_batch.npy")
        self.emotion_tokens_path = os.path.join(database_path, "emotion_tokens_batch.npy")
        self.load_samples_list()

        self.mfcc_features_lengths = None
        self.mfcc_features = None
        self.emotion_tokens = None

        self.batch_index = 0

    def load_samples_list(self):
        # If the list file doesn't exist, we create it
        if not os.path.exists(self.samples_list_path):
            self.samples_names = self.build_samples_list()
            return self.samples_names
        # Then we load it inside a python list
        samples_names = []
        with open(self.samples_list_path, 'r') as samples_list_file:
            lines = samples_list_file.readlines()
            for line in lines:
                samples_names.append(line.replace('\n', ''))
        # Keeping the list inside the object
        self.samples_names = samples_names
        if len(self.samples_names) == 0:
            raise FileNotFoundError("No samples were found in samples list (%s)" % (self.samples_list_path))
        return self.samples_names

    def load_samples(self,
                    load_mfcc = False, load_wav = False):
        samples = []
        for sample_name in self.samples_names:
            sample = BerlinSample(sample_name, self.database_path)
            sample.load(
                      load_mfcc = load_mfcc,
                      load_wav = load_wav)
            if sample.emotion_token != None:
                samples.append(sample)
        return samples

    def build_samples_list(self):
        samples_names = []
        base_path_len = len(self.database_path)
        # Walking inside dataset
        for root, dirs, files in os.walk(self.database_path):
            if len(files) == 0:
                continue
            sample_root = root[base_path_len:]
            for file in files:
                if not file.endswith(".wav"):
                    continue
                samples_names.append(sample_root + '\\' + file[:-4])
        if len(samples_names) == 0:
            raise FileNotFoundError("No samples were found in database path (%s)" % (self.database_path))
        # Building of final string to write
        samples_names_string = ""
        for sample_name in samples_names:
            samples_names_string += sample_name + '\n'
        # Writing to file
        with open(self.samples_list_path, 'w') as samples_list_file:
            samples_list_file.write(samples_names_string)
        return samples_names

    def build_samples_mfcc_features(self, winstep = 0.01, winlen = 0.025):
        for sample_name in self.samples_names:
            sample = BerlinSample(sample_name, self.database_path)
            sample.preprocess_wav_file_to_mfcc(self.mfcc_features_count, winstep, winlen)

    def load_mfcc_lengths(self):
        self.mfcc_features_lengths = np.load(self.mfcc_lengths_path)

    def build_batch(self):
        print("Building Berlin EMO-DB batch : MFCCs Lengths")
        self.build_mfcc_lengths_batch()
        print("Building Berlin EMO-DB batch : MFCCs")
        self.build_mfcc_batch()
        print("Building Berlin EMO-DB batch : Emotion tokens")
        self.build_emotion_tokens_batch()

    def build_mfcc_lengths_batch(self):
        lengths = []
        samples = self.load_samples(load_mfcc = True)
        for sample in samples:
            lengths.append(sample.mfcc_length)
        lengths = sorted(lengths)
        np.save(self.mfcc_lengths_path, lengths)

    def build_mfcc_batch(self):
        # Getting max length across complete database (for padding)
        max_mfcc_features_length = self.get_max_mfcc_features_length()
        samples = self.load_samples(load_mfcc = True)
        lengths = []
        for sample in samples:
            lengths.append(sample.mfcc_length)
        # Sorting
        lengths, samples = zip(*sorted(zip(lengths, samples), key=lambda pair: pair[0]))
        # Padding
        batch_size = len(samples)
        mfcc = []
        for i in range(batch_size):
          mfcc.append(samples[i].mfcc.astype(np.float32))
        mfcc = np.asarray(mfcc)
        #mfcc_batch = np.zeros([batch_size, max_mfcc_features_length, self.mfcc_features_count], dtype = np.float32)
        #print("build_mfcc batch dimensions: ", [batch_size, max_mfcc_features_length, self.mfcc_features_count])
        #for i in range(batch_size):
        #    for j in range(lengths[i]):
        #        for k in range(self.mfcc_features_count):
        #            mfcc_batch[i, j, k] = samples[i].mfcc[j, k]
        # Saving
        np.save(self.mfcc_features_path, mfcc)

    def get_max_mfcc_features_length(self):
        if self.mfcc_features_lengths is None:
            self.load_mfcc_lengths()
        return np.max(self.mfcc_features_lengths)

    def build_emotion_tokens_batch(self):
        emotion_tokens = []
        samples = self.load_samples()
        for sample in samples:
            emotion_tokens.append(sample.emotion_token)
        np.save(self.emotion_tokens_path, emotion_tokens)

    def load_batch(self, padding = None):
        self.mfcc_features = np.load(self.mfcc_features_path)
        self.mfcc_features_lengths = np.load(self.mfcc_lengths_path)
        self.emotion_tokens = np.load(self.emotion_tokens_path)
        if padding is not None:
            samples_count = len(self.mfcc_features)
            tmp = np.zeros(shape = [samples_count, padding, self.mfcc_features_count], dtype = np.float32)
            for i in range(samples_count):
                length = min(self.mfcc_features_lengths[i], padding)
                for j in range(length):
                    for k in range(self.mfcc_features_count):
                        tmp[i, j, k] = self.mfcc_features[i][j, k]
            self.mfcc_features = tmp
        return self.mfcc_features, self.mfcc_features_lengths, self.emotion_tokens

    def shuffle_batch(self):
        order = np.arange(len(self.mfcc_features))
        np.random.shuffle(order)
        self.mfcc_features = self.mfcc_features[order]
        self.mfcc_features_lengths = self.mfcc_features_lengths[order]
        self.emotion_tokens = self.emotion_tokens[order]

    def next_batch(self, batch_size : int):
        if self.mfcc_features is None:
            self.load_batch()
            self.shuffle_batch()

        start = self.batch_index
        end = min(self.batch_index + batch_size, self.samples_count)
        mfcc_features_batch         = self.mfcc_features[start : end]
        mfcc_features_lengths_batch = self.mfcc_features_lengths[start : end]
        emotion_tokens_batch        = self.emotion_tokens[start : end]

        self.batch_index += batch_size
        if self.batch_index >= self.samples_count:
            self.batch_index = 0
            self.shuffle_batch()
        return (mfcc_features_batch, mfcc_features_lengths_batch, emotion_tokens_batch)

    @property
    def samples_count(self):
        if self.mfcc_features_lengths is None:
            self.load_mfcc_lengths()
        return len(self.mfcc_features_lengths)

def build_timit_database(database_path, mfcc_features_count = 40, winstep = 0.01, winlen = 0.025):
    print("Build Berlin EMO-DB database : initializing...")
    data = BerlinDatabase(database_path, mfcc_features_count)
    print("Build Berlin EMO-DB database : starting ...")
    print("Build Berlin EMO-DB database : building MFCCs ...")
    data.build_samples_mfcc_features(winstep = winstep, winlen = winlen)
    print("Build Berlin EMO-DB database : building datasets batch ...")
    data.build_batch()
    print("Build Berlin EMO-DB database : finished !")

if __name__ == "__main__":
    database_path = r"C:\tmp\Berlin"
    build_timit_database(database_path)