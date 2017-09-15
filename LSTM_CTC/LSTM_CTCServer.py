import tensorflow as tf
import numpy as np
import socket
from python_speech_features import mfcc

from LSTM_CTCModel import LSTM_CTCModel
from LSTM_CTCConfig import LSTM_CTCModelConfig, LSTM_CTCServerConfig

class LSTM_CTCServer(object):
    def __init__(self, model : LSTM_CTCModel, server_config : LSTM_CTCServerConfig):
        self.model = model
        self.config = server_config

        self.graph = tf.Graph()
        self.graph.as_default()

        self.mfcc_features = np.zeros([self.model.config.input_max_time, self.model.config.input_frame_size])

    def start_server(self):
        self.init_model()
        self.init_server()
        self.mainloop()

    def init_server(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((self.config.listen_addr, self.config.listen_addr))

    def init_model(self):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(self.model.config.checkpoints_path)

        if not checkpoint or not checkpoint.model_checkpoint_path:
            print("No checkpoint of model found in", self.model.config.checkpoints_path)
            print("Aborting ...")
            input()
            exit()

        self.session = tf.Session(config = LSTM_CTCServer.session_config)
        print("Loading model...")
        self.saver.restore(self.session, checkpoint.model_checkpoint_path)
        print("Model loaded")

    def mainloop(self):
        while True:
            raw_data, _ = self.get_data()
            #### SOUND PROCESS
            features_count = self.model.config.input_frame_size
            mfcc_frames = mfcc(raw_data, self.config.sound_sample_rate, nfilt = features_count, numcep = features_count)
            mfcc_frames = np.array(mfcc_frames)
            mfcc_frames_count = mfcc_frames.shape[0]
            self.mfcc_features[:mfcc_frames_count] = mfcc_frames
            self.mfcc_features = np.roll(-self.mfcc_features, mfcc_frames_count, axis = 0)

            #### INFERENCE
            feed_dict = \
            {
                self.model.input_placeholder : self.mfcc_features,
                self.model.input_lengths_placeholder : self.model.config.input_max_time,
                self.model.output_lengths_placeholder : self.model.config.output_max_time
            }

            inference = self.session.run(self.model.decoded_inference, feed_dict = feed_dict)
            inference = inference[0].values

            #### TEXT
            result_words = ""
            for idx in inference:
                word = timit_database.id_to_phoneme_dictionary[idx]
                if word != "<EOS>":
                    result_words += word + ' '

            #### SEND BACK
            self.send_data(result_words.encode())

    @property
    def session_config():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        return config

    def recv_ack(self, size):
        data, ip = self.socket.recvfrom(size)
        ack = "ACK".encode()
        self.send_data(ack)
        return data, ip

    def get_data(self):
        recv_size = 32768
        data, _ = self.recv_ack(recv_size)
        length = int.from_bytes(data[:4], 'little')

        recv_count = math.ceil(length / recv_size)

        for i in range(1, recv_count):
          tmp, _ = self.recv_ack(recv_size)
          data += tmp
        return data, length

    def send_data(self, data : bytes):
        self.socket.sendto(data, (self.config.send_addr, self.config.send_port))

def main():
    model_config = LSTM_CTCModelConfig()
    model = LSTM_CTCModel(model_config)

    server_config = LSTM_CTCServerConfig()
    server = LSTM_CTCServer(model, server_config)
    server.start_server()

if __name__ == '__main__':
  main()