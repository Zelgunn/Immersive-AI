from configparser import SafeConfigParser

class CLDNNConfig():
  def __init__(self, max_timesteps, mfcc_features_count, 
               max_output_length, dictionary_size, 
               config_path = "CLDNN.ini"):
    self.max_timesteps = max_timesteps
    self.mfcc_features_count = mfcc_features_count
    self.max_output_length = max_output_length
    self.dictionary_size = dictionary_size

    config_parser = SafeConfigParser()
    config_parser.read(config_path)

    training_config = config_parser["Training"]
    self.learning_rate = float(training_config["learning_rate"])

    cnn_config = config_parser["CNN"]
    self.conv_kernel_size = int(cnn_config["conv_kernel_size"])
    self.conv_features_count = int(cnn_config["conv_features_count"])

    maxpooling_config = config_parser["MaxPooling"]
    self.max_pooling_size = int(maxpooling_config["max_pooling_size"])

    reduction_config = config_parser["Reduction"]
    self.dimension_reduction_output_size = int(reduction_config["dimension_reduction_output_size"])
    self.time_reduction_factor = int(reduction_config["time_reduction_factor"])

    lstm_config = config_parser["LSTM"]
    self.lstm1_hidden_units_count = int(lstm_config["lstm1_hidden_units_count"])
    self.lstm2_hidden_units_count = int(lstm_config["lstm2_hidden_units_count"])

    dnn_config = config_parser["DNN"]
    self.fully_connected1_size = int(dnn_config["fully_connected1_size"])