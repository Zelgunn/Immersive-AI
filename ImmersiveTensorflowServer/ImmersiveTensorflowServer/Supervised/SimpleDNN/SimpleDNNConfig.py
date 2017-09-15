from configparser import SafeConfigParser

class SimpleDNNConfig(object):
  def __init__(self, config_path = "SimpleDNNConfig.ini"):
    config_parser = SafeConfigParser()
    config_parser.read(config_path)

    layers_config = config_parser["Layers"]
    self.layer_count = int(layers_config["layer_count"])
    self.layers_size = int(layers_config["layers_size"])

    training_config = config_parser["Training"]
    self.learning_rate = float(training_config["learning_rate"])

#      [Input]
#width = 480
#height = 270
#depth = 3

#[Actions]
#count = 5

#[Training]
#learning_rate = 0.5