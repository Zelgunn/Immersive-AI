from configparser import SafeConfigParser

class DQNConfig(object):
  def __init__(self, config_path = "ModelConfig.ini"):
    config_parser = SafeConfigParser()
    config_parser.read(config_path)

    input_config = config_parser["Input"]
    self.input_width                        = int(input_config["width"])
    self.input_height                       = int(input_config["height"])
    self.input_depth                        = int(input_config["depth"])

    actions_config = config_parser["Actions"]
    self.actions_count                      = int(actions_config["action_count"])

    training_config = config_parser["Training"]
    self.learning_rate                      = float(training_config["learning_rate"])