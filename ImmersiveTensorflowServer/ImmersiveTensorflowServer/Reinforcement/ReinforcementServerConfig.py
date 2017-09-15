from configparser import SafeConfigParser

class ReinforcementServerConfig():
  def __init__(self, config_path = "ServerConfig.ini"):
    config_parser = SafeConfigParser()
    config_parser.read(config_path)

    ### Server
    server_config = config_parser["Server"]
    self.ip                                 = server_config["ip"]
    self.send_port                          = int(server_config["send_port"])
    self.listen_port                        = int(server_config["listen_port"])

    ### Session
    session_config = config_parser["Session"]
    self.gpu_allow_growth                   = session_config["gpu_allow_growth"] == "True"
    self.checkpoint_path                    = session_config["checkpoint_path"]
    self.save_every_x_steps                 = int(session_config["save_every_x_steps"])

    ### Actions
    actions_config = config_parser["Actions"]
    self.initial_random_action_probability  = float(actions_config["initial_random_action_probability"])
    self.final_random_action_probability    = float(actions_config["final_random_action_probability"])

    ### Memory
    memory_config = config_parser["Memory"]
    self.max_frame_count                    = int(memory_config["max_frame_count"])
    self.max_reward_count                   = int(memory_config["max_reward_count"])
    self.mini_batch_size                    = int(memory_config["mini_batch_size"])

    ### Training
    training_config = config_parser["Training"]
    self.future_reward_factor               = float(training_config["future_reward_factor"])
    self.observation_count_before_training  = int(training_config["observation_count_before_training"])
    self.exploration_steps_count            = int(training_config["exploration_steps_count"])