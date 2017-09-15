from configparser import SafeConfigParser

class SupervisedServerConfig():
  def __init__(self, config_path = "ServerConfig.ini"):
    config_parser = SafeConfigParser()
    config_parser.read(config_path)

    ### Server
    server_config = config_parser["Server"]

    self.ip = server_config["ip"]
    self.send_port = int(server_config["send_port"])
    self.listen_port = int(server_config["listen_port"])

    ### Session
    session_config = config_parser["Session"]

    self.gpu_allow_growth = session_config["gpu_allow_growth"] == "True"