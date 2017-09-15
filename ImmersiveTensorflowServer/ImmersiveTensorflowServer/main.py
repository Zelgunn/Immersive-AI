def start_supervised_server(server_config_path = "ServerConfig.ini", model_config_path = "Supervised/SimpleDNN/ModelConfig.ini"):
  from Supervised.SimpleDNN.SimpleDNNModel import SimpleDNNModel, SimpleDNNConfig
  from Supervised.SupervisedServer import SupervisedServer, SupervisedServerConfig

  model_config = SimpleDNNConfig(model_config_path)
  model = SimpleDNNModel(model_config, 480*270*3, 4)

  server_config = SupervisedServerConfig(server_config_path)
  immersiveTensorflowServer = SupervisedServer(server_config, model)
  immersiveTensorflowServer.run_model_training()

def start_reinforcement_server(server_config_path = "Reinforcement/ServerConfig.ini", model_config_path = "Reinforcement/DQN/ModelConfig.ini"):
  from Reinforcement.DQN.DQNModel import DQNModel, DQNConfig
  from Reinforcement.ReinforcementServer import ReinforcementServer, ReinforcementServerConfig

  model_config = DQNConfig(model_config_path)
  model = DQNModel(model_config)

  server_config = ReinforcementServerConfig(server_config_path)
  server = ReinforcementServer(server_config, model)
  server.run_model_training()

def main():
  #start_supervised_server()
  start_reinforcement_server()

if __name__ == '__main__':
  main()