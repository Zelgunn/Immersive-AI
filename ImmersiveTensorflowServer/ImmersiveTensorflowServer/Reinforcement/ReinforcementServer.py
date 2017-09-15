import tensorflow as tf
from tensorflow.python.client import timeline
import numpy as np
import socket
import struct
import math
import random
import gc
import os
from collections import deque

from ImmersiveTensorflowServer import ImmersiveTensorflowServer
from Reinforcement.ReinforcementServerConfig import ReinforcementServerConfig
from Reinforcement.DQN.DQNModel import DQNModel, DQNConfig

from Reinforcement.ModelSkeleton import ModelSkeleton

class Observation():
  pass

class ReinforcementServer(ImmersiveTensorflowServer):
  def __init__(self, config : ReinforcementServerConfig, model : ModelSkeleton):
    super().__init__()

    self.config = config

    self.graph = tf.Graph()
    self.graph.as_default()

    self.model = model

    self.last_state = None
    self.last_action = 0

    self.probability_of_random_action = self.config.initial_random_action_probability

    self.observations = deque()
    self.observations_count = 0

    self.score = 0
    self.rewards = deque()
    self.rewards_count = 0

    self.test = True

  def run_model_training(self):
    session_config = self.get_session_config()
    self.session = tf.Session(config = session_config)

    self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    self.run_metadata = tf.RunMetadata()

    self.init_model()
    self.start_listen_training()

  def init_model(self):
    self.summary = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    print("Initializing variables...")
    self.session.run(init)
    print("Variables initialized !")

    self.summary_writer = tf.summary.FileWriter(self.config.checkpoint_path + "/logs", self.session.graph)

    if not os.path.exists(self.config.checkpoint_path):
      os.mkdir(self.config.checkpoint_path)

    self.saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state(self.config.checkpoint_path)

    if checkpoint and checkpoint.model_checkpoint_path:
      self.saver.restore(self.session, checkpoint.model_checkpoint_path)
      print("Loaded checkpoints", checkpoint.model_checkpoint_path)
    else:
      print("No checkpoint found. Starting from scratch...")

  def get_session_config(self):
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = self.config.gpu_allow_growth
    return session_config

  def start_listen_training(self):
    if self.listening:
      print("Already listening.")
      return

    self.listening = True
    self.sock.bind((self.config.ip, self.config.listen_port))

    #placeholders = self.model.placeholders

    #inference_op = self.model.inference
    #train_op = self.model.training
    #loss_op = self.model.loss
    #eval_op = self.model.evaluation

    #1] Get Frame
    #2] If Reward : Append Reward
    #3] Append Observation (Last_State : last Frame, Last_Action : last Action, Reward, Current_State : Frame, ?)
    #4] If not observing : TRAIN
    #5] Update (State : Frame, Action : new Action)
    #6] If not observing : Reduce random action probability (until a minimum)

    while self.listening:
      recv_type, recv_frame, recv_reward, recv_action = self.get_frame()
      self.observing = (recv_type == 0)
      self.listening = (recv_type != 2)
      if not self.listening:
        break

      self.register_reward(recv_reward)
      self.create_observation(recv_frame, recv_reward, recv_action)

      if not self.observing:
        self.train()

      self.last_state = recv_frame
      if self.observing:
        self.last_action = recv_action
      else:
        self.last_action = self.choose_next_action()

      if not self.observing:
        self.update_random_action_probability()

      self.send_data(struct.pack('i', self.last_action))

      if (self.observations_count % 1000) == 0:
        print(self.observations_count, ')\tScore =', self.score)
      
      if (self.observations_count % 50) == 0:
        gc.collect()

  def get_frame(self):
    recv_data, length = self.get_data()
    i = 4 # length size (int => 4 bytes)

    recv_type = recv_data[i : i + 4]
    recv_type = int.from_bytes(recv_type, byteorder = 'little')
    if recv_type == 2:
      return recv_type, None, None, None
    i += 4 # recv_type size (int => 4 bytes)

    recv_frame = recv_data[i : i + self.model.input_size]
    recv_frame = np.fromstring(recv_frame, dtype=np.uint8).astype(float)
    i += self.model.input_size # pixel count size (1 pixel => 1 byte)

    recv_reward = recv_data[i : i + 4]
    recv_reward = struct.unpack('f', recv_reward)[0]
    i += 4 # recv_reward size (float => 4 bytes)

    recv_action = recv_data[i : i + 4]
    recv_action = int.from_bytes(recv_action, byteorder = 'little')
    #i += 4  # recv_action size (int => 4 bytes)
    return recv_type, recv_frame, recv_reward, recv_action

  def register_reward(self, reward : float):
    if reward != 0:
      if reward >= 1:
        self.score += 1
      if self.rewards_count > self.config.max_reward_count:
        self.rewards.popleft()
      self.rewards.append(reward)

  def create_observation(self, frame, reward, action):
    # First frame ever ?
    if self.last_state is None:
      self.last_state = frame
    # Building observation
    observation = (self.last_state, self.action_to_onehot(action), reward, frame)
    observation = np.array(observation)
    # Appending observation, if full then we pop the oldest one to make room for the new one
    if self.observations_count > self.config.max_frame_count:
      self.observations.popleft()
    self.observations.append(observation)
    self.observations_count += 1

  def choose_next_action(self):
    take_random_action = random.random() <= self.probability_of_random_action
    if take_random_action :
      action = random.randrange(self.model.action_count)
    else :
      last_state_as_array = np.reshape(self.last_state, (1, self.model.input_size))
      feed_dict = \
      {
        self.model.input_placeholder : last_state_as_array
      }
      inference = self.session.run(self.model.inference, feed_dict = feed_dict)[0]
      action = np.argmax(inference)
    return action

  def train(self):
    mini_batch = random.sample(self.observations, self.config.mini_batch_size)
    previous_states = [d[0] for d in mini_batch]
    actions         = [d[1] for d in mini_batch]
    rewards         = [d[2] for d in mini_batch]
    current_states  = [d[3] for d in mini_batch]

    feed_dict = \
    {
      self.model.input_placeholder : current_states
    }
    #print(len(current_states))
    #input()
    expected_rewards_by_agent = []
    if self.test:
      inferenced_rewards = self.session.run(self.model.inference, feed_dict = feed_dict, options = self.run_options, run_metadata = self.run_metadata)
      tl = timeline.Timeline(self.run_metadata.step_stats)
      ctf = tl.generate_chrome_trace_format()
      with open('timeline.json', 'w') as f:
        f.write(ctf)
      self.test = False
    else:
      inferenced_rewards = self.session.run(self.model.inference, feed_dict = feed_dict)

    for i in range(len(mini_batch)):
      expected_rewards_by_agent.append(rewards[i] + self.config.future_reward_factor * np.max(inferenced_rewards[i]))
    feed_dict = \
    {
      self.model.input_placeholder : previous_states,
      self.model.action_placeholder : actions,
      self.model.target_placeholder : expected_rewards_by_agent
    }
    self.session.run(self.model.training, feed_dict = feed_dict)

    if (self.observations_count % 100) == 0:
      summary_str = self.session.run(self.summary, feed_dict = feed_dict)
      self.summary_writer.add_summary(summary_str, self.observations_count)
      self.summary_writer.flush()

    if (self.observations_count % self.config.save_every_x_steps) == 0:
      self.saver.save(self.session, self.config.checkpoint_path + r"\network", global_step=self.observations_count)

  def update_random_action_probability(self):
    if self.probability_of_random_action > self.config.final_random_action_probability:
      self.probability_of_random_action -= (self.config.initial_random_action_probability - self.config.final_random_action_probability) / self.config.exploration_steps_count

  def action_to_onehot(self, action : int):
    onehot = np.zeros(self.model.action_count).astype(float)
    onehot[action] = 1.0
    return onehot