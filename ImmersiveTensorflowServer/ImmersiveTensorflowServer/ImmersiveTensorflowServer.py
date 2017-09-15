import numpy as np
import socket
import struct
import math

class ImmersiveTensorflowServer(object):
  def __init__(self):
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self.listening = False

  def get_data(self):
    recv_size = 32768
    data, _ = self.recv_ack(recv_size)
    length = int.from_bytes(data[:4], 'little')

    recv_count = math.ceil(length / recv_size)

    for i in range(1, recv_count):
      tmp, _ = self.recv_ack(recv_size)
      data += tmp
    return data, length

  def recv_ack(self, size):
    data, ip = self.sock.recvfrom(size)
    ack = "ACK".encode()
    self.send_data(ack)
    return data, ip

  def stop_listening(self):
    self.listening = False
    self.sock.close()

  def send_data(self, data : bytes):
    self.sock.sendto(data, (self.config.ip, self.config.send_port))

