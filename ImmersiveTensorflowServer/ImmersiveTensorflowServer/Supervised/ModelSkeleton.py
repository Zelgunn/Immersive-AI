import tensorflow as tf
import functools

def define_scope(function):
  attribute = '_cache_' + function.__name__

  @property
  @functools.wraps(function)
  def decorator(self):
    if not hasattr(self, attribute):
      with tf.variable_scope(function.__name__):
        setattr(self, attribute, function(self))
    return getattr(self, attribute)

  return decorator

class ModelSkeleton(object):
  @define_scope
  def inference(self):
    # To inherit
    pass

  @define_scope
  def loss(self):
    # To inherit
    pass

  @define_scope
  def training(self):
    # To inherit
    pass

  @define_scope
  def evaluation(self):
    # To inherit
    pass

  @define_scope
  def placeholders(self):
    # To inherit
    pass