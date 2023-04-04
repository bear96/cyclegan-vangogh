class Hyperparameters(object):
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)
  
  def __str__(self):
    return str(self.__class__) + ": " + str(self.__dict__)