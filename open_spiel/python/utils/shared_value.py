"""A shared value without deep copy."""


class SharedValue(object):
  """A shared value without deep copy."""

  def __init__(self, value):
    self.value = value

  def __deepcopy__(self, memo):
    return SharedValue(self.value)
