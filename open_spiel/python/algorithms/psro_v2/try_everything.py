import numpy as np

a = np.ones((5,3))
b = np.array([1,2,3])

c = np.reshape(b,[-1,1])

def softmax_on_range(number_policies):
  x = np.array(list(range(number_policies)))
  x = np.exp(x-x.max())
  x /= np.sum(x)
  return x

print(softmax_on_range(5))



