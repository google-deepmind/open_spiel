import numpy as np

# a = np.ones((5,3))
# b = np.array([1,2,3])
#
# c = np.reshape(b,[-1,1])
#
# def softmax(x):
#     return np.exp(x)/np.sum(np.exp(x))
#
# d = a.dot(c)
# print(softmax(d))
#
# class Normalizer():
#
#     def __init__(self, nb_inputs):
#         self.n = np.zeros(nb_inputs)
#         self.mean = np.zeros(nb_inputs)
#         self.mean_diff = np.zeros(nb_inputs)
#         self.var = np.zeros(nb_inputs)
#
#     def observe(self, x):
#         self.n += 1.
#         last_mean = self.mean.copy()
#         self.mean += (x - self.mean) / self.n
#         self.mean_diff += (x - last_mean) * (x - self.mean)
#         self.var = (self.mean_diff / self.n).clip(min=1e-2)
#
#     def normalize(self, inputs):
#         obs_mean = self.mean
#         obs_std = np.sqrt(self.var)
#         return (inputs - obs_mean) / obs_std
#
#
# normalizer = Normalizer(3)
#
# noise = np.random.normal(size=(5,3))
#
# for x in list(noise):
#     normalizer.observe(x)
#
# print(normalizer.normalize(b))
# print(normalizer.mean, normalizer.var)

def renormalize(probabilities):
  """Replaces all non-zero entries with zeroes and normalizes the result.

  Args:
    probabilities: probability vector to renormalize. Has to be one-dimensional.

  Returns:
    Renormalized probabilities.
  """
  probabilities[probabilities < 0] = 0
  probabilities = probabilities / np.sum(probabilities)
  return probabilities

def normalize_ne(eq):
    for p in range(len(eq)):
        for i, str in enumerate(eq[p]):
            eq[p][i] = renormalize(str)
    return eq

a = [[np.array([-0.1,0.9]), np.array([-0.01,0.98]), np.array([-0.03,0.99])]]
print(normalize_ne(a))

