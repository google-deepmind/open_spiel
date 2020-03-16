import numpy as np

a = np.ones((5,3))
b = np.array([1,2,3])

c = np.reshape(b,[-1,1])

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

d = a.dot(c)
print(softmax(d))

class Normalizer():

    def __init__(self, nb_inputs):
        self.n = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)

    def observe(self, x):
        self.n += 1.
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min=1e-2)

    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std


normalizer = Normalizer(3)

noise = np.random.normal(size=(5,3))

for x in list(noise):
    normalizer.observe(x)

print(normalizer.normalize(b))
print(normalizer.mean, normalizer.var)



