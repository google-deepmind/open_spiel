import numpy as np

a = np.ones((5,3))
b = np.array([1,2,3])

c = np.reshape(b,[-1,1])

def hello():
    print("hello world.")
a = {"a": hello}
b = "hello world."
print("use {}".format(b.__name__))



