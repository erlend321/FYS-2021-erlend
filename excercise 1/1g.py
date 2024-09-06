import numpy as np

w = np.random.random((3, 3))
x = np.random.random((3, 3))

v = np.dot(x, w.T)

v = np.dot(w, x)

a = np.array([1, 2, 3])

y = np.dot(v, a)

print(y)