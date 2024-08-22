import numpy as np

# 1d og 1e

a = np.array([1, 2, 3])
x = np.array([(1, 2, 3), (4, 5, 6)])

y = np.dot(x, a)

# 1d
print(y)

#1e
print(sum(y))

