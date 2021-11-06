import numpy as np

x = [[1, 2], [3, 4]]
print(np.linalg.norm(x, ord=2, axis=0))
print(np.linalg.norm(x, ord=2, axis=1))