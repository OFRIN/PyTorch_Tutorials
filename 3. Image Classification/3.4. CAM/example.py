
# def sum(a, b, c):
#     return a + b + c

# inputs = [1, 2, 3]

# sums = sum(*inputs)
# print(sums)

import numpy as np

# weights = np.zeros((512))
# features = np.zeros((512,7,7))
# print(weights[:, np.newaxis, np.newaxis]*features)

# [512, 7, 7] -> [7, 7]
# np.sum(features, axis=0) 

vector = np.random.randint(0, 100, 10)

# [23 76 22 53 75 82 35 83  4 16]

min_value = np.min(vector) # 4
max_value = np.max(vector) # 83

vector = vector - np.min(vector)
vector = vector / np.max(vector)
print(vector * 255)