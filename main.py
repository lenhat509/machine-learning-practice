import numpy as np
import pandas as pd

pd.set_option('display.float_format', lambda x: '%.3f' % x)

learning_rate = 0.1
data = pd.read_csv('data.csv', header=None)
weights = np.array([0, 0])
output = data[1]

input = pd.DataFrame()
input1 = np.full(100, 1)
input[0] = input1
input = input.join(data[[0]].rename(columns={0: 1}))

next_weights = np.array([1, 1])

while not (weights == next_weights).all():
    weights = next_weights
    partial_derivative = (input * ((input * weights).sum(axis=1) - output).values.reshape((100, 1))).sum(axis=0)
    next_weights = weights - learning_rate * partial_derivative
    print(partial_derivative)
