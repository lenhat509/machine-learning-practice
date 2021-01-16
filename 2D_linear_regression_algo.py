import numpy as np
import pandas as pd

pd.set_option('display.float_format', lambda x: '%.5f' % x)

learning_rate = 0.005
data = pd.read_csv('data.csv', header=None)
weights = np.array([0, 0])
output = data[1]

input = pd.DataFrame()
input1 = np.full(100, 1)
input[0] = input1
input = input.join(data[[0]].rename(columns={0: 1}))

partial_derivative = np.array([1, 1])

while not (partial_derivative == [0, 0]).all():
    partial_derivative = (input * ((input * weights).sum(axis=1) - output).values.reshape((100, 1))).sum(axis=0).round(5)
    weights = weights - learning_rate * partial_derivative
    print(partial_derivative)

print(weights)
