from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import cross_val_score
import numpy as np
data = datasets.load_iris()
X = data['data']
y = data['target']

lasso = Lasso(alpha = 0.01, normalize=True)

lasso.fit(X, y)

plt.plot(data['feature_names'], lasso.coef_)
plt.show()