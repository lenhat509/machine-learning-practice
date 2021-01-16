from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

knn = KNeighborsClassifier(5)

data = datasets.load_iris()
# print(data.DESCR)
# print(data['feature_names'])
X = data['data']
y = data['target']
print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print(knn.score(X_test, y_test))

# petal_width is likely to be the most important factor to determine
# the type of iris (according to lasso.py, it has the largest coefficient)
X_petal_width = X[:, 3]

plt.scatter(X_petal_width, y)
plt.show()