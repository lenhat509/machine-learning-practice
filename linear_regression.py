from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

data = datasets.load_boston()
# print(data.keys())
df = pd.DataFrame(data['data'], columns=data['feature_names'])
print(df.head())
print(df.shape)

X = data['data']
y = data['target']

reg = LinearRegression()

# regression with all features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

mse = mean_squared_error(y_pred, y_test)
print('R^2 : {}'.format(mse))

# regression with 1 feature : RM
X_rm = df['RM'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X_rm, y, test_size=0.3, random_state=42)

reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

plt.scatter(X_rm, y)
plt.plot(X_test, y_pred)
plt.xlabel('Number of rooms')
plt.ylabel('House\'s price')
plt.show()



