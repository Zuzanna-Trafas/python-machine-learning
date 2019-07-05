import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Adaline(object):
    """
    Parameters:
        eta = learning rate, between 0.0 and 1.0
        n_iter = number of iterations
        random_state = seed
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    """
    Attributes:
        w_ = weights after fitting
        errors_ = number of misclassification in every epoch
        X = training vector
        Y = target values
    """

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta*errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def activation(self, X):
        return X

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

# reading iris names and exchanging setosa with -1 and versicolor with 1
y = df.iloc[0:100, 4].values
y = np.where(y == "Iris-setosa", -1, 1)

# reading sepal and petal lengths
X = df.iloc[0:100, [0, 2]].values
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

# plotting flowers
plt.scatter(X_std[:50,0], X_std[:50, 1], color='red', marker='o', label='Setosa')
plt.scatter(X_std[50:100, 0], X_std[50:100, 1], color='blue', marker='o', label='Versicolor')
plt.xlabel("Sepal length [cm]")
plt.ylabel("Petal length [cm]")
plt.legend(loc='upper left')
plt.show()

# predicting
ada = Adaline(n_iter=15, eta=0.01)
ada.fit(X_std, y)
plt.plot(range(1, len(ada.cost_)+1), ada.cost_, marker='o')
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.show()
