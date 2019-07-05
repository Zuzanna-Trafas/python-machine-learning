import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Perceptron(object):
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

    def fit(self, X, Y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, Y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        # print results
        #for prediction, correct in zip(X, Y):
        #    print("Predicted: " + str(self.predict(prediction)) + "  Correct: " + str(correct))
        return self

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

# plotting flowers
plt.scatter(X[:50,0], X[:50, 1], color='red', marker='o', label='Setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='o', label='Versicolor')
plt.xlabel("Sepal length [cm]")
plt.ylabel("Petal length [cm]")
plt.legend(loc='upper left')
plt.show()

# predicting
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel("Epochs")
plt.ylabel("Errors")
plt.show()

