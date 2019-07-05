import numpy as np


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
        for prediction, correct in zip(X, Y):
            print("Predicted: " + str(self.predict(prediction)) + "  Correct: " + str(correct))

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


x = np.array([[2.7810836, 2.550537003],
              [1.465489372, 2.362125076],
              [3.396561688, 4.400293529],
              [1.38807019, 1.850220317],
              [3.06407232, 3.005305973],
              [7.627531214, 2.759262235],
              [5.332441248, 2.088626775],
              [6.922596716, 1.77106367],
              [8.675418651, -0.242068655],
              [7.673756466, 3.508563011]])

y = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1])

perceptron = Perceptron()
perceptron.fit(x, y)
