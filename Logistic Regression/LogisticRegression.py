import pandas as pd
import numpy as np
class LogisticRegression1:
    def __init__(self, X, Y, iterations, alpha=0.01):
        self.X = np.c_[np.ones((X.shape[0], 1)), X]
        self.Y = Y.reshape(-1, 1)
        self.iterations = iterations
        self.alpha = alpha
        self.theta = None
        self.m = self.X.shape[0]

    def initialize_params(self):
        self.theta = np.zeros((self.X.shape[1], 1))

    def sigmoid(self, z):
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def gradient_descent(self):
        z = self.X @ self.theta
        h = self.sigmoid(z)
        dtheta = (1 / self.m) * (self.X.T @ (h - self.Y))
        self.theta -= self.alpha * dtheta
        cost = -(1 / self.m) * np.sum(
            self.Y * np.log(h + 1e-8) + (1 - self.Y) * np.log(1 - h + 1e-8)
        )
        return cost

    def train(self):
        self.initialize_params()
        costs = []
        for i in range(self.iterations):
            cost = self.gradient_descent()
            if i % 100 == 0:
                print(f"Cost at iteration {i}: {cost}")
            costs.append(cost)
        return self.theta, costs

    def predict(self, X):
        X = np.c_[np.ones((X.shape[0], 1)), X]
        z = X @ self.theta
        h = self.sigmoid(z)
        return (h >= 0.5).astype(int)
    