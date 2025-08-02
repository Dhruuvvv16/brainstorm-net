import pandas as pd
import numpy as np
class DeepNN:
    def __init__(self, X, Y, loss_fn, alpha=0.01, lambda_=0.01):
        self.X = X
        self.Y = Y
        self.L = int(input("Enter number of layers: "))
        self.n_h = {0: X.shape[1]}
        self.activations = {}
        for l in range(0, self.L):
            self.n_h[l + 1] = int(input(f"how many neurons in layer {l + 1} : "))
            self.activations[l + 1] = input(f"Activation function for layer {l + 1} : ")
        self.alpha = alpha
        self.m = X.shape[0]
        self.W = {}
        self.B = {}
        self.Z = {}
        self.A = {0: self.X.T}
        self.loss_fn = loss_fn
        self.cost = []
        self.lambda_ = lambda_
        self.initialize_params()
        self.VdW = {}
        self.VdB = {}
        self.SdW = {}
        self.SdB = {}
        self.t = 0
        for l in range(1, self.L + 1):
            self.VdW[l] = np.zeros_like(self.W[l])
            self.VdB[l] = np.zeros_like(self.B[l])
            self.SdW[l] = np.zeros_like(self.W[l])
            self.SdB[l] = np.zeros_like(self.B[l])

    def initialize_params(self):
        for l in range(1, self.L + 1):
            self.W[l] = np.random.randn(self.n_h[l], self.n_h[l - 1]) * np.sqrt(2. / self.n_h[l - 1])
            self.B[l] = np.zeros((self.n_h[l], 1))

    def linear(self, Z):
        return Z

    def sigmoid(self, Z):
        Z = np.clip(Z, -500, 500)
        S = 1 / (1 + np.exp(-Z))
        return S

    def tanh(self, Z):
        T = np.tanh(Z)
        return T

    def relu(self, Z):
        R = np.maximum(0, Z)
        return R

    def leakyrelu(self, Z, alpha=0.01):
        R = np.where(Z > 0, Z, alpha * Z)
        return R

    def softmax(self, Z):
        Z_stable = Z - np.max(Z, axis=0, keepdims=True)
        exp_Z = np.exp(Z_stable)
        S = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        return S

    def apply_activation(self, z, activation_type):
        if activation_type.lower() == 'sigmoid':
            return self.sigmoid(z)
        elif activation_type.lower() == 'tanh':
            return self.tanh(z)
        elif activation_type.lower() == 'relu':
            return self.relu(z)
        elif activation_type.lower() == 'leakyrelu':
            return self.leakyrelu(z)
        elif activation_type.lower() == 'softmax':
            return self.softmax(z)
        elif activation_type == 'linear':
            return self.linear(z)
        else:
            raise ValueError(f"Unknown activation function: {activation_type}")

    def forward_propagation(self):
        for l in range(1, self.L + 1):
            # print(f"Layer {l} ; Shape of A{l-1} - {self.A[l-1].shape} ")
            self.Z[l] = (self.W[l] @ self.A[l - 1]) + self.B[l]
            self.A[l] = self.apply_activation(self.Z[l], self.activations[l])
        return self.Z

    def binary_cross_entropy(self, Y, A):
        m = Y.shape[0]
        cost = -np.sum(Y * np.log(A + 1e-8) + (1 - Y) * np.log(1 - A + 1e-8)) / m
        return cost

    def categorical_cross_entropy(self, Y, A):
        m = Y.shape[0]
        cost = -np.sum(Y * np.log(A + 1e-8)) / m
        return cost

    def mean_squared_error(self, Y, A):
        m = Y.shape[0]
        cost = np.sum((A - Y) ** 2) / (2 * m)
        return cost

    def compute_cost(self):
        L2_cost = (self.lambda_ / (2 * self.m)) * sum([np.sum(np.square(self.W[l])) for l in range(1, self.L + 1)])
        if self.loss_fn == "binary_crossentropy":
            cost = self.binary_cross_entropy(self.Y, self.A[self.L])
        elif self.loss_fn == "categorical_crossentropy":
            cost = self.categorical_cross_entropy(self.Y, self.A[self.L])
        elif self.loss_fn == "mse":
            cost = self.mean_squared_error(self.Y, self.A[self.L])
        else:
            raise ValueError("Unsupported loss function")
        cost = cost + L2_cost
        return cost

    def linear_derivative(self, Z):
        return np.ones_like(Z)

    def sigmoid_derivative(self, Z):
        A = self.sigmoid(Z)
        return A * (1 - A)

    def relu_derivative(self, Z):
        Z = np.array(Z)
        return (Z > 0).astype(float)

    def tanh_derivative(self, Z):
        return 1 - np.tanh(Z) ** 2

    def leakyrelu_derivative(self, Z, alpha=0.01):
        dZ = np.ones_like(Z)
        dZ[Z < 0] = alpha
        return dZ

    def apply_activation_derivative(self, z, activation_type):
        if activation_type.lower() == 'sigmoid':
            return self.sigmoid_derivative(z)
        elif activation_type.lower() == 'tanh':
            return self.tanh_derivative(z)
        elif activation_type.lower() == 'relu':
            return self.relu_derivative(z)
        elif activation_type.lower() == 'leakyrelu':
            return self.leakyrelu_derivative(z)
        elif activation_type == 'linear':
            return self.linear_derivative(z)
        else:
            raise ValueError(f"Unknown activation function: {activation_type}")

    def backpropagation(self):
        cost = self.compute_cost()
        self.cost.append(cost)

        dW = {}
        dB = {}
        dZ = {}
        dA = {}

        if self.loss_fn == "binary_crossentropy":
            dA[self.L] = -(self.Y.T / (self.A[self.L] + 1e-8) - (1 - self.Y.T) / (1 - self.A[self.L] + 1e-8))
        elif self.loss_fn == "categorical_crossentropy":
            dA[self.L] = self.A[self.L] - self.Y
        elif self.loss_fn == "mse":
            dA[self.L] = self.A[self.L] - self.Y.T

        for l in range(self.L, 0, -1):
            if self.activations[l] == "softmax" and self.loss_fn == "categorical_crossentropy":
                dZ[l] = self.A[l] - self.Y
            else:
                dZ[l] = dA[l] * self.apply_activation_derivative(self.Z[l], self.activations[l])
            dW[l] = (1 / self.m) * np.dot(dZ[l], self.A[l - 1].T) + (self.lambda_ / self.m) * self.W[l]
            dB[l] = (1 / self.m) * np.sum(dZ[l], axis=1, keepdims=True)
            if l > 0:
                dA[l - 1] = np.dot(self.W[l].T, dZ[l])

        # ADAM implementation
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        self.t += 1

        for l in range(1, self.L + 1):
            self.VdW[l] = beta1 * self.VdW[l] + (1 - beta1) * dW[l]
            self.VdB[l] = beta1 * self.VdB[l] + (1 - beta1) * dB[l]
            self.SdW[l] = beta2 * self.SdW[l] + (1 - beta2) * (dW[l] ** 2)
            self.SdB[l] = beta2 * self.SdB[l] + (1 - beta2) * (dB[l] ** 2)

            VdW_corr = self.VdW[l] / (1 - beta1 ** self.t)
            VdB_corr = self.VdB[l] / (1 - beta1 ** self.t)
            SdW_corr = self.SdW[l] / (1 - beta2 ** self.t)
            SdB_corr = self.SdB[l] / (1 - beta2 ** self.t)

            self.W[l] -= self.alpha * VdW_corr / (np.sqrt(SdW_corr) + epsilon)
            self.B[l] -= self.alpha * VdB_corr / (np.sqrt(SdB_corr) + epsilon)
        return cost

    def train(self, epochs=1000, print_cost=True):
        for i in range(epochs):
            self.forward_propagation()
            cost = self.backpropagation()
            if print_cost and i % 100 == 0:
                print(f"Cost after iteration {i}: {cost}")

    def predict(self, X):
        A = X.T
        for l in range(1, self.L + 1):
            Z = np.dot(self.W[l], A) + self.B[l]
            A = self.apply_activation(Z, self.activations[l])

        if self.loss_fn == "binary_crossentropy":
            return (A > 0.5).astype(int)
        elif self.loss_fn == "categorical_crossentropy":
            return np.argmax(A, axis=0)
        else:
            return A