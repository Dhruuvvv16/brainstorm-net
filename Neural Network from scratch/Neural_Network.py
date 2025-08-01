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