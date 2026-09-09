import numpy as np
import math,random
# Loss Function of SVM : Hinge Loss
def hinge_loss(X,y,w,b):
    total_loss = 0
    for i in range(len(X)):
        margin = y[i] * (np.dot(X[i], w) + b)
        total_loss += max(0.0, 1.0 - margin)
    return total_loss / len(X)

# Linear SVM Class
class LinearSVM:
    def __init__(self, lr=0.001, lam=0.01, epochs=1000):
        self.lr = lr
        self.lam = lam
        self.epochs = epochs
        self.w = None
        self.b = 0.0
    def fit(self, X, y):
        n_features = len(X[0])
        self.w = np.zeros(n_features)
        self.b = 0.0
        for epoch in range(self.epochs):
            for i in range(len(X)):
                margin= y[i] * (np.dot(X[i], self.w) + self.b)
                if margin < 1:
                    self.w -= self.lr * (self.lam * self.w - y[i] * X[i])
                    self.b += self.lr * (y[i])
                else:
                    self.w -= self.lr * (self.lam * self.w)
    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

# Find Support Vectors Function and Kernel Functions

def find_support_vectors(X,y,w,b,threshold=1e-3):
    support_vectors = []
    for i in range(len(X)):
        margin = y[i] * (np.dot(X[i], w) + b)
        if abs(margin - 1.0) < threshold:
            support_vectors.append((X[i], y[i]))
    return support_vectors

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x1, x2, degree=3,c=1.0):
    return (np.dot(x1, x2) + c) ** degree

def rbf_kernel(x1, x2, gamma=0.1):
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)