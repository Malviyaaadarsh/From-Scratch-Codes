import numpy as np
# Logistic Function 
def sigmoid(z):
    return 1/(1+ np.exp(-z))

# Logistic Regression Class
class LogisticRegression:
    def __init__(self, n_features, lr=0.01):
        self.bias = 0.0
        self.weights = np.zeros(n_features)
        self.lr = lr
        self.losses = []

    def predict_prob(self, x):
        z = np.dot(x, self.weights) + self.bias
        return sigmoid(z)

    def predict(self, x, threshold=0.5):
        return 1 if self.predict_prob(x) >= threshold else 0

    def compute_loss(self, X, y):
        p = np.clip([self.predict_prob(x) for x in X], 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def fit(self, X, y, epochs=1000):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        n, n_features = X.shape

        for epoch in range(epochs):
            preds = sigmoid(X @ self.weights + self.bias)
            errors = preds - y

            dw = X.T @ errors / n
            db = np.mean(errors)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            loss = self.compute_loss(X, y)
            self.losses.append(loss)

        return self

    def accuracy(self, X, y):
        preds = [self.predict(x) for x in X]
        return np.mean(preds == y)

# Classification Metrics
class ClassificationMetrics: 
    def __init__(self,y_true,y_pred):
        self.tp = sum(1 for t,p in zip(y_true,y_pred) if t==1 and p==1)
        self.tn = sum(1 for t,p in zip(y_true,y_pred) if t==0 and p==0)
        self.fp = sum(1 for t,p in zip(y_true,y_pred) if t==0 and p==1)
        self.fn = sum(1 for t,p in zip(y_true,y_pred) if t==1 and p==0)

    def accuracy(self):
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0.0

    def error(self):
        total = self.tp + self.tn + self.fp + self.fn
        return (self.fp + self.fn) / total if total > 0 else 0.0

    def precision(self):
        total = self.tp + self.fp
        return self.tp / total if total > 0 else 0.0

    def recall(self):
        total = self.tp + self.fn
        return self.tp / total if total > 0 else 0.0

    def f1_score(self):
        precision = self.precision()
        recall = self.recall()
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0