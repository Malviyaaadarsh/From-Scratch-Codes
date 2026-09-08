import math, random
import numpy as np 

def gini_impurity(labels):
    if len(labels) == 0:
        return 0.0
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return 1.0 - sum((count/len(labels))**2 for count in counts.values())

def entropy(labels):
    if len(labels)==0:
        return 0.0
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return -sum((count/len(labels))*math.log2(count/len(labels)) for count in counts.values() if count > 0)

def info_gain(parent_labels,left_labels,right_labels,criterion='gini'):
    measure = gini_impurity if criterion == 'gini' else entropy
    n = len(parent_labels)
    n_left = len(left_labels)
    n_right = len(right_labels)
    if n_left == 0 or n_right == 0:
        return 0.0
    parent_impurity = measure(parent_labels)
    child_impurity = ((n_left/n)*measure(left_labels) + (n_right/n)*measure(right_labels))
    return parent_impurity - child_impurity

class DecisionTree:
    def __init__(self,max_depth=None, min_samples_split=2,min_samples_leaf=1,criterion='gini',max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.tree = None
        self.feature_importances_ = None

    def _split(self,X,y,feature,threshold):
        left_X,left_y,right_X,right_y = [],[],[],[]
        for i in range(len(X)):
            if X[i][feature] <= threshold:
                left_X.append(X[i])
                left_y.append(y[i])
            else:
                right_X.append(X[i])
                right_y.append(y[i])
        return left_X,left_y,right_X,right_y

    def _predict_one(self,x,node):
        if node['leaf']:
            return node['value']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x,node['left'])
        return self._predict_one(x,node['right'])

    def predict(self,X):
        return [self._predict_one(x,self.tree) for x in X]

    def fit(self,X,y):
        self.n_features = len(X[0])
        self.feature_importances_ = [0.0] * self.n_features
        self.n_samples = len(X)
        self.tree = self._build_tree(X,y,depth=0)
        total = sum(self.feature_importances_)
        if total > 0:
            self.feature_importances_ = [imp/total for imp in self.feature_importances_]

    def _build_tree(self,X,y,depth):
        if len(set(y)) == 1:
            return {'leaf': True, 'value': y[0]}
        if self.max_depth is not None and depth >= self.max_depth:
            return self._make_leaf(y)
        if len(y) < self.min_samples_split:
            return self._make_leaf(y)
        best_feature, best_threshold, best_gain = self._best_split(X,y)
        if best_feature is None or best_gain <= 0:
            return self._make_leaf(y)
        left_X,left_y,right_X,right_y = self._split(X,y,best_feature,best_threshold)
        if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
            return self._make_leaf(y)
        weight = len(y) / self.n_samples
        self.feature_importances_[best_feature] += best_gain * weight
        return {'leaf': False, 'feature': best_feature, 'threshold': best_threshold,
                'left': self._build_tree(left_X,left_y,depth+1),
                'right': self._build_tree(right_X,right_y,depth+1)}


    def _make_leaf(self,y):
        counts = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1
        majority_label = max(counts, key=counts.get)
        return {'leaf': True, 'value': majority_label}

    def _best_split(self,X,y):
        best_gain = -1
        best_feature = None
        best_threshold = None
        if self.max_features == 'sqrt':
            k = max(1, int(math.sqrt(self.n_features)))
            feature_indices = random.sample(range(self.n_features), k)
        elif isinstance(self.max_features, int):
            if self.max_features < 1:
                raise ValueError("max_features must be at least 1")
            k = min(self.max_features, self.n_features)
            feature_indices = random.sample(range(self.n_features), k)
        else:
            feature_indices = range(self.n_features)
        for feature in feature_indices:
            values = sorted(set(X[i][feature] for i in range(len(X))))
            if len(values) <= 1:
                continue
            for i in range(len(values)-1):
                threshold = (values[i] + values[i+1]) / 2.0
                left_y = [y[j] for j in range(len(X)) if X[j][feature] <= threshold]
                right_y = [y[j] for j in range(len(X)) if X[j][feature] > threshold]
                if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
                    continue
                gain = info_gain(y,left_y,right_y,self.criterion)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold, best_gain


class RandomForest:
    def __init__(self, n_trees=100, max_depth=None,
                 min_samples_split=2, max_features="sqrt",
                 criterion="gini"):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            indices = [random.randint(0, len(X) - 1) for _ in range(len(X))]
            X_bootstrap = [X[i] for i in indices]
            y_bootstrap = [y[i] for i in indices]
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                criterion=self.criterion,
                                max_features=self.max_features)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X):
        all_predictions = [tree.predict(X) for tree in self.trees]
        predictions = []
        for i in range(len(X)):
            votes = {}
            for tree_predictions in all_predictions:
                label = tree_predictions[i]
                votes[label] = votes.get(label, 0) + 1
            predictions.append(max(votes, key=votes.get))
        return predictions


