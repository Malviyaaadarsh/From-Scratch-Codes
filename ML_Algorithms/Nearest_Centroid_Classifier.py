import numpy as np
class NearestCentroid:

    def __init__(self):
        self.classes=None
        self.centroids=None

    def fit(self,x,y):  # compute the two means (centers)
        self.classes=np.unique(y)
        self.centroids= np.array([x[y==c].mean(axis=0) for c in self.classes])

    def predict(self,x): # compute distance
        distances= np.array([np.sqrt(np.sum((x-c)**2,axis=1)) for c in self.centroids])
        return self.classes[np.argmin(distances,axis=0)]

    def score(self,x,y):
        return np.mean(self.predict(x)==y)



# sklearn implementation
from sklearn.neighbors import NearestCentroid
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
X,y = make_classification(n_samples=1000, n_features=3, n_redundant=0, n_clusters_per_class=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = NearestCentroid()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))