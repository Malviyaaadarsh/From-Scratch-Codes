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

