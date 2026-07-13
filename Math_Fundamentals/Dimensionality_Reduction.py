"""
Summary of Notes : 

Curse of Dimensionality : Distance,Volume, and Data density behave counterintuitively as dimension grows. 
PCA : Rotating coordinate system so that axes align with direcn of max variance,dropping lower variance axes.
Principal Component : Eigenvector of covariance matrix in direcn of data in feature space that varies the most.
Covariance Matrix : Symmetric matrix that measure of two feature move together.
t-SNE : Non linear method that maps high dimensional data into 2D preserving pairwise neighbourhood probabilities.
UMAP : Non linear method based on topological graphical data analysis. Better than t-SNE at scaling.
Explained Variance Ratio : Fraction of total variance captured by one principal component.
Manifold : lower dimensional embedded in higher dimension space. 
"""

import numpy as np 

class PCA :
    def __init__(self,n_components):
        self.n_components = n_components
        self.mean = None 
        self.components = None 
        self.eigenvals = None 
        self.explained_variance_ratio = None 
    
    def fit(self,X):
        self.mean = np.mean(X,axis=0)
        X_centered = X-self.mean 
        covariance_matrix = np.cov(X_centered,rowvar=False)
        eigenvals,eigenvecs = np.linalg.eigh(covariance_matrix)
        sorted = np.argsort(eigenvals)[::-1]
        eigenvecs = eigenvecs[:,sorted]
        eigenvals = eigenvals[sorted]
        self.components = eigenvecs[:,: self.n_components].T
        self.eigenvals = eigenvals[:self.n_components]
        total_variance = np.sum(eigenvals)
        self.explained_variance_ratio= self.eigenvals/total_variance
        return self 
    
    def transform(self,X):
        X_centered = X-self.mean
        return X_centered@self.components.T 
    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)
    
# np.random.seed(42)
# n_samples = 500
# t = np.random.uniform(0, 2 * np.pi, n_samples)
# x1 = np.tan(t) + np.random.normal(0, 0.2, n_samples)
# x2 = 2*np.cos(t) + np.random.normal(0, 0.2, n_samples)
# x3 = 0.4 * x1 + 0.7 * x2 + np.random.normal(0, 0.1, n_samples)
# X_synthetic = np.column_stack([x1, x2, x3])
# pca = PCA(n_components=2)
# X_reduced = pca.fit_transform(X_synthetic)    
# print(X_synthetic.shape,X_reduced.shape,pca.explained_variance_ratio,sum(pca.explained_variance_ratio))
