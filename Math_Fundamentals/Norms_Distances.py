"""
Summary of Notes : 

Norm : Function that maps vector to non negative scalar traingle inequality,absolute homogeneity,zero for zero vector.
L1 Norm (Manhattan Dist.) : Sum of absolute component values.Produces sparsityin optimization.Robust to outliers.
L2 Norm (Euclidean Dist.) : Sqrt of sum of sq. components.
Lp Norm (Generalized) : pth root of sum of pth powers of absolute components.
L-infinity Norm (Chebyshev Dist.): Max. Absolute component value.Limit of Lp as p approaches infinity.
Cosine Similarity : Dot products, normalized by magnitudes of both vectors. Ranges [-1,1].
Cosine Distance : Convert Cosine similarity to distance. Ranges [0,2]. equals 1 - Cosine Similarity.
Mahalanobis Distance : L2 dist. in space that has whitened ( decorrelated & normalized ) using a covariance matrix.
Jaccard Similarity : (Size of Intersetion)/(Size of Union) for sets.
Levenshtein Distance (Edit Dist.) : Min. insertion,Deletion,Substitution to transform one string to another.
KL Divergence : Measure extra bits from using Q to encode P.
Wasserstein Distance (Earth mover) : Min. work to transport mass from one distribution to another.
Approximate Nearest Neighbor (ANN) : find approx. closest points much faster than exact search.
HNSW (Hierarchial NAvigablle World Graph) : multi layered graph for fast ann search. Used in Vector DBs.
L1 Regularization (Lasso) : Adding L1 norm of weights to loss. Drives weights to zero.
L2 Regularization (Ridge) : Adding sq. L2 Norm of weights to loss. Shrinks weights towards zero without sparsity.
Elastic Net : L1+L2 Regularization combined. Handles correlated groups better.
"""

# Definitions of various norms and distances used in mathematics and machine learning.
import math
import numpy as np

def l1_norm(vector):
    return np.sum(np.abs(vector))

def l2_norm(vector):
    return np.sqrt(np.sum(np.square(vector)))

def lp_norm(vector, p):
    if p <= 0:
        raise ValueError("p must be a positive integer.")
    elif p == np.inf:
        return np.max(np.abs(vector))
    else:
        return np.power(np.sum(np.abs(vector) ** p), 1/p)
    
def lp_distance(vector_a, vector_b, p):
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must be of the same length.")
    return lp_norm(vector_a - vector_b, p)

def l_infinity_norm(vector):
    return np.max(np.abs(vector))

def l_infinity_distance(vector_a, vector_b):
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must be of the same length.")
    return l_infinity_norm(vector_a - vector_b)

def cosine_similarity(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    norm_a = l2_norm(vector_a)
    norm_b = l2_norm(vector_b)
    if norm_a == 0 or norm_b == 0:
        raise ValueError("One of the vectors is zero, cannot compute cosine similarity.")
    return dot_product / (norm_a * norm_b)

def cosine_distance(vector_a, vector_b):
    return 1.0 - cosine_similarity(vector_a, vector_b)

def normalized_vector(vector):
    norm = l2_norm(vector)
    if norm == 0:
        return vector[:]
    return vector / norm

def mahalanobis_distance(vector_a, vector_b, covariance_matrix):
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must be of the same length.")
    diff = vector_a - vector_b
    inv_cov_matrix = np.linalg.inv(covariance_matrix)
    return np.sqrt(np.dot(np.dot(diff.T, inv_cov_matrix), diff))

def jaccard_similarity(set_a, set_b):
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 1.0  
    return intersection / union

def jaccard_distance(set_a, set_b):
    return 1.0 - jaccard_similarity(set_a, set_b)

# Levenshtein distance implementation ( By dynamic programming )
def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    m,n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

    return dp[m][n]

def kl_divergence(p, q):
    total = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            if qi <= 0:
                return float('inf')
            total += pi * math.log(pi / qi)
    return total

def wasserstein_distance(p, q):
    if len(p) != len(q):
        raise ValueError("Distributions must be of the same length.")
    return np.sum(np.abs(np.cumsum(p) - np.cumsum(q)))

def nearest_neighbor_search(query,dataset,distance_function,k=5,**kwargs):
    distances = []
    for i,data_point in enumerate(dataset):
        dist = distance_function(query, data_point, **kwargs)
        distances.append((i,dist))
    distances.sort(key=lambda x: x[1])
    return distances[:k]

def l1_regularization(weights, alpha):
    return alpha * l1_norm(weights)

def l2_regularization(weights, alpha):
    return alpha * l2_norm(weights) ** 2

def elastic_net_regularization(weights, alpha, l1_ratio):
    l1_component = l1_regularization(weights, alpha * l1_ratio)
    l2_component = l2_regularization(weights, alpha * (1 - l1_ratio))
    return l1_component + l2_component
