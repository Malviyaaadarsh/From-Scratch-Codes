"""
Summary of Notes : 

SVD : Decomposition of A (of any shape) into U*D*(V.T).
Singular Value : ith diagonal entry of D, measure stretch along ith principal direction in matrix.
Left Singular Vector : Column of U (Direcn in output space that ith right singular vector maps to)
Right Singular Vector : Column of V (direcn in input space that matrix maps to ith left singular vector)
Truncated-SVD : by low rank approximation (keeping only top k singular values and vectors)
Power Iteration : Repeatedly multiply a random vector by the matrix and normalize. Converges to eigenvector with largest value.
Pseudoinverse : V * D * U.T , inverts non zero singular values, solves least squares problem.
"""

import numpy as np 
import math,random

# Power Iteration Definition 
def power_iteration(M,iterations=100):
    n= M.shape[1]
    v= np.random.randn(n)
    v /= np.linalg.norm(v)       
    for _ in range(iterations):
        Mv = M @ v
        v = Mv/np.linalg.norm(Mv)
    eigenvalue = v @ M @ v 
    return eigenvalue,v

def svd(A,k=None):
    m,n=A.shape
    if k is None:
        k=min(m,n)
    D0,U0,V0 = [],[],[]
    A_copy = A.copy().astype(float)
    for _ in range(k):
        Atrans_A = A_copy.T @ A_copy 
        eigenvals,v= power_iteration(Atrans_A,iterations=200)
        if eigenvals<1e-8:
            break
        d = np.sqrt(eigenvals)
        u = A_copy @ v / d
        D0.append(d)
        U0.append(u)
        V0.append(v)
        A_copy -= (d*np.outer(u,v))
    U = np.column_stack(U0) if U0 else np.empty((m,0))
    V= np.column_stack(V0) if V0 else np.empty((n,0))
    D = np.array(D0)
    return U,D,V

# Image Compression using SVD 
def compress_image(img,k):
    u,d,vt = svd(img)
    compressed = u[:, :k] @ np.diag(d[:k]) @ vt[:k,:]
    return compressed

# image = np.random.seed(42)
# rows, cols = 300, 200
# image = np.random.randn(rows, cols)
# for k in [1, 5, 10, 20, 50]:
#     compressed = compress_image(image, k)
#     error = np.linalg.norm(image - compressed) / np.linalg.norm(image)
#     original_size = rows * cols
#     compressed_size = k * (rows + cols + 1)
#     ratio = compressed_size / original_size
#     print(f"k={k:>3d}  error={error:.3f}  storage={ratio:.2%}")
