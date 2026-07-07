"""
Summary of Notes : 

Rotation Matrix : Orthogonal matrix that moves points along circular arcs while preserving distance and angles.(Determinant= 1)
Scaling Matrix : Diagonal matrix that stretches/compress independently along each axis.(Determinant = product of scalar factors)
Shearing Matrix : Matrix that shift one coordinate proportionally to another.(Determinant = 1) (Rectangles -> Parallelogram)
Reflection Matrix : flips space across an axis or plane.(Determinant = -1)
Composition : Multiplying transformation matrices to chain operations. Orders matters. (A @ B means apply B first then A)
Eigenvector : direction that matrix only scales ( never rotate) 
Eigenvalue : scalar factor by which matrix scales its eigenvector. It can be negative (flipped) or complex (rotated).
EigenDecomposition : matrix as V @ D @ V^-1 
Determinant : factor by which transformation scales area or volume. Det=0 means transformation is irreversible. 
Characteristic Equation : det(A-lambda*(I)). Its roots are eigenvalues. 
"""

# Definitions 
import random,math 
def rotation_2d(theta):
    c,s= math.cos(theta),math.sin(theta)
    return [[c,-s],[s,c]]

def scaling_2d(sx,sy):
    return [[sx,0],[0,sy]]

def reflection_x():
    return [[1,0],[0,-1]]

def reflection_y():
    return [[-1,0],[0,1]]

def shearing_2d(kx,ky):
    return [[1,kx],[ky,1]]

def mat_x_vec(mat,vec):
    return [sum(mat[i][j]*vec[j] for j in range(len(vec))) for i in range(len(mat))]

angle = math.pi 

# print(rotation_2d(angle),scaling_2d(1,1),shearing_2d(4,3),mat_x_vec([[2,3],[3,4]],[1,2])) 

# EigenValues and EigenVectors from Scratch 

def eigen_value_2d(matrix):
    a,b = matrix[0]
    c,d = matrix[1] 
    trace = a + d 
    det = a*d - b*c 
    discriminant = trace**2 - 4*det   # (Discriminat positive -> 2 real eigenvalues , negative -> complex conjugate eigenvalues , zero -> real repeated values)
    if discriminant<0 : 
        real = trace/2 
        img = (-discriminant)**0.5 /2 
        return (complex(real,img),complex(real,-img))
    disc_sqrt = discriminant**0.5 
    return ((trace+disc_sqrt)/2,(trace-disc_sqrt)/2)

def eigen_vectors_2d(matrix,eigenvalue):
    a,b = matrix[0]
    c,d = matrix[1]
    if(abs(b)>1e-10):
        v = [b,eigenvalue-a]
    elif abs(c)>1e-10:
        v = [eigenvalue-d,c]
    else:
        if(abs(a-eigenvalue)<1e-10):
            v=[1,0]
        else:
            v=[0,1]
    mag = (v[0]**2 + v[1]**2)**0.5
    return [v[0]/mag,v[1]/mag]


# Numpy implementation 
import numpy as np
a = np.array([[1,2],[5,6]])
eigenval,eigenvec = np.linalg.eig(a)
# print(eigenval,eigenvec,np.linalg.det(a))
for i in range(len(eigenval)):
    v,lam = eigenvec[:,i],eigenval[i]
    # print(a@v,lam*v)
