"""
Summary of Notes : 

Vector : List of numbers representing a point or direction in a muti-dimensional space.
Matrix : A transformation that maps vector from one to another space
Dot Product : measures how aligned (similar) two vectors are 
Embedding : vector representing a meaning of sonething
Linear Independence : No vector in set can be represented as combination of other vectors.
Rank : No. of linearly independent columns or rows 
Basis : Minimal set of independent vectors that spans the space 
Projection : Component of a vector in direction of another 
Orthonormal : Unit vectors that are mutually perpendicular
"""
# Definition of a vector
class Vector:
    def __init__(self,components):
        self.components = list(components)
        self.dim = len(components)
    def __repr__(self):
        return f"Vector = ({self.components})"
    def __add__(self, other):
        return Vector([a+b for a,b in zip(self.components,other.components)])
    def __sub__(self, other):
        return Vector([a-b for a,b in zip(self.components,other.components)])
    def dot(self,other):
        return sum([a*b for a,b in zip(self.components,other.components)])
    def magnitude(self):
        return sum(x**2 for x in self.components)**0.5 
    def normalize(self):
        mag = self.magnitude()
        return Vector([x/mag for x in self.components])
    def cosine_similarity(self,other):
        return self.dot(other) / (self.magnitude() * other.magnitude())

# a = Vector([10,1,14])
# b= Vector([12,23,2])
# print(a+b)
# print(a-b)
# print(a.dot(b))
# print(a.magnitude())
# print(b.normalize())
# print(a.cosine_similarity(b))


# Definition of a matrix 
class Matrix:
    def __init__(self,rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]))
    def __repr__(self):
        return f"Matrix = ({self.rows})"
    def transpose(self):
        return Matrix([[self.rows[j][i] for j in range(self.shape[0])] for i in range(self.shape[1])])
    def __matmul__(self, other):
        if isinstance(other,Vector):
            return Vector([sum(self.rows[i][j] * other.components[j] for j in range(self.shape[1])) for i in range(self.shape[0])])
        rows = []
        for i in range(self.shape[0]):
            row = []
            for j in range(self.shape[1]):
                row.append(sum(self.rows[i][k]*other.rows[k][j] for k in range(self.shape[1])))
            rows.append(row)
        return Matrix(rows)
    
# m = Matrix([[0,-1],[1,0]])
# v = Vector([2,1])
# rotated = m @ v
# transposed = m.transpose()
# print(m,v,rotated,transposed)

# Definition of Projection 
def project(a,b):
    scalar = a.dot(b) / b.dot(b) 
    return Vector([scalar*i for i in b.components])

# print(project(Vector([1,7,3]),Vector([3,4,9])))

# Definition of Linearly Independent Vectors 
def is_linearly_independent(vectors):
    n = len(vectors)
    dim = len(vectors[0].components)
    mat = Matrix([v.components[:] for v in vectors])
    rows =  [row[:] for row in mat.rows]
    rank = 0 
    for col in range(dim):
        pivot = None 
        for row in range(rank,len(rows)):
            if abs(rows[row][col])>1e-10 :
                pivot = row 
                break 
        if pivot is None : 
            continue
        rows[rank],rows[pivot] = rows[pivot],rows[rank]   
        scale = rows[rank][col]
        rows[rank] = [x/scale for x in rows[rank]]
        for row in range(len(rows)):
            if row != rank and abs(rows[row][col])>1e-10 :
                factor = rows[row][col]
                rows[row] = [rows[row][j] - factor * rows[rank][j] for j in range(dim)]
        rank+=1 
    return rank==n 

# print(is_linearly_independent([Vector([1,0,0]),Vector([2,0,0])]))


# Definition of Gram Schmidt 
def gram_schmidt(vectors):
    orthonormal = []
    for v in vectors : 
        w = v 
        for u in orthonormal : 
            proj = project(w,u)
            w -= proj 
        if w.magnitude()<1e-10:
            continue
        orthonormal.append(w.normalize())
    return orthonormal 
# print(gram_schmidt([Vector([1,0,1]),Vector([2,0,0]),Vector([3,6,8])]))



# Definitions in Numpy 
# import numpy as np 
# a = np.array([1,2,4],dtype=float) 
# b = np.array([1,4,7],dtype=float) 
# c = np.array([[1,2,3],[7,8,9]])
# proj = (np.dot(a,b)/np.dot(b,b))*b 
# q,r = np.linalg.qr(np.random.randn(3,3))
# print(a+b , np.dot(a,b), np.linalg.norm(a),np.linalg.matrix_rank(c),proj)