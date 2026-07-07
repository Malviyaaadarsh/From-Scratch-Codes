"""
Summary of Notes : 

Vector : Ordered list of numbers. (Point in high dmensional space)
Matrix : A linear transformation. Maps vector from one space to another. 
Transpose : Swap rows and columns. 
Matrix Multiplication : Dot product between every row of first matrix with every column of second matrix.
Determinant : determines how much the matrix scales the area(2D) or volume(3D) 
Inverse : Matrix that reverses the transformation. Does not exist when determinant is zero.
Identity : Used as 1 in matrices. Useful in residual connections. 
Broadcasting : Stretching smaller array to a larger one by repeating along missing dimension. 
"""
# Matrix Definition with operations 
class Matrix:
    def __init__(self,data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows,self.cols)
    @staticmethod
    def identity(n):
        return Matrix([[1 if i==j else 0 for j in range(n)]for i in range(n)])
    def __repr__(self):
        rows_as_str = "\n ".join(str(row) for row in self.data)
        return f"Matrix - ({self.shape}) : \n {rows_as_str}"
    def __add__(self,other):
        return Matrix([[self.data[i][j] + other.data[i][j] for j in range(self.cols)] for i in range(self.rows)])
    def __sub__(self,other):
        return Matrix([[self.data[i][j] - other.data[i][j] for j in range(self.cols)] for i in range(self.rows)])
    def multiply_scalar(self,scalar):
        return Matrix([[self.data[i][j] * scalar for j in range(self.cols)] for i in range(self.rows)])
    def multiply_element_wise(self,other):
        return Matrix([[self.data[i][j] * other.data[i][j] for j in range(self.cols)] for i in range(self.rows)])
    def matmul(self,other):
        return Matrix([[sum(self.data[i][k] * other.data[k][j] for k in range(self.cols)) for j in range(other.cols)]
                        for i in range(self.rows)])
    def transpose(self):
        return Matrix([[self.data[j][i] for j in range(self.rows)] for i in range(self.cols)])
    def determinant(self):
        if self.shape == (1,1):
            return self.data[0][0]
        if self.shape == (2,2):
            return self.data[0][0]*self.data[1][1] - self.data[0][1]*self.data[1][0]
        det = 0 
        for j in range(self.cols):
            minor = Matrix([[self.data[i][k] for k in range(self.cols) if k!=j] for i in range(1,self.rows)])
            det += ((-1)**j)*self.data[0][j]*minor.determinant()
        return det 
    def inverse_2d(self):
        det = self.determinant() 
        if det == 0 : 
            raise ValueError("No Inverse Exists")
        return Matrix([[self.data[1][1]/det, -self.data[0][1]/det],[-self.data[1][0]/det,self.data[0][0]/det]])

# Sample Example : 
# a = Matrix([[1,2],[2,4]])
# b = Matrix([[3,6],[8,9]])
# i = Matrix.identity(3)
# print(a.data , (a+b).data, a.matmul(b).data, a.determinant(),b.inverse_2d().data, i.data)



# Usage connection to Neural Networks 
import random 
inputs = Matrix([[0.5],[0.4],[0.3]])
weights = Matrix([[random.uniform(-1,1) for _ in range(3)]])
bias = Matrix([[0.1],[0.1]])
def relu_matrix(mat):
    return Matrix([[max(0,val) for val in row] for row in mat.data])
pre_activation = weights.matmul(inputs) + bias    # Forward
output = relu_matrix(pre_activation)

# print(inputs.shape,output.shape,weights.shape , bias.shape, output.data)


# Numpy Usage 
import numpy as np 
A = np.array([[1,2],[3,4]])
B = np.array([[2,4],[5,6]])
# print(A+B , A*B , A@B , A.T , np.linalg.inv(A), np.linalg.det(B), np.eye(3))
i = np.random.randn(3,1)
w = np.random.randn(2,3)
b = np.array([[0.1],[0.1]])
o = np.maximum(0,w@i + b )
# print(o)
