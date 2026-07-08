"""
Summary of Notes : 

Derivative : Rate of change of function at a point. How much output change per unit change in input.
Partial Derivative : Derivative w.r.t one variable keeping other constant.
Gradient : Vector of all partial derivatives.Pointss in direction of steepest ascent. 
Gradient Descent : Negative of gradient added to parameters to reduce loss in neural network training.
Learning Rate : Scalar that control how big each gradient step is. Diverge if large.Slowly Converge if small.
Chain Rule : Rule for differentiation composite functions.
Jacobian : Matrix of all partial derivatees of outputs w.r.t inputs.
Hessian : Matrix of second order partial derivativees. Tells Curvature of a funcn. 
Integral : Accumulation of quantity. Tells Area under curve.
Taylor Series : Polynomial approximation. Approximating a function near a point using its derivatives.
"""
# Definition of Derivative
def derivative(f,x,h=1e-7):
    return (f(x+h)-f(x-h))/2*h

# def f(x):
#     return x**2
# x = 3
# print(derivative(f,x))



# Definition of gradient 
def gradient(f,points,h=1e-7):
    grad=[]
    for i in range(len(points)):
        point_to_add,point_to_subtract = list(points),list(points)
        point_to_add[i]+=h 
        point_to_subtract[i]-=h 
        partial_derivative = (f(point_to_add)-f(point_to_subtract))/(2*h)
        grad.append(partial_derivative) 
    return grad 

# def f(points):
#     x,y = points
#     return x**2 + y**2 + x*y*2
# print(gradient(f,[2.0,2.0]))


# Definition of Gradient Descent 
points = [10.0,4.0]
lr = 0.2
def f(points):
    x,y = points
    return x**3 + y**3
for step in range(21):
    grad = gradient(f,points)
    points = [p - lr*g for p,g in zip(points,grad)]
    loss = f(points)
    if step%4==0:
        pass
        # print(step,points,loss)


# Definition of 2D Hessian Matrix 
def hessian(f,x,y,h=1e-7):
    fxx = (f(x+h,y)-2*f(x,y)+f(x-h,y))/h**2
    fyy = (f(x,y+h)-2*f(x,y)+f(x,y-h))/h**2
    fxy = (f(x+h,y+h)-f(x-h,y+h)-f(x+h,y-h)+f(x-h,y-h))
    return [[fxx,fxy],[fxy,fyy]]

# def saddle(x,y):return x**2 - y**2 
# def bowl(x,y):return x**2 + y**2 
# print(hessian(saddle,0.0,0.0),hessian(bowl,0.0,0.0))


# Definition of Taylor Approximation 
import math 
def taylor_approximation(f,f1,f2,x0,h,order=2):
    result = f(x0)
    if order>=1:result+=f1(x0)+h
    if order>=2:result+= 0.5*f2(x0)*h**2
    return result 
x0 = 0.0 
for h in [0.1,0.5,1.0]:
    val = math.sin(h)
    taylor = taylor_approximation(lambda x : math.sin(x),lambda x : math.cos(x),lambda x : -math.sin(x),x0,h,order=1)
    # print(val,taylor)




# Pattern for Gradient based Training Loop 
import random 
random.seed(42)
w,b = random.gauss(0,1),random.gauss(0,1)
lr = 0.01 
xs = [1.0,2.0,3.0,4.0]
ys = [2.0,4.0,6.0,8.0]
for epoch in range(101):
    total_loss = 0 
    dw = db = 0 
    for x,y in zip(xs,ys):
        pred = w*x + b              # Prediction
        error = pred - y            
        total_loss+= error**2       # Compute Loss 
        dw += 2*x*error 
        db += 2*error
    dw/=len(xs)                     # Compute Gradients
    db/=len(xs)
    total_loss /= len(xs)
    w-= lr*dw                      # Update Weights 
    b-=lr*db
    if epoch%10==0:
        pass
        #print(epoch,w,b,total_loss)


# Numpy Implementation 
import numpy as np 
x = np.array([1,2,3,4],dtype=float)
y = np.array([2,4,6,8],dtype=float)
w,b = np.random.randn(),np.random.randn()
lr = 0.01 
for epoch in range(101):
    pred = w*x+b 
    error = pred-y 
    loss = np.mean(error**2)
    dw = np.mean(2*x*error)
    db = np.mean(2*error)
    w-= lr*dw 
    b-= lr*db 
    #print(epoch,w,b)

