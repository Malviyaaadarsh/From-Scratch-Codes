import numpy as np
""" 
Linear Regression is a supervised regression algorithm that aims to model 
relationship between a dependent variable (y) and one or more explanatory independent
variables (X) by fitting a linear equation.
-----------------------------------------------------------------------------------------
Method 1 : Linear Regression using Normal Equation

Normal equation = (X^T * X)^-1 * X^T * y 
It provides solution to find coefficients (weights) that minimize 
the cost function for regression.
(No feature scaling, no learning rate, no iterations , but can be computationally
expensive due to (X^T*X)^-1 operation for large datasets)

------------------------------------------------------------------------------------------

Method 2 : Linear Regression using Gradient Descent
Gradient descent updates the weights by moving in the direction that reduces the error
most quickly (the negative gradient direction). The negative is because we want to
minimize the cost function, going down the slope of the cost function.


Loss Function : Mean Squared Error (MSE) = (1/2m) * Σ(y_i - (h_θ(x_i)))^2
Gradient : ∇J(θ) = (1/m) * X^T * (Xθ - y)
Update Rule : θ = θ - α * ∇J(θ)    
Learning Rate (α) : A hyperparameter that controls step size during the update of weights.
------------------------------------------------------------------------------------------
 """


def lin_reg_normal_eq(X:list[list[float]],y:list[float])->list[float]:
    """
    Linear Regression using Normal Equation
    :param X: 2D list of independent variables (features)
    :param y: 1D list of dependent variable (target)
    :return: list of coefficients (weights)
    """
    try: 
        X_a = np.array(X)  # m x n
        y_a = np.array(y).reshape(-1,1)  # m x 1
        if X_a.ndim != 2 or y_a.ndim != 2:
            return None
        if X_a.shape[0] != y_a.shape[0]:
            return None
        theta = np.linalg.pinv(X_a.T @ X_a) @ (X_a.T @ y_a)    # m >= n for invertibility
        theta = np.round(theta, 4)
        return theta.flatten()   # or theta.tolist() 
    except: 
        return None

def lin_reg_grad_desc(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    """
    Linear Regression using Gradient Descent
    :param X: 2D numpy array of independent variables (features)
    :param y: 1D numpy array of dependent variable (target)
    :param alpha: Learning rate
    :param iterations: Number of iterations for gradient descent
    :return: numpy array of coefficients (weights)
    """
    m, n = X.shape
    theta = np.zeros((n, 1))  
    for _ in range(iterations):
        predictions = X @ theta   # hypothesis xiθi 
        errors = predictions - y  # h_θ(xi) - yi
        gradient = (1/m) * (X.T @ errors)  # gradient ∇J(θ) = (1/m) * X^T * (Xθ - y)
        theta -= alpha * gradient  # Update Rule : θ = θ - α * ∇J(θ)
    return theta



# Sample Output :
# print(lin_reg_normal_eq([[1, 1], [1, 2], [2, 2], [2, 3]], [1, 2, 2, 3]))
# print(lin_reg_grad_desc(np.array([[1, 1], [1, 2], [2, 2], [2, 3]]), np.array([[1], [2], [2], [3]]), alpha=0.01, iterations=1000))


