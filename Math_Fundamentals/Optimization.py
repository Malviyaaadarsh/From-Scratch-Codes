"""
Summary of Notes : 

Learning Rate : Scalar that control how far each update moves with weight. Too small -> more Computation , too large -> divergence.
Gradient Descent : update weights by subtracting (grad*learning_rate)
SGD : Compute gradient on random subset insted of full dataset. 
Mini-Batch GD : Small Subset of data(30-250 samples) used to estimate gradient. Most reliable.
Momentum : Accumulating past gradients into a velocity vector. Keep Rollin ! 
ADAM : Adaptive Moment Estimation : Per weight running averages of gradient and squared gradients to give each weights its own adaptive learning rate.
Bias Correction : Adam moments initialized to 0. Bias Correction divides by (1/beta^t) to compensate early steps. Fixin the cold start! 
Learning Rate Schedule : Large steps early , smaller steps late. 
Convex function : local -> minima = global minima. one minima. Gd always got it. NN functs aren't convex. 
Saddle Point : Point where grad is 0. Neither minima nor maxima. Common in multidimensions. 
Convergence : Further steps don't meaningfully reduce loss. 
Loss Landscape : Loss function plotted over weighted space. 
"""

# Definition of Vanilla GD 
class GradientDescent: 
    def __init__(self,lr=0.001):
        self.lr = lr
    def step(self,params,grads):
        return [param-self.lr*grad for param,grad in zip(params,grads)]

# Definition of SGD with momentum 0.9 
class SGD:
    def __init__(self,lr=0.001,momentum=0.9):
        self.lr = lr 
        self.momentum=momentum 
        self.velocity = None 
    def step(self,params,grads):
        if self.velocity is None:
            self.velocity = [0.0]*len(params)
        self.velocity = [self.momentum*v + g for v,g in zip(self.velocity,grads)]
        return [param-self.lr*velocity for param,velocity in zip(params,self.velocity)]
    
# Definition of Adam Optimizer with the two moments 
class Adam:
    def __init__(self,lr=0.001,beta1=0.9,beta2=0.999,epsilon=1e-8):
        self.lr = lr 
        self.b1 = beta1 
        self.b2 = beta2 
        self.m= None 
        self.v = None 
        self.t = 0 
        self.epsilon = epsilon 
    def step(self,params,grads):
        if self.m is None:
            self.m=[0.0]*len(params)
            self.v = [0.0]*len(params)
        self.t+=1 
        self.m = [self.b1*m + (1-self.b1)*g for m,g in zip(self.m,grads)]
        self.v = [self.b2*v + (1-self.b2)*g**2 for v,g in zip(self.v,grads)]
        m_hat = [m/(1-self.b1**self.t) for m in self.m]
        v_hat = [v/(1-self.b2**self.t) for v in self.v]
        return [param-self.lr*m_h/(v_h**0.5+self.epsilon) for param,m_h,v_h in zip(params,m_hat,v_hat)]
    
# Definiton of RosenBrock Benchmark 
def rosenbrock(params):
    x,y = params
    return (1-x)**2 + 100 * (y-x**2)**2
def rosenbrock_grad(params):
    x,y=params
    df_dx = -2*(1-x) + 200 *(y-x**2)*(-2*x)
    df_dy = 200*(y-x**2)
    return [df_dx,df_dy]

# Let's Use our optimizers 
def optimize(optimizer,func,func_grad,start,steps=3000):
    params=list(start)
    history = [params[:]]
    for _ in range(steps):
        grads = func_grad(params)
        params = optimizer.step(params, grads)
        history.append(params[:])
    return history

# start = [-1.0, 1.0]
# gd = optimize(GradientDescent(lr=0.0005), rosenbrock, rosenbrock_grad, start)
# sgd = optimize(SGD(lr=0.0001, momentum=0.9), rosenbrock, rosenbrock_grad, start)
# adam = optimize(Adam(lr=0.01), rosenbrock, rosenbrock_grad, start)

# for name, history in [("GD", gd), ("SGD+M", sgd), ("Adam", adam)]:
#     final = history[-1]
#     loss = rosenbrock(final)
#     print(f"{name:6s} : x={final[0]:.6f}, y={final[1]:.6f}, loss={loss:.8f}")

# Pytorch Implementation 
import torch
model = torch.nn.Linear(784, 10)
sgd = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
adam = torch.optim.Adam(model.parameters(), lr=0.001)
adamw = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(adam, T_max=100)
