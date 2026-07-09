"""
Summary of Notes : 
Chain Rule : Derivative of composed functions equals product of each function's local derivative.
Computational Graph : Directed acyclic graph where nodes are opearations, edges carry values(forward) or gradients(backward). 
Forward Mode : Autodiff that propagates derivatives from inputs to outputs. Efficient with dual number. 
Reverse Mode : Autodiff that propagates gradients from outputs to inputs.
Autograd : System that records operations on values, builds a graph, and computes exact gradients via the chain rule.
Dual Number : a+b*epsilon where epsilon^2=0. Carry derivative information through arithmetic rules.
Topological sort : Ordering graph nodes so every node comes after all its dependencies. Required for correct gradient propagation.
Gradient Accumulation : Current gradient for a feeded value is sum of all incoming gradient contribution. Adding not Replacing. 
Dynamic Graph : Computation graph rebuilt on every forward pass, allowing Python control flow inside models.
Gradient Checking : Comparing Autodiff against numerical derivatives.
"""

import random,math
# Definition of Value Class (stores numeric data,gradient=0,backward function,pointers to children)
class Value:

    def __init__(self,data,children=(),op=''):
        self.data = data
        self.grad = 0.0 
        self._backward = lambda : None 
        self._op = op 
        self._prev = set(children)

    def __repr__(self):
        return f"Value(data = {self.data:.4f} , grad = {self.grad:.4f})"
    
    def __neg__(self):
        return Value(-self.data,(self,),'-')
    
    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        upstream = Value(self.data + other.data,(self,other),'+')
        def _backward():
            self.grad += upstream.grad
            other.grad += upstream.grad
        upstream._backward = _backward
        return upstream 
    
    def __radd__(self,other):
        return self+other
    
    def __sub__(self,other):
        return self + (-other)
    
    def __rsub__(self,other):
        return other + (-self)
    
    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        upstream = Value(self.data * other.data,(self,other),'*')
        def _backward():
            self.grad += other.data*upstream.grad
            other.grad += self.data*upstream.grad
        upstream._backward = _backward
        return upstream
    
    def __rmul__(self,other):
        return other*self 
    
    def __truediv__(self, other):
        return self * (other ** -1) if isinstance(other,Value) else self * (Value(other)**-1)
    
    def __pow__(self,n):
        upstream = Value(self.data**n , (self,),f'**{n}')
        def _backward():
            self.grad += n*(self.data**(n-1))*upstream.grad
        upstream._backward = _backward 
        return upstream
    
    def exp(self):
        upstream = Value(math.exp(self.data),(self,),'exp')
        def _backward():
            self.grad += math.exp(self.data)*upstream.grad
        upstream._backward = _backward 
        return upstream 
    
    def log(self):
        upstream = Value(math.log(self.data),(self,),'log')
        def _backward():
            self.grad += (1.0/self.data)*upstream.grad 
        upstream._backward = _backward 
        return upstream 
    
    def tanh(self):
        upstream = Value(math.tanh(self.data),(self,),'tanh')
        def _backward():
            self.grad += (1-(math.tanh(self.data)**2))*upstream.grad 
        upstream._backward= _backward 
        return upstream 
    
    def relu(self):
        upstream = Value(max(0,self.data),(self,),'relu')
        def _backward():
            self.grad += (1.0 if upstream.data >0 else 0.0) * upstream.grad
        upstream._backward = _backward 
        return upstream 

    def backward(self):
        topo=[]
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0 
        for v in reversed(topo):
            v._backward()

# Definition of Neuron  

class Neuron:
    def __init__(self,n_inputs):
        self.w = [Value(random.uniform(-1,1)) for _ in range(n_inputs)]
        self.b = Value(0.0)

    def __call__(self,x):
        pred = sum((wi*xi for wi,xi in zip(self.w,x)),self.b)
        return pred.tanh()
    
    def params(self):
        return self.w + [self.b]
    
# Definition of layer 

class Layer : 
    def __init__(self,n_inputs,n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self,x):
        return [n(x) for n in self.neurons]
    
    def params(self):
        return [p for n in self.neurons for p in n.params()]
    

class MLP : 
    def __init__(self,sizes):
        self.layers = [Layer(sizes[i],sizes[i+1]) for i in range(len(sizes)-1)]

    def __call__(self,x):
        for l in self.layers:
            x = l(x) 
        return x[0] if len(x)==1 else x 

    def params(self):
        return [p for l in self.layers for p in l.params()]
    
    
# Training on XOR 
random.seed(42)
model = MLP([2,4,1]) #  2I,4H,1O
xs = [[0,0],[0,1],[1,0],[1,1]]
ys = [-1,1,1,-1]    # Replaced 0 with -1 for tanh 

for epoch in range(101):
    preds = [model(x) for x in xs]
    loss = sum((pred-y)**2 for pred,y in zip(preds,ys))
    for p in model.params():
        p.grad = 0.0
    loss.backward()
    lr = 0.05 
    for p in model.params():
        p.data -= lr*p.grad
    if epoch%20 == 0:
        print(epoch,loss.data)

for x,y in zip(xs,ys):
    print(x,y,model(x).data)



# Gradient Checking 
def grad_check(expr,xval,h=1e-7):
    x = Value(xval)
    y= expr(x)
    y.backward()
    autodiff_grad = x.grad 
    y_pos = expr(Value(xval+h)).data
    y_neg = expr(Value(xval-h)).data
    numerical_grad = (y_pos-y_neg)/(2*h)
    diff = abs(autodiff_grad-numerical_grad)
    return autodiff_grad,numerical_grad,diff

# Sample Expr (Testing)
def expr(x):
    return (x**3 + x*2 + 6).tanh()

print(grad_check(expr,0.5))
