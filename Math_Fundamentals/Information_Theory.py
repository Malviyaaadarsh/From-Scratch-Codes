"""
Summary Of Notes : 

Information : No. of bits/nats needed to encode event. = -log(p)
Entropy : Avg. surprises across outcomes of distributions.It measure irreducible uncertainity.
Cross Entropy : Avg. surprises when using model distribution Q to encode events of tru event P. Negative Log Likelihood for one hot labels. 
KL Divergence : Cross Entropy - Entropy , tells extra bits wasted by Q insted of P.
Mutual Info. : Reduction in uncertainity about X from knowing Y.
Perplexity : Exponential of cross entropy. 
Bits/Nats : Info. measured with log base 2 / info. measured with napiers log 
"""

# Definition of Information and Entropy 
import math,random 

def information(p,base=2):
    if p<=0 or p>1:
        return float('inf') if p<=0 else 0.0
    return -math.log(p)/math.log(base)

def entropy(probs,base=2):
    return sum(p*information(p,base) for p in probs if p>0 )
 
def cross_entropy(p,q,base=2):
    total = 0.0 
    for pi,qi in zip(p,q):
        if pi>0 : 
            if qi<=0 : 
                return float('inf')
            total+= pi*(-math.log(qi)/math.log(base)) 
    return total 

def kl_divergence(p,q,base=2):
    return cross_entropy(p,q,base)-entropy(p,base)

def softmax(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def cross_entropy_loss(true_class, logits):
    probs = softmax(logits)
    return -math.log(probs[true_class])

