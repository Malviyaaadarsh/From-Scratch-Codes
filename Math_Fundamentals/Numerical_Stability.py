"""
Summary of Notes : 

NaN : Not a number. Specific float value from undefined operation like 0/0, inf-inf,sqrt(-neg no.) etc.
inf : Special float from overflow or division by zero. 
overflow : exceeding max. representational value exp(89) in case of float32. The exceeded value becomes inf.
underflow : result closer to zero than smallest representable pos. no. and becomes zero. For ex. exp(-105) underflows float32
Machine epsilon : Smallest e such that 1.0+e! = 1.0. for float32, its about 1.19e-7
bfloat16 : google's 16 bit format with 8 exp bits and 7 mantissa bits. Preferred for training.
Gradient Clipping : Scaling gradient vector so that its norm does not exceed a threshold. 
Loss Scaling : Multiplying loss by a large constant before backprop so gradient stay in representation range, then dividing by same constant before weight updates.
Log Sum Exp Trick : Computing log9sum(exp(x)) by factoring out exp(max(x)), prevent overflow and underflow.
Stable Softmax : Subtracting max(logits) before exponentiating. No possible overflow and identical results.
Mixed Precision : Using lower precision floats for speed critical operations , and higher for numerically sensitive operations.
"""
import math,struct

# They are not equal 
# print(0.1+0.2==0.3) 

def softmax(logits):
    exps= [math.exp(z) for z in logits]
    total = sum(exps)
    return [e/total for e in exps]

def stable_softmax(logits):
    maxL = max(logits)
    exps = [math.exp(z - maxL) for z in logits]
    total = sum(exps)
    return [e/total for e in exps]

# print(softmax([100.0,101.0,102.0]))
# print(stable_softmax([1000.0,1010.0,1020.0]))


def logsumexp(vals):
    return math.log(sum(math.exp(v) for v in vals))

def logsumexp_stable(vals):
    maxV = max(vals)
    return maxV + math.log(sum(math.exp(v-maxV) for v in vals))

# print(logsumexp([500.0,501.0]))
# print(logsumexp_stable([5000.0,5010.0]))

def cross_entropy(y_true,logits):
    probs = softmax(logits)
    return -math.log(probs[y_true])

def cross_entropy_stable(y_true,logits):
    maxL = max(logits)
    shiftedL = [l - maxL for l in logits]
    log_sum_exp= logsumexp_stable(shiftedL)
    return -shiftedL[y_true] + log_sum_exp

# Mixed Precision 
def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]

# Gradient Clipping
def clip_gradients(gradients, max_norm):
    norm = math.sqrt(sum(g**2 for g in gradients))
    if norm > max_norm:
        scale = max_norm / norm
        return [g * scale for g in gradients]
    return gradients