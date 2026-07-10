"""
Summary of Notes : 

Sample Space : set of every possible outcome 
PMF : function that give exact probabilities for discrete outcomes.
PDF : Density function for continuous variables, integrated over an interval.
Conditional Probability : Probability of something  given something. Foundation for bayes theorem.
Independence : Probability of one doesn't affect other. 
Expected Value : Probability weighted sum of all outcomes.
Variance : Expected Square deviation from mean. 
Central Limit Theorem : Meanof many independent  samples converges to normal distribution.
Joint Probability : P(X,Y) 
Marginal Probability : Recovers one variable's distribution from joint. P(X)= sum(y in P(X,Y))
Log Probability : log(P(X)). Product -> Sum. Prevent Numerical Overflow.
Cross Entropy : -sum(p_true*log(p_pred)) measures how different two distributions are. 
Logits : Raw scores by model 
Softmax : Maps logits to valid probability distributions.
Sampling : Generating values acc. to probability distribution.
Prior : P(hypothesis) before observing evidence. Regularization in ML.
Posterior : P(hypothesis|evidence). Updated Belief.
Likelihood : P(evidence|hypothesis). 
Evidence : P(data) across hypotheses. 
Naive Bayes : Classifier that assume features are independent given the class.
Laplace Smoothing : Adding small count to ever feature to prevent zero probability. 
MLE : Choose param that maximizes P(data|param)
MAP : Choose param that maximizes P(param|data)
False Positive : Wrong Alarm. Test-> Positive. Actual -> Negative.
"""

# Basic Probability Functions 
import math,random

def fact(n):
    res = 1 
    for i in range(2,n+1):
        res*=i 
    return res 

def combination(n,k):
    return fact(n)/(fact(k)*fact(n-k))

def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(r, lam):
    return (lam ** r) * math.exp(-lam) / fact(r)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exp = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exp)

def expectation(vals,probs):
    return sum(xi*pi for xi,pi in zip(vals,probs))

def var(vals,probs):
    mu =  expectation(vals,probs)
    return sum(p*(v-mu)**2 for v,p in zip(vals,probs))

def sample_bernoulli(p,n=1):
    return [1 if random.random() <p else 0 for _ in range(n)]

def sample_categorical(probs,n=1):
    cumulative = []
    tot = 0 
    for p in probs:
        tot+=p 
        cumulative.append(tot)
    samples=[]
    for _ in range(n):
        r = random.random()
        for i,c in enumerate(cumulative):
            if r<=c:
                samples.append(i)
            break
    return samples

def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]


# In Numpy 
import numpy as np 
from scipy import stats 
from scipy.special import softmax,log_softmax
normal = stats.norm(loc=0,scale=1)
samples=  normal.rvs(size=10000)
logits = np.array([1.0,0.9,3.0,2.5])
probs = softmax(logits)
log_probs = log_softmax(logits)
# print(np.mean(samples),np.std(samples),normal.cdf(1.5))
# print(probs,log_probs,np.sum(probs))



# Bayes Theorem 
def bayes(prior,likelihood,fp_rate):
    evidence = likelihood * prior + fp_rate*(1-prior)
    posterior = likelihood*prior / evidence
    return posterior
# print(bayes(0.0001,0.99,0.01))

# Naive Bayes Classifier from scratch 
from collections import defaultdict

class NaiveBayesClassifier:

    def __init__(self,smoothing=1.0):
        self.smoothing = smoothing
        self.vocab = set()
        self.class_counts = defaultdict(int)
        self.word_counts = defaultdict(lambda:defaultdict(int))
        self.class_word_total = defaultdict(int)
    
    def train(self,docs,labels):
        for doc,label in zip(docs,labels):
            self.class_counts[label]+=1
            words = doc.lower().split()
            for word in words:
                self.word_counts[label][word] += 1
                self.class_word_total[label] += 1
                self.vocab.add(word)

    def predict(self,doc):
        words = doc.lower().split()
        total_docs = sum(self.class_counts.values())
        vocab_size= len(self.vocab)
        best_class= None
        best_score= float("-inf")
        for c in self.class_counts:
            score = math.log(self.class_counts[c]/total_docs)
            for word in words:
                cnt = self.word_counts[c].get(word,0)
                tot = self.class_word_total[c]
                score+=math.log((cnt+self.smoothing)/(tot+self.smoothing*vocab_size))
            if score>best_score:
                best_score = score 
                best_class = c
        return best_class


#testing 
train_data = ["Get $10000 for free by opening account.Free Reward",
              "You won a prize worth 1 Lakh. Reedem Now",
              "Check the festial offers now.",
              "Subscribe to our newsletter now and get 25 percent off for 1 year",
              "Discover Best product deals at Instamart. Join Now.",
              "Congrats!, you are selected for prize of 10000 as bonus. Accept invite and Join now.",
              "Discover best prices. See our catalog.",
              "You're selected for grand lottery of 20000. Check Now.",
              "Choose our monthly plans for tension free shopping."]

train_labels = ["spam","spam","ham","ham","ham","spam","ham","spam","ham"]

classifier = NaiveBayesClassifier()
classifier.train(train_data,train_labels)

test_data = ["Congrats!, grab your monthly reward.",
             "You won special prizes for your entries. Check Now.",
             "See all available buying options. Know More."]

for msg in test_data:
    print(classifier.predict(msg))

def show_top_words(classifier, cls, n=5):
    vocab_size = len(classifier.vocab)
    total = classifier.class_word_total[cls]
    probs = {}
    for word in classifier.vocab:
        count = classifier.word_counts[cls].get(word, 0)
        probs[word] = (count + classifier.smoothing) / (total + classifier.smoothing * vocab_size)
    sorted_words = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    for word, prob in sorted_words[:n]:
        print(f"    {word}: {prob:.4f}")
print("\nTop spam words:")
show_top_words(classifier, "spam")
print("\nTop ham words:")
show_top_words(classifier, "ham")


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_data)
nb = MultinomialNB()
nb.fit(X_train, train_labels)

X_test = vectorizer.transform(test_data)
predictions = nb.predict(X_test)
for msg, pred in zip(test_data, predictions):
    print(f"  '{msg}' -> {pred}")