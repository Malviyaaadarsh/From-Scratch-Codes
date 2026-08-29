import math,random
random.seed(42)

def mean(data):
    return sum(data) / len(data)

def variance(data,sample=True):
    mu = mean(data)
    squared_diffs = [(x - mu) ** 2 for x in data]
    if sample and len(data) > 1:
        return sum(squared_diffs) / (len(data) - 1)
    else:
        return sum(squared_diffs) / len(data)

def percentile(data,percentile):
    data = sorted(data)
    index = (len(data) - 1) * percentile / 100
    lower_index = math.floor(index)
    upper_index = math.ceil(index)
    if lower_index == upper_index:
        return data[int(index)]
    else:
        weight = index - lower_index
        return data[lower_index] * (1 - weight) + data[upper_index] * weight


def covariance(x,y,sample=True):
    if len(x) != len(y):
        return None
    mu_x = mean(x)
    mu_y = mean(y)
    cov = sum((x[i] - mu_x) * (y[i] - mu_y) for i in range(len(x)))
    if sample and len(x) > 1:
        return cov / (len(x) - 1)
    else:
        return cov / len(x)

def covariance_matrix(data):
    n = len(data)
    c = len(data[0])
    means = [mean(data[i]) for i in range(n)]
    matrix = [[0] * c for _ in range(c)]
    for i in range(c):
        col_i = [row[i] for row in data]
        for j in range(c):
            col_j = [row[j] for row in data]
            matrix[i][j] = covariance(col_i, col_j)
    return matrix

def cohens_d(x,y):
    n_x, n_y = len(x), len(y)
    mean_x, mean_y = mean(x), mean(y)
    var_x, var_y = variance(x), variance(y)
    pooled = math.sqrt(((n_x - 1) * var_x + (n_y - 1) * var_y) / (n_x + n_y - 2))
    if pooled == 0:
        return 0
    return (mean_x - mean_y) / pooled