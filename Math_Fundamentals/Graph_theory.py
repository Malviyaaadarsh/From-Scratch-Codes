import numpy as np 
from collections import deque
from sklearn.cluster import KMeans

class Graph: 
    def __init__(self,n, directed=False): 
        self.n = n 
        self.directed = directed 
        self.adj = {i : {} for i in range(n)}

    def add_edge(self,u,v,w=1.0):
        self.adj[u][v] = w 
        if not self.directed: 
            self.adj[v][u] = w

    def neighbors(self,u):
        return list(self.adj[u].keys())

    def deg(self,u):
        return len(self.adj[u])

    def adjacency_matrix(self):
        matrix = [[0]*self.n for _ in range(self.n)]
        for u in range(self.n):
            for v, w in self.adj[u].items():
                matrix[u][v] = w
        return matrix

    def degree_matrix(self):
        matrix = [[0]*self.n for _ in range(self.n)]
        for u in range(self.n):
            matrix[u][u] = self.deg(u)
        return matrix

    def laplacian_matrix(self):
        d = np.array(self.degree_matrix())
        a = np.array(self.adjacency_matrix())
        return d-a

def bfs(graph, start):
    visited = set()
    visited.add(start)
    queue = deque([(start,0)])
    path = []
    distances = {}
    while queue:
        node,dist = queue.popleft()
        path.append(node)
        distances[node] = dist
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return path, distances

def dfs(graph, start):
    visited = set()
    path = []
    stack = [start]
    while stack:
        node = stack.pop()
        if node in visited: 
            continue
        visited.add(node)
        path.append(node)
        for neighbour in reversed(graph.neighbors(node)):
            if neighbour not in visited:
                stack.append(neighbour)
    return path

def connected_components(graph):
    visited = set()
    components = []
    for node in range(graph.n):
        if node not in visited:
            component = dfs(graph, node)
            components.append(component)
            visited.update(component)
    return components

def spectral_clustering(graph,k=2):
    l=graph.laplacian_matrix()
    eigenvals, eigenvecs = np.linalg.eigh(l) # l is symmetric,so eigh
    features = eigenvecs[:,1:k+1] # skip the first eigenvector
    kmeans = KMeans(n_clusters=k, n_init=10).fit(features)
    return kmeans.labels_

def message_passing(graph,features,weight_matrix):
    a = graph.adjacency_matrix()
    row_sum = np.sum(a,axis=1,keepdims=True)
    row_sum[row_sum==0] = 1
    normalized_a = a / row_sum
    aggregated_features = normalized_a @ features
    updated_features = aggregated_features @ weight_matrix
    return updated_features



    
    