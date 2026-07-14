from typing import List,Tuple,Dict,Any 

class Graph:
    def __init__(self,vertices : int):
        self.vertices = vertices
        self.edges : List[Tuple[int,int,int]]=[]
    
    def add_edge_in_graph(self,src:int,dest:int,wt:int):
        self.edges.append((src,dest,wt))

class DisjointSet:
    def __init__(self,vertices:int)->None:
        self.parent = list(range(vertices))
        self.rank = [0] * vertices

    def find(self,vertex:int)->int:
        if self.parent[vertex] != vertex:
            self.parent[vertex] = self.find(self.parent[vertex])
        return self.parent[vertex]
    
    def union(self,v1:int,v2:int)->bool:
        r1 = self.find(v1)
        r2 = self.find(v2)
        if r1 == r2:
            return False 
        if self.rank[r1] < self.rank[r2]:
            self.parent[r1] = r2
        elif self.rank[r1] > self.rank[r2]:
            self.parent[r2] = r1
        else:
            self.parent[r2]=r1
            self.rank[r1] += 1
        return True
    
class Kruskal:
    def __init__(self,graph:Graph):
        self.graph = graph
    
    def find_mst(self):
        ds = DisjointSet(self.graph.vertices)
        mst = []
        cost = 0 
        self.graph.edges.sort(key = lambda x: x[2])
        for src,dest,wt in self.graph.edges:
            if ds.union(src,dest):
                mst.append((src,dest,wt))
                cost+=wt
        return mst, cost


def main():
    g = Graph(5)
    g.add_edge_in_graph(0,1,2)
    g.add_edge_in_graph(0,3,4)
    g.add_edge_in_graph(1,2,5)
    g.add_edge_in_graph(2,4,1)
    g.add_edge_in_graph(3,4,2)
    g.add_edge_in_graph(2,3,6)
    k = Kruskal(g)
    mst, cost = k.find_mst()
    for src,dest,wt in mst:
        print(f"{src} - {dest} => {wt}")
    print(cost)

if __name__ == "__main__":
    main()