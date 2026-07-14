from typing import List, Tuple, Any, Dict 
import heapq 

class Graph:
    def __init__(self, vertices: int):
        self.vertices = vertices 
        self.graph = [[] for _ in range(vertices)]
        
    def add_edge_in_graph(self, src: int, dest: int, wt: int):
        self.graph[src].append((dest, wt))
        self.graph[dest].append((src, wt))
    
class Prim:
    def __init__(self,graph:Graph):
        self.graph=graph 
    
    def find_mst(self):
        visited = [False]*self.graph.vertices
        mst=[]
        cost = 0 
        min_heap=[]
        heapq.heappush(min_heap,(0,0,-1)) # (wt,v2,v1)
        while min_heap:
            wt,vertex,parent = heapq.heappop(min_heap)
            if visited[vertex]:
                continue
            visited[vertex] = True  
            cost += wt 
            if parent != -1:
                mst.append((parent,vertex,wt))
            for neighbor,wt in self.graph.graph[vertex]:
                if not visited[neighbor]:
                    heapq.heappush(min_heap,(wt,neighbor,vertex))
        return mst,cost

def main():
    g = Graph(5)
    g.add_edge_in_graph(0,1,2)
    g.add_edge_in_graph(0,3,4)
    g.add_edge_in_graph(1,2,5)
    g.add_edge_in_graph(2,4,1)
    g.add_edge_in_graph(3,4,2)
    g.add_edge_in_graph(2,3,6)
    p = Prim(g)
    mst, cost = p.find_mst()
    for src,dest,wt in mst:
        print(f"{src} - {dest} => {wt}")
    print(cost)

if __name__ == "__main__":
    main()