import heapq
from typing import List, Tuple

class Graph:
    def __init__(self,vertices:int):
        self.V = vertices
        self.adjList = [[] for _ in range(vertices)]
    def addEdge(self,src:int,dest:int,wt:int):
        self.adjList[src].append((dest,wt))
        self.adjList[dest].append((src,wt))

class Dijkstra:
    def __init__(self,graph:Graph):
        self.graph = graph

    def shortestPath(self,src:int):
        dist = [float('inf')]*self.graph.V
        parent = [-1]*self.graph.V
        dist[src] = 0
        pq = [(0,src)]
        while pq:
            currentDist, currentV = heapq.heappop(pq)
            if currentDist > dist[currentV]:
                continue
            for neighbor, weight in self.graph.adjList[currentV]:
                distance = currentDist + weight
                if distance < dist[neighbor]:
                    dist[neighbor] = distance
                    parent[neighbor] = currentV
                    heapq.heappush(pq, (distance, neighbor))

        return dist, parent

    def reconstuctPath(self,parent:int,dest:int):
        path = []
        while dest != -1:
            path.append(dest)
            dest = parent[dest]
        return path[::-1]

def main():
    g = Graph(7);
    g.addEdge(0,1,3); g.addEdge(1,2,7); 
    g.addEdge(2,4,15); g.addEdge(4,5,6); 
    g.addEdge(5,6,9); g.addEdge(6,1,14); 
    g.addEdge(1,3,9); g.addEdge(3,4,11); 
    g.addEdge(2,3,10); g.addEdge(6,3,2); 
    dijkstra = Dijkstra(g)
    dist,parent = dijkstra.shortestPath(0)
    for v in range(g.V):
        print(f"Vertex : {v} : {dist[v]}")

    print("Path from 0 to 5:", dijkstra.reconstuctPath(parent,5))

if __name__ == "__main__":
    main()

