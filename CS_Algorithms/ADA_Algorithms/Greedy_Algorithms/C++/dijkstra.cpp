#include<iostream>
#include<limits>
#include<vector>
#include<algorithm>
#include<queue>
using namespace std; 
using pii = pair<int,int>;

class Graph{
    int V; vector<vector<pii>> adjList; 
public : 
    explicit Graph(int V): V(V){adjList.resize(V);}

    void addEdge(int src, int dest, int wt){
        adjList[src].push_back({dest, wt});
        adjList[dest].push_back({src, wt});
    }

    int getV() const { return V; }

    const vector<pii>& getNeighbors(int v) const{
        return adjList[v];
    }
};

class Dijkstra{
    const Graph& graph;
public:
    explicit Dijkstra(const Graph& g): graph(g){}
    pair<vector<int>,vector<int>>shortestPath(int src){
        const int inf = numeric_limits<int>::max();
        vector<int>dist(graph.getV(), inf);
        vector<int>parent(graph.getV(), -1);
        priority_queue<pii, vector<pii>, greater<pii>> pq;
        dist[src] = 0; pq.push({0, src});
        while(!pq.empty()){
            auto currentNode = pq.top(); pq.pop();
            int currentDist = currentNode.first;
            int currentV = currentNode.second;
            if(currentDist > dist[currentV]) continue;
            for(const auto& neighbor : graph.getNeighbors(currentV)){
                int nextV = neighbor.first;
                int weight = neighbor.second;
                int newDist = currentDist + weight;
                if(newDist < dist[nextV]){
                    dist[nextV] = newDist;
                    parent[nextV] = currentV;
                    pq.push({newDist, nextV});
                }
            }
        }
        return {dist, parent};
    }

    vector<int> reconstructPath(vector<int>&parent,int dest){
        vector<int>path; 
        if(parent[dest] == -1) return path;
        while(dest != -1){
            path.push_back(dest);
            dest = parent[dest];
        }
        reverse(path.begin(), path.end());
        return path;
    }
};


int main(){
    Graph g(7);
    g.addEdge(0,1,3); g.addEdge(1,2,7); 
    g.addEdge(2,4,15); g.addEdge(4,5,6); 
    g.addEdge(5,6,9); g.addEdge(6,1,14); 
    g.addEdge(1,3,9); g.addEdge(3,4,11); 
    g.addEdge(2,3,10); g.addEdge(6,3,2); 
 
    Dijkstra dijkstra(g);
    int src = 0 ; auto path = dijkstra.shortestPath(src);
    auto dist = path.first; auto parent = path.second;

    for(int i=0; i<dist.size(); i++){
        cout<<i << " "<< dist[i] << endl;
    }
    // Reconstructing path from source (0) to destination (5)
    int dest = 5;
    auto reconstructedPath = dijkstra.reconstructPath(parent, dest);
    for(int v : reconstructedPath){ cout << v << " ";}
    cout << endl; return 0 ; 
}