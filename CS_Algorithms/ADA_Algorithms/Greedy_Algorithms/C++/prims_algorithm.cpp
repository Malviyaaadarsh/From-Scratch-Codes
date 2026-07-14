#include<iostream>
#include<vector>
#include<queue>
using namespace std ; 

class Graph{
    int vertices; vector<vector<pair<int,int>>>adjList; 
public: 
    Graph(int v){ vertices=v ; adjList.resize(v); }
    void addEdgeinGraph(int src, int des, int wt){
        adjList[src].push_back({wt,des});
        adjList[des].push_back({wt,src});  
    }
    int getVertices() const{ return vertices; }
    const vector<pair<int,int>>& getNeighbors(int vertex) const{ return adjList[vertex];}
};

class Prim{
    Graph& g; 
public:
    explicit Prim(Graph&graph): g(graph){}
    void findMST(){
        vector<bool>visited(g.getVertices(),false);
        using Edge = pair<int,pair<int,int>>; 
        priority_queue<Edge,vector<Edge>,greater<Edge>>min_heap;
        min_heap.push({0,{0,-1}});
        int cost = 0 ; 
        while(!min_heap.empty()){
            Edge curr = min_heap.top(); min_heap.pop(); 
            int wt = curr.first; int vertex = curr.second.first; int par = curr.second.second;
            if(visited[vertex])continue; 
            visited[vertex]=true; cost+= wt; 
            if(par!=-1)cout<<par<<" "<<vertex<<" "<<wt<<endl;
            for(const auto& neighbor : g.getNeighbors(vertex)){
                int w = neighbor.first; int v1 = neighbor.second; 
                if(!visited[v1]){min_heap.push({w,{v1,vertex}});}
            }
        }
        cout<<cost<<endl ; 
    }
};

int main(){
    Graph g(5); 
    g.addEdgeinGraph(0,1,2); g.addEdgeinGraph(0,3,4);
    g.addEdgeinGraph(1,2,5); g.addEdgeinGraph(2,4,1); 
    g.addEdgeinGraph(3,4,2); g.addEdgeinGraph(2,3,6); 

    Prim p(g); 
    p.findMST(); return 0 ; 
}