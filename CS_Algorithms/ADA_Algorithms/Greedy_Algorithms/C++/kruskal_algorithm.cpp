#include<iostream>
#include<algorithm>
#include<vector>
using namespace std  ; 

class Edge{
public : 
    int src,dest,wt; 
    bool operator<(const Edge& other) const{ return wt < other.wt;}
};

class DisjointSet{
    vector<int>parent,rank; 
public: 
    explicit DisjointSet(int vertices){
        parent.resize(vertices); rank.resize(vertices,0); 
        for(int i=0 ; i<vertices; i++)parent[i]=i; 
    }

    int find(int vertex){
        if(parent[vertex]!=vertex)parent[vertex]=find(parent[vertex]);
        return parent[vertex]; 
    }

    bool U(int v1, int v2){
        int r1 = find(v1); int r2 = find(v2); 
        if(r1==r2)return false; 
        else if(rank[r1]<rank[r2])parent[r1]=r2; 
        else{
            parent[r2]=r1; rank[r1]++; 
        }
        return true ; 
    }
};

class Graph{
    int vertices ; vector<Edge>edges; 
public:
    explicit Graph(int vertices){this->vertices= vertices;}

    void addEdgeinGraph(int src,int dest,int wt){
        edges.push_back({src,dest,wt}); 
    }
    
    void findMST(){
        sort(edges.begin(),edges.end()); 
        DisjointSet ds(vertices); int cost = 0; 
        for(const Edge& e : edges){
            if(ds.U(e.src,e.dest)){
            cout<<e.src<<" "<<e.dest<<" "<<e.wt<<endl;   cost+=e.wt; }
        }
        cout<< cost << endl ; 
    }
}; 


int main(){
    Graph g(5); 
    g.addEdgeinGraph(0,1,2); g.addEdgeinGraph(0,3,4);
    g.addEdgeinGraph(1,2,5); g.addEdgeinGraph(2,4,1); 
    g.addEdgeinGraph(3,4,2); g.addEdgeinGraph(2,3,6); 

    g.findMST(); 
}