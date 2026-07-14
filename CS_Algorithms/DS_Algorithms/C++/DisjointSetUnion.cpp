#include<iostream>
#include<algorithm>
#include<vector>
#include<numeric>
using namespace std ; 

class DisjointSetUnion{
    vector<int>parent,rank;
public:
    explicit DisjointSetUnion(int vertices){
        parent.resize(vertices); rank.resize(vertices,0); 
        iota(parent.begin(),parent.end(),0); // same as for(int i=0 ; i<vertices; i++)parent[i]=i;
    }

    int find(int vertex){
        if(parent[vertex]==vertex)return vertex;
        return parent[vertex]=find(parent[vertex]); // Compressing the path
    }

    bool unionSet(int v1,int v2){
        int r1 = find(v1); int r2 = find(v2); 
        if(r1==r2)return false; 
        else if(rank[r1]<rank[r2])parent[r1]=r2; 
        else{
            parent[r2]=r1; rank[r1]++; 
        }
        return true ; 
    }

    bool isconnected(int v1,int v2){ return find(v1)==find(v2); }
};

int main(){
    DisjointSetUnion dsu(5); // [{0}, {1}, {2}, {3}, {4}]
    dsu.unionSet(4, 1); // [{0}, {1, 4}, {2}, {3}, {4}]
    dsu.unionSet(1,3); // [{0}, {1, 4, 3}, {2}]
    cout << dsu.isconnected(1, 3) << endl;
    cout << dsu.isconnected(4, 2) << endl; 
    return 0;
}

