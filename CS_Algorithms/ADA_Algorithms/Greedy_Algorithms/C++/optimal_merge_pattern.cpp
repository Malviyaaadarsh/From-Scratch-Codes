#include <iostream>
#include <vector>
#include <queue>
using namespace std ; 

class OptimalMergePattern{
vector<int>file_sizes; 
public: 
explicit OptimalMergePattern(const vector<int>&files): file_sizes(files){}
int minimum_cost(){
    if(file_sizes.size()<2)return 0;
    priority_queue<int,vector<int>,greater<int>>min_heap(file_sizes.begin(),file_sizes.end());
    int cost = 0 ;
    while(min_heap.size()>1){
        int first_smallest = min_heap.top(); min_heap.pop(); 
        int second_smallest = min_heap.top(); min_heap.pop(); 
        int merged_size= first_smallest+second_smallest; 
        cost+=merged_size;
        min_heap.push(merged_size); 
    }

    return cost; 
}
};

int main(){
    vector<int>file_sizes = {10,55,32,8,24}; 
    OptimalMergePattern optimal_cost_finder(file_sizes); 
    cout<<"Set of File Sizes : "; 
    for(int size : file_sizes)cout<<size<<" "; cout << endl ; 
    cout<<"Minimum optimal Merge Cost : "<<optimal_cost_finder.minimum_cost()<<endl ; 
}