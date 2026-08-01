#include<iostream>
#include<algorithm>
#include<vector>
using namespace std;

class Item {
public: 
    int index ; int value ; int weight ; 
    Item(const int& idx,int val,int wt):index(idx),value(val),weight(wt) {}
};

class Knapsack {
    int capacity ; vector<Item> itemlist ; 
public : 
    Knapsack(int cap,vector<Item> items):capacity(cap),itemlist(items) {}

    // Helper Function for 0-1 Knapsack Problem using Recursion
    int recursiveKnapsack(int index, int remainingCapacity) {
        if(index==itemlist.size() || remainingCapacity==0)return 0 ; 
        int includeCurrent = 0 ;
        int excludeCurrent = recursiveKnapsack(index+1,remainingCapacity);
        if(itemlist[index].weight <= remainingCapacity) {
            includeCurrent = itemlist[index].value + recursiveKnapsack(index+1,remainingCapacity-itemlist[index].weight);
        }
        return max(includeCurrent, excludeCurrent);
    }

    // Helper Function for 0-1 Knapsack Problem using Memoization
    int memoizedKnapsack(int index,int remainingCapacity, vector<vector<int>>& dp) {
        if(index==itemlist.size() || remainingCapacity==0)return 0 ;
        if(dp[index][remainingCapacity]!=-1)return dp[index][remainingCapacity] ;
        int includeCurrent = 0 ;
        int excludeCurrent = memoizedKnapsack(index+1,remainingCapacity,dp);
        if(itemlist[index].weight <= remainingCapacity) {
            includeCurrent = itemlist[index].value + memoizedKnapsack(index+1,remainingCapacity-itemlist[index].weight,dp); 
        }
        return dp[index][remainingCapacity] = max(includeCurrent, excludeCurrent);
    }

    int initializeMemoization() {
        vector<vector<int>> dp(itemlist.size(),vector<int>(capacity+1,-1));
        return memoizedKnapsack(0,capacity,dp);
    }

    // Helper Function for 0-1 Knapsack Problem using Tabulation
    int tabulateKnapsack() {
        int n = itemlist.size();
        vector<vector<int>> dp(n+1,vector<int>(capacity+1,0));
        for(int i=1 ; i<=n ; i++){
            for(int w=0 ; w<= capacity; w++){
                if(itemlist[i-1].weight<=w)dp[i][w] = max(itemlist[i-1].value + dp[i-1][w-itemlist[i-1].weight], dp[i-1][w]);
                else dp[i][w] = dp[i-1][w] ;
            }
        }
        return dp[n][capacity];
    }

    // Helper Function for 0-1 Knapsack Problem with Space Optimization
    int optimizeSpace() {
        vector<int> dp(capacity+1,0);
        for(const auto& item : itemlist) {
            for(int w=capacity ; w>=item.weight ; w--){
                dp[w] = max(dp[w], item.value + dp[w-item.weight]);
            }
        }
        return dp[capacity];
    }
};



int main(){
    vector<Item> items = {{0,25,18}, {1,24,15}, {2,15,10}};
    int m = 20 ; int n = items.size() ; // m = capacity of knapsack, n = number of items
    Knapsack k(m,items); 
    cout << "Maximization Result (Recursive): " << k.recursiveKnapsack(0, m) << endl;
    cout << "Maximization Result (Memoized): " << k.initializeMemoization() << endl;
    cout << "Maximization Result (Tabulated): " << k.tabulateKnapsack() << endl;
    cout << "Maximization Result (Space Optimized): " << k.optimizeSpace() << endl;
}