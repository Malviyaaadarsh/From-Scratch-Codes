#include<iostream>
#include<iomanip>
#include<algorithm>
#include<vector>
using namespace std;

class Item {
public:
    string index; double value; double weight;
    Item(const string&idx,double wt,double val):index(idx),weight(wt),value(val){}
    double ratio() const{ return value/weight; }
};

class FractionalKnapsack {
public:
    double knapsack_capacity; vector<Item>items; 
    FractionalKnapsack(double knapsack_capacity, const vector<Item>& items): knapsack_capacity(knapsack_capacity), items(items) {}
    void maximize_profit(){
        sort(items.begin(),items.end(),[](const Item&a,const Item&b){ return a.ratio()>b.ratio(); });
        double total_value = 0; double remaining_capacity = knapsack_capacity;
        for(const auto& item: items){
            if(remaining_capacity<=0) break;
            if(item.weight<=remaining_capacity){
                total_value += item.value;
                remaining_capacity -= item.weight;
                cout << "Taking full item: " << item.index << " (Value: " << item.value << ", Weight: " << item.weight << ")\n";
            } else {
                double fraction = remaining_capacity / item.weight;
                total_value += item.value * fraction;
                cout << "Taking fraction of item: " << item.index << " (Value: " << item.value * fraction << ", Weight: " << remaining_capacity << ")\n";
                remaining_capacity = 0; 
            }  
    }
    cout<< total_value << endl;
}
};

int main(){
    vector<Item>items={Item("A",5,50),Item("B",15,80),Item("C",35,90)};
    FractionalKnapsack fk(45, items);
    fk.maximize_profit();
}