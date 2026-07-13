#include<iostream>
#include<string>
#include<queue>
#include<unordered_map>
using namespace std; 

class HuffmanNode{
public:  char ch; int freq ; HuffmanNode *left,*right;
HuffmanNode(char c,int f): ch(c),freq(f),left(NULL),right(NULL){}
}; 

class Compare{
public: 
    bool operator()(HuffmanNode* a,HuffmanNode* b){
        return a->freq > b->freq; }
};

class HuffmanCoding{
    HuffmanNode* root; unordered_map<char,string>codes;
public: 
    HuffmanCoding(){root=nullptr;}

    void build_tree(const unordered_map<char,int>& freq_map){
        priority_queue<HuffmanNode*,vector<HuffmanNode*>,Compare>min_heap;
        for(auto& pair: freq_map)min_heap.push(new HuffmanNode(pair.first,pair.second));
        while(min_heap.size()>1){
            HuffmanNode* left = min_heap.top(); min_heap.pop();
            HuffmanNode* right = min_heap.top(); min_heap.pop();
            HuffmanNode* merged = new HuffmanNode('\0',left->freq+right->freq);
            merged->left=left; merged->right=right;
            min_heap.push(merged);
        }
        root = min_heap.top();
    }

    void assign_codes(){
        codes.clear();
        assign_codes_helper(root,"");
    }
    void assign_codes_helper(HuffmanNode*node,string code){
        if(!node)return; 
        if(node->ch != '\0'){
            codes[node->ch]=code; return;
        }
        assign_codes_helper(node->left,code+'0');
        assign_codes_helper(node->right,code+'1');
    }

    string encode(const string& text){
        string encoded; 
        for(char ch: text)encoded+=codes[ch];
        return encoded; 
    }

    string decode(const string& encoded){
        string decoded; 
        HuffmanNode* current = root; 
        for(char bit: encoded){
            if(bit=='0')current=current->left;
            else current=current->right;
            if(current->ch != '\0'){
                decoded+=current->ch; 
                current=root; 
            }
        }
        return decoded; 
    }

    void print_codes(){
        for(auto& pair: codes)cout<<pair.first<<": "<<pair.second<<endl; 
    }
};
int main() 
{ 
     unordered_map<char,int> freq_map = {{'A',3},{'B',4},{'C',6},{'D',2},{'E',2}};
     HuffmanCoding huffman; 
     huffman.build_tree(freq_map);
     huffman.assign_codes();
     cout<<"Huffman Codes: "<<endl;
     huffman.print_codes();
     string text = "AABCCDEEDDEC";
     string encoded = huffman.encode(text);  cout<<"Encoded Text: "<<encoded<<endl; 
     string decoded = huffman.decode(encoded); cout<<"Decoded Text: "<<decoded<<endl;
}