class HashTable:
    def __init__(self):
        self.collection ={}

    def hash(self,s:str):
        total = 0 
        for c in s:
            total += ord(c)
        return total

    def add(self,k,v):
        val = self.hash(k)
        if val in self.collection:
            self.collection[val][k]=v 
        else:
            self.collection[val]={k:v}

    def remove(self,k):
        val = self.hash(k)
        if val in self.collection and k in self.collection[val]:
            del self.collection[val][k]

    def lookup(self,k):
        val = self.hash(k)
        return self.collection[val][k] if val in self.collection and k in self.collection[val] else None

    def __repr__(self):
        return str(self.collection)