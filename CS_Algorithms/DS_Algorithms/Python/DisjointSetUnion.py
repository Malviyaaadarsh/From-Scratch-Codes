class DisjoinSetUnion: 

    def __init__(self,vertices:int)->None:
        self.parent = list(range(vertices))
        self.rank = [0]*vertices
    
    def find(self,vertex:int)->int:
        if self.parent[vertex] != vertex:
            self.parent[vertex] = self.find(self.parent[vertex])
        return self.parent[vertex]
    
    def union(self,v1:int,v2:int)->bool:
        r1 = self.find(v1)
        r2 = self.find(v2)
        if r1 == r2:
            return False 
        if self.rank[r1] < self.rank[r2]:
            self.parent[r1] = r2
        elif self.rank[r1] > self.rank[r2]:
            self.parent[r2] = r1
        else:
            self.parent[r2]=r1
            self.rank[r1] += 1
        return True
    
    def isconnected(self,v1:int,v2:int)->bool:
        return self.find(v1) == self.find(v2)
    

def main():
    dsu = DisjoinSetUnion(5)
    dsu.union(4,1)
    dsu.union(1,3)
    print(dsu.isconnected(1,3))
    print(dsu.isconnected(4,2))

if __name__ == "__main__":
    main()