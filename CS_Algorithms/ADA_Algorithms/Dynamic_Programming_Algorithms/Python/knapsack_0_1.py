from typing import List 

class Item: 
    def __init__(self, index:int, value: int , weight: int):
        self.index = index 
        self.value = value 
        self.weight = weight

class Knapsack:
    def __init__(self,capacity:int, itemlist:List[Item]):
        self.capacity = capacity
        self.itemlist = itemlist

    # 0/1 Knapsack by Recursion
    def recursiveKnapsack(self,index:int, capacity:int):
        if(index==len(self.itemlist) or capacity==0):
            return 0
        includeCurrent = 0 
        excludeCurrent = self.recursiveKnapsack(index+1, capacity)
        if(self.itemlist[index].weight <= capacity):
            includeCurrent = self.itemlist[index].value + self.recursiveKnapsack(index+1, capacity-self.itemlist[index].weight)
        return max(includeCurrent, excludeCurrent)

    # 0/1 Knapsack by Memoization
    def memoizedKnapsack(self):
        n = len(self.itemlist)
        dp = [[-1]*(self.capacity+1) for _ in range(n)]
        def helper(index:int,remainingCapacity:int):
            if(index == n or remainingCapacity == 0):
                return 0
            if(dp[index][remainingCapacity] != -1):
                return dp[index][remainingCapacity]
            includeCurrent = 0
            excludeCurrent = helper(index+1, remainingCapacity)
            if(self.itemlist[index].weight <= remainingCapacity):
                includeCurrent = self.itemlist[index].value + helper(index+1, remainingCapacity - self.itemlist[index].weight)
            dp[index][remainingCapacity] = max(includeCurrent, excludeCurrent)
            return dp[index][remainingCapacity]
        return helper(0, self.capacity)

    # 0/1 Knapsack by Tabulation
    def tabulatedKnapsack(self):
        n = len(self.itemlist)
        dp = [[0]*(self.capacity+1) for _ in range(n+1)]
        for i in range(1, n+1):
            for w in range(1, self.capacity+1):
                if(self.itemlist[i-1].weight <= w):
                    dp[i][w] = max(self.itemlist[i-1].value + dp[i-1][w-self.itemlist[i-1].weight], dp[i-1][w])
                else:
                    dp[i][w] = dp[i-1][w]
        return dp[n][self.capacity]

    # 0/1 Knapsack with Space Optimization
    def optimizeSpace(self):
        dp = [0]*(self.capacity+1)
        for i in self.itemlist:
            for w in range(self.capacity, i.weight-1, -1):
                dp[w] = max(dp[w], i.value + dp[w-i.weight])
        return dp[self.capacity]


def main():
    items = [Item(0,25,18),Item(1,24,15),Item(2,15,10)]
    capacity = 20
    k = Knapsack(capacity, items)
    print("Maximization Result (Recursive):", k.recursiveKnapsack(0, capacity))
    print("Maximization Result (Memoized):", k.memoizedKnapsack())
    print("Maximization Result (Tabulated):", k.tabulatedKnapsack())
    print("Maximization Result (Space Optimized):", k.optimizeSpace())

if __name__ == "__main__":
    main()