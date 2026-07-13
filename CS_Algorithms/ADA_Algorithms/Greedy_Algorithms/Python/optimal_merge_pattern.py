import heapq 
from typing import List

from sympy import sec 

class OptimalMergePattern:
    def __init__(self,file_sizes:List[int]):
        self.file_sizes=file_sizes.copy() # Set of files with diff. sizes 

    def minimum_cost(self)->int:
        if(len(self.file_sizes)<2):
            return 0
        min_heap = self.file_sizes.copy()
        heapq.heapify(min_heap)

        cost = 0 
        while len(min_heap)>1:
            first_smallest = heapq.heappop(min_heap)
            second_smallest = heapq.heappop(min_heap)
            merged_size = first_smallest + second_smallest
            cost+=merged_size
            heapq.heappush(min_heap,merged_size)
        return cost 
    
def main():
    file_sizes = [10,55,32,8,24]
    optimal_cost_finder = OptimalMergePattern(file_sizes)
    print(f"Set of Files with Sizes : {file_sizes} \n Optimal minimum merge cost : {optimal_cost_finder.minimum_cost()}")

if __name__ =="__main__":
    main()