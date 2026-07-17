from typing import List 

class Item:
    def __init__(self,index:int,value:float,weight:float):
        self.index = index
        self.value = value
        self.weight = weight
    @property
    def ratio(self):
        return self.value / self.weight
    
class FractionalKnapsack:

    def __init__(self,capacity:float, items:List[Item]):
        self.capacity = capacity
        self.items = items
    def maximize_profit(self):
        sorted_items = sorted(self.items, key=lambda x: x.ratio, reverse=True)
        remaining_capacity = self.capacity
        total_profit = 0.0
        selected_items = []
        for item in sorted_items:
            if remaining_capacity <= 0:
                break
            if item.weight <= remaining_capacity:
                selected_items.append((item.index,1.0,item.value))
                total_profit += item.value
                remaining_capacity -= item.weight
            else:
                fraction = remaining_capacity / item.weight
                selected_items.append((item.index,fraction,fraction*item.value))
                total_profit += fraction * item.value
                remaining_capacity = 0
        return total_profit, selected_items
    
def main():
    items=[Item(1,5,50),Item(2,15,80),Item(3,35,90)]
    capacity=45
    profit, selected_items = FractionalKnapsack(capacity, items).maximize_profit()
    print(f"Total Profit: {profit}")
    print("Selected Items:")
    for index, fraction, item_profit in selected_items:
        print(f"Item {index}: {fraction:.2f} (Profit: {item_profit:.2f})")  

if __name__ == "__main__":
    main()
