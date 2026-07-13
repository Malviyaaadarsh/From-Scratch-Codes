import heapq 
from typing import Dict, Optional, Tuple, List

class HuffmanNode:
    def __init__(self,char: Optional[str], freq: int):
        self.char = char
        self.freq = freq
        self.left: Optional['HuffmanNode'] = None
        self.right: Optional['HuffmanNode'] = None
    def __lt__(self, other: 'HuffmanNode') -> bool:
        return self.freq < other.freq
    
class HuffmanCoding:
    def __init__(self,freq_dict: Dict[str,int]) -> None:
        self.freq_dict=freq_dict
        self.codes = {}
        self.root: Optional[HuffmanNode] = None
    
    def build_tree(self) -> None:
        heap: List[HuffmanNode] = [HuffmanNode(char, freq) for char, freq in self.freq_dict.items()]
        heapq.heapify(heap)
        while len(heap)>1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            merged = HuffmanNode(None, left.freq + right.freq)
            merged.left = left
            merged.right = right
            heapq.heappush(heap, merged)
        self.root = heap[0]
    
    def assign_codes(self):
        self.codes.clear()
        def assign(node: Optional[HuffmanNode], code: str) -> None:
            if node is not None:
                if node.char is not None:
                    self.codes[node.char] = code
                    return
                assign(node.left, code + '0')
                assign(node.right, code + '1')
            else:
                return 
        assign(self.root,"")

    def encode(self, text: str) -> str:
        encoded = "".join(self.codes[char] for char in text)
        return encoded
    
    def decode(self, encoded_text: str) -> str:
        decoded = ""
        current_node = self.root
        for bit in encoded_text:
            if bit == '0':
                current_node = current_node.left
            else:
                current_node = current_node.right
            if current_node.char is not None:
                decoded += current_node.char
                current_node = self.root
        return decoded
    
def main():
    freq_dict = {'A':3,'B':4,'C':6,'D':2,'E':2}
    huffman_coding = HuffmanCoding(freq_dict)
    huffman_coding.build_tree()
    huffman_coding.assign_codes()
    print("Huffman Codes:", huffman_coding.codes)
    sample_text = "AABCCDEEDDEC"
    encoded_text = huffman_coding.encode(sample_text)
    decoded_text = huffman_coding.decode(encoded_text)
    print("Sample Text:", sample_text)
    print("Encoded Text:", encoded_text)
    print("Decoded Text:", decoded_text)

if __name__ == "__main__":
    main()