import heapq
from collections import Counter, namedtuple

# Define a Node structure
class Node(namedtuple("Node", ["char", "freq", "left", "right"])):
    def __lt__(self, other):
        return self.freq < other.freq

# Helper function to build the Huffman Tree
def build_huffman_tree(frequencies):
    heap = [Node(char, freq, None, None) for char, freq in frequencies.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        # Pop the two nodes with the smallest frequency
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        
        # Merge these two nodes
        merged = Node(None, left.freq + right.freq, left, right)
        heapq.heappush(heap, merged)
    
    # The remaining node is the root of the Huffman Tree
    return heap[0]

# Helper function to generate Huffman codes by traversing the tree
def generate_codes(node, prefix="", codebook={}):
    if node is None:
        return
    
    # Leaf node: contains a character
    if node.char is not None:
        codebook[node.char] = prefix
    else:
        # Internal node: traverse left and right
        generate_codes(node.left, prefix + "0", codebook)
        generate_codes(node.right, prefix + "1", codebook)
    
    return codebook

# Main function to implement Huffman Encoding
def huffman_encoding(data):
    if not data:
        return "", {}
    
    # Step 1: Calculate frequency of each character
    frequencies = Counter(data)
    
    # Step 2: Build the Huffman Tree
    huffman_tree = build_huffman_tree(frequencies)
    
    # Step 3: Generate Huffman Codes
    huffman_codes = generate_codes(huffman_tree)
    
    # Step 4: Encode the input data using the generated codes
    encoded_data = ''.join(huffman_codes[char] for char in data)
    
    return encoded_data, huffman_codes

# Function to decode the encoded data using the Huffman Tree
def huffman_decoding(encoded_data, huffman_codes):
    reverse_codes = {v: k for k, v in huffman_codes.items()}
    decoded_data = ""
    code = ""
    
    for bit in encoded_data:
        code += bit
        if code in reverse_codes:
            decoded_data += reverse_codes[code]
            code = ""
    
    return decoded_data

# Example Usage
data = "hello tejas hande"
encoded_data, huffman_codes = huffman_encoding(data)
decoded_data = huffman_decoding(encoded_data, huffman_codes)

print("Original Data:", data)
print("Encoded Data:", encoded_data)
print("Huffman Codes:", huffman_codes)
print("Decoded Data:", decoded_data)
