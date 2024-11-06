# Define an Item class for clarity
class Item:
    def __init__(self, value, weight):
        self.value = value
        self.weight = weight
        self.ratio = value / weight

def fractional_knapsack(items, capacity):
    # Step 1: Sort items by value-to-weight ratio in descending order
    items.sort(key=lambda item: item.ratio, reverse=True)
    
    total_value = 0.0  # Total value accumulated
    for item in items:
        if capacity <= 0:  # Knapsack is full
            break
        if item.weight <= capacity:
            # Take the whole item
            total_value += item.value
            capacity -= item.weight
        else:
            # Take a fraction of the item
            fraction = capacity / item.weight
            total_value += item.value * fraction
            capacity = 0  # Knapsack is now full

    return total_value

# Example usage
items = [
    Item(value=60, weight=10),
    Item(value=100, weight=20),
    Item(value=120, weight=30)
]
capacity = 50

max_value = fractional_knapsack(items, capacity)
print(f"Maximum value in the knapsack = {max_value}")
