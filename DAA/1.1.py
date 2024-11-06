def fibonacci_iterative(n):
    if n <= 0:
        return []  # Return an empty list for invalid input
    elif n == 1:
        return [0]  # Return just [0] for the first Fibonacci number
    
    series = [0, 1]  # Start the series with the first two Fibonacci numbers
    a, b = 0, 1
    for _ in range(2, n):
        a, b = b, a + b
        series.append(b)  # Add the next Fibonacci number to the series
    
    return series

# Test the function
n = int(input("How many nTerms:- "))
print(f"Fibonacci series up to {n}:", fibonacci_iterative(n))
