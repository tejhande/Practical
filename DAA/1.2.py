def fibonacci_recursive(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

# Test the function
n = 10
print(f"Recursive Fibonacci of {n}:", fibonacci_recursive(n))
