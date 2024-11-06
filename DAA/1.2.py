def fibonacci_recursive_series(n, series=None):
    if series is None:
        series = [0, 1]  # Start with the first two Fibonacci numbers (0 and 1)

    # Base case for recursion
    if len(series) == n:
        return series
    
    # Recursive case: calculate Fibonacci for n and append to series
    next_fib = series[-1] + series[-2]
    series.append(next_fib)

    return fibonacci_recursive_series(n, series)

# Test the function
n = int(input("How many nTerms:- "))
print(f"Fibonacci series up to {n}:", fibonacci_recursive_series(n))
