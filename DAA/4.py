def knapsack_0_1(values, weights, W):
    n = len(values)
    # Create a DP table to store maximum value at each sub-capacity up to W
    dp = [[0] * (W + 1) for _ in range(n + 1)]
    
    # Build the DP table
    for i in range(1, n + 1):
        for w in range(1, W + 1):
            if weights[i - 1] <= w:
                # Include the item or exclude it
                dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
            else:
                # Cannot include the item
                dp[i][w] = dp[i - 1][w]
    
    return dp[n][W]

# Example usage
values = [60, 100, 120]
weights = [10, 20, 30]
W = 50

max_value = knapsack_0_1(values, weights, W)
print(f"Maximum value in the knapsack = {max_value}")
