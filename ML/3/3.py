import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

# Objective function
def objective(x):
    return (x + 3)**2

# Derivative of the objective function
def derivative(x):
    return 2 * (x + 3)

# Gradient Descent using sympy (symbolic computation)
def gradient_descent(expr, alpha, start, max_iter):
    x = sym.symbols('x')
    grad = sym.Derivative(expr, x).doit()  # Compute the gradient (derivative)
    x_val = start
    x_list = [x_val]

    for _ in range(max_iter):
        gradient = grad.subs(x, x_val)  # Evaluate the gradient at x_val
        x_val = x_val - (alpha * gradient)  # Update x based on the gradient and alpha
        x_list.append(x_val)

    return x_list

# Parameters
alpha = 0.1
start = 2
max_iter = 30

# Define the expression
x = sym.symbols('x')
expr = (x + 3)**2

# Perform gradient descent
x_values = gradient_descent(expr, alpha, start, max_iter)

# Create an array of x coordinates for plotting
x_cordinate = np.linspace(-5, 5, 100)

# Plot the objective function
plt.plot(x_cordinate, objective(x_cordinate), label="Objective Function")


# Plot the path taken by gradient descent (red dots)
x_arr = np.array(x_values)
plt.plot(x_arr, objective(x_arr), 'ro-', label="Gradient Descent Path")

# Highlight the starting point
plt.plot(start, objective(start), 'bo', label="Start Point")

# Add labels and title
plt.xlabel("x")
plt.ylabel("Objective Value")
plt.title("Gradient Descent Optimization")
plt.legend()

# Show the plot
plt.show()