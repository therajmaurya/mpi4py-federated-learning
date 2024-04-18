import numpy as np
import matplotlib.pyplot as plt


# Define the function
def f(x):
    return x**3 - 3*x**2 - x + 3


# Generate x values
x = np.linspace(-5, 5, 400)

# Generate y values
y = f(x)

# Plot the function
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='f(x) = x^3 - 3x^2 - x + 3')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Cubic Polynomial Function')
plt.grid(True)
plt.legend()
plt.show()
