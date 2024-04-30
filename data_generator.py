"""
Write script to generate the dataset to be used for the modelling purpose.
This data generation is generated in a way that would help us bring forward
the properties that we want to study in federated learning.
"""
import csv
import numpy as np
import matplotlib.pyplot as plt


def generate_data_points():
    np.random.seed(42)  # Set seed for reproducibility
    # Generate 500 linear data points
    linear_xs = np.linspace(-5, -3, 500)
    linear_ys = f(linear_xs) + np.random.normal(0, 20, 500)
    linear_points = list(zip(linear_xs, linear_ys))

    # Generate 500 polynomial data points
    polynomial_xs = np.linspace(-5, 5, 500)
    polynomial_ys = f(polynomial_xs) + np.random.normal(20, 15, 500)
    polynomial_points = list(zip(polynomial_xs, polynomial_ys))

    # Generate 500 polynomial data points
    polynomial_xs = np.linspace(-5, 5, 500)
    polynomial_ys = f(polynomial_xs) + np.random.normal(-20, 15, 500)
    polynomial_points2 = list(zip(polynomial_xs, polynomial_ys))

    # Combine the data points
    data_points = []
    for i in range(500):
        data_points.append(linear_points[i])
        data_points.append(polynomial_points[i])
        data_points.append(polynomial_points2[i])
    return data_points


def f(x):
    return x**3 - 3*x**2 - x + 3


data = generate_data_points()

# Save data to CSV file
with open('training_data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['x', 'y'])
    for point in data:
        writer.writerow(point)

linear_xs = [point[0] for i, point in enumerate(data) if i % 3 == 0]
linear_ys = [point[1] for i, point in enumerate(data) if i % 3 == 0]

polynomial_xs = [point[0] for i, point in enumerate(data) if i % 3 == 1]
polynomial_ys = [point[1] for i, point in enumerate(data) if i % 3 == 1]

polynomial_xs2 = [point[0] for i, point in enumerate(data) if i % 3 == 2]
polynomial_ys2 = [point[1] for i, point in enumerate(data) if i % 3 == 2]

# Plotting the polynomial curve
x_values = np.linspace(-5, 5, 100)
y_values = f(x_values)  # Polynomial function

plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, color='blue', label='Actual Function Curve')


def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')


# Plotting linear points and polynomial points
plt.scatter(linear_xs, linear_ys, color='red', label='Linear Points')
# plt.plot(linear_xs, linear_ys, color='red', label='Linear Points Curve')
# x_av, y_av = movingaverage(linear_xs, 10), linear_ys
# plt.plot(x_av, y_av)

plt.scatter(polynomial_xs, polynomial_ys, color='green', label='Polynomial Points')
# plt.plot(polynomial_xs, polynomial_ys, color='green', label='Polynomial Points Curve')

plt.scatter(polynomial_xs2, polynomial_ys2, color='black', label='Polynomial Points Type 2')
# plt.plot(polynomial_xs2, polynomial_ys2, color='black', label='Polynomial Points Type 2 Curve')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear and Polynomial Data Points')
plt.legend()
plt.grid(True)
plt.show()
