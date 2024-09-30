import numpy as np
import matplotlib.pyplot as plt

"""
Lager funksjon
"""

def log_reg(wt, x, w0):

    lin_dec_bound = (wt * x + w0)
    
    y = 1 / (1 +np.exp(-lin_dec_bound))

    return y

# Step 1: Generate the data
import numpy as np

# Define mean vectors and identity covariance matrices
mu1 = [0, 0]
mu2 = [1, 1]
cov = [[1, 0], [0, 1]]  # Identity matrix

# Generate 5 random points for each class
class1_points = np.random.multivariate_normal(mu1, cov, 5)
class2_points = np.random.multivariate_normal(mu2, cov, 5)

# Step 2: Plot the data
import matplotlib.pyplot as plt

# Plot points for class 1
plt.scatter(class1_points[:, 0], class1_points[:, 1], color='blue', label='Class 1')
# Plot points for class 2
plt.scatter(class2_points[:, 0], class2_points[:, 1], color='red', label='Class 2')

# Step 3: Define a linear decision boundary
# We will use an arbitrary boundary, e.g., y = -x + 0.5
x_vals = np.linspace(-2, 3, 100)  # Range of x values for the line
y_vals = -x_vals + 0.5  # Arbitrary slope and intercept

# Step 4: Plot the decision boundary
plt.plot(x_vals, y_vals, label='Decision Boundary', color='green')

# Add labels and legend
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()


