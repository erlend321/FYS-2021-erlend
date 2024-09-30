import numpy as np
import matplotlib.pyplot as plt

# Creating sample data for each case
x = np.linspace(0, 10, 100)

# Case 1: Perfect fit (R² = 1)
y_perfect = 2 * x + 1  # A perfect linear relationship

# Case 2: No fit (R² = 0)
y_no_fit = np.random.normal(5, 2, len(x))  # Completely random data

# Case 3: Worse than mean (Negative R²)
y_negative_fit = -2 * x + np.random.normal(0, 2, len(x))  # Inverse trend with noise

# Plotting the three cases
plt.figure(figsize=(8, 6))

# Perfect fit plot
plt.subplot(3, 1, 1)
plt.scatter(x, y_perfect, color='blue', label='Perfect fit: R² = 1')
plt.plot(x, y_perfect, color='red')
plt.title('Perfect Fit (R² = 1)')
plt.legend()

# No fit plot
plt.subplot(3, 1, 2)
plt.scatter(x, y_no_fit, color='green', label='No fit: R² = 0')
plt.plot(x, np.ones_like(x) * np.mean(y_no_fit), color='red', label='Mean line')
plt.title('No Fit (R² = 0)')
plt.legend()

# Negative fit plot
plt.subplot(3, 1, 3)
plt.scatter(x, y_negative_fit, color='purple', label='Worse than mean: R² < 0')
plt.plot(x, -2 * x, color='red', label='Regression line')
plt.title('Worse than Mean (Negative R²)')
plt.legend()

plt.tight_layout()
plt.show()
