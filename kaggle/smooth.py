import numpy as np
import matplotlib.pyplot as plt

# Generate data for a smooth loss function (e.g., quadratic)
x = np.linspace(-3, 3, 400)
smooth_loss = x**2

# Generate data for a non-smooth loss function (e.g., absolute loss)
non_smooth_loss = np.abs(x)

# Plotting the smooth and non-smooth loss functions
plt.figure(figsize=(12, 6))

# Plot smooth loss function
plt.subplot(1, 2, 1)
plt.plot(x, smooth_loss, label="Smooth Loss (MSE)", color='blue')
plt.title("Smooth Loss Function")
plt.xlabel("Weight values")
plt.ylabel("Loss")
plt.grid(True)

# Plot non-smooth loss function
plt.subplot(1, 2, 2)
plt.plot(x, non_smooth_loss, label="Non-Smooth Loss (MAE)", color='red')
plt.title("Non-Smooth Loss Function")
plt.xlabel("Weight values")
plt.grid(True)

plt.tight_layout()
plt.show()
