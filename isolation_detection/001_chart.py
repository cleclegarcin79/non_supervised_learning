import numpy as np
import matplotlib.pyplot as plt

# fix seed for reproducibility
np.random.seed(seed=15)

# Generate random data points from a normal distribution
m = np.random.normal(loc=10, scale=2, size=(2, 100))

# Add an anomaly point to the dataset
m_noisy = np.append(m, np.array((4, 14)).reshape(2, 1), axis=1)

# Plot the data points
plt.plot(m[0], m[1], '.', color= 'black', alpha=0.5)
# Plot the anomaly point in red
plt.plot(m_noisy[0, -1], m_noisy[1, -1], '.', color='red', alpha=0.8)

# Plot an example normal point in blue
choose=75
plt.plot(m[0, choose], m[1, choose], '.', color='blue', alpha=0.8)
plt.grid()
plt.show()