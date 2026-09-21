import numpy as np
import matplotlib.pyplot as plt

# fix seed for reproducibility
np.random.seed(seed=15)

# Generate random data points from a normal distribution
m = np.random.normal(loc=10, scale=2, size=(2, 100))

# Add an anomaly point to the dataset
m_noisy = np.append(m, np.array((4, 14)).reshape(2, 1), axis=1)

# Define the index of the normal point
CHOOSE=75


def isolation_detection(pts, choosen_point):
    """
    Perform isolation detection on a given set of points.
    It randomly selects an axis and a value to filter the points until only one point remains.
    The function returns the number of iterations (loops) it took to isolate the chosen point.
    Additionally, it returns the cuts made during the isolation process for visualization purposes.
    """
    
    m_filtered = pts.copy()
    loop_index = 0
    cuts = []
    x_min = pts[0].min()
    x_max = pts[0].max()
    y_min = pts[1].min()
    y_max = pts[1].max()
    while m_filtered.shape[1] > 1:
        # Random select from axis
        axis = np.random.choice([0, 1], size=1)[0]

        # Random select from the min and max of the axis
        min_val = np.min(m_filtered[axis])
        max_val = np.max(m_filtered[axis])
        random_val = np.random.uniform(min_val, max_val)

        if pts[axis, choosen_point] < random_val:
            m_filtered = m_filtered[:, m_filtered[axis] < random_val]
            if axis == 0:
                cuts.append((axis, (random_val, (y_min, y_max))))
                x_max = random_val
            if axis == 1:
                cuts.append((axis, (random_val, (x_min, x_max))))
                y_max = random_val
        else:
            m_filtered = m_filtered[:, m_filtered[axis] > random_val]
            if axis == 0:
                cuts.append((axis, (random_val, (y_min, y_max))))
                x_min = random_val
            if axis == 1:
                cuts.append((axis, (random_val, (x_min, x_max))))
                y_min = random_val
        loop_index += 1
    return loop_index, cuts

# Run isolation detection for the normal point and the anomaly point
normal_pt_detection, normal_pt_cuts = isolation_detection(m_noisy, CHOOSE)
anomaly_pts_detection, anomaly_pts_cuts = isolation_detection(m_noisy, -1)

print(f'Normal point detection took {normal_pt_detection} iterations.')
print(f'Anomaly point detection took {anomaly_pts_detection} iterations.')
### Observations:
# Il faut moins d'itérations pour isoler le point d'anomalie que le point normal, ce qui est 
# conforme avec le papier "Isolation Forest" de Liu et al. (2008).

# Plots the cuts for the normal point and anomaly point
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot the data points
ax[0].plot(m[0], m[1], '.', color= 'black', alpha=0.5)
ax[1].plot(m[0], m[1], '.', color= 'black', alpha=0.5)

# Plot the anomaly point in red
ax[1].plot(m_noisy[0, -1], m_noisy[1, -1], '.', color='red', alpha=0.8)

# Plot an example normal point in blue
choose=75
ax[0].plot(m[0, choose], m[1, choose], '.', color='blue', alpha=0.8)

# Plot the cuts for the normal point
for cut in normal_pt_cuts:
    axis, (value, (min_val, max_val)) = cut
    if axis == 0:
        ax[0].vlines(x=value, ymin=min_val, ymax=max_val, alpha=0.5)
    else:
        ax[0].hlines(y=value, xmin=min_val, xmax=max_val, alpha=0.5)

# Plot the cuts for the anomaly point
for cut in anomaly_pts_cuts:
    axis, (value, (min_val, max_val)) = cut
    if axis == 0:
        ax[1].vlines(x=value, ymin=min_val, ymax=max_val, alpha=0.5)
    else:
        ax[1].hlines(y=value, xmin=min_val, xmax=max_val, alpha=0.5)

ax[0].set_title('Normal Point Cuts')
ax[1].set_title('Anomaly Point Cuts')

ax[0].grid()
ax[1].grid()
plt.show()
