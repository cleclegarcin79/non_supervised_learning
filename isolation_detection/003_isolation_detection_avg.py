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
    Function to perform isolation detection on a given set of points.
    It randomly selects an axis and a value to filter the points until only one point remains.
    The function returns the number of iterations (loops) it took to isolate the chosen point."""
    
    m_filtered = pts.copy()
    loop_index = 0
    while m_filtered.shape[1] > 1:
        # Random select from axis
        axis = np.random.choice([0, 1], size=1)[0]

        # Random select from the min and max of the axis
        min_val = np.min(m_filtered[axis])
        max_val = np.max(m_filtered[axis])
        random_val = np.random.uniform(min_val, max_val)
        if pts[axis, choosen_point] < random_val:
            m_filtered = m_filtered[:, m_filtered[axis] < random_val]
        else:
            m_filtered = m_filtered[:, m_filtered[axis] > random_val]
        loop_index += 1
    return loop_index

def isolation_detection_average(pts, choosen_point, n=100):
    """
    Function to perform isolation detection multiple times and calculate the average number of iterations (loops) it took to isolate the chosen point.
    It returns a list of mean loops after each iteration."""
    
    loops = []
    mean_loops = []
    for i in range(n):
        loops.append(isolation_detection(pts, choosen_point))
        mean_loops.append(np.mean(loops))
    return mean_loops

# Run isolation detection for the normal point and the anomaly point multiple times and calculate the average number of iterations (loops) it took to isolate the chosen point.
n_iterations = 500
normal_pts_avg = isolation_detection_average(m, CHOOSE, n_iterations)
anomaly_pts_avg = isolation_detection_average(m_noisy, -1, n_iterations)
arange = np.arange(1, n_iterations + 1)

# Plot the results
plt.plot(arange, normal_pts_avg, label='Normal Point', color='blue')
# Add annotation to calculate the average number of iterations (loops) it took to isolate the normal point.
plt.annotate(
    f'{normal_pts_avg[-1]}',
    xy=(arange[-1], normal_pts_avg[-1]),
    xytext=(n_iterations, normal_pts_avg[-1] + 0.1),
    color='blue',
)
plt.plot(arange, anomaly_pts_avg, label='Anomaly Point', color='red')
# Add annotation to calculate the average number of iterations (loops) it took to isolate the anomaly point.
plt.annotate(
    f'{anomaly_pts_avg[-1]}',
    xy=(arange[-1], anomaly_pts_avg[-1]),
    xytext=(n_iterations, anomaly_pts_avg[-1] + 0.1),
    color='red',
)
plt.xlabel('Number of Iterations')
plt.ylabel('Average Number of Loops')
plt.title('Isolation Detection Average Loops')
plt.legend()
plt.grid()
plt.show()