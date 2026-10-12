from sklearn.datasets import fetch_kddcup99
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

# Load the KDD Cup 1999 dataset filtered for the 'http' subset
subset = 'http'
data, target = fetch_kddcup99(subset=subset, return_X_y=True, as_frame=True, percent10=False)

# Data Prep
target_encoded = (target != b"normal.").astype(np.int32)
data['is_attack'] = target_encoded

# Display the first few rows of the dataset
data.head()

# Display the shape of the dataset
data.shape
### Observations:
# Le dataset contient bien 567 498 lignes et 3 colonnes, ce qui est conforme avec le papier.

# Display the distribution of the target variable
target_encoded.value_counts() / target_encoded.shape[0]
### Observations:
# Nous observons un nombre d'attaques de l'ordre de 0.4% ce qui est conforme avec la table 2 du papier.

# Sampling the data to reduce the number of points for visualization
sampled_data = data.sample(n=10000, random_state=42)

# Plotting the sampled data points for the 'duration', 'src_bytes', and 'dst_bytes' features 2 by 2
columns = ['duration', 'src_bytes', 'dst_bytes']
tuple_columns = list(combinations(columns, 2))
fig, axs = plt.subplots(len(tuple_columns), 1, figsize=(12, 10))
for i, (col1, col2) in enumerate(tuple_columns):
    ax = axs[i]
    # Plotting the normal points
    ax.scatter(sampled_data[sampled_data['is_attack'] == 0][col1], 
               sampled_data[sampled_data['is_attack'] == 0][col2], 
               color='blue', label='Normal', alpha=0.3)
    # Plotting the anomaly points
    ax.scatter(sampled_data[sampled_data['is_attack'] == 1][col1], 
               sampled_data[sampled_data['is_attack'] == 1][col2], 
               color='red', label='Anomaly', alpha=0.3)
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    ax.set_title(f'KDD Cup 1999 on ({col1}, {col2} subset)')
    ax.legend()
plt.show()
