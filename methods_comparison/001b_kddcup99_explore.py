from sklearn.datasets import fetch_kddcup99
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.decomposition import PCA

# Load the KDD Cup 1999 dataset filtered for the 'http' subset
subset = 'http' # or 'smtp'
data, target = fetch_kddcup99(subset=subset, return_X_y=True, as_frame=True, percent10=True)

# Data Prep
target_encoded = (target != b"normal.").astype(np.int32)
data['is_attack'] = target_encoded

# Sampling the data to reduce the number of points for visualization
sampled_data = data.sample(n=10000, random_state=42)

# PCA visualization of the data
pca = PCA(n_components=2)
columns = ['duration', 'src_bytes', 'dst_bytes']

data_sampled_data = sampled_data[columns]
data_pca = pca.fit_transform(data_sampled_data)

sampled_data['pca1'] = data_pca[:, 0]
sampled_data['pca2'] = data_pca[:, 1]

# Plotting the PCA results
plt.figure(figsize=(12, 10))
plt.scatter(sampled_data[sampled_data['is_attack'] == 0]['pca1'], 
            sampled_data[sampled_data['is_attack'] == 0]['pca2'], 
            color='blue', label='Normal', alpha=0.3)
plt.scatter(sampled_data[sampled_data['is_attack'] == 1]['pca1'], 
            sampled_data[sampled_data['is_attack'] == 1]['pca2'], 
            color='red', label='Anomaly', alpha=0.3)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('KDD Cup 1999 - PCA Visualization')
plt.legend()
plt.show()

### Observations:
# Nous observons que les points d'anomalie (en rouge) se distinguent des points normaux (en bleu) dans l'espace PCA.
# Cependant, il y a certaines points normaux qui sont isolés et pourraient être confondus avec des anomalies.
# De plus, les points d'anomalie sont très concentrés dans certaines régions, ce qui pourrait indiquer des patterns spécifiques d'attaques.