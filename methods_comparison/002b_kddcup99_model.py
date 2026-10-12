from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor as LOF
from sklearn.datasets import fetch_kddcup99
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# Load the KDD Cup 1999 dataset filtered for the 'http' subset
subset = 'http'
data, target = fetch_kddcup99(subset=subset, return_X_y=True, as_frame=True, percent10=False)
target_encoded = (target != b"normal.").astype(np.int32)

# Model definition with respect to the parameters of the paper "Isolation Forest" de Liu et al. (2008)
isolation_forest = IsolationForest(n_estimators=100, max_samples=256, n_jobs=-1)
lof = LOF(n_neighbors=10)

# Training the models
isolation_forest.fit(data)
lof.fit(data)

# Apply anomaly scores on the training data
anomaly_score = -isolation_forest.decision_function(data)
anomaly_score_lof = -lof.negative_outlier_factor_


# Plotting PCA results with isolation forest anomaly scores and LOF anomaly scores
data['anomaly_score_iforest'] = anomaly_score
data['anomaly_score_lof'] = anomaly_score_lof
data['target'] = target_encoded

# Sample a subset of the data for visualization
data_sampled = data.sample(n=10000, random_state=42)

# Apply Min-Max scaling to the anomaly scores for better visualization for LOF
vmin_lof = data_sampled['anomaly_score_lof'].min()
vmax_lof = data_sampled['anomaly_score_lof'].max()
data_sampled['anomaly_score_lof_scaled'] = (data_sampled['anomaly_score_lof'] - vmin_lof) / (vmax_lof - vmin_lof)

# Apply PCA to reduce the dimensionality of the data for visualization
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_sampled[['duration', 'src_bytes', 'dst_bytes']])
data_sampled['pca1'] = data_pca[:, 0]
data_sampled['pca2'] = data_pca[:, 1]

# Create subplots for the PCA visualization with anomaly scores
fig, axs = plt.subplots(2, 1, figsize=(12, 10))

for i, anomaly_score_col in enumerate(['anomaly_score_iforest', 'anomaly_score_lof_scaled']):
    vmin = data_sampled[anomaly_score_col].min()
    vmax = data_sampled[anomaly_score_col].max()
    sc_normal = axs[i].scatter(data_sampled[data_sampled['target'] == 0]['pca1'], 
                   data_sampled[data_sampled['target'] == 0]['pca2'], 
                   c=data_sampled[data_sampled['target'] == 0][anomaly_score_col], cmap='viridis',
                   vmin=vmin, vmax=vmax, label='Normal', alpha=0.5, marker='o')
    sc_anomaly = axs[i].scatter(data_sampled[data_sampled['target'] == 1]['pca1'], 
                   data_sampled[data_sampled['target'] == 1]['pca2'], 
                   c=data_sampled[data_sampled['target'] == 1][anomaly_score_col], cmap='viridis',
                   vmin=vmin, vmax=vmax, label='Anomaly', alpha=0.5, marker='*')
    axs[i].set_title(f'PCA Visualization with {anomaly_score_col}')
    axs[i].set_xlabel('First Principal Component')
    axs[i].set_ylabel('Second Principal Component')
    axs[i].legend()
    fig.colorbar(sc_normal, ax=axs[i], label='Anomaly Score')
plt.tight_layout()
plt.show()

# Concernant Iforest, on remarque que les points anormaux sont bien détectés et ont des scores d'anomalie plus élevés que les points normaux. 
# De plus, plus les points sont éloignés de la majorité des points normaux, plus leur score d'anomalie est élevé. Ce qui fait que certains points "normaux"
# mais isolés ont des scores d'anomalie élevés.
# En revanche, pour LOF, on remarque aucune différence de scores notables entre les points normaux et les points anormaux.