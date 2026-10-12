from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor as LOF
from sklearn.datasets import fetch_kddcup99
from pyod.models.knn import KNN
import numpy as np
from time import perf_counter

# Load the KDD Cup 1999 dataset filtered for the 'http' subset
subset = 'http' # smtp
data, target = fetch_kddcup99(subset=subset, return_X_y=True, as_frame=True, percent10=False)
target_encoded = (target != b"normal.").astype(np.int32)

# Model definition with respect to the parameters of the paper "Isolation Forest" de Liu et al. (2008)
isolation_forest = IsolationForest(n_estimators=100, max_samples=256, n_jobs=-1)

lof = LOF(n_neighbors=10)
orca_eq = KNN(n_neighbors=10)

# Training the models and measuring the time taken for training
st_if_train = perf_counter()
isolation_forest.fit(data)
end_if_train = perf_counter()

st_lof_train = perf_counter()
lof.fit(data)
end_lof_train = perf_counter()

st_orca_eq_train = perf_counter()
orca_eq.fit(data)
end_orca_eq_train = perf_counter()

# Apply anomaly scores on the training data and measuring the time taken for prediction (only for Iforest)
# Calculate the auc score
# Attention au piège de la fonction decision_function qui retourne des valeurs négatives pour les anomalies et positives pour les normales. 
# Pour obtenir un score d'anomalie, on prend l'opposé de cette valeur.
st_if_pred = perf_counter()
anomaly_score_if = -isolation_forest.decision_function(data)
end_if_pred = perf_counter()
roc_auc_if = roc_auc_score(target_encoded, anomaly_score_if)

anomaly_score_lof = -lof.negative_outlier_factor_
roc_auc_lof = roc_auc_score(target_encoded, anomaly_score_lof)

anomaly_score_orca_eq = orca_eq.decision_scores_
roc_auc_orca_eq = roc_auc_score(target_encoded, anomaly_score_orca_eq)


# Display the results
if __name__ == "__main__":
    print("###### IForest ######")
    print(f'ROC AUC score: {roc_auc_if:.4f}')
    print(f"Total time: {(end_if_train - st_if_train) + (end_if_pred - st_if_pred):.4f} seconds")

    print("###### LOF ######")
    print(f'ROC AUC score : {roc_auc_lof:.4f}')
    print(f"Total time: {end_lof_train - st_lof_train:.4f} seconds")

    print("###### ORCA-EQ ######")
    print(f'ROC AUC score: {roc_auc_orca_eq:.4f}')
    print(f"Total time: {end_orca_eq_train - st_orca_eq_train:.4f} seconds")

# En observant les points anormaux on remarque qu'ils sont pour la plupart très proches. 
# Ce qui en résulte est que les méthodes de détection d'anomalies basées sur la distance comme
# KNN/LOF ont du mal à les détecter.
# Dans ce cas l'isolation forest est plus efficace car chaque itree est construit sur un sous-échantillon aléatoire de l'ensemble de données.
# Voir les explications sur le "masking effect" dans l'article de Liu et al. (2008) "Isolation Forest" (cf Partie 3.).
# De plus, si deux points sont identiques, ils seront isolés au même niveau dans l'arbre et donc auront le même score d'anomalie.
