from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_kddcup99

import numpy as np

# Load the KDD Cup 1999 dataset filtered for the 'http' subset
subset = 'http' # smtp
data, target = fetch_kddcup99(subset=subset, return_X_y=True, as_frame=True, percent10=False)
target_encoded = (target != b"normal.").astype(np.int32)


# Model definition with respect to the parameters of the paper "Isolation Forest" de Liu et al. (2008)
isolation_forest = IsolationForest(n_estimators=100, max_samples=256, n_jobs=-1)

## Augmenting max samples to 8192 as mentioned in the paper "Isolation Forest" de Liu et al. (2008) to improve the performance of the model.
isolation_forest_max = IsolationForest(n_estimators=100, max_samples=8192, n_jobs=-1)

# Training the models and measuring the time taken for training
# Repeat the operation ten times and take the average AUC score for each model.
auc_scores_if = []
auc_scores_if_max = []


for i in range(10):  
    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, target_encoded, test_size=0.33, random_state=42)

    # Isolate only normal data for training the models
    X_train_normal = X_train[y_train == 0]
  
    # Train the models on the normal data only
    isolation_forest.fit(X_train_normal)
    isolation_forest_max.fit(X_train_normal)

    # Apply anomaly scores on the test sets data 
    anomaly_score_if = -isolation_forest.decision_function(X_test)
    auc_scores_if.append(roc_auc_score(y_test, anomaly_score_if))

    anomaly_score_if_max = -isolation_forest_max.decision_function(X_test)
    auc_scores_if_max.append(roc_auc_score(y_test, anomaly_score_if_max))


# Display the results
if __name__ == "__main__":
    print("###### IForest ######")
    print(f'Mean ROC AUC score: {np.mean(auc_scores_if):.4f}')

    print("###### IForest (max samples=8192) ######")
    print(f'Mean ROC AUC score: {np.mean(auc_scores_if_max):.4f}')

## Observations :
# Le fait d'avoir retiré les anomalies du jeu de données d'entraînement 
# dégrade légèrement le score ROC AUC pour l'isolation forest.
# Par contre, l'augementation du paramètre max_samples à 8192 ramène le score ROC AUC 
# pour l'isolation forest au même niveau que lorsque l'entraînement est effectué 
# sur l'ensemble des données.