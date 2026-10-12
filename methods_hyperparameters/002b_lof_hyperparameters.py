import pandas as pd
from sklearn.neighbors import LocalOutlierFactor as LOF
from time import perf_counter
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

train_path = "unsupervised_ml/anomaly_detection/methods_hyperparameters/annthyroid/ann-train.data"
test_path = "unsupervised_ml/anomaly_detection/methods_hyperparameters/annthyroid/ann-test.data"

df_train = pd.read_csv(train_path, header=None, sep=" ")
df_test = pd.read_csv(test_path, header=None, sep=" ")
df = pd.concat([df_train, df_test], axis=0)


# only keep cotinuous features as mentionned in the paper "Isolation Forest" de Liu et al. (2008)
continuous_features = [0, 16, 17, 18, 19, 20, 21]
df = df[continuous_features]

# Create a binary target variable where normal = 0 and anomaly = 1
df[21] = df[21].map({1: 1, 2: 1, 3: 0})

# Split the dataset into features and target
y = df[21]
X = df.drop(columns=[21])

# Print the shape of the dataset and the proportion of anomalies
print(X.shape)
print(y.value_counts()/len(y))

# Train LOF model and measure the time taken for training
lof = LOF(n_neighbors=10)

st_lof_train = perf_counter()
lof.fit(X)
end_lof_train = perf_counter()

anomaly_score_lof = -lof.negative_outlier_factor_
roc_auc_lof = roc_auc_score(y, anomaly_score_lof)

print("###### LOF ######")
print(f'ROC AUC score: {roc_auc_lof:.4f}')
print(f"Total time: {(end_lof_train - st_lof_train):.4f} seconds")

# Varying the n° of estimators and measuring the time taken and the ROC AUC score
def neighbors_vary_and_plot(n_neighbors_values, X, y):
    results = []
    for arg_value in n_neighbors_values:
        lof = LOF(n_neighbors=arg_value)
        st_lof_train = perf_counter()
        lof.fit(X)
        end_lof_train = perf_counter()

        anomaly_score_lof = -lof.negative_outlier_factor_
        roc_auc_lof = roc_auc_score(y, anomaly_score_lof)


        total_time = (end_lof_train - st_lof_train)
        results.append((arg_value, roc_auc_lof, total_time))
        
    # Plotting the results in the same plot
    arg_values, roc_auc_values, time_values = zip(*results)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(arg_values, roc_auc_values, marker='o', label='ROC AUC')
    ax1.set_xlabel(f'Number of neighbors')
    ax1.set_ylabel('ROC AUC')
    ax1.set_title('LOF Performance')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(arg_values, time_values, marker='s', color='red', label='Training Time')
    ax2.set_ylabel('Training Time (s)')
    ax2.grid(True)
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.show()



# On fait varier le nombre de voisins et on mesure le temps pris et le score ROC AUC
n_neighbors_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                    55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

neighbors_vary_and_plot(n_neighbors_list, X, y)

## Observations :
# Idem au KNN
# Augmenter le nombre de voisins dégradent a la fois le score ROC AUC et le temps d'exécution.
# Il n'y a que sur les très petites valeurs de n_neighbors où on observe une 
# amélioration du score ROC AUC en augmentant le nombre de voisins. 