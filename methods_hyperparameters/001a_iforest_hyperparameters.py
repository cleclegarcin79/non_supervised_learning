import pandas as pd
from sklearn.ensemble import IsolationForest
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

# Train Isolation Forest model and measure the time taken for training
isolation_forest = IsolationForest(n_estimators=100, max_samples=4, n_jobs=-1)

st_if_train = perf_counter()
isolation_forest.fit(X)
end_if_train = perf_counter()

st_if_pred = perf_counter()
anomaly_score_if = -isolation_forest.decision_function(X)
end_if_pred = perf_counter()
roc_auc_if = roc_auc_score(y, anomaly_score_if)

print("###### IForest ######")
print(f'ROC AUC score: {roc_auc_if:.4f}')
print(f"Total time: {(end_if_train - st_if_train) + (end_if_pred - st_if_pred):.4f} seconds")

# Varying the hyperparameters and measuring the time taken and the ROC AUC score
def iforest_one_args_vary_and_plot(arg_name, arg_values, X, y):
    results = []
    for arg_value in arg_values:
        if arg_name == "n_estimators":
            isolation_forest = IsolationForest(n_estimators=arg_value, max_samples=256, n_jobs=-1)
        elif arg_name == "max_samples":
            isolation_forest = IsolationForest(n_estimators=100, max_samples=arg_value, n_jobs=-1)
        else:
            raise ValueError(f"Unknown argument name: {arg_name}")
        st_if_train = perf_counter()
        isolation_forest.fit(X)
        end_if_train = perf_counter()

        st_if_pred = perf_counter()
        anomaly_score_if = -isolation_forest.decision_function(X)
        end_if_pred = perf_counter()
        roc_auc_if = roc_auc_score(y, anomaly_score_if)

        total_time = (end_if_train - st_if_train) + (end_if_pred - st_if_pred)
        results.append((arg_value, roc_auc_if, total_time))
        
    # Plotting the results in the same plot
    arg_values, roc_auc_values, time_values = zip(*results)
    arg_values = np.log(arg_values) if arg_name == "max_samples" else arg_values  # Log scale for max_samples
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(arg_values, roc_auc_values, marker='o', label='ROC AUC')
    ax1.set_xlabel(f'Number of {arg_name}')
    ax1.set_ylabel('ROC AUC')
    ax1.set_title('IForest Performance')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(arg_values, time_values, marker='s', color='red', label='Training Time')
    ax2.set_ylabel('Training Time (s)')
    ax2.grid(True)
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


# On fait varier le nombre d'estimateurs et on mesure le temps pris et le score ROC AUC
n_estimators_list = [10, 15, 20, 25, 30, 35, 40, 45, 50, 
                     55, 60, 65, 70, 75, 80, 85, 90, 95, 
                     100, 110, 120, 130, 140, 150, 160,
                     170, 180, 190, 200, 250, 300, 350, 400,
                     450, 500]

#iforest_one_args_vary_and_plot("n_estimators", n_estimators_list, X, y)
## Observations :
# Un nombre faibles d'estimateurs rend l'AUC score assez instable
# A partir de 100 estimateurs, l'ajout de nouveaux estimateurs n'améliore pas significativement le score AUC.
# Le temps de calcul augmente linéairement avec le nombre d'estimateurs, ce qui est attendu.


# Réitère l'expérience en variant le nombre d'échantillons max_samples
max_samples_list = [2,4,8,16,32,64,128,256,512,1024,2048,4096]
iforest_one_args_vary_and_plot("max_samples", max_samples_list, X, y)
## Observations :
# Un nombre faibles d'échantillons permet d'obtenir un score AUC plus élevé
# Avec un temps de calcul plus faible.