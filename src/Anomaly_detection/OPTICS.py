from sklearn.cluster import OPTICS
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import os

optics = OPTICS(min_samples=3, xi=0.005, min_cluster_size=4)
df['cluster_optics'] = optics.fit_predict(X_scaled)
df['predicted_anomaly_optics'] = (df['cluster_optics'] == -1).astype(int)

# Visualization
X_pca = PCA(n_components=2).fit_transform(X_scaled)
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster_optics'], cmap='tab10', alpha=0.7)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Clustering con OPTICS')
plt.colorbar(label="Cluster")
plt.show()

# Evaluation
y_true = df['Upgrade']
y_pred = df['predicted_anomaly_optics']
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
conf_matrix = confusion_matrix(y_true, y_pred)

print(f"OPTICS - Precision: {precision:.4f}")
print(f"OPTICS - Recall: {recall:.4f}")
print(f"OPTICS - F1-Score: {f1:.4f}")
print("Matrice di Confusione:\n", conf_matrix)


