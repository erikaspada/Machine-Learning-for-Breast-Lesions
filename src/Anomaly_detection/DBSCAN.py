from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def detect_anomalies_db(df):
    """
    Run DBSCAN clustering on the dataset, plot PCA visualization, 
    and print evaluation metrics without returning anything.

    Args:
        df (pd.DataFrame): DataFrame containing features and 'Upgrade' target column.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.drop(columns=["Upgrade"]))

    # Clustering DBSCAN
    dbscan = DBSCAN(eps=8, min_samples=2)
    clusters = dbscan.fit_predict(X_scaled)
    df['cluster'] = clusters

    # PCA for visualization
    pca_2d = PCA(n_components=2)
    pca_components_2d = pca_2d.fit_transform(X_scaled)
    df_pca_2d = pd.DataFrame(pca_components_2d, columns=['PC1', 'PC2'])

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(df_pca_2d['PC1'], df_pca_2d['PC2'], c=df['cluster'], cmap='tab20', marker='o', s=40)
    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
    ax.add_artist(legend1)
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_title('Clustering DBSCAN sui dati')
    plt.tight_layout()
    plt.show()

    # Evaluation
    df['predicted_anomaly_dbscan'] = (df['cluster'] == -1).astype(int)
    y_true = df['Upgrade']
    y_pred = df['predicted_anomaly_dbscan']
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)

    print(f"DBSCAN - Precision: {precision:.4f}")
    print(f"DBSCAN - Recall: {recall:.4f}")
    print(f"DBSCAN - F1-Score: {f1:.4f}")
    print("Matrice di Confusione:\n", conf_matrix)



