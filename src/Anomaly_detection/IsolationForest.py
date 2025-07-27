import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def detect_anomalies(df):
    features = df.drop(columns=['Upgrade'])
    X = features.values

    # Fit Isolation Forest
    model = IsolationForest(n_estimators=100, contamination=9/62)
    model.fit(X)
    df['anomaly'] = model.predict(X)
    df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})

    # PCA for 3D visualization
    pca = PCA(n_components=3)
    pca_components = pca.fit_transform(X)
    df_pca = pd.DataFrame(pca_components, columns=['PC1', 'PC2', 'PC3'])

    # Plot anomalies in PCA space
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df_pca['PC1'], df_pca['PC2'], df_pca['PC3'], c=df['anomaly'], cmap='coolwarm', marker='o')
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    ax.set_title('Anomaly distribution in PCA 3D')
    plt.colorbar(scatter, label='Anomaly (1=anomaly)')
    plt.show()

    # Evaluation metrics
    y_true = df['Upgrade']
    y_pred = df['anomaly']
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomaly"])
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    plt.show()

 

