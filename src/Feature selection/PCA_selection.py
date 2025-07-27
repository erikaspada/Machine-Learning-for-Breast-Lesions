"""
PCA feature reduction, returning the number of components explaining
a given variance threshold (default 95%)
"""
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def pca_components_by_variance(X, variance_threshold=0.95):

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Normalize before PCA

    pca = PCA()
    pca.fit(X_scaled)

    cum_variance = pca.explained_variance_ratio_.cumsum()
    n_components = (cum_variance < variance_threshold).sum() + 1

    return pca, n_components
