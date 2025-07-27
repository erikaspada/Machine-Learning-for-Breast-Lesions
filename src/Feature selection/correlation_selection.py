"""
Feature selection based on correlation threshold.
Removes features with correlation above threshold.
"""

import numpy as np

def remove_highly_correlated_features(X, threshold=0.8):
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    dropped_features = [col for col in upper.columns if any(upper[col] > threshold)]
    X_filtered = X.drop(columns=dropped_features)
    return X_filtered, dropped_features
