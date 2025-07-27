"""
Feature selection based on AUC scores per feature.
Returns features ordered by their AUC score descending.
"""
from sklearn.metrics import roc_auc_score
import pandas as pd

def select_features_by_auc(X, y):
    auc_scores = {}
    for col in X.columns:
        try:
            auc = roc_auc_score(y, X[col])
            auc_scores[col] = auc
        except ValueError:
            # Handle cases where AUC can't be computed (e.g. constant features)
            continue

    sorted_features = sorted(auc_scores, key=auc_scores.get, reverse=True)
    return sorted_features
