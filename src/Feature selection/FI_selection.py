"""
Feature selection using tree-based model feature importance.
"""

def select_features_by_tree_importance(X, y, model, top_k=50):
    model.fit(X, y)
    importances = model.feature_importances_
    sorted_idx = importances.argsort()[::-1]
    top_features = X.columns[sorted_idx[:top_k]].tolist()
    return top_features
