"""
Training functions with grid search for different feature selection methods.
Includes incremental feature addition for all except correlation method.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from pipeline_utils import build_pipeline  

def grid_search(X_train, y_train, pipe, param_grid):
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid

def train_incremental_features(X, y, classifier, param_grid, features_list, normalize_in_pipeline=True, method_name=""):
    best_result = None
    for i in range(1, len(features_list) + 1):
        selected = features_list[:i]
        X_sel = X[selected]
        X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.3, random_state=42)

        pipe = build_pipeline(classifier, apply_normalization=normalize_in_pipeline)
        grid = grid_search(X_train, y_train, pipe, param_grid)

        y_pred = grid.best_estimator_.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)

        if (best_result is None) or (test_acc > best_result['test_score']):
            best_result = {
                'method': method_name,
                'num_features': i,
                'features': selected,
                'train_score': grid.best_score_,
                'test_score': test_acc,
                'best_params': grid.best_params_
            }
    return best_result

def train_correlation_only(X, y, classifier, param_grid, normalize_in_pipeline=True, method_name="Correlation"):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    pipe = build_pipeline(classifier, apply_normalization=normalize_in_pipeline)
    grid = grid_search(X_train, y_train, pipe, param_grid)

    y_pred = grid.best_estimator_.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    return {
        'method': method_name,
        'num_features': X.shape[1],
        'features': X.columns.tolist(),
        'train_score': grid.best_score_,
        'test_score': test_acc,
        'best_params': grid.best_params_
    }

def train_pca_incremental(X_pca, y, classifier, param_grid, normalize_in_pipeline=False):
    best_result = None
    for i in range(1, X_pca.shape[1] + 1):
        X_sel = X_pca[:, :i]
        X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.3, random_state=42)

        pipe = build_pipeline(classifier, apply_normalization=normalize_in_pipeline)
        grid = grid_search(X_train, y_train, pipe, param_grid)

        y_pred = grid.best_estimator_.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)

        if (best_result is None) or (test_acc > best_result['test_score']):
            best_result = {
                'method': "PCA",
                'num_components': i,
                'train_score': grid.best_score_,
                'test_score': test_acc,
                'best_params': grid.best_params_
            }
    return best_result
