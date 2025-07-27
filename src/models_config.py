from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

models = {
    "DecisionTree": {
        "estimator": DecisionTreeClassifier(random_state=42),
        "param_grid": {
            "clf__max_depth": [None, 10, 20, 30, 40, 50],
            "clf__min_samples_split": [2, 5, 10, 15, 20],
            "clf__min_samples_leaf": [1, 2, 4, 10],
            "classifier__max_features": [None, 'sqrt', 'log2']
        },
        "normalize": False
    },
    "RandomForest": {
        "estimator": RandomForestClassifier(random_state=42),
        "param_grid": {
            "clf__n_estimators": [50, 100, 200, 500],
            "clf__max_depth": [None, 10, 20],
            "clf__min_samples_split": [2, 5, 10],  
            "clf__min_samples_leaf": [1, 2, 4],  
            "clf__max_features": ['sqrt', 'log2'],  
            "clf__criterion": ['gini', 'entropy']
        },
        "normalize": False
    },
    "GradientBoosting": {
        "estimator": GradientBoostingClassifier(random_state=42),
        "param_grid": {
            "clf__n_estimators": [50, 100,150],
            "clf__learning_rate":  [0.01, 0.1, 0.2, 0.5],
            "clf__max_depth": [3, 4, 5],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 5]
        },
        "normalize": False
    },
    "XGBoost": {
        "estimator": xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss"),
        "param_grid": {
            "clf__n_estimators": [50, 100,200],
            "clf__learning_rate": [0.01, 0.1, 0.2],
            "clf__max_depth": [3, 5, 7, 10],
            "clf__colsample_bytree": [0.8, 1.0], 
        },
        "normalize": False
    },
    "AdaBoost": {
        "estimator": AdaBoostClassifier(random_state=42),
        "param_grid": {
            "clf__base_estimator__max_depth": [1, 2, 3, 4, 5],
            "clf__n_estimators": [50, 100, 200, 300],
            "clf__learning_rate": [0.01, 0.1, 0.5, 1.0, 1.5] 
        },
        "normalize": False
    },
    "NaiveBayes": {
        "estimator": GaussianNB(),
        "param_grid": {
            "clf__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
        },
        "normalize": True
    },
    "LogisticRegression": {
        "estimator": LogisticRegression(max_iter=1000, random_state=42),
        "param_grid": {
            "clf__C": [0.001, 0.01, 0.1, 1, 10, 100],
            "clf__solver": ['lbfgs', 'liblinear', 'saga'],
            "clf__max_iter": [100, 200, 500, 1000,10000],  
            "clf__penalty": ['l2', 'l1']
        },
        "normalize": True    
    },
    "SVM": {
        "estimator": SVC(probability=True, random_state=42),
        "param_grid": {
            "clf__C": [0.1, 1, 10],
            "clf__kernel": ["linear", "rbf"]
        },
        "normalize": True
    },
    "SGDClassifier": {
        "estimator": SGDClassifier(random_state=42),
        "param_grid": {
            "clf__loss": ["hinge", "log_loss"],
            "clf__alpha": [0.0001, 0.001, 0.01, 0.1]
        },
        "normalize": True
    },
    "KNN": {
        "estimator": KNeighborsClassifier(),
        "param_grid": {
            "clf__n_neighbors": [3, 5, 7, 10, 15],
            "clf__weights": ["uniform", "distance"],
            "clf__algorithm": ['auto', 'ball_tree', 'kd_tree'],
            "clf__leaf_size": [20, 30, 40, 50],
            "clf__p": [1, 2]

        },
        "normalize": True
    }
}
