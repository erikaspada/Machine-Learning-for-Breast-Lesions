from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def build_pipeline(classifier, apply_normalization=False):
    steps = []
    
    if apply_normalization:
        # Add a standard scaler as first step
        steps.append(("scaler", StandardScaler()))
    
    # Add the classifier as the last step
    steps.append(("clf", classifier))
    
    return Pipeline(steps)
