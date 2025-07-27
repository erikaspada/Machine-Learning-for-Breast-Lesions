# Breast Lesion Classification Pipeline

This project implements a machine learning pipeline for classifying breast tissue lesions using radiomic features extracted from mammography images. The pipeline includes:

- Image preprocessing and mask creation
- Radiomic feature extraction
- Feature selection using various methods (AUC, tree-based importance, PCA, correlation)
- Handling imbalanced datasets with SMOTE
- Model training with hyperparameter tuning using GridSearchCV
- Visualization of clustering results and class distributions

## Project Structure

- `mask_creation.py`: Creates masks for breast tissue in images.
- `feature_extractor.py`: Extracts radiomic features from images using masks.
- `AUC_selection.py`, `FI_selection.py`, `PCA_selection.py`, `correlation_selection.py`: Feature selection methods.
- `SMOTE.py`: Resamples data using SMOTE to balance classes.
- `models_config.py`: Configuration of ML models and their hyperparameters.
- `pipeline.py`: Constructs ML pipelines with optional normalization.
- `train_test.py`: Functions for training models with incremental feature selection and grid search.
- `class_distribution_plot.py`: Visualizes class distribution.
- `database_configs.py`: Dataset configurations.
- `OPTICS.py`: Clustering and anomaly detection using OPTICS.
- `status_upgrade_utils.py`: Utility functions to assign labels/status to data.

## Requirements

See `requirements.txt` for Python packages.

## Usage

1. Configure datasets in `database_configs.py`.
2. Create masks if needed.
3. Extract features.
4. Perform feature selection.
5. Train models using `train_test.py`.
6. Evaluate and visualize results.

## Example

```python
from mask_creation import creation_mask
from feature_extractor import init_feature_extractor, extract_features
from status_upgrade_utils import assign_status, assign_upgrade
from SMOTE import apply_smote
from train_test import train_incremental_features
from models_config import models
from pipeline import build_pipeline

# Step 1: Mask creation (if needed)
creation_mask(image_path, mask_save_dir, breast_save_dir)

# Step 2: Feature extraction
extractor = init_feature_extractor()
df_features = extract_features(image_dir, mask_dir, extractor, feature_prefixes=["original_firstorder", "original_glcm"])

# Step 3: Label assignment
df_labeled = assign_status(df_features, positive_ids)
df_labeled = assign_upgrade(df_labeled, upgrade_ids)

# Step 4: Handle imbalanced data
X = df_labeled.drop(columns=['Status', 'Upgrade'])
y = df_labeled['Status']
X_resampled, y_resampled = apply_smote(X, y)

# Step 5: Train model example
model_config = models["RandomForest"]
pipeline = build_pipeline(model_config["estimator"], apply_normalization=model_config["normalize"])
best_result = train_incremental_features(X_resampled, y_resampled, model_config["estimator"], model_config["param_grid"], features_list=X.columns.tolist())

print(best_result)

