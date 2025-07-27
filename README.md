# Breast Lesion Classification Pipeline

This project implements a machine learning pipeline for classifying breast lesions B3 using radiomic features extracted from mammography images. The pipeline includes:

- Image preprocessing and mask creation
- Radiomic feature extraction
- Feature selection using various methods (AUC, tree-based importance, PCA, correlation)
- Handling imbalanced datasets with SMOTE
- Model training with hyperparameter tuning using GridSearchCV
- Visualization of clustering results and class distributions

## Project Structure

- `mask_creation.py`: Creates masks for breast tissue in images.
- `feature_extractor.py`: Extracts radiomic features from images using masks using pyradiomics.
- `AUC_selection.py`, `FI_selection.py`, `PCA_selection.py`, `correlation_selection.py`: Feature selection methods.
- `SMOTE.py`: Resamples data using SMOTE to balance classes.
- `models_config.py`: Configuration of ML models and their hyperparameters.
- `pipeline.py`: Constructs ML pipelines with optional normalization.
- `train_test.py`: Functions for training models with incremental feature selection and grid search.
- `class_distribution_plot.py`: Visualizes class distribution.
- `database_configs.py`: Creation database configurations.
- `OPTICS.py` 'DBSCAN.py', 'Isolation_forest.py': Anomaly detection methods.
- `status_upgrade_utils.py`: Utility functions to assign labels/status to data.

## Requirements

See `requirements.txt` for Python packages.




