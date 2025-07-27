# Breast Lesion Classification Pipeline

Repository for Master Degree Thesis.
Please note that Input data are not availablle due to privacy.

This project implements a machine learning pipeline for classifying breast lesions B3 using radiomic features extracted from mammography images. The pipeline includes:

- Image preprocessing and mask creation
- Radiomic feature extraction
- Feature selection using various methods (AUC, tree-based importance, PCA, correlation)
- Handling imbalanced datasets with SMOTE
- Model training with hyperparameter tuning using GridSearchCV
- Visualization of clustering results and class distributions

## Project Structure
The `config` folder contains:
- `database_configs.py`: Creation database configurations.
 The `plot` folder contains:
- `class_distribution_plot.py`: Visualizes class distribution.
The `src` folder contains:
- `SMOTE.py`: Resamples data using SMOTE to balance classes.
- `models_config.py`: Configuration of ML models and their hyperparameters.
- `pipeline.py`: Constructs ML pipelines with optional normalization.
- `train_test.py`: Functions for training models with incremental feature selection and grid search.
- folder `Feature_selection`
- folder `feature_extraction`
- folder `Anomaly_detection`
The folder `Feature_selection` contains:
- `AUC_selection.py`, `FI_selection.py`, `PCA_selection.py`, `correlation_selection.py`: Feature selection methods.
The folder `feature_extraction` contains: 
- `mask_creation.py`: Creates masks for breast tissue in images.
- `feature_extractor.py`: Extracts radiomic features from images using masks using pyradiomics.
- `status_upgrade_utils.py`: Utility functions to assign labels/status to data.
The folder `Anomaly_detection` contains:
- `OPTICS.py` `DBSCAN.py`, `Isolation_forest.py`: Anomaly detection methods.


## Requirements

See `requirements.txt` for Python packages.


## Author 
Erika Spada – s318375@studenti.polito.it


