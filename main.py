import os
from mask_creation import creation_mask
from feature_extractor import init_feature_extractor, extract_features
from status_upgrade_utils import assign_status
from SMOTE import apply_smote
from train_test import train_incremental_features
from models_config import models
from pipeline import build_pipeline
import pandas as pd

def main():
    # --- Configure paths and dataset ---
    image_dir = "path_to_images"
    mask_dir = "path_to_masks"
    positive_ids = ["patient1_L", "patient2_R"]  # example list of positive patients
    
    # Create masks (only if create_masks=True) 
    for image_name in os.listdir(image_dir):
        if image_name.endswith(".jpg"):
            creation_mask(os.path.join(image_dir, image_name), mask_dir, "path_to_save_breast_images")
    
    # Step 2: Initialize feature extractor 
    extractor = init_feature_extractor()
    
    # Step 3: Extract features
    df_features = extract_features(image_dir, mask_dir, extractor, feature_prefixes=["original_firstorder", "original_glcm"])
    
    # Step 4: Assign labels based on the dataset
    df_labeled = assign_status(df_features, positive_ids)
    
    # Step 5: Prepare data for modeling 
    X = df_labeled.drop(columns=["Status"])
    y = df_labeled["Status"]
    
    # Step 6: Balance dataset with SMOTE (only for dataset 3)
    X_resampled, y_resampled = apply_smote(X, y)
    
    # Step 7: Select model and train 
    model_key = "RandomForest"
    model_conf = models[model_key]
    pipeline = build_pipeline(model_conf["estimator"], apply_normalization=model_conf["normalize"])
    
    # Train with incremental feature selection (all features here)
    best_result = train_incremental_features(
        X_resampled, y_resampled,
        model_conf["estimator"], model_conf["param_grid"],
        features_list=X.columns.tolist(),
        normalize_in_pipeline=model_conf["normalize"],
        method_name=model_key
    )
    
    print("Best training result:")
    print(best_result)

if __name__ == "__main__":
    main()
