import os
import cv2
import numpy as np
import SimpleITK as sitk
import pandas as pd
from radiomics import featureextractor

def init_feature_extractor():
    params = {
        "setting": {
            "binWidth": 25,
            "label": 1,
            "interpolator": "sitkBSpline",
            "resampledPixelSpacing": None,
        },
        "imageType": {
            "Original": {},
        },
    }
    return featureextractor.RadiomicsFeatureExtractor(params)

def extract_features(image_dir, mask_dir, extractor, feature_prefixes):
    data = []
    for image_name in os.listdir(image_dir):
        if not image_name.endswith(".jpg"):
            continue

        image_path = os.path.join(image_dir, image_name)
        mask_name = image_name.replace(".jpg", "_mask.jpeg")
        mask_path = os.path.join(mask_dir, mask_name)

        if not os.path.exists(mask_path):
            continue

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        masked_pixels = image[mask == 255]

        if masked_pixels.size == 0:
            continue

        p5, p95 = np.percentile(masked_pixels, [5, 95])
        extractor.settings['resegmentRange'] = [p5, p95]

        image_sitk = sitk.GetImageFromArray(image)
        mask_sitk = sitk.GetImageFromArray(mask)

        features = extractor.execute(image_sitk, mask_sitk)
        filtered = {k: float(v) for k, v in features.items() if k.startswith(feature_prefixes)}
        patient_id_parts = image_name.split("_")
        patient_id = f"{patient_id_parts[0]}_{patient_id_parts[1][0]}"
        filtered["PatientID"] = patient_id
        data.append(filtered)

    return pd.DataFrame(data)
