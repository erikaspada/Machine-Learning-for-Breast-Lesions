database_configs = {
  #ROI:entire breast
  #Goal: determine if the algorithm can distinguish between a lesion-present breast (status 1) and a lesion-free breast (status 0)
    "db1": {
        "create_masks": True,
        "image_dir": "...", #insert directory of images
        "mask_dir": "...", #directory of the output mask function
        "positive_ids": [...],  #insert list of breasts with lesion
        "output_csv": "data/output/db1.csv"
    },
  
  #ROI:Boxes of tissue surrounding the lesions
  #Goal: determine if the algorithm can distinguish between a lesion-present tissue (status 1) and a lesion-free tissue (status 0)
    "db2": {
        "create_masks": False,
        "image_dir": "...", #insert directory of images
        "mask_dir": "...", #insert directory of masks
        "positive_ids": [...],  #insert list of tissues with lesion
        "output_csv": "data/output/db2.csv"
    },
  
  #ROI:Boxes of tissue surrounding the lesions
  #Goal: determine if the algorithm can determine if the lesion is benign (Upgrade 0) or malignant (Upgrade 1)
    "db3": {
        "from_csv": "data/output/db2.csv",
        "filter": "[...]", #insert list of patient who have undergone surgery
        "upgrade_ids": [],  # #insert list of patients with malignant B3 lesion
        "output_csv": "data/output/db3.csv"
    }
}
