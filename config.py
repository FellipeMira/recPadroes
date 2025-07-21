ROI = 'PA'
FILE = 'df_pa.parquet'
SAMPLER_TYPE = 'under'

if ROI == 'PA':
    SAMPLING_STRATEGY = {0: 24000, 1: 24000}
elif ROI == 'TK':
    SAMPLING_STRATEGY = {0: 3000, 1: 3000}
else:
    SAMPLING_STRATEGY = {0: 1000, 1: 1000}

import os
ROOT = os.getcwd()
DATA_PATH = os.path.join(ROOT, FILE)

MODEL_DIR = os.path.join(ROOT, f"model_SFFS_{ROI}")
MODEL_DIR_PCA = os.path.join(ROOT, f"model_PCA_{ROI}")
MODEL_DIR_FULL = os.path.join(ROOT, f"model_FULL_{ROI}")
FEATS_PATH = os.path.join(ROOT, f"selected_features_{ROI}.json")
PCA_PATH = os.path.join(ROOT, f"pca_scaler_{ROI}.joblib")
