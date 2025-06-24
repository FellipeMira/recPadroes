#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
from joblib import load

from workflow3 import load_data, split_data, final_predictions


def load_models(model_dir):
    models = {}
    for fname in os.listdir(model_dir):
        if fname.endswith('.joblib'):
            key = os.path.splitext(fname)[0]
            models[key] = load(os.path.join(model_dir, fname))
    return models


def main():
    ROI = 'TK'
    file = 'df_tk.parquet'

    ROOT = os.getcwd()
    MODEL_DIR = os.path.join(ROOT, f"model_SFFS_{ROI}")
    MODEL_DIR_PCA = os.path.join(ROOT, f"model_PCA_{ROI}")
    FEATS_PATH = os.path.join(ROOT, f"selected_features_{ROI}.json")
    PCA_PATH = os.path.join(ROOT, f"pca_scaler_{ROI}.joblib")

    path = os.path.join(ROOT, file)
    df = load_data(path)
    X_train, X_test, y_train, y_test, X_full, y_full = split_data(df, test_size=0.9)

    with open(FEATS_PATH) as f:
        selected_cols = json.load(f)

    X_train_sel = X_train[selected_cols]
    X_test_sel = X_test[selected_cols]
    X_full_sel = X_full[selected_cols]

    models = load_models(MODEL_DIR)
    final_predictions(df, X_full_sel, y_full, X_test_sel, y_test, models,
                      top_n=10, output_csv=f'predictions_SFFS_{ROI}.csv')

    trans = load(PCA_PATH)
    scaler = trans['scaler']
    pca = trans['pca']
    def transform(df_):
        X_s = scaler.transform(df_)
        cols = [f'PC{i+1}' for i in range(pca.n_components_)]
        return pd.DataFrame(pca.transform(X_s), columns=cols, index=df_.index)

    X_train_p = transform(X_train_sel)
    X_test_p = transform(X_test_sel)
    X_full_p = transform(X_full_sel)

    models_pca = load_models(MODEL_DIR_PCA)
    final_predictions(df, X_full_p, y_full, X_test_p, y_test, models_pca,
                      top_n=10, output_csv=f'predictions_PCA_{ROI}.csv')


if __name__ == '__main__':
    main()
