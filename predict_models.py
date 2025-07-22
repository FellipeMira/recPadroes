#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import argparse
import pandas as pd
from joblib import load

from workflow import load_data, split_data, final_predictions

from config import (
    ROI, DATA_PATH, MODEL_DIR, MODEL_DIR_PCA, MODEL_DIR_FULL,
    FEATS_PATH, PCA_PATH
)

path_pca = f'predictions_PCA_{ROI}_Filtered.csv'
path_sffs = f'predictions_SFFS_{ROI}_Filtered.csv'
path_full = f'predictions_FULL_{ROI}_Filtered.csv'

def load_models(model_dir):
    models = {}
    for fname in os.listdir(model_dir):
        if fname.endswith('.joblib'):
            key = os.path.splitext(fname)[0]
            models[key] = load(os.path.join(model_dir, fname))
    return models


def main(full=False):
    df, X = load_data(DATA_PATH)
    X_train, X_test, _, y_test, _, _ = split_data(df, test_size=0.9)
    _, _, _, _, X_full, y_full = split_data(X, test_size=0.9)
    
    print(f"\n\nX_train: {X_train.shape}\nX_test:{X_test.shape}\nX_full:{X_full.shape}\ny_full: {y_full.shape}\ny_test: {y_test.shape}\n\n")
    
    print(f"Dados: {df.shape[0]} linhas, {df.shape[1]} colunas")
    
    print(f"cols {X_full.columns.tolist()}")

    if full:
        X_train_sel = X_train
        X_test_sel = X_test
        X_full_sel = X_full
        models = load_models(MODEL_DIR_FULL)
        print(f"\n\nModels loaded from {MODEL_DIR_FULL}:\n{list(models.keys())}\n\n")
        final_predictions(df, X_full_sel, y_full, X_test_sel, y_test, models,
                          top_n=10, output_csv=path_full)
        return

    with open(FEATS_PATH) as f:
        selected_cols = json.load(f)

    print(f"features selected: {selected_cols}")
    print(f'\n\n\nX_train: {X_train.columns.tolist()}\n\nX_test: {X_test.columns.tolist()}\n\nX_full: {X_full.columns.tolist()}')

    X_train_sel = X_train[selected_cols]
    X_test_sel = X_test[selected_cols]
    X_full_sel = X_full[selected_cols]

    models = load_models(MODEL_DIR)

    print(f"\n\nModels loaded from {MODEL_DIR}:\n{list(models.keys())}\n\n")

    final_predictions(df, X_full_sel, y_full, X_test_sel, y_test, models,
                      top_n=10, output_csv=path_sffs)

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
                      top_n=10, output_csv=path_pca)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gera previsões usando modelos treinados')
    parser.add_argument('--full', action='store_true', help='Usa modelos treinados com todas as features')
    args = parser.parse_args()
    main(full=args.full)
