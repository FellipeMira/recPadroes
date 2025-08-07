#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calcula métricas de FP, FN, TP e TN para modelos treinados."""
import os
import json
import pandas as pd
from joblib import load
from sklearn.metrics import confusion_matrix

from workflow import load_data, split_data

# Mapear métodos para diretórios de modelos
MODEL_DIRS = {
    'SFFS': lambda roi: os.path.join(os.getcwd(), f"model_SFFS_{roi}"),
    'PCA': lambda roi: os.path.join(os.getcwd(), f"model_PCA_{roi}"),
    'ALL': lambda roi: os.path.join(os.getcwd(), f"model_FULL_{roi}")
}

# Mapear ROI para arquivos de dados
DATA_FILES = {
    'PA': 'df_pa.parquet',
    'TK': 'df_tk.parquet'
}

def load_models(model_dir: str):
    """Carrega todos os modelos .joblib de um diretório."""
    models = {}
    if not os.path.isdir(model_dir):
        return models
    for fname in os.listdir(model_dir):
        if fname.endswith('.joblib'):
            key = os.path.splitext(fname)[0]
            models[key] = load(os.path.join(model_dir, fname))
    return models

def compute_confusion(models, X_test, y_test):
    """Retorna média de FP, FN, TP e TN para um conjunto de modelos."""
    metrics = []
    for mdl in models.values():
        # Alinha as colunas de X_test com as utilizadas no treino do modelo
        Xt = X_test
        if hasattr(mdl, 'feature_names_in_'):
            cols = [c for c in mdl.feature_names_in_ if c in X_test.columns]
            if cols:
                Xt = X_test[cols]
        preds = mdl.predict(Xt)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        metrics.append({'FP': fp, 'FN': fn, 'TP': tp, 'TN': tn})
    if not metrics:
        return {'FP': 0, 'FN': 0, 'TP': 0, 'TN': 0}
    df = pd.DataFrame(metrics)
    return df.mean().to_dict()

def main():
    rows = []
    for roi, data_file in DATA_FILES.items():
        data_path = os.path.join(os.getcwd(), data_file)
        if not os.path.exists(data_path):
            continue
        df, _ = load_data(data_path)
        _, X_test, _, y_test, _, _ = split_data(df, test_size=0.5)
        for method, dir_fn in MODEL_DIRS.items():
            model_dir = dir_fn(roi)
            models = load_models(model_dir)
            if not models:
                continue
            Xt = X_test
            # Ajusta os dados de teste conforme o método utilizado
            if method in {'SFFS', 'PCA'}:
                feats_path = os.path.join(os.getcwd(), f"selected_features_{roi}.json")
                if os.path.exists(feats_path):
                    with open(feats_path) as f:
                        selected = json.load(f)
                    Xt = Xt[selected]
            if method == 'PCA':
                pca_path = os.path.join(os.getcwd(), f"pca_scaler_{roi}.joblib")
                if os.path.exists(pca_path):
                    obj = load(pca_path)
                    scaler, pca = obj['scaler'], obj['pca']
                    Xt_s = scaler.transform(Xt)
                    cols = [f'PC{i+1}' for i in range(pca.n_components_)]
                    Xt = pd.DataFrame(pca.transform(Xt_s), columns=cols, index=Xt.index)
            cm = compute_confusion(models, Xt, y_test)
            rows.append({
                'ROI': roi,
                'Method': method,
                'FP': cm['FP'],
                'FN': cm['FN'],
                'TP': cm['TP'],
                'TN': cm['TN']
            })
    if rows:
        df_out = pd.DataFrame(rows)
        print(df_out)
    else:
        print("Nenhum dado ou modelo encontrado.")

if __name__ == '__main__':
    main()
