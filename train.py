#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import pandas as pd
from joblib import dump

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score,
    precision_score, cohen_kappa_score, make_scorer
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier,
    BaggingClassifier, StackingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from scipy.stats import uniform, randint

from workflow import (
    load_data, split_data, select_features_sffs, pca_analysis,
    apply_pca, compute_sampling_strategy,
    run_model, evaluate_on_test, build_rbf_classifier
)

from config import (
    ROI, SAMPLER_TYPE, SAMPLING_STRATEGY, DATA_PATH,
    MODEL_DIR, MODEL_DIR_PCA, FEATS_PATH, PCA_PATH
)

from sklearnex import patch_sklearn
patch_sklearn()



def main():
    fold = 5
    n_iter = 50

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR_PCA, exist_ok=True)
    os.makedirs(FEATS_PATH, exist_ok=True)
    os.makedirs(PCA_PATH, exist_ok=True)
    print(f"\n\nCarregando dados de {DATA_PATH}...\n\n")

    df, X  = load_data(DATA_PATH)
    
    print(f"Dados: {df.shape[0]} linhas, {df.shape[1]} colunas; classes:\n{df['label'].value_counts(normalize=True)}")
    print(f"Contagem:\n{X.groupby('label').count()}")
    
    X_train, X_test, y_train, y_test, _, _ = split_data(df, test_size=0.5)
    _, _, _, _, X_full, y_full = split_data(X, test_size=0.5)
    
    if SAMPLER_TYPE == 'smote':
        sampling_strategy = None
    else:
        sampling_strategy = compute_sampling_strategy(y_train)

    print("\n>>> Selecionando atributos (SFFS)")
    selected_cols = select_features_sffs(X_train, y_train)
    with open(FEATS_PATH, 'w') as f:
        json.dump(selected_cols, f)

    X_train = X_train[selected_cols]
    X_test = X_test[selected_cols]
    X_full = X_full[selected_cols]

    skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=123)
    scorers = {
        'accuracy': make_scorer(accuracy_score),
        'f1_macro': make_scorer(f1_score, average='macro'),
        'f1_weighted': make_scorer(f1_score, average='weighted'),
        'recall_macro': make_scorer(recall_score, average='macro'),
        'precision_macro': make_scorer(precision_score, average='macro'),
        'kappa': make_scorer(cohen_kappa_score)
    }

    models = {
        'SVM-Linear': (
            SVC(kernel='linear', probability=True, random_state=42),
            {
                'model__C': uniform(1, 1000),
                'model__gamma': [0.01, 0.05, 0.1, 0.5, 1, 1.5]
            }
        ),
        'SVM-RBF': (
            SVC(kernel='rbf', probability=True, random_state=42),
            {
                'model__C': uniform(1, 1000),
                'model__gamma': [0.01, 0.05, 0.1, 0.5, 1, 1.5]
            }
        ),
        'KNN': (
            KNeighborsClassifier(),
            {'model__n_neighbors': randint(1, 31)}
        ),
        'RF': (
            RandomForestClassifier(random_state=42),
            {
                'model__n_estimators': randint(10, 200),
                'model__max_depth': [2, 5, 10, 25, 50],
                'model__max_features': ['auto','sqrt','log2'],
                'model__min_impurity_decrease':  [0.001, 0.00001, 0.00001]
            }
        ),
        'MLP': (
            MLPClassifier(max_iter=1000, random_state=42),
            {
                'model__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
                'model__alpha': 10.0 ** -np.arange(1, 5),
                'model__learning_rate_init': [0.01, 0.001, 0.005],
                'model__activation': ['relu', 'tanh', 'logistic']
            }
        ),
        'AdaBoost': (
            AdaBoostClassifier(random_state=42),
            {
                'model__n_estimators': randint(10, 250),
                'model__learning_rate': [0.01, 0.001, 0.0001]
            }
        ),
        'GNB': (GaussianNB(),
                {'model__var_smoothing': [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]}),
        'RBF-Net': (
            build_rbf_classifier(),
            {
                'model__n_clusters': randint(5, 61),
                'model__epsilon': [0.1, 0.01, 0.5,1.0]
            }
        )
    }

    trained = {}
    print(f'n_iter: {n_iter}')

    for name, (est, params) in models.items():
        trained[name] = run_model(
            name, est, params, X_train, y_train, skf, scorers, n_iter,
            model_dir=MODEL_DIR,
            sampling_strategy=sampling_strategy,
            sampler_type=SAMPLER_TYPE
        )

    bag_base = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    bag = BaggingClassifier(estimator=bag_base, random_state=42)
    trained['Bagging'] = run_model(
        'Bagging', bag, {'model__n_estimators': [10,20,30,40,50]},
        X_train, y_train, skf, scorers, n_iter, model_dir=MODEL_DIR,
        sampling_strategy=sampling_strategy, sampler_type=SAMPLER_TYPE
    )

    stack = StackingClassifier(
        estimators=[(n, trained[n]) for n in ['RF','SVM-Linear','MLP','KNN'] if n in trained],
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=skf, n_jobs=1
    )
    trained['Stacking'] = run_model(
        'Stacking', stack, {'final_estimator__C': np.logspace(-2, 1, 10)},
        X_train, y_train, skf, scorers, n_iter, model_dir=MODEL_DIR,
        sampling_strategy=sampling_strategy, sampler_type=SAMPLER_TYPE,
        n_jobs=1
    )

    print("\n>>> Avaliação no conjunto de teste")
    df_test_eval = evaluate_on_test(trained, X_test, y_test)
    print(df_test_eval.sort_values('F1_Macro', ascending=False).to_string(index=False))
    df_test_eval.to_csv(f'model_performance_test_SFFS_{ROI}.csv', index=False)

    pca_analysis(X_train)
    X_train_p, X_test_p, X_full_p, scaler_p, pca = apply_pca(X_train, X_test, X_full)
    dump({'scaler': scaler_p, 'pca': pca}, PCA_PATH)

    trained = {}
    for name, (est, params) in models.items():
        trained[name] = run_model(
            name, est, params, X_train_p, y_train, skf, scorers, n_iter,
            model_dir=MODEL_DIR_PCA, sampling_strategy=sampling_strategy,
            sampler_type=SAMPLER_TYPE
        )

    bag_base = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    bag = BaggingClassifier(estimator=bag_base, random_state=42)
    trained['Bagging'] = run_model(
        'Bagging', bag, {'model__n_estimators': [10,20,30]},
        X_train_p, y_train, skf, scorers, n_iter, model_dir=MODEL_DIR_PCA,
        sampling_strategy=sampling_strategy, sampler_type=SAMPLER_TYPE
    )

    stack = StackingClassifier(
        estimators=[(n, trained[n]) for n in ['RF','SVM-Linear','MLP','KNN'] if n in trained],
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=skf, n_jobs=1
    )
    trained['Stacking'] = run_model(
        'Stacking', stack, {'final_estimator__C': np.logspace(-2, 1, 10)},
        X_train_p, y_train, skf, scorers, n_iter, model_dir=MODEL_DIR_PCA,
        sampling_strategy=sampling_strategy, sampler_type=SAMPLER_TYPE,
        n_jobs=1
    )

    print("\n>>> Avaliação no conjunto de teste (PCA)")
    df_test_eval = evaluate_on_test(trained, X_test_p, y_test)
    print(df_test_eval.sort_values('F1_Macro', ascending=False).to_string(index=False))
    df_test_eval.to_csv(f'model_performance_test_PCA_{ROI}.csv', index=False)


if __name__ == '__main__':
    main()
