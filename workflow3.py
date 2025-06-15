#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
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
from sklearn.cluster import KMeans
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA

from sffs import sffs
import warnings
from joblib import Memory, dump

#memory = Memory(location='cache_dir', verbose=1)

warnings.filterwarnings('ignore')
ROOT = os.getcwd()
MODEL_DIR = os.path.join(ROOT, "model")
MODEL_DIR_PCA = r"/home/mira/recPadroes/model_dir_2"


def load_data(path: str):
    """Carrega e prepara o DataFrame, mapeando labels."""
    df = pd.read_parquet(path)
    df = df.iloc[:, 3:32].copy()
    df = df.rename(columns={df.columns[28]: 'label'})
    # mapeamento de -1,0,1 para 0,1,2
    df['label'] = df['label'].map({-1: 0, 0: 1, 1: 2})
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.9, random_state: int = 42):
    """Divide df em treino e teste estratificados."""
    X = df.drop('label', axis=1)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, X, y


def select_features_sffs(X, y, max_features=8):
    """Seleciona atributos usando o algoritmo SFFS baseado em informação."""
    subset, score = sffs(X.values, y.values, max_features=max_features)
    selected = X.columns[subset].tolist()
    print(f"Melhores atributos ({len(selected)}) → {selected}")
    return selected


def pca_analysis(X):
    """Exibe a variância explicada dos componentes principais."""
    pca = PCA(n_components=X.shape[1])
    pca.fit(X)
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Cumulated explained variance:", np.cumsum(pca.explained_variance_ratio_))


def apply_pca(X_train, X_test, X_full, variance=0.95):
    """Aplica padronização e PCA preservando a variância desejada."""
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_full_s = scaler.transform(X_full)

    pca = PCA(n_components=variance, random_state=42)
    pca.fit(X_train_s)
    cols = [f'PC{i+1}' for i in range(pca.n_components_)]
    X_train_p = pd.DataFrame(pca.transform(X_train_s), columns=cols, index=X_train.index)
    X_test_p = pd.DataFrame(pca.transform(X_test_s), columns=cols, index=X_test.index)
    X_full_p = pd.DataFrame(pca.transform(X_full_s), columns=cols, index=X_full.index)
    print(f"PCA manteve {pca.n_components_} componentes")
    return X_train_p, X_test_p, X_full_p


SAMPLING_STRATEGY = {0: 3452, 1: 30000, 2: 38694}


def compute_sampling_strategy(y, desired=SAMPLING_STRATEGY):
    """Retorna uma estratégia de amostragem sem exceder o número disponível."""
    counts = y.value_counts().to_dict()
    strategy = {}
    for cls, count in counts.items():
        target = desired.get(cls, count)
        strategy[cls] = min(target, count)
    return strategy


def make_pipeline(estimator, sampling_strategy):
    """Cria um pipeline com RandomUnderSampler para balancear os dados."""
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    return Pipeline([
        ('undersample', rus),
        ('scaler', StandardScaler()),
        ('model', estimator),
    ])


def run_model(name, estimator, param_dist, X_train, y_train, cv, scorers, n_iter,
              model_dir="model", sampling_strategy=SAMPLING_STRATEGY):
    """
    Ajusta RandomizedSearchCV e retorna o melhor pipeline treinado.
    """
    print(f"\n>>> Otimizando {name}")
    # Evita empilhar pipelines já pré-processados
    if isinstance(estimator, Pipeline) or isinstance(estimator, StackingClassifier):
        pipe = estimator
    else:
        pipe = make_pipeline(estimator, sampling_strategy)


    rs = RandomizedSearchCV(
        pipe, param_dist, n_iter=n_iter, cv=cv,
        scoring=scorers, refit='f1_macro',
        random_state=42, verbose=3, n_jobs=8
    )
    rs.fit(X_train, y_train)
    print(f"{name} → f1_macro CV: {rs.best_score_:.4f}, params: {rs.best_params_}")

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{name.replace(' ', '_')}.joblib")
    dump(rs.best_estimator_, model_path)
    print(f"Modelo salvo em {model_path}")

    return rs.best_estimator_


def evaluate_on_test(trained: dict, X_test, y_test):
    """Avalia cada modelo treinado no conjunto de teste."""
    results = []
    for name, model in trained.items():
        y_pred = model.predict(X_test)
        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'F1_Macro': f1_score(y_test, y_pred, average='macro'),
            'F1_Weighted': f1_score(y_test, y_pred, average='weighted'),
            'Recall_Macro': recall_score(y_test, y_pred, average='macro'),
            'Precision_Macro': precision_score(y_test, y_pred, average='macro'),
            'Kappa': cohen_kappa_score(y_test, y_pred)
        })
    return pd.DataFrame(results)


def final_predictions(df_full, X_full, y_full, X_test, y_test, trained: dict,
                      top_n: int = 3,
                      output_csv: str = 'full_predictions_top3.csv'):
    """
    Gera predict/predict_proba para os ``top_n`` modelos (ordenados pelo
    ``F1_Macro`` obtido no conjunto de teste) sobre o dataset completo,
    anexa colunas ao ``df_full`` e salva CSV.
    """
    # Recarregar avaliação de teste para ordenar
    test_df = evaluate_on_test(trained, X_test, y_test)
    top_models = test_df.sort_values('F1_Macro', ascending=False).head(top_n)['Model'].tolist()
    print(f"\nTop {top_n} modelos: {top_models}")

    df_out = df_full.copy()
    df_out['true_label'] = df_out['label']

    for name in top_models:
        mdl = trained[name]
        preds = mdl.predict(X_full)
        probs = mdl.predict_proba(X_full)
        df_out[f'pred_{name}'] = preds
        # adicionar uma coluna de probabilidade por classe
        classes = mdl.named_steps['model'].classes_
        for idx, cls in enumerate(classes):
            df_out[f'prob_{cls}_{name}'] = probs[:, idx]

    df_out.to_csv(output_csv, index=False)
    print(f"Predições completas salvas em {output_csv}")


def build_rbf_classifier():
    """Retorna um pipeline encapsulando o RBF customizado."""
    class RBFClassifier:
        def __init__(self, n_clusters=10, rbf_gamma=1.0, solver='lbfgs', max_iter=1000):
            self.n_clusters = n_clusters
            self.rbf_gamma = rbf_gamma
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.logreg = LogisticRegression(solver=solver, max_iter=max_iter, random_state=42)

        def _rbf(self, X):
            X = pd.DataFrame(X, columns=self.feature_names_)
            d = self.kmeans.transform(X)
            return np.exp(-self.rbf_gamma * d**2)

        def fit(self, X, y):
            self.feature_names_ = X.columns
            self.kmeans.fit(X)
            Z = self._rbf(X)
            self.logreg.fit(Z, y)
            return self

        def predict(self, X):
            Z = self._rbf(X)
            return self.logreg.predict(Z)

        def predict_proba(self, X):
            Z = self._rbf(X)
            return self.logreg.predict_proba(Z)

    # inserir no pipeline a mesma sequência de pré-processamento
    return make_pipeline(RBFClassifier())


def main():
    # 1. Carrega dados
    path = os.path.join(ROOT, 'df_tk.parquet')
    df = load_data(path)
    print(f"Dados: {df.shape[0]} linhas, {df.shape[1]} colunas; classes:\n{df['label'].value_counts(normalize=True)}")

    # 2. Divide treino/teste
    X_train, X_test, y_train, y_test, X_full, y_full = split_data(df, test_size=0.9)

    sampling_strategy = compute_sampling_strategy(y_train)

    # 3. Seleção de atributos (SFFS) e PCA
    print("\n>>> Selecionando atributos (SFFS)")
    selected_cols = select_features_sffs(X_train, y_train)
    X_train = X_train[selected_cols]
    X_test = X_test[selected_cols]
    X_full = X_full[selected_cols]

    # 4. CV e métricas
    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=123)
    scorers = {
        'accuracy': make_scorer(accuracy_score),
        'f1_macro': make_scorer(f1_score, average='macro'),
        'f1_weighted': make_scorer(f1_score, average='weighted'),
        'recall_macro': make_scorer(recall_score, average='macro'),
        'precision_macro': make_scorer(precision_score, average='macro'),
        'kappa': make_scorer(cohen_kappa_score)
    }
    n_iter = 5

    # 5. Define modelos e parâmetros
    models = {
        'SVM-Linear': (SVC(kernel='linear', probability=True, random_state=42),
                       {'model__C': np.logspace(-2, 1, 10)}),
        'SVM-RBF':    (SVC(kernel='rbf', probability=True, random_state=42),
                       {'model__C': np.logspace(-2, 1, 10),
                        'model__gamma': ['scale', 'auto'] + list(np.logspace(-4, 0, 5))}),
        'KNN':        (KNeighborsClassifier(),
                       {'model__n_neighbors': np.arange(3, 15, 2)}),
        'RF':         (RandomForestClassifier(random_state=42),
                       {'model__n_estimators': np.arange(100, 500, 50),
                        'model__max_depth': [10,20,30,None]}),
        'MLP':        (MLPClassifier(max_iter=1000, random_state=42),
                       {'model__hidden_layer_sizes': [(50,),(100,),(50,25)],
                        'model__activation': ['relu','tanh'],
                        'model__alpha': np.logspace(-5,-2,10)}),
        'AdaBoost':   (AdaBoostClassifier(random_state=42),
                       {'model__n_estimators': np.arange(50,200,25)}),
        'GNB':        (GaussianNB(), {}),
        #'RBF-Net':    (build_rbf_classifier(), {'model__n_clusters': [10,20], 'model__rbf_gamma': [0.1,1.0]})
    }

    # 6. Treina e otimiza modelos
    trained = {}
    for name, (est, params) in models.items():
        trained[name] = run_model(
            name, est, params, X_train, y_train, skf, scorers, n_iter,
            model_dir=MODEL_DIR, sampling_strategy=sampling_strategy
        )

    # 7. Ensemble: Bagging
    bag_base = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    bag = BaggingClassifier(estimator=bag_base, random_state=42)
    trained['Bagging'] = run_model(
        'Bagging', bag, {'model__n_estimators': [10,20,30]},
        X_train, y_train, skf, scorers, n_iter, model_dir=MODEL_DIR,
        sampling_strategy=sampling_strategy
    )

    # 8. Ensemble: Stacking
    stack = StackingClassifier(
        estimators=[(n, trained[n]) for n in ['RF','SVM-Linear','MLP','KNN'] if n in trained],
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=skf, n_jobs=-1
    )
    trained['Stacking'] = run_model(
        'Stacking', stack, {'model__final_estimator__C': np.logspace(-2,1,10)},
        X_train, y_train, skf, scorers, n_iter, model_dir=MODEL_DIR,
        sampling_strategy=sampling_strategy
    )

    # 9. Avaliação final no teste
    print("\n>>> Avaliação no conjunto de teste")
    df_test_eval = evaluate_on_test(trained, X_test, y_test)
    print(df_test_eval.sort_values('F1_Macro', ascending=False).to_string(index=False))

    # salvar métricas de teste
    df_test_eval.to_csv('model_performance_test_SFFS.csv', index=False)

    # 10. Predições finais no dataset completo
    final_predictions(df, X_full, y_full, X_test, y_test, trained, top_n=3)

    
    # PCA
    print("\n>>> PCA dos atributos selecionados")
    pca_analysis(X_train)
    X_train, X_test, X_full = apply_pca(X_train, X_test, X_full)    
    

    # 6. Treina e otimiza modelos
    trained = {}
    for name, (est, params) in models.items():
        trained[name] = run_model(
            name, est, params, X_train, y_train, skf, scorers, n_iter,
            model_dir=MODEL_DIR_PCA, sampling_strategy=sampling_strategy
        )

    # 7. Ensemble: Bagging
    bag_base = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    bag = BaggingClassifier(estimator=bag_base, random_state=42)
    trained['Bagging'] = run_model(
        'Bagging', bag, {'model__n_estimators': [10,20,30]},
        X_train, y_train, skf, scorers, n_iter, model_dir=MODEL_DIR_PCA,
        sampling_strategy=sampling_strategy
    )

    # 8. Ensemble: Stacking
    stack = StackingClassifier(
        estimators=[(n, trained[n]) for n in ['RF','SVM-Linear','MLP','KNN'] if n in trained],
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=skf, n_jobs=-1
    )
    trained['Stacking'] = run_model(
        'Stacking', stack, {'model__final_estimator__C': np.logspace(-2,1,10)},
        X_train, y_train, skf, scorers, n_iter, model_dir=MODEL_DIR_PCA,
        sampling_strategy=sampling_strategy
    )

    # 9. Avaliação final no teste
    print("\n>>> Avaliação no conjunto de teste")
    df_test_eval = evaluate_on_test(trained, X_test, y_test)
    print(df_test_eval.sort_values('F1_Macro', ascending=False).to_string(index=False))

    # salvar métricas de teste
    df_test_eval.to_csv('model_performance_test_PCA.csv', index=False)

    # 10. Predições finais no dataset completo
    final_predictions(df, X_full, y_full, X_test, y_test, trained, top_n=3)

    print("\n>>> Script concluído!")

if __name__ == '__main__':
    main()
