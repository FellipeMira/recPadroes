#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd

from sklearnex import patch_sklearn
patch_sklearn()

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
from sklearn.base import BaseEstimator, ClassifierMixin
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from scipy.stats import loguniform, randint, expon, uniform

from sffs import sffs
import warnings
from joblib import Memory, dump

from config import (
    ROI, SAMPLER_TYPE, SAMPLING_STRATEGY, DATA_PATH,
    MODEL_DIR, MODEL_DIR_PCA
)

warnings.filterwarnings('ignore')

fold = 5
n_iter = 50

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(MODEL_DIR_PCA, exist_ok=True)

print(f"\n\tUsando estratégia de amostragem {ROI}: \n\n\t{SAMPLING_STRATEGY}")

def load_data(path: str):
    """Carrega e prepara o DataFrame selecionando colunas automaticamente."""
    df = pd.read_parquet(path)

    label_col = 'pseudosamples_rho2'
    feature_cols = [c for c in df.columns if c.endswith('_VV') or c.endswith('_VH')]

    df = df[feature_cols + [label_col]].copy()
    df = df.rename(columns={label_col: 'label'})
    # remove registros rotulados como 0 e faz mapeamento -1 -> 1, 1 -> 0
    df_1 = df[df['label'] != 0]
    df_1['label'] = df_1['label'].map({-1: 1, 1: 0})
    return (df_1, df)


def split_data(df: pd.DataFrame, test_size: float = 0.9, random_state: int = 42):
    """Divide df em treino e teste estratificados."""
    X = df.drop('label', axis=1)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, X, y


def select_features_sffs(X, y, max_features=6):
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
    return X_train_p, X_test_p, X_full_p, scaler, pca



def compute_sampling_strategy(y, desired=SAMPLING_STRATEGY):
    """Retorna estratégia sem exceder o número disponível em ``y``."""

    counts = pd.Series(y).value_counts().to_dict()
    strategy = {}
    for cls, count in counts.items():
        target = desired.get(cls, count)
        strategy[cls] = min(target, count)
    return strategy


class DynamicSamplingStrategy:
    """Callable usado para ajustar a estrategia de amostragem dinamicamente."""

    def __init__(self, desired):
        self.desired = desired

    def __call__(self, y):
        return compute_sampling_strategy(y, desired=self.desired)


def make_pipeline(estimator, sampling_strategy, sampler_type='under'):
    """Cria pipeline com escolha de amostragem SMOTE ou RandomUnderSampler."""

    if sampler_type == 'smote':
        if sampling_strategy is None:
            sampler = SMOTE(random_state=42)
        else:
            sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    else:
        sampler = RandomUnderSampler(
            sampling_strategy=DynamicSamplingStrategy(sampling_strategy),
            random_state=42
        )

    return Pipeline([
        ('sampler', sampler),
        ('scaler', StandardScaler()),
        ('model', estimator),
    ])


def run_model(name, estimator, param_dist, X_train, y_train, cv, scorers, n_iter,
              model_dir="model", sampling_strategy=SAMPLING_STRATEGY,
              sampler_type='under', n_jobs=8):
    """Ajusta ``RandomizedSearchCV`` e retorna o melhor pipeline treinado.

    Parameters
    ----------
    sampler_type : str
        "under" usa ``RandomUnderSampler``; "smote" usa ``SMOTE``.
    n_jobs : int
        Número de processos para ``RandomizedSearchCV``.
    """
    print(f"\n>>> Otimizando {name}")
    # Evita empilhar pipelines já pré-processados
    if isinstance(estimator, Pipeline) or isinstance(estimator, StackingClassifier):
        pipe = estimator
    else:
        pipe = make_pipeline(estimator, sampling_strategy, sampler_type)


    rs = RandomizedSearchCV(
        pipe, param_dist, n_iter=n_iter, cv=cv,
        scoring=scorers, refit='f1_macro',
        random_state=42, verbose=3, n_jobs=n_jobs
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
                      top_n: int = 10,
                      output_csv: str = 'full_predictions_top10.csv'):
    """
    Gera predict/predict_proba para os ``top_n`` modelos (ordenados pelo
    ``F1_Macro`` obtido no conjunto de teste) sobre o dataset completo,
    anexa colunas ao ``df_full`` e salva CSV.
    """
    # Recarregar avaliação de teste para ordenar
    test_df = evaluate_on_test(trained, X_test, y_test)
    top_models = test_df.sort_values('F1_Macro', ascending=False).head(top_n)['Model'].tolist()
    print(f"\nTop {top_n} modelos: {top_models}")

    df_out = X_full.copy()
    #df_out['true_label'] = df_out['label']

    for name in top_models:
        mdl = trained[name]
        preds = mdl.predict(X_full)
        #probs = mdl.predict_proba(X_full)
        df_out[f'pred_{name}'] = preds
        
        # adicionar uma coluna de probabilidade por classe
        # if hasattr(mdl, 'named_steps'):
        #     classes = mdl.named_steps['model'].classes_
        # else:
        #     classes = mdl.classes_
        # for idx, cls in enumerate(classes):
        #     df_out[f'prob_{cls}_{name}'] = probs[:, idx]

    df_out.to_csv(output_csv, index=False)
    print(f"Predições completas salvas em {output_csv}")


# Funções auxiliares para a RBF Network

def init_kmeans(data, k, epsilon):
    """Obtém centros e variâncias usando um K-Means simplificado."""
    m, dim = data.shape
    rand_pos = np.random.randint(0, m, k)
    cent = data[rand_pos, :].copy()
    prev = cent.copy()
    dist = np.zeros(k)
    it = 0
    while True:
        count = np.zeros(k)
        acc = np.zeros_like(cent)
        for i in range(m):
            for j in range(k):
                dist[j] = np.linalg.norm(prev[j, :] - data[i, :])
            idx = np.argmin(dist)
            acc[idx] += data[i, :]
            count[idx] += 1
        for j in range(k):
            if count[j] > 0:
                cent[j, :] = acc[j] / count[j]
        desloc = np.linalg.norm(cent - prev)
        it += 1
        if desloc < epsilon or it > 1000:
            break
        prev = cent.copy()
    campos = np.ones(k)
    count_campos = np.zeros(k)
    for i in range(m):
        for j in range(k):
            dist[j] = np.linalg.norm(cent[j, :] - data[i, :])
        idx = np.argmin(dist)
        count_campos[idx] += 1
        campos[idx] += np.linalg.norm(cent[idx, :] - data[i, :])
    for j in range(k):
        campos[j] /= (count_campos[j] + 1)
    return cent, campos


def expected_output(y):
    """Converte rótulos em representação vetorial."""
    uy = np.unique(y)
    nclass = uy.size
    m = y.shape[0]
    vecY = np.zeros((nclass, m))
    for i in range(m):
        pos = np.where(uy == y[i])
        vecY[pos, i] = 1
    return vecY


def map_rbf(x, mu, sig):
    m = x.shape[0]
    k = mu.shape[0]
    mappedX = np.empty((m, k))
    for i in range(m):
        for j in range(k):
            mappedX[i, j] = np.exp(-(np.linalg.norm(x[i, :] - mu[j, :]) ** 2) / (2 * (sig[j] ** 2)))
    return mappedX


def train_rbf_net(x, y, k, epsilon):
    codeY = expected_output(y)
    cent, campo = init_kmeans(x, k, epsilon)
    mapX = map_rbf(x, cent, campo)
    A = mapX.T @ mapX
    W = np.linalg.inv(A) @ mapX.T @ codeY.T
    return W, cent, campo


class RBFClassifier(BaseEstimator, ClassifierMixin):
    """Classificador RBF simples com treinamento analítico."""

    def __init__(self, n_clusters=10, epsilon=1e-2):
        self.n_clusters = n_clusters
        self.epsilon = epsilon

    def _check_X(self, X):
        if isinstance(X, pd.DataFrame):
            return X.values
        return np.asarray(X)

    def fit(self, X, y):
        X = self._check_X(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.W_, self.centers_, self.sigmas_ = train_rbf_net(X, y, self.n_clusters, self.epsilon)
        return self

    def _rbf(self, X):
        return map_rbf(X, self.centers_, self.sigmas_)

    def predict(self, X):
        X = self._check_X(X)
        Z = self._rbf(X)
        scores = Z @ self.W_
        idx = np.argmax(scores, axis=1)
        return self.classes_[idx]

    def predict_proba(self, X):
        X = self._check_X(X)
        Z = self._rbf(X)
        scores = Z @ self.W_
        scores -= scores.max(axis=1, keepdims=True)
        exp_s = np.exp(scores)
        return exp_s / exp_s.sum(axis=1, keepdims=True)



def build_rbf_classifier():
    """Retorna uma instância do ``RBFClassifier``."""
    return RBFClassifier()


def main():
    # 1. Carrega dados
    df, X = load_data(DATA_PATH)
    print(f"Dados: {df.shape[0]} linhas, {df.shape[1]} colunas; classes:\n{df['label'].value_counts(normalize=True)}")

    # 2. Divide treino/teste
    X_train, X_test, y_train, y_test, _, _ = split_data(df, test_size=0.9)

    _, _, _, _, X_full, y_full = split_data(X, test_size=0.9)

    if SAMPLER_TYPE == 'smote':
        sampling_strategy = None
    else:
        sampling_strategy = compute_sampling_strategy(y_train)

    # 3. Seleção de atributos (SFFS) e PCA
    print("\n>>> Selecionando atributos (SFFS)")
    selected_cols = select_features_sffs(X_train, y_train)
    X_train = X_train[selected_cols]
    X_test = X_test[selected_cols]
    X_full = X_full[selected_cols]

    # 4. CV e métricas
    skf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=123)
    scorers = {
        'accuracy': make_scorer(accuracy_score),
        'f1_macro': make_scorer(f1_score, average='macro'),
        'f1_weighted': make_scorer(f1_score, average='weighted'),
        'recall_macro': make_scorer(recall_score, average='macro'),
        'precision_macro': make_scorer(precision_score, average='macro'),
        'kappa': make_scorer(cohen_kappa_score)
    }

    # 5. Define modelos e parâmetros
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
    # 6. Treina e otimiza modelos
    trained = {}
    
    print(f'n_iter: {n_iter}')
    
    for name, (est, params) in models.items():
        trained[name] = run_model(
            name, est, params, X_train, y_train, skf, scorers, n_iter,
            model_dir=MODEL_DIR, 
            sampling_strategy=sampling_strategy,
            sampler_type=SAMPLER_TYPE
        )

    # 7. Ensemble: Bagging
    bag_base = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    bag = BaggingClassifier(estimator=bag_base, random_state=42)
    trained['Bagging'] = run_model(
        'Bagging', bag, {'model__n_estimators': [10,20,30]},
        X_train, y_train, skf, scorers, n_iter, model_dir=MODEL_DIR,
        sampling_strategy=sampling_strategy, sampler_type=SAMPLER_TYPE
    )

    # 8. Ensemble: Stacking
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

    # 9. Avaliação final no teste
    print("\n>>> Avaliação no conjunto de teste")
    df_test_eval = evaluate_on_test(trained, X_test, y_test)
    print(df_test_eval.sort_values('F1_Macro', ascending=False).to_string(index=False))

    # salvar métricas de teste
    df_test_eval.to_csv(f'model_performance_test_SFFS_{ROI}.csv', index=False)

    # 10. Predições finais no dataset completo
    final_predictions(df, X_full, y_full, X_test, y_test, trained, top_n=10, output_csv=f'full_predictions_top10_SFFS_{ROI}.csv')

    
    # PCA
    print("\n>>> PCA dos atributos selecionados")
    pca_analysis(X_train)
    X_train, X_test, X_full, _, _ = apply_pca(X_train, X_test, X_full)
    

    # 6. Treina e otimiza modelos
    trained = {}
    for name, (est, params) in models.items():
        trained[name] = run_model(
            name, est, params, X_train, y_train, skf, scorers, n_iter,
            model_dir=MODEL_DIR_PCA, sampling_strategy=sampling_strategy,
            sampler_type=SAMPLER_TYPE
        )

    # 7. Ensemble: Bagging
    bag_base = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    bag = BaggingClassifier(estimator=bag_base, random_state=42)
    trained['Bagging'] = run_model(
        'Bagging', bag, {'model__n_estimators': [10,20,30]},
        X_train, y_train, skf, scorers, n_iter, model_dir=MODEL_DIR_PCA,
        sampling_strategy=sampling_strategy, sampler_type=SAMPLER_TYPE
    )

    # 8. Ensemble: Stacking
    stack = StackingClassifier(
        estimators=[(n, trained[n]) for n in ['RF','SVM-Linear','MLP','KNN'] if n in trained],
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=skf, n_jobs=1
    )
    trained['Stacking'] = run_model(
        'Stacking', stack, {'final_estimator__C': np.logspace(-2, 1, 10)},
        X_train, y_train, skf, scorers, n_iter, model_dir=MODEL_DIR_PCA,
        sampling_strategy=sampling_strategy, sampler_type=SAMPLER_TYPE,
        n_jobs=1
    )

    # 9. Avaliação final no teste
    print("\n>>> Avaliação no conjunto de teste")
    df_test_eval = evaluate_on_test(trained, X_test, y_test)
    print(df_test_eval.sort_values('F1_Macro', ascending=False).to_string(index=False))

    # salvar métricas de teste
    df_test_eval.to_csv(f'model_performance_test_PCA_{ROI}.csv', index=False)

    # 10. Predições finais no dataset completo
    final_predictions(df, X_full, y_full, X_test, y_test, trained, top_n=10, output_csv=f'full_predictions_top10_PCA_{ROI}.csv')

    print("\n>>> Script concluído!")

# if __name__ == '__main__':
#     main()




