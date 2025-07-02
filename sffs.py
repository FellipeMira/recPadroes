import numpy as np
from typing import List, Tuple


def compute_info(Z: List[int], x: np.ndarray, posY: dict) -> float:
    m, _ = x.shape
    mu = np.mean(x[:, Z], axis=0).reshape((1, len(Z)))
    Sb = np.zeros((len(Z), len(Z)))
    Sw = np.zeros((len(Z), len(Z)))
    for j in range(min(posY.keys()), max(posY.keys()) + 1):
        _x = x[posY[j], :][:, Z]
        mu_j = np.mean(_x, axis=0).reshape((1, len(Z)))
        sig_j = np.cov(_x.T).reshape((len(Z), len(Z)))
        Sb += (len(posY[j]) / m) * np.dot((mu - mu_j).T, (mu - mu_j))
        Sw += (len(posY[j]) / m) * sig_j
    return np.linalg.det(Sb + Sw) / np.linalg.det(Sw)


def compute_info_gain_add(Z: List[int], W: List[int], x: np.ndarray, posY: dict) -> np.ndarray:
    m, _ = x.shape
    vecJ = np.zeros(len(W))
    for at in range(len(W)):
        S = np.union1d(Z, W[at]).astype(int).tolist()
        mu = np.mean(x[:, S], axis=0).reshape((1, len(S)))
        Sb = np.zeros((len(S), len(S)))
        Sw = np.zeros((len(S), len(S)))
        for j in range(min(posY.keys()), max(posY.keys()) + 1):
            _x = x[posY[j], :][:, S]
            mu_j = np.mean(_x, axis=0).reshape((1, len(S)))
            sig_j = np.cov(_x.T).reshape((len(S), len(S)))
            Sb += (len(posY[j]) / m) * np.dot((mu - mu_j).T, (mu - mu_j))
            Sw += (len(posY[j]) / m) * sig_j
        vecJ[at] = np.linalg.det(Sb + Sw) / np.linalg.det(Sw)
    return vecJ


def compute_info_gain_remove(Z: List[int], x: np.ndarray, posY: dict) -> np.ndarray:
    m, _ = x.shape
    vecJ = np.zeros(len(Z))
    for item in range(len(Z)):
        at = Z[item]
        S = Z.copy(); S.remove(at)
        mu = np.mean(x[:, S], axis=0).reshape((1, len(S)))
        Sb = np.zeros((len(S), len(S)))
        Sw = np.zeros((len(S), len(S)))
        for j in range(min(posY.keys()), max(posY.keys()) + 1):
            _x = x[posY[j], :][:, S]
            mu_j = np.mean(_x, axis=0).reshape((1, len(S)))
            sig_j = np.cov(_x.T).reshape((len(S), len(S)))
            Sb += (len(posY[j]) / m) * np.dot((mu - mu_j).T, (mu - mu_j))
            Sw += (len(posY[j]) / m) * sig_j
        vecJ[item] = np.linalg.det(Sb + Sw) / np.linalg.det(Sw)
    return vecJ


def sffs(X: np.ndarray, y: np.ndarray, max_features: int = None) -> Tuple[List[int], float]:
    """Sequential Forward Floating Selection using the info criterion.

    Parameters
    ----------
    X : np.ndarray
        Matriz de atributos (amostras x features). Não deve conter ``NaN``.
    y : np.ndarray
        Vetor de rótulos.

    Raises
    ------
    ValueError
        Se ``X`` contiver valores ``NaN``.
    """
    if np.isnan(X).any():
        raise ValueError(
            "Input data for SFFS contains NaN values. "
            "Remova ou impute valores ausentes antes da seleção de atributos."
        )
    n_feats = X.shape[1]
    max_features = n_feats if max_features is None else max_features
    posY = {c: np.where(y == c)[0] for c in np.unique(y)}

    Z: List[int] = []
    W: List[int] = list(range(n_feats))
    best_subset: List[int] = []
    best_score = -np.inf

    while W and len(Z) < max_features:
        add_scores = compute_info_gain_add(Z, W, X, posY)
        idx_best = int(np.argmax(add_scores))
        f_best = W.pop(idx_best)
        Z.append(f_best)
        J_cur = add_scores[idx_best]

        improved = True
        while improved and len(Z) > 2:
            remove_scores = compute_info_gain_remove(Z, X, posY)
            idx_rm = int(np.argmax(remove_scores))
            if remove_scores[idx_rm] > J_cur:
                f_rm = Z.pop(idx_rm)
                W.append(f_rm)
                J_cur = remove_scores[idx_rm]
            else:
                improved = False

        if J_cur > best_score:
            best_score = J_cur
            best_subset = Z.copy()

    return best_subset, best_score
