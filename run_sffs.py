#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run SFFS feature selection for PA and TK regions."""

import json
import os
from typing import List

import pandas as pd
from sklearn.model_selection import train_test_split

from sffs import sffs


def load_data(path: str) -> pd.DataFrame:
    """Load dataset and keep only VV/VH features with mapped labels."""
    df = pd.read_parquet(path)
    label_col = "pseudosamples_rho2"
    feature_cols = [c for c in df.columns if c.endswith("_VV") or c.endswith("_VH")]
    df = df[feature_cols + [label_col]].copy()
    df = df.rename(columns={label_col: "label"})
    df = df[df["label"] != 0]
    df["label"] = df["label"].map({-1: 1, 1: 0})
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.5, random_state: int = 42):
    """Split DataFrame into train and test sets."""
    X = df.drop("label", axis=1)
    y = df["label"]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def select_features_sffs(X: pd.DataFrame, y: pd.Series, max_features: int = 6) -> List[str]:
    """Select features using SFFS algorithm."""
    subset, _ = sffs(X.values, y.values, max_features=max_features)
    return X.columns[subset].tolist()


def run_sffs_for_roi(roi: str) -> None:
    """Run SFFS for a specific region of interest and save selected features."""
    file_map = {"PA": "df_pa.parquet", "TK": "df_tk.parquet"}
    data_path = os.path.join(os.getcwd(), file_map[roi])
    df = load_data(data_path)
    X_train, _, y_train, _ = split_data(df, test_size=0.5)
    feats = select_features_sffs(X_train, y_train)
    out_file = f"selected_features_{roi}_sffs_filtered.json"
    with open(out_file, "w") as f:
        json.dump(feats, f, indent=2)
    print(f"{roi}: saved {len(feats)} features to {out_file}")


if __name__ == "__main__":
    for region in ["PA", "TK"]:
        run_sffs_for_roi(region)
