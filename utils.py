"""
utils.py — Shared data loading, preprocessing, and evaluation helpers.
Wisconsin Breast Cancer Dataset (sklearn built-in, no download needed).
"""

import numpy as np
import json
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_STATE = 42
N_FEATURES   = 10      # PCA components fed to both classical and quantum models
TEST_SIZE    = 0.20    # 80/20 split


def load_and_preprocess():
    data = load_breast_cancer()
    X, y = data.data, data.target

    scaler_pre = MinMaxScaler()
    X_scaled   = scaler_pre.fit_transform(X)

    pca   = PCA(n_components=N_FEATURES, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    # Scale to [0, pi] for quantum angle encoding
    scaler_post = MinMaxScaler(feature_range=(0, np.pi))
    X_final     = scaler_post.fit_transform(X_pca)

    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    feature_names = [f"PC{i+1}" for i in range(N_FEATURES)]
    return X_train, X_test, y_train, y_test, feature_names


def compute_metrics(y_true, y_pred, y_prob=None, model_name="Model"):
    metrics = {
        "model":     model_name,
        "accuracy":  round(accuracy_score(y_true, y_pred),  4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred,    zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred,        zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_prob)   if y_prob is not None else float("nan"), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }
    return metrics


def save_results(metrics_dict, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"  Results saved to {filepath}")


def print_metrics(m):
    print(f"\n{'='*50}")
    print(f"  {m['model']}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {m['accuracy']:.4f}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  F1 Score  : {m['f1']:.4f}")
    if m['roc_auc'] == m['roc_auc']:
        print(f"  ROC-AUC   : {m['roc_auc']:.4f}")
    cm = m['confusion_matrix']
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"    FN={cm[1][0]}  TP={cm[1][1]}")
    print(f"{'='*50}\n")
