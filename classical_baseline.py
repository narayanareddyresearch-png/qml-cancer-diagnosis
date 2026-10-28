"""
classical_baseline.py
Classical ML baselines for breast cancer classification.
Models: Logistic Regression (linear) and SVM with linear kernel.
These are deliberately chosen to mirror the quantum linear classifier
for a fair comparison.
"""

import time
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from utils import (
    load_and_preprocess, compute_metrics,
    save_results, print_metrics
)


def run_logistic_regression(X_train, X_test, y_train, y_test):
    print("\n[1/2] Training Logistic Regression ...")
    t0    = time.time()
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob,
                              model_name="Logistic Regression (classical)")
    metrics["train_time_seconds"] = round(train_time, 4)
    metrics["n_parameters"]       = int(np.prod(model.coef_.shape) + model.intercept_.shape[0])
    return metrics, model


def run_svm_linear(X_train, X_test, y_train, y_test):
    print("\n[2/2] Training Linear SVM ...")
    t0    = time.time()
    model = SVC(kernel="linear", probability=True, random_state=42)
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob,
                              model_name="Linear SVM (classical)")
    metrics["train_time_seconds"] = round(train_time, 4)
    metrics["n_support_vectors"]  = int(model.n_support_.sum())
    return metrics, model


if __name__ == "__main__":
    print("=" * 60)
    print("  CLASSICAL BASELINE — Breast Cancer Classification")
    print("  Dataset : Wisconsin Breast Cancer (sklearn built-in)")
    print("  Features: 10 PCA components, scaled to [0, pi]")
    print("=" * 60)

    X_train, X_test, y_train, y_test, _ = load_and_preprocess()
    print(f"\n  Train samples : {len(X_train)}")
    print(f"  Test  samples : {len(X_test)}")
    print(f"  Features      : {X_train.shape[1]}")

    lr_metrics, _  = run_logistic_regression(X_train, X_test, y_train, y_test)
    svm_metrics, _ = run_svm_linear(X_train, X_test, y_train, y_test)

    print_metrics(lr_metrics)
    print_metrics(svm_metrics)

    save_results(lr_metrics,  "results/classical_lr_results.json")
    save_results(svm_metrics, "results/classical_svm_results.json")

    print("\nClassical baseline complete.")
