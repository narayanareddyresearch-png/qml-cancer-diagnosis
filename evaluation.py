"""
evaluation.py
Loads saved results and produces comparison plots + summary table.

Two comparison tiers:
  Tier 1 (Full dataset) : LR vs Linear SVM — establishes strong classical baseline
  Tier 2 (Subset, fair) : LR_subset vs SVM_subset vs VQC — apples-to-apples quantum comparison
"""

import json, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

RESULTS_DIR = "results"
PLOT_PATH   = os.path.join(RESULTS_DIR, "comparison_plots.png")
METRIC_KEYS    = ["accuracy", "precision", "recall", "f1", "roc_auc"]
METRIC_LABELS  = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]


def load(fname):
    fpath = os.path.join(RESULTS_DIR, fname)
    if os.path.exists(fpath):
        with open(fpath) as f:
            return json.load(f)
    return None


def print_table(title, results):
    print(f"\n{'='*80}")
    print(f"  {title:^76}")
    print("="*80)
    hdr = f"  {'Model':<36}" + "".join(f"  {k.upper():<10}" for k in METRIC_KEYS)
    print(hdr); print("-"*80)
    for r in results:
        if r is None: continue
        row = f"  {r['model']:<36}"
        for k in METRIC_KEYS:
            v = r.get(k, float("nan"))
            row += f"  {v:<10.4f}" if v == v else f"  {'N/A':<10}"
        print(row)
    print("="*80)


def bar_chart(ax, results, title):
    models   = [r["model"].replace(" (classical)", "").replace(" (subset, 6 features)", "\n(subset)") for r in results]
    n        = len(models)
    x        = np.arange(len(METRIC_KEYS))
    w        = 0.22
    colors   = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    for i, (model, res) in enumerate(zip(models, results)):
        vals = [res.get(k, float("nan")) for k in METRIC_KEYS]
        bars = ax.bar(x + i*w, vals, w, label=model,
                      color=colors[i % len(colors)], alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            if v == v:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=6, rotation=90)
    ax.set_xticks(x + w*(n-1)/2)
    ax.set_xticklabels(METRIC_LABELS, fontsize=9)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.12)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7); ax.grid(axis="y", linestyle="--", alpha=0.4)


def cm_plot(ax, res, title):
    if res is None: ax.axis("off"); return
    cm = np.array(res["confusion_matrix"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Malig.", "Benign"],
                yticklabels=["Malig.", "Benign"],
                ax=ax, cbar=False, linewidths=0.5)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=7)
    ax.set_ylabel("Actual", fontsize=7)


def loss_plot(ax, vqc_res):
    if vqc_res is None: ax.axis("off"); return
    loss = vqc_res.get("loss_history", [])
    if not loss: ax.text(0.5,0.5,"No loss history",ha="center",transform=ax.transAxes); return
    ax.plot(loss, color="#C44E52", linewidth=1.5, alpha=0.9, label="VQC loss")
    ax.axhline(y=min(loss), color="#C44E52", linestyle="--", alpha=0.4,
               label=f"Min loss: {min(loss):.4f}")
    ax.set_xlabel("Iteration", fontsize=9); ax.set_ylabel("COBYLA Objective", fontsize=9)
    ax.set_title("VQC Training Loss Curve", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8); ax.grid(linestyle="--", alpha=0.4)


def resource_plot(ax, vqc_res):
    if vqc_res is None: ax.axis("off"); return
    qc     = vqc_res.get("quantum_circuit", {})
    labels = ["Qubits", "Circuit\nDepth", "Decomp.\nDepth", "Trainable\nParams", "Total\nGates"]
    values = [qc.get("n_qubits",0), qc.get("circuit_depth",0),
              qc.get("decomposed_depth",0), qc.get("n_trainable_params",0),
              qc.get("total_gates",0)]
    colors = ["#4C72B0","#55A868","#8172B2","#C44E52","#CCB974"]
    bars   = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                str(v), ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_title("Quantum Resource Metrics", fontsize=10, fontweight="bold")
    ax.set_ylabel("Count"); ax.grid(axis="y", linestyle="--", alpha=0.4)


if __name__ == "__main__":
    print("=" * 60)
    print("  EVALUATION — Generating plots & summary tables")
    print("=" * 60)

    lr_full   = load("classical_lr_results.json")
    svm_full  = load("classical_svm_results.json")
    lr_sub    = load("classical_lr_subset_results.json")
    svm_sub   = load("classical_svm_subset_results.json")
    vqc_res   = load("quantum_vqc_results.json")

    if lr_full and svm_full:
        print_table("TIER 1 — Full Dataset Classical Baseline (455 train / 114 test, 10 features)",
                    [lr_full, svm_full])

    fair = [r for r in [lr_sub, svm_sub, vqc_res] if r]
    if fair:
        print_table("TIER 2 — Fair Subset Comparison (80 train / 40 test, 6 features)",
                    fair)

    if vqc_res:
        qc = vqc_res.get("quantum_circuit", {})
        print(f"\n  Quantum Resource Summary:")
        print(f"    Qubits            : {qc.get('n_qubits')}")
        print(f"    Circuit depth     : {qc.get('circuit_depth')}")
        print(f"    Decomposed depth  : {qc.get('decomposed_depth')}")
        print(f"    Trainable params  : {qc.get('n_trainable_params')}")
        print(f"    Total gates       : {qc.get('total_gates')}")
        print(f"    Training time     : {vqc_res.get('train_time_seconds')}s")

    # ── Plots ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 13))
    fig.suptitle(
        "Quantum ML for Healthcare: Breast Cancer Classification\n"
        "Wisconsin Breast Cancer Dataset — VQC (ZZFeatureMap) vs Classical Baselines",
        fontsize=13, fontweight="bold", y=0.99
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.35)

    ax_full = fig.add_subplot(gs[0, :])
    ax_sub  = fig.add_subplot(gs[1, :2])
    ax_res  = fig.add_subplot(gs[1, 2])
    ax_cm   = [fig.add_subplot(gs[2, i]) for i in range(3)]
    ax_loss = None

    if lr_full and svm_full:
        bar_chart(ax_full, [lr_full, svm_full],
                  "Tier 1: Full Dataset Classical Baseline (455 train, 10 PCA features)")

    if fair:
        bar_chart(ax_sub, fair,
                  "Tier 2: Fair Comparison on Same Subset (80 train, 6 features)")

    resource_plot(ax_res, vqc_res)

    cm_models = [(lr_sub, "LR Subset"), (svm_sub, "SVM Subset"), (vqc_res, "Quantum VQC")]
    for ax, (res, lbl) in zip(ax_cm, cm_models):
        cm_plot(ax, res, lbl)

    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"\n  Plot saved → {PLOT_PATH}")
    plt.close()
    print("Evaluation complete.")
