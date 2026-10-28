"""
run_all.py — End-to-end runner.
Step 1: classical_baseline.py   (full dataset, ~2s)
Step 2: quantum_vqc.py          (subset + VQC, ~5-8 min)
Step 3: evaluation.py           (plots + tables, ~5s)
"""
import subprocess, sys

steps = [
    ("Classical Baseline (full dataset)", ["python", "classical_baseline.py"]),
    ("Quantum VQC + Subset Comparison",   ["python", "quantum_vqc.py"]),
    ("Evaluation & Plots",                ["python", "evaluation.py"]),
]

for name, cmd in steps:
    print(f"\n{'#'*60}\n# {name}\n{'#'*60}")
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        print(f"\nERROR in: {name}")
        sys.exit(r.returncode)

print("\n" + "="*60)
print("  ALL STEPS COMPLETE")
print("  Results : results/")
print("  Plot    : results/comparison_plots.png")
print("="*60)
