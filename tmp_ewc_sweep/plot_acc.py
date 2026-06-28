"""Quick accuracy matrix plot for exp<IDX>_cnn_* directories.
No GPU needed, just reads performance_history.json.

Usage:
    python tmp_ewc_sweep/plot_acc.py 8
    python tmp_ewc_sweep/plot_acc.py 1 2 3
"""
import sys
import os
import glob

WORK_DIR = "/data/run01/scxk458/drift/cnn"
sys.path.insert(0, WORK_DIR)
os.chdir(WORK_DIR)

from src.analysis.performance import plot_cnn_performance

if len(sys.argv) < 2:
    print("Usage: python plot_acc.py <idx> [<idx2> ...]")
    sys.exit(1)

indices = sys.argv[1:]
exp_root = "experiments"

for idx in indices:
    pattern = os.path.join(exp_root, "exp{}_cnn_*/".format(idx))
    dirs = sorted(glob.glob(pattern))
    if not dirs:
        print("No dirs matched: {}".format(pattern))
        continue
    for d in dirs:
        d = d.rstrip("/")
        out = os.path.join(d, "drift_analysis")
        os.makedirs(out, exist_ok=True)
        try:
            plot_cnn_performance(d, out)
            print("[done] {}".format(os.path.basename(d)))
        except Exception as e:
            print("[fail] {}: {}".format(os.path.basename(d), e))
