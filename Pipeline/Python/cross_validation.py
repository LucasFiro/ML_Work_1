import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "Pipeline", "Python"))

import analysis as an
import implementations as imp
from load_data import load_data
from tools import best_threshold

LAMBDAS = [0.0, 1e-4, 1e-3, 3e-3, 1e-2]  # values of lambda compared
K = 5  # number of folds
SEED = 0  # seed of the fold assignment
GAMMA = 0.5  # gradient descent step size (same as run.py)
MAX_ITERS = 1000  # gradient descent iterations (same as run.py)
OUTPUT = os.path.join(ROOT, "figures", "cv_results.csv")


def k_fold_indices(n, k, seed):
    """
    Randomly split the indices 0..n-1 into k folds of (almost) equal size.
    ---
    Input:
        n (int) - Number of samples.
        k (int) - Number of folds.
        seed (int) - Random seed for reproducibility.
    ---
    Returns:
        A list of k numpy arrays containing the indices of each fold.
    """
    return np.array_split(np.random.default_rng(seed).permutation(n), k)


def main():
    """Run the cross-validation and save the F1 score of every (fold, lambda) pair."""
    x, _, y, _, _, names = load_data(data_dir=os.path.join(ROOT, "Data", "dataset"))
    folds = k_fold_indices(len(y), K, SEED)
    f1 = np.zeros((K, len(LAMBDAS)))
    thr = np.zeros((K, len(LAMBDAS)))

    for k in range(K):
        va = folds[k]
        tr = np.concatenate([folds[i] for i in range(K) if i != k])
        # Cleaning and preprocessing fitted on the training folds only
        x_tr, x_va = an.clean_special_codes(x[tr], x[va])
        tx_tr, tx_va, _ = an.preprocess(x_tr, x_va, names)
        tx_tr, tx_va = tx_tr.astype(np.float64), tx_va.astype(np.float64)

        for j, lambda_ in enumerate(LAMBDAS):
            start = time.time()
            w, _ = imp.reg_logistic_regression(
                y[tr], tx_tr, lambda_, np.zeros(tx_tr.shape[1]), MAX_ITERS, GAMMA
            )
            thr[k, j], f1[k, j] = best_threshold(y[va], imp.sigmoid(tx_va @ w))
            print(
                f"fold {k} | lambda={lambda_:g} | F1={f1[k, j]:.4f} "
                f"| threshold={thr[k, j]:.3f} | {time.time() - start:.0f}s",
                flush=True,
            )

    print("\nlambda      mean F1   std F1   mean threshold")
    for j, lambda_ in enumerate(LAMBDAS):
        print(
            f"{lambda_:<10g}  {f1[:, j].mean():.4f}    {f1[:, j].std():.4f}"
            f"   {thr[:, j].mean():.3f}"
        )

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    header = "lambda," + ",".join(f"f1_fold{k}" for k in range(K))
    rows = np.column_stack([LAMBDAS, f1.T])
    np.savetxt(OUTPUT, rows, delimiter=",", header=header, comments="", fmt="%.6g")
    print(f"Results saved to {OUTPUT}")


if __name__ == "__main__":
    main()
