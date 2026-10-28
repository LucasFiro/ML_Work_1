import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "Pipeline", "Python"))

import analysis as an
import implementations as imp
import plotting as pl
from load_data import load_data
from tools import accuracy, best_threshold, f1_score, ridge_fit, train_val_split

LAMBDA, GAMMA, MAX_ITERS, SEED = 1e-3, 0.5, 1000, 0  # same as run.py
FIG_DIR = os.path.join(ROOT, "figures")


def prepare(x, x_test, y, names, clean=True, flags=True):
    """
    Clean and preprocess the data, then split it into train/validation.
    ---
    Input:
        x, x_test (numpy arrays) - Raw training and test features.
        y (numpy array) - Training labels in {0, 1}.
        names (numpy array) - Feature names.
        clean (bool) - Whether to decode the BRFSS special answer codes.
        flags (bool) - Whether to add missing-value indicators.
    ---
    Returns:
        tr_x, tr_y, va_x, va_y (numpy arrays) - The train/validation split in float64.
    """
    if clean:
        x, x_test = an.clean_special_codes(x, x_test)
    tx, _, _ = an.preprocess(x, x_test, names, add_missing_flags=flags)
    return train_val_split(tx.astype(np.float64), y, 0.2, SEED)


def logistic(y, tx, lambda_=LAMBDA, max_iters=MAX_ITERS):
    """Train the (regularized) logistic regression from w = 0 and return w."""
    w, _ = imp.reg_logistic_regression(
        y, tx, lambda_, np.zeros(tx.shape[1]), max_iters, GAMMA
    )
    return w


def evaluate(y_val, scores):
    """Return (F1, accuracy, threshold) with the F1-optimal threshold."""
    threshold, f1 = best_threshold(y_val, scores)
    return f1, accuracy(y_val, (scores >= threshold).astype(float)), threshold


def main():
    """Run all experiments and print the comparison and ablation tables."""
    x, x_test, y, _, _, names = load_data(
        data_dir=os.path.join(ROOT, "Data", "dataset")
    )
    results = {}

    # ---- Model comparison (cleaned features + missing flags) ----
    tr_x, tr_y, va_x, va_y = prepare(x, x_test, y, names)
    results["always negative"] = (0.0, 1 - va_y.mean(), np.nan)

    nb = imp.gaussian_naive_bayes(tr_y, tr_x)
    pred = imp.gaussian_naive_bayes_predict(nb, va_x).astype(float)
    results["gaussian naive bayes"] = (
        f1_score(va_y, pred),
        accuracy(va_y, pred),
        np.nan,
    )

    try:
        w, _ = imp.least_squares(tr_y, tr_x)
        results["least squares"] = evaluate(va_y, va_x @ w)
    except np.linalg.LinAlgError:
        # Identical missing-value indicators make tx^T tx singular
        results["least squares"] = (np.nan, np.nan, np.nan)
        print("least squares: tx^T tx is singular (duplicated columns)")

    w = ridge_fit(tr_y, tr_x, 1e-6)
    results["ridge (lambda=1e-6)"] = evaluate(va_y, va_x @ w)

    w = logistic(tr_y, tr_x, lambda_=0.0)
    results["logistic regression"] = evaluate(va_y, imp.sigmoid(va_x @ w))

    w = logistic(tr_y, tr_x)
    scores = imp.sigmoid(va_x @ w)
    results["final: L2 logistic (lambda=1e-3)"] = evaluate(va_y, scores)
    fixed = (scores >= 0.5).astype(float)
    results["final, threshold 0.5"] = (
        f1_score(va_y, fixed),
        accuracy(va_y, fixed),
        0.5,
    )
    final_threshold = results["final: L2 logistic (lambda=1e-3)"][2]

    # ---- Ablation: remove one component at a time ----
    tr_x, tr_y, va_x, va_y_nc = prepare(x, x_test, y, names, clean=False)
    w = logistic(tr_y, tr_x)
    results["final without code cleaning"] = evaluate(va_y_nc, imp.sigmoid(va_x @ w))

    tr_x, tr_y, va_x, va_y_nf = prepare(x, x_test, y, names, flags=False)
    w = logistic(tr_y, tr_x)
    results["final without missing flags"] = evaluate(va_y_nf, imp.sigmoid(va_x @ w))

    print(f"\n{'experiment':40s} {'F1':>7s} {'acc':>7s} {'thr':>7s}")
    for name, (f1, acc, thr) in results.items():
        print(f"{name:40s} {f1:7.4f} {acc:7.4f} {thr:7.3f}")

    os.makedirs(FIG_DIR, exist_ok=True)
    with open(os.path.join(FIG_DIR, "ablation_results.csv"), "w") as f:
        f.write("experiment,f1,accuracy,threshold\n")
        for name, (f1, acc, thr) in results.items():
            f.write(f"{name},{f1:.6f},{acc:.6f},{thr:.6f}\n")

    # ---- Report figure ----
    cv_path = os.path.join(FIG_DIR, "cv_results.csv")
    if os.path.exists(cv_path):
        cv = np.loadtxt(cv_path, delimiter=",", skiprows=1, ndmin=2)
        pl.plot_report_figure(
            list(cv[:, 0]),
            cv[:, 1:].T,
            va_y,
            scores,
            final_threshold,
            os.path.join(FIG_DIR, "fig_tuning.pdf"),
        )
        print("Report figure saved to figures/fig_tuning.pdf")


if __name__ == "__main__":
    main()
