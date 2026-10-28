"""Train and compare all models of the report.

For every model (logistic regression without regularization, L1 / L2
regularized logistic regression, k-NN, Gaussian Naive Bayes) this script:
  1. optionally selects the hyperparameters on a held-out validation split,
  2. selects the decision threshold maximizing the validation F1-score,
  3. refits the model on ALL training data and predicts the test set,
  4. writes one submission CSV per model,
  5. records every number needed for the report (threshold, F1, accuracy,
     hyperparameters, ...) in results/results.json, results/summary.csv and
     results/table_rows.tex (rows ready to paste in the LaTeX table).

Usage (from the repository root):
    python compare_models.py                 # all models
    python compare_models.py l2 knn          # only some models
"""

import csv
import json
import os
import sys
import time

import numpy as np

# Make the modules in Pipeline/Python importable from the repository root
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "Pipeline", "Python"))

import analysis as an
import implementations as imp
from helpers import create_csv_submission
from load_data import load_data
from tools import train_val_split

# ----------------------------------------------------------------------------
# Global settings (all of them should be reported in the paper)
# ----------------------------------------------------------------------------
SEED = 0  # seed of the train/validation split
VAL_RATIO = 0.2  # fraction of the training data kept for validation
CLEAN_CODES = True  # replace BRFSS "don't know / refused / none" codes
TUNE_HYPERPARAMS = True  # False: use the "default" params of each model
THRESHOLDS = np.linspace(0.05, 0.95, 181)  # candidate thresholds (step 0.005)

GAMMA = 0.5  # gradient step size (logistic regressions)
MAX_ITERS = 1000  # number of gradient steps (logistic regressions)

KNN_MAX_TRAIN = 50_000  # k-NN reference set size (None = all, but very slow)
KNN_CHUNK = 256  # number of queries processed at once by k-NN

OUT_DIR = os.path.join(ROOT, "results")


# ----------------------------------------------------------------------------
# Metrics and threshold selection
# ----------------------------------------------------------------------------
def sigmoid(z):
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def metrics(y, y_hat):
    """Accuracy, precision, recall and F1 for labels in {0, 1}."""
    tp = np.sum((y_hat == 1) & (y == 1))
    fp = np.sum((y_hat == 1) & (y == 0))
    fn = np.sum((y_hat == 0) & (y == 1))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 0.0
    return {
        "accuracy": float(np.mean(y_hat == y)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def sweep_threshold(y, scores):
    """Return (best threshold, its metrics, F1 curve over THRESHOLDS)."""
    f1_curve = np.array(
        [metrics(y, (scores >= t).astype(float))["f1"] for t in THRESHOLDS]
    )
    best = int(np.argmax(f1_curve))
    thr = float(THRESHOLDS[best])
    return thr, metrics(y, (scores >= thr).astype(float)), f1_curve


# ----------------------------------------------------------------------------
# Models. Each fit function takes (y, tx, params) and returns
# (score_fn, info) where score_fn(x) gives a score in [0, 1] for each row of x.
# ----------------------------------------------------------------------------
def fit_logreg(y, tx, params):
    """Logistic regression without regularization, starting from w = 0."""
    w, _ = imp.logistic_regression(y, tx, np.zeros(tx.shape[1]), MAX_ITERS, GAMMA)
    return lambda x: imp.sigmoid(x @ w), {"n_nonzero_weights": int(np.sum(w != 0))}


def fit_l2(y, tx, params):
    """L2-regularized logistic regression, starting from w = 0."""
    w, _ = imp.reg_logistic_regression(
        y, tx, params["lambda"], np.zeros(tx.shape[1]), MAX_ITERS, GAMMA
    )
    return lambda x: imp.sigmoid(x @ w), {"n_nonzero_weights": int(np.sum(w != 0))}


def fit_l1(y, tx, params):
    """L1-regularized logistic regression by proximal gradient (ISTA).

    Objective: mean logistic loss + lambda * ||w||_1. Each iteration does a
    gradient step on the loss, then soft-thresholding. Constant columns (the
    bias, if any) are not penalized.
    """
    w, _ = imp.logistic_regression_l1(
        y, tx, np.zeros(tx.shape[1]), MAX_ITERS, GAMMA, params["lambda"]
    )
    return lambda x: imp.sigmoid(x @ w), {"n_nonzero_weights": int(np.sum(w != 0))}


def fit_knn(y, tx, params):
    """k-NN: score = fraction of positives among the k nearest neighbours.

    Euclidean distance on the standardized features. The reference set is a
    random subsample of at most KNN_MAX_TRAIN training points (fixed seed).
    """
    k = params["k"]
    keep = tx.std(axis=0) > 0  # drop constant columns
    idx = np.arange(len(y))
    if KNN_MAX_TRAIN is not None and len(y) > KNN_MAX_TRAIN:
        idx = np.random.RandomState(SEED).choice(len(y), KNN_MAX_TRAIN, replace=False)
    ref = tx[idx][:, keep].astype(np.float32)
    ref_y = y[idx].astype(np.float32)
    ref_sq = (ref**2).sum(axis=1)

    def score_fn(x):
        x = x[:, keep].astype(np.float32)
        out = np.empty(len(x))
        for s in range(0, len(x), KNN_CHUNK):
            q = x[s : s + KNN_CHUNK]
            # squared distance up to a per-query constant (irrelevant for ranking)
            d2 = ref_sq[None, :] - 2.0 * (q @ ref.T)
            nn = np.argpartition(d2, k, axis=1)[:, :k]
            out[s : s + KNN_CHUNK] = ref_y[nn].mean(axis=1)
        return out

    return score_fn, {"n_reference_points": int(len(idx))}


def fit_gnb(y, tx, params):
    """Gaussian Naive Bayes: per-class mean/variance of each feature.

    The variance of each feature gets + var_smoothing * (largest feature
    variance) for numerical stability. Score = posterior P(y = 1 | x).
    """
    keep = tx.std(axis=0) > 0  # drop constant columns
    x = tx[:, keep]
    eps = params["var_smoothing"] * x.var(axis=0).max()
    stats = []
    for c in (0, 1):
        xc = x[y == c]
        stats.append((xc.mean(axis=0), xc.var(axis=0) + eps, np.log(len(xc) / len(x))))

    def score_fn(xt):
        xt = xt[:, keep]
        log_post = []
        for mu, var, log_prior in stats:
            ll = -0.5 * np.sum(np.log(2 * np.pi * var) + (xt - mu) ** 2 / var, axis=1)
            log_post.append(log_prior + ll)
        return sigmoid(log_post[1] - log_post[0])

    return score_fn, {}


MODELS = {
    "logreg": {
        "label": "Logistic regr. (no reg.)",
        "fit": fit_logreg,
        "default": {},
        "grid": [{}],
    },
    "l2": {
        "label": "L2 logistic regr.",
        "fit": fit_l2,
        "default": {"lambda": 1e-3},
        "grid": [{"lambda": v} for v in (1e-4, 1e-3, 1e-2)],
    },
    "l1": {
        "label": "L1 logistic regr.",
        "fit": fit_l1,
        "default": {"lambda": 1e-4},
        "grid": [{"lambda": v} for v in (1e-5, 1e-4, 1e-3)],
    },
    "knn": {
        "label": "k-NN",
        "fit": fit_knn,
        "default": {"k": 51},
        "grid": [{"k": v} for v in (15, 51, 101)],
    },
    "gnb": {
        "label": "Gaussian Naive Bayes",
        "fit": fit_gnb,
        "default": {"var_smoothing": 1e-6},
        "grid": [{"var_smoothing": v} for v in (1e-9, 1e-6, 1e-3)],
    },
}


# ----------------------------------------------------------------------------
# Experiment for one model
# ----------------------------------------------------------------------------
def run_model(name, data):
    """Tune, choose the threshold, refit on all data, write the submission."""
    spec = MODELS[name]
    grid = spec["grid"] if TUNE_HYPERPARAMS else [spec["default"]]
    print(f"\n=== {spec['label']} ({len(grid)} configuration(s)) ===")

    # 1-2. Hyperparameters and threshold on the validation split
    tried, best = [], None
    for params in grid:
        t0 = time.time()
        score_fn, _ = spec["fit"](data["tr_y"], data["tr_x"], params)
        scores = score_fn(data["va_x"])
        thr, m, curve = sweep_threshold(data["va_y"], scores)
        dt = time.time() - t0
        print(
            f"  params={params}  thr={thr:.3f}  F1={m['f1']:.4f}  "
            f"acc={m['accuracy']:.4f}  ({dt:.0f}s)"
        )
        tried.append({"params": params, "threshold": thr, **m})
        if best is None or m["f1"] > best["metrics"]["f1"]:
            best = {
                "params": params,
                "threshold": thr,
                "metrics": m,
                "curve": curve,
                "val_pos_rate": float(np.mean(scores >= thr)),
            }

    # 3. Refit on all training data and predict the test set
    t0 = time.time()
    score_fn, info = spec["fit"](data["y_train"], data["tx_train"], best["params"])
    test_scores = score_fn(data["tx_test"])
    fit_time = time.time() - t0
    y_pred = np.where(test_scores >= best["threshold"], 1, -1)

    # 4. Submission and F1-vs-threshold curve
    os.makedirs(OUT_DIR, exist_ok=True)
    sub_path = os.path.join(OUT_DIR, f"submission_{name}.csv")
    create_csv_submission(data["test_ids"], y_pred, sub_path)
    np.savetxt(
        os.path.join(OUT_DIR, f"f1_curve_{name}.csv"),
        np.column_stack([THRESHOLDS, best["curve"]]),
        delimiter=",",
        header="threshold,val_f1",
        comments="",
    )
    print(
        f"  -> best params {best['params']}, threshold {best['threshold']:.3f}, "
        f"val F1 {best['metrics']['f1']:.4f}; written {sub_path}"
    )

    return {
        "model": name,
        "label": spec["label"],
        "best_params": best["params"],
        "threshold": best["threshold"],
        **{f"val_{k}": v for k, v in best["metrics"].items()},
        "val_predicted_positive_rate": best["val_pos_rate"],
        "test_predicted_positive_rate": float(np.mean(y_pred == 1)),
        "refit_and_predict_time_s": round(fit_time, 1),
        "info": info,
        "all_configurations": tried,
        "submission_file": os.path.relpath(sub_path, ROOT),
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    names = sys.argv[1:] or list(MODELS)
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        sys.exit(f"Unknown model(s) {unknown}. Choose among {list(MODELS)}")
    np.random.seed(SEED)

    # Load and preprocess (statistics computed on train only)
    x_train, x_test, y_train, _, test_ids, feat_names = load_data()
    print(f"Loaded x_train {x_train.shape}, x_test {x_test.shape}")
    if CLEAN_CODES:
        x_train, x_test = an.clean_special_codes(x_train, x_test)
    tx_train, tx_test, _ = an.preprocess(x_train, x_test, feat_names)
    tx_train, tx_test = tx_train.astype(np.float64), tx_test.astype(np.float64)

    # Same train/validation split for every model
    tr_x, tr_y, va_x, va_y = train_val_split(tx_train, y_train, VAL_RATIO, SEED)
    data = {
        "tx_train": tx_train,
        "y_train": y_train,
        "tx_test": tx_test,
        "test_ids": test_ids,
        "tr_x": tr_x,
        "tr_y": tr_y,
        "va_x": va_x,
        "va_y": va_y,
    }

    results = [run_model(n, data) for n in names]

    # Global setup (to report in the paper)
    setup = {
        "seed": SEED,
        "val_ratio": VAL_RATIO,
        "clean_special_codes": CLEAN_CODES,
        "tune_hyperparams": TUNE_HYPERPARAMS,
        "threshold_grid": [
            float(THRESHOLDS[0]),
            float(THRESHOLDS[-1]),
            float(THRESHOLDS[1] - THRESHOLDS[0]),
        ],
        "gamma": GAMMA,
        "max_iters": MAX_ITERS,
        "knn_max_train": KNN_MAX_TRAIN,
        "n_features_after_preprocessing": int(tx_train.shape[1]),
        "n_train_total": int(len(y_train)),
        "n_train_split": int(len(tr_y)),
        "n_validation": int(len(va_y)),
        "positive_rate_train": float(np.mean(y_train == 1)),
        "positive_rate_validation": float(np.mean(va_y == 1)),
    }
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump({"setup": setup, "results": results}, f, indent=2)

    cols = [
        "model",
        "best_params",
        "threshold",
        "val_f1",
        "val_accuracy",
        "val_precision",
        "val_recall",
        "test_predicted_positive_rate",
    ]
    with open(os.path.join(OUT_DIR, "summary.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for r in results:
            writer.writerow(
                [json.dumps(r[c]) if c == "best_params" else r[c] for c in cols]
            )

    # Rows for the LaTeX results table
    rows = [
        f"{r['label']} & z-score & {r['threshold']:.3f} & "
        f"{r['val_f1']:.3f} & {r['val_accuracy']:.3f} \\\\"
        for r in results
    ]
    with open(os.path.join(OUT_DIR, "table_rows.tex"), "w") as f:
        f.write("\n".join(rows) + "\n")

    print("\n=== Summary (validation) ===")
    print(f"{'model':<26}{'thr':>7}{'F1':>8}{'acc':>8}{'prec':>8}{'rec':>8}  params")
    for r in results:
        print(
            f"{r['label']:<26}{r['threshold']:>7.3f}{r['val_f1']:>8.4f}"
            f"{r['val_accuracy']:>8.4f}{r['val_precision']:>8.4f}"
            f"{r['val_recall']:>8.4f}  {r['best_params']}"
        )
    print(f"\nAll outputs are in {OUT_DIR}")
    print("LaTeX rows:\n" + "\n".join(rows))


if __name__ == "__main__":
    main()
