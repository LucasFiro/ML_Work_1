import os
import sys

import numpy as np

# Make the modules in Pipeline/Python importable from the repository root
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "Pipeline", "Python"))

import analysis as an
import implementations as imp
from helpers import create_csv_submission
from load_data import load_data
from tools import accuracy, best_threshold, train_val_split

LAMBDA = 1e-3  # L2 regularization strength
GAMMA = 0.5  # gradient descent step size
MAX_ITERS = 1000  # number of gradient descent steps
SEED = 0  # seed of the train/validation split, for reproducibility
CLEAN_CODES = True  # replace BRFSS "don't know / refused / none" codes
OUTPUT = "submission.csv"


def train(y, tx):
    """Train the L2-regularized logistic regression, starting from w = 0."""
    w, _ = imp.reg_logistic_regression(
        y, tx, LAMBDA, np.zeros(tx.shape[1]), MAX_ITERS, GAMMA
    )
    return w


def main():
    """Train the model, choose the threshold and write the submission file."""
    np.random.seed(SEED)

    # 1. Load data (labels y_train are in {0, 1})
    x_train, x_test, y_train, _, test_ids, names = load_data()
    print(f"Loaded x_train {x_train.shape}, x_test {x_test.shape}")

    # 2. Clean BRFSS special codes, then preprocess train and test
    #    (all statistics are computed on train only)
    if CLEAN_CODES:
        x_train, x_test = an.clean_special_codes(x_train, x_test)
    tx_train, tx_test, _ = an.preprocess(x_train, x_test, names)
    # float64 for numerically accurate gradients (the data is stored in float32)
    tx_train, tx_test = tx_train.astype(np.float64), tx_test.astype(np.float64)

    # 3. Threshold selection on a held-out validation split
    tr_x, tr_y, va_x, va_y = train_val_split(tx_train, y_train, 0.2, SEED)
    w = train(tr_y, tr_x)
    scores = imp.sigmoid(va_x @ w)
    threshold, f1_val = best_threshold(va_y, scores)
    acc_val = accuracy(va_y, (scores >= threshold).astype(float))
    print(f"Validation: F1 = {f1_val:.4f}, accuracy = {acc_val:.4f}")
    print(f"Chosen threshold: {threshold:.3f}")

    # 4. Refit on all training data, predict the test set
    w = train(y_train, tx_train)
    y_pred = np.where(imp.sigmoid(tx_test @ w) >= threshold, 1, -1)
    print(f"Predicted positives on test: {np.mean(y_pred == 1):.1%}")

    # 5. Write the submission ({-1, 1} labels, same ids as x_test.csv)
    create_csv_submission(test_ids, y_pred, OUTPUT)
    print(f"Submission written to {OUTPUT}")


if __name__ == "__main__":
    main()
