import os
import sys

import numpy as np

# Make the modules in Pipeline/Python importable from the repository root
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "Pipeline", "Python"))

import analysis as an
from helpers import create_csv_submission
from load_data import load_data
from tools import accuracy, best_threshold, ridge_fit, train_val_split

LAMBDA = 1e-6  # ridge regularization strength
SEED = 0  # seed of the train/validation split, for reproducibility
OUTPUT = "submission.csv"


def main():
    """Train the model, choose the threshold and write the submission file."""
    np.random.seed(SEED)

    # 1. Load data (labels y_train are in {0, 1})
    x_train, x_test, y_train, _, test_ids, names = load_data()
    print(f"Loaded x_train {x_train.shape}, x_test {x_test.shape}")

    # 2. Preprocess train and test with statistics computed on train only
    tx_train, tx_test, _ = an.preprocess(x_train, x_test, names)

    # 3. Threshold selection on a held-out validation split
    tr_x, tr_y, va_x, va_y = train_val_split(tx_train, y_train, 0.2, SEED)
    w = ridge_fit(tr_y, tr_x, LAMBDA)
    threshold, f1_val = best_threshold(va_y, va_x @ w)
    acc_val = accuracy(va_y, (va_x @ w >= threshold).astype(float))
    print(f"Validation: F1 = {f1_val:.4f}, accuracy = {acc_val:.4f}")

    # 4. Refit on all training data, predict the test set
    w = ridge_fit(y_train, tx_train, LAMBDA)
    y_pred = np.where(tx_test @ w >= threshold, 1, -1)
    print(f"Predicted positives on test: {np.mean(y_pred == 1):.1%}")

    # 5. Write the submission ({-1, 1} labels, same ids as x_test.csv)
    create_csv_submission(test_ids, y_pred, OUTPUT)
    print(f"Submission written to {OUTPUT}")


if __name__ == "__main__":
    main()