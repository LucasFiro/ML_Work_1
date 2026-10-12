"""
How to use the code:
After the standard "python3 main.py", add one of the following steps:
Steps : load | explore | plot | features | model | all   (default : all)
Examples :
    python3 main.py load
    python3 main.py plot
    python3 main.py features model
"""
import sys
import numpy as np
from load_data import load_data
import analysis as an
import plotting as pl
from tools import train_val_split, ridge_fit, f1_score, accuracy, best_threshold


def step_load(d):
    """
    Description: Load the dataset and display basic information about the training and testing data.
    ---
    Input:
        d (tuple) - A tuple containing the loaded data: (x_train, x_test, y_train, train_ids, test_ids, feature_names).
    ---
    Returns:
        None
    """
    x_train, x_test, y_train, *_ = d
    print(f"x_train {x_train.shape} | x_test {x_test.shape} | y_train {y_train.shape}")
    print("Classes :", an.class_balance(y_train))


def step_explore(d):
    """
    Description: Perform exploratory data analysis on the training data, including missing value statistics and feature correlations with the target variable.
    ---
    Input:
        d (tuple) - A tuple containing the loaded data: (x_train, x_test, y_train, train_ids, test_ids, feature_names).
    ---
    Returns:
        None
    """
    x_train, _, y_train, _, _, names = d
    miss = an.missing_ratio(x_train)
    print(f"NaN globaux : {np.isnan(x_train).mean():.1%} | "
          f"features >80% NaN : {(miss > 0.8).sum()} | "
          f"features sans NaN : {(miss == 0).sum()}")
    corr = an.target_correlation(x_train, y_train)
    top = np.argsort(-np.abs(corr))[:10]
    print("Top 10 corrélations :", [(str(names[i]), round(float(corr[i]), 3)) for i in top])


def step_plot(d):
    """
    Description: Generate and save visualizations for class balance, missing values, top feature correlations, and F1 score vs. decision threshold.
    ---
    Input:
        d (tuple) - A tuple containing the loaded data: (x_train, x_test, y_train, train_ids, test_ids, feature_names).
    ---
    Returns:
        None
    """
    x_train, _, y_train, _, _, names = d
    pl.plot_class_balance(y_train)
    pl.plot_missing(an.missing_ratio(x_train))
    pl.plot_top_correlations(an.target_correlation(x_train, y_train), names)
    print("Figures sauvegardées dans figures/")


def step_features(d):
    """
    Description: Perform feature extraction, including selection, imputation, and standardization, on the training and testing data.
    ---
    Input:
        d (tuple) - A tuple containing the loaded data: (x_train, x_test, y_train, train_ids, test_ids, feature_names).
    ---
    Returns:
        tx_train (numpy array) - The preprocessed training features array.
        tx_test (numpy array) - The preprocessed testing features array.
        fnames (list) - The names of the selected features after preprocessing.
    """
    x_train, x_test, _, _, _, names = d
    tx_train, tx_test, fnames = an.preprocess(x_train, x_test, names)
    print(f"Features : {x_train.shape[1]} -> {tx_train.shape[1]} (biais et indicateurs de NaN inclus)")
    print(f"Train {tx_train.shape} | Test {tx_test.shape}")
    return tx_train, tx_test, fnames


def step_model(d):
    """
    Description: Test a Ridge regression model with an optimized threshold for the F1 score.
    ---
    Input:
        d (tuple) - A tuple containing the loaded data: (x_train, x_test, y_train, train_ids, test_ids, feature_names).
    ---
    Returns:
        None
    """
    _, _, y_train, *_ = d
    tx_train, _, _ = step_features(d)
    tr_x, tr_y, va_x, va_y = train_val_split(tx_train, y_train, val_ratio=0.2, seed=0)
    for lambda_ in [1e-6, 1e-4, 1e-2]:
        w = ridge_fit(tr_y, tr_x, lambda_)
        scores = va_x @ w
        th, f1 = best_threshold(va_y, scores)
        acc = accuracy(va_y, (scores >= th).astype(float))
        print(f"ridge lambda={lambda_:.0e} | seuil={th:.3f} | F1 val={f1:.4f} | acc={acc:.4f}")
    pl.plot_threshold_curve(va_y, va_x @ ridge_fit(tr_y, tr_x, 1e-4), f1_score)


STEPS = {"load": step_load, "explore": step_explore, "plot": step_plot,
         "features": step_features, "model": step_model}


def main():
    """
    Description: Main function to run the pipeline steps.
    """
    args = sys.argv[1:] or ["all"]
    if "all" in args:
        args = list(STEPS)
    unknown = [a for a in args if a not in STEPS]
    if unknown:
        sys.exit(f"Unknown steps : {unknown}. Choice : {list(STEPS) + ['all']}")
    d = load_data()
    for a in args:
        print(f"\n=== {a} ===")
        STEPS[a](d)


if __name__ == "__main__":
    main()
