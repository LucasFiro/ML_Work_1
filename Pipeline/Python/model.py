import implementations as imp
import numpy as np


def compare_models(
    y_train,
    tx_train,
    y_test,
    tx_test,
    max_iters=1000,
    gamma=0.01,
    lambda_l2=0.01,
    lambda_l1=0.01,
    k=5,
    threshold=0.5,
):
    """
    Train and compare:
    - Logistic Regression
    - Logistic Regression L2
    - Logistic Regression L1
    - KNN
    - Gaussian Naive Bayes
    - Baseline: predict every label as 0

    Returns:
        results: dictionary of metrics for each model
    """

    y_train = np.asarray(y_train).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)
    tx_train = np.asarray(tx_train)
    tx_test = np.asarray(tx_test)

    if tx_train.shape[1] != tx_test.shape[1]:
        raise ValueError("Train and test must have the same features.")

    if len(y_train) != len(tx_train) or len(y_test) != len(tx_test):
        raise ValueError("Labels and feature matrices have incompatible sizes.")

    if not np.all(np.isin(y_train, [0, 1])) or not np.all(np.isin(y_test, [0, 1])):
        raise ValueError("Labels must be binary: 0 or 1.")

    initial_w = np.zeros(tx_train.shape[1])

    # Predictions from all models
    predictions = {}

    # 1. Logistic Regression
    w_log, loss_log = imp.logistic_regression(
        y_train, tx_train, initial_w.copy(), max_iters, gamma
    )
    predictions["Logistic Regression"] = (
        imp.sigmoid(tx_test.dot(w_log)) >= threshold
    ).astype(int)

    # 2. Logistic Regression L2
    w_l2, loss_l2 = imp.reg_logistic_regression(
        y_train, tx_train, lambda_l2, initial_w.copy(), max_iters, gamma
    )
    predictions["Logistic Regression L2"] = (
        imp.sigmoid(tx_test.dot(w_l2)) >= threshold
    ).astype(int)

    # 3. Logistic Regression L1
    loss_l1, w_l1 = imp.logistic_regression_l1(
        y_train, tx_train, initial_w.copy(), max_iters, gamma, lambda_l1
    )
    predictions["Logistic Regression L1"] = (
        imp.sigmoid(tx_test.dot(w_l1)) >= threshold
    ).astype(int)

    # 4. KNN
    predictions["KNN"] = imp.knn_predict(tx_train, y_train, tx_test, k=k)

    # 5. Gaussian Naive Bayes
    nb_model = imp.gaussian_naive_bayes(y_train, tx_train)
    predictions["Gaussian Naive Bayes"] = imp.gaussian_naive_bayes_predict(
        nb_model, tx_test
    )

    # 6. Baseline: always predict 0
    predictions["Baseline (all zeros)"] = np.zeros(len(y_test), dtype=int)

    def compute_metrics(y_true, y_pred):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        accuracy = (tp + tn) / len(y_true)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        specificity = tn / (tn + fp) if tn + fp > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        balanced_accuracy = (recall + specificity) / 2

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "F1": f1,
            "balanced_accuracy": balanced_accuracy,
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
        }

    results = {}

    for name, y_pred in predictions.items():
        results[name] = compute_metrics(y_test, y_pred)

    # Display results
    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "F1",
        "balanced_accuracy",
    ]

    print("\nMODEL COMPARISON")
    print("-" * 100)
    print(f"{'Model':30s}" + "".join(f"{m:>13s}" for m in metric_names))
    print("-" * 100)

    for name, metrics in results.items():
        print(f"{name:30s}" + "".join(f"{metrics[m]:13.4f}" for m in metric_names))

    print("\nCONFUSION COUNTS")
    print("-" * 60)
    print(f"{'Model':30s}{'TP':>8s}{'TN':>8s}{'FP':>8s}{'FN':>8s}")

    for name, metrics in results.items():
        print(
            f"{name:30s}"
            f"{metrics['TP']:8d}"
            f"{metrics['TN']:8d}"
            f"{metrics['FP']:8d}"
            f"{metrics['FN']:8d}"
        )

    return results
