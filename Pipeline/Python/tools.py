import numpy as np


def train_val_split(x, y, val_ratio=0.2, seed=0):
    """
    Description: Randomly split the input data (x, y) into training and validation sets based on the specified validation ratio.
    ---
    Input:
        x (numpy array) - The input features array.
        y (numpy array) - The target variable array.
        val_ratio (float) - The proportion of the data to be used for validation.
        seed (int) - The random seed for reproducibility.
    ---
    Returns:
        x_train (numpy array) - The training features array.
        y_train (numpy array) - The training target variable array.
        x_val (numpy array) - The validation features array.
        y_val (numpy array) - The validation target variable array.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_val = int(len(y) * val_ratio)
    val, tr = idx[:n_val], idx[n_val:]
    return x[tr], y[tr], x[val], y[val]


def standardize(x_train, *others):
    """
    Description: Standardize the input training data and other datasets using the mean and standard deviation of the training data.
    ---
    Input:
        x_train (numpy array) - The training features array to be standardized.
        *others (numpy arrays) - Additional datasets to be standardized using the same mean and standard deviation as x_train.
    ---
    Returns:
        A list of standardized datasets, including the standardized x_train and any other datasets provided.
    """
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    return [(a - mean) / std for a in (x_train, *others)]


def add_bias(x):
    """
    Description: Add a bias term (column of ones) to the input features array.
    ---
    Input:
        x (numpy array) - The input features array to which the bias term will be added.
    ---
    Returns:
        A new numpy array with a bias term added as the first column.
    """
    return np.hstack([np.ones((x.shape[0], 1)), x])


def ridge_fit(y, tx, lambda_):
    """
    Description: Fit a Ridge regression model using the normal equations, with the bias term (first column) not penalized.
    ---
    Input:
        y (numpy array) - The target variable array.
        tx (numpy array) - The input features array.
        lambda_ (float) - The regularization parameter for Ridge regression.
    ---
    Returns:
        A numpy array containing the fitted coefficients for the Ridge regression model.
    """
    reg = lambda_ * len(y) * np.eye(tx.shape[1])
    reg[0, 0] = 0.0
    gram = (tx.T @ tx).astype(np.float64)  # (D, D), petit
    return np.linalg.solve(gram + reg, (tx.T @ y).astype(np.float64))


def confusion(y_true, y_pred):
    """
    Description: Calculate the confusion matrix components (true positives, false positives, false negatives, true negatives) for binary classification.
    ---
    Input:
        y_true (numpy array) - The true labels of the target variable.
        y_pred (numpy array) - The predicted labels of the target variable.
    ---
    Returns:
        tp (int) - The number of true positives.
        fp (int) - The number of false positives.
        fn (int) - The number of false negatives.
        tn (int) - The number of true negatives.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return tp, fp, fn, tn


def accuracy(y_true, y_pred):
    """
    Description: Calculate the accuracy of predictions by comparing the true labels with the predicted labels.
    ---
    Input:
        y_true (numpy array) - The true labels of the target variable.
        y_pred (numpy array) - The predicted labels of the target variable.
    ---
    Returns:
        A float representing the accuracy of the predictions, calculated as the proportion of correct predictions.
    """
    return np.mean(y_true == y_pred)


def f1_score(y_true, y_pred):
    """
    Description: Calculate the F1 score, which is the harmonic mean of precision and recall, for binary classification.
    ---
    Input:
        y_true (numpy array) - The true labels of the target variable.
        y_pred (numpy array) - The predicted labels of the target variable.
    ---
    Returns:
        A float representing the F1 score, which balances precision and recall in binary classification tasks.
    """
    tp, fp, fn, _ = confusion(y_true, y_pred)
    if tp == 0:
        return 0.0
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def best_threshold(y_true, scores, n_grid=200):
    """
    Description: Find the optimal threshold for a binary classifier that maximizes the F1 score.
    ---
    Input:
        y_true (numpy array) - The true labels of the target variable.
        scores (numpy array) - The predicted scores of the target variable.
        n_grid (int) - The number of thresholds to consider.
    ---
    Returns:
        A tuple containing the optimal threshold and the corresponding F1 score.
    """
    ths = np.quantile(scores, np.linspace(0.5, 0.999, n_grid))
    f1s = [f1_score(y_true, (scores >= t).astype(float)) for t in ths]
    i = int(np.argmax(f1s))
    return ths[i], f1s[i]
