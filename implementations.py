import numpy as np


## utils
def calculate_mse(e):
    """Description: Calculate the mse for vector e.
    ---
    Input:
        e (array) - Difference between actual value and predictoion (e = y - tx.dot(w))
    ---
    Return:
        Scalar with the MSE loss value"""

    return 1 / 2 * np.mean(e**2)


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


def batch_iter(y, tx, batch_size, num_batches=1):
    """
    Generate a minibatch iterator for a dataset.
    """
    data_size = len(y)  # NUmber of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    # Generate an array of indexes indicating the start of each batch
    idxs = np.random.randint(max_batches, size=num_batches) * batch_size
    if remainder != 0:
        # Add an random offset to the start of each batch to eventually consider the remainder points
        idxs += np.random.randint(remainder + 1, size=num_batches)

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]


def sigmoid(t):
    return np.exp(-np.logaddexp(0, -t))


def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    # return np.squeeze(-loss).item() * (1 / y.shape[0])
    return -loss / y.shape[0]


### asked models

"""Linear regression with gradient descent """


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w
    for _ in range(max_iters):
        # compute loss, gradient
        grad, _ = compute_gradient(y, tx, w)
        # gradient w by descent update
        w = w - gamma * grad
    loss = calculate_mse(y - tx.dot(w))

    return w, loss


"""Linear regression using stochastic gradient descent"""


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD)"""

    w = initial_w

    for _ in range(max_iters):

        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            # compute gradient
            grad, _ = compute_gradient(y_batch, tx_batch, w)

            # update w
            w = w - gamma * grad

        # calculate loss
    loss = calculate_mse(y - tx.dot(w))

    return w, loss


"""Least squares regression using normal equation"""


def least_squares(y, tx):
    """Calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    mse = calculate_mse(y - tx.dot(w))
    return w, mse


"""Ridge regression using normal equations"""


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    mse = calculate_mse(y - tx.dot(w))
    return w, mse


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent (y∈{0,1})"""
    w = initial_w
    for _ in range(max_iters):
        # get loss and update w.
        pred = sigmoid(tx.dot(w))
        grad = tx.T.dot(pred - y) / y.shape[0]
        w = w - gamma * grad
    loss = calculate_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent (y∈{0,1},with regularization term λ∥w∥^2)"""

    w = initial_w

    # start the logistic regression
    for _ in range(max_iters):
        # get gradiente
        pred = sigmoid(tx.dot(w))
        gradient = tx.T.dot(pred - y) / y.shape[0] + 2 * lambda_ * w
        # update w
        w = w - gamma * gradient
    loss = calculate_loss(y, tx, w)
    return w, loss


### additional models


def logistic_regression_l1(y, tx, initial_w, max_iters, gamma, lambda_):
    """
    Logistic regression with L1 regularization using gradient descent.

    Args:
        y: labels (N,)
        tx: features matrix (N, D)
        initial_w: initial weights (D,)
        max_iters: number of iterations
        gamma: learning rate
        lambda_: L1 regularization strength

    Returns:
        loss: final regularized loss
        w: learned weights
    """
    w = initial_w
    N = y.shape[0]

    for _ in range(max_iters):
        pred = sigmoid(tx.dot(w))

        # Gradient of logistic loss
        grad = tx.T.dot(pred - y) / N

        # Subgradient of L1 penalty
        grad += lambda_ * np.sign(w)

        w -= gamma * grad

    # Logistic loss + L1 penalty
    eps = 1e-15
    pred = np.clip(sigmoid(tx.dot(w)), eps, 1 - eps)

    loss = calculate_loss(y, tx, w)
    # loss += lambda_ * np.sum(np.abs(w))
    return loss, w


def knn_train(y, tx):
    """
    KNN training: store the training data.

    Returns:
        model: tuple containing training features and labels
    """
    return tx.copy(), y.copy()


def knn_predict(tx_train, y_train, tx_test, k=5, batch_size=128):
    """
    KNN prediction using vectorized NumPy operations.

    batch_size controls memory usage.
    """
    predictions = np.zeros(tx_test.shape[0], dtype=int)
    k = min(k, len(y_train))

    # Squared norms of training samples
    train_norms = np.sum(tx_train**2, axis=1)

    for start in range(0, len(tx_test), batch_size):
        end = min(start + batch_size, len(tx_test))
        X_batch = tx_test[start:end]

        # Squared Euclidean distances:
        # ||x - z||² = ||x||² + ||z||² - 2*x.z
        distances = (
            np.sum(X_batch**2, axis=1)[:, None]
            + train_norms[None, :]
            - 2 * X_batch @ tx_train.T
        )

        # Numerical errors can produce tiny negative distances
        np.maximum(distances, 0, out=distances)

        neighbors = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]

        votes = np.mean(y_train[neighbors], axis=1)
        predictions[start:end] = (votes >= 0.5).astype(int)

    return predictions


def gaussian_naive_bayes(y, tx, eps=1e-9):
    """
    Train Gaussian Naive Bayes for binary classification.

    Args:
        y: labels (N,), values 0 or 1
        tx: features matrix (N, D)
        eps: minimum variance for numerical stability

    Returns:
        model: dictionary containing class priors, means and variances
    """
    classes = np.array([0, 1])

    priors = np.zeros(2)
    means = np.zeros((2, tx.shape[1]))
    variances = np.zeros((2, tx.shape[1]))

    for c in classes:
        X_c = tx[y == c]

        if len(X_c) == 0:
            raise ValueError(f"No training samples for class {c}")

        priors[c] = len(X_c) / len(y)
        means[c] = np.mean(X_c, axis=0)
        variances[c] = np.maximum(np.var(X_c, axis=0), eps)

    return {"priors": priors, "means": means, "variances": variances}


def gaussian_naive_bayes_predict(model, tx):
    """
    Predict binary labels using Gaussian Naive Bayes.

    Returns:
        predictions: predicted labels (N,)
    """
    priors = model["priors"]
    means = model["means"]
    variances = model["variances"]

    log_probs = np.zeros((tx.shape[0], 2))

    for c in [0, 1]:
        # Log prior
        log_probs[:, c] = np.log(priors[c])

        # Log Gaussian likelihood, summed over features
        log_probs[:, c] += -0.5 * np.sum(
            np.log(2 * np.pi * variances[c]) + ((tx - means[c]) ** 2) / variances[c],
            axis=1,
        )

    return np.argmax(log_probs, axis=1)
