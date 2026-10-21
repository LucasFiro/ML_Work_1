import numpy as np


def calculate_mse(e):
    """Calculate the mse for vector e."""
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
    return 1.0 / (1 + np.exp(-t))


def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(-loss).item() * (1 / y.shape[0])


"""Linear regression with gradient descent """

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        loss = calculate_mse(err)
        # gradient w by descent update
        w = w - gamma * grad

    return w, loss



"""Linear regression using stochastic gradient descent"""

def mean_squared_error_sgd(y, tx, initial_w, batch_size, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD) """

    w = initial_w

    for _ in range(max_iters):

        for y_batch, tx_batch in batch_iter(
            y, tx, batch_size=batch_size, num_batches=1
        ):
            # compute gradient 
            grad, err = compute_gradient(y_batch, tx_batch, w)

            # update w 
            w = w - gamma * grad
    
        # calculate loss
        loss = calculate_mse(err)


    return w, loss



"""Least squares regression using normal equation"""

def least_squares(y, tx):
    """Calculate the least squares solution."""
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    mse = calculate_mse(y, tx, w)
    return w, mse



"""Ridge regression using normal equations"""

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    mse = calculate_mse(y, tx, w)
    return w, mse



"""Logistic regression using gradient descent (y∈{0,1})"""
def logistic_regression(y,tx,initial_w, max_iters,gamma):
    w = initial_w
    for _ in range(max_iters):
            # get loss and update w.
            pred = sigmoid(tx.dot(w))
            grad = tx.T.dot(pred - y) * (1 / y.shape[0])
            w -= gamma * grad

    return calculate_loss(y, tx, w), w



"""Regularized logistic regression using gradient descent
(y∈{0,1},with regularization term λ∥w∥^2)"""

def reg_logistic_regression(y,tx,lambda_, initial_w,max_iters,gamma):
    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        #get gradiente
        pred = sigmoid(tx.dot(w))
        gradient = tx.T.dot(pred - y) * (1 / y.shape[0]) + 2 * lambda_ * w
        # update w
        w -= gamma * gradient 

    return calculate_loss(y, tx, w), w