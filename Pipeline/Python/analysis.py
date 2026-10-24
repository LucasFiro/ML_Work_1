import numpy as np

# Column that contains admin information (no need to keep them)
ADMIN_COLS = ["SEQNO", "_PSU", "IDATE", "IMONTH", "IDAY", "IYEAR", "FMONTH", "DISPCODE"]


# -------------- Data exploration --------------
def missing_ratio(x):
    """
    Description: Calculate the ratio of missing values (NaN) for each column in the input array.
    ---
    Input:
        x (numpy array) - The input array to analyze.
    ---
    Returns:
        A numpy array containing the ratio of missing values for each column.
    """
    return np.isnan(x).mean(axis=0)


def n_unique(x):
    """
    Description: Calculate the number of unique values (excluding NaN) for each column in the input array.
    ---
    Input:
        x (numpy array) - The input array to analyze.
    ---
    Returns:
        A numpy array containing the number of unique values for each column.
    """
    return np.array([len(np.unique(c[~np.isnan(c)])) for c in x.T])


def class_balance(y):
    return {"n": len(y), "positifs": int(y.sum()), "ratio_positifs": float(y.mean())}


def target_correlation(x, y):
    """
    Description: Calculate the Pearson correlation coefficient between each feature in x and the target variable y, ignoring NaN values.
    ---
    Input:
        x (numpy array) - The input features array.
        y (numpy array) - The target variable array.
    ---
    Returns:
        A numpy array containing the Pearson correlation coefficients between each feature and the target variable.
    """
    corr = np.zeros(x.shape[1])
    for j in range(x.shape[1]):
        m = ~np.isnan(x[:, j])
        c = x[m, j].astype(np.float64)
        if m.sum() < 2 or c.std() == 0:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            corr[j] = np.nan_to_num(np.corrcoef(c, y[m])[0, 1])
    return corr


# -------------- Data cleaning --------------
def clean_special_codes(x_train, x_test):
    """
    Replace the BRFSS special answer codes by meaningful values.
    """
    x_train, x_test = x_train.copy(), x_test.copy()
    for j in range(x_train.shape[1]):
        col = x_train[:, j]
        values = np.unique(col[~np.isnan(col)])
        if len(values) == 0 or not np.all(values == np.round(values)):
            continue  # empty or continuous column
        values = set(values.astype(int))
        to_nan, to_zero = [], False
        if max(values) <= 9:
            if 9 in values:
                to_nan.append(9)
            if 7 in values and 6 not in values and 8 not in values:
                to_nan.append(7)
        elif max(values) <= 99 and (77 in values or 99 in values):
            if any(31 <= v <= 76 for v in values):
                continue  # 77 may be a real value (e.g. an age)
            to_nan += [77, 99]
            to_zero = True
        for X in (x_train, x_test):
            X[np.isin(X[:, j], to_nan), j] = np.nan
            if to_zero:
                X[X[:, j] == 88, j] = 0
    return x_train, x_test


# # -------------- Feature extraction --------------
def select_features(x_train, names, max_missing=0.8):
    """
    Description: Select features based on certain criteria.
    ---
    Input:
        x_train (numpy array) - The training features array.
        names (list) - The names of the features.
        max_missing (float) - The maximum ratio of missing values allowed.
    ---
    Returns:
        A numpy array containing the indices of the selected features.
    """
    keep = missing_ratio(x_train) <= max_missing
    keep &= np.array([n not in ADMIN_COLS for n in names])
    with np.errstate(invalid="ignore"):
        keep &= np.nan_to_num(np.nanstd(x_train, axis=0)) > 0
    return np.where(keep)[0]


def preprocess(
    x_train, x_test, names, max_missing=0.8, add_missing_flags=True, min_flag_ratio=0.01
):
    """
    Description: Preprocess the training and testing data by selecting features, adding missing value indicators, imputing missing values with the median, and standardizing the features.
    ---
    Input:
        x_train (numpy array) - The training features array.
        x_test (numpy array) - The testing features array.
        names (list) - The names of the features.
        max_missing (float) - The maximum ratio of missing values allowed for feature selection.
        add_missing_flags (bool) - Whether to add missing value indicators as additional features.
        min_flag_ratio (float) - The minimum ratio of missing values required to add a missing value indicator.
    ---
    Returns:
        tx_train (numpy array) - The preprocessed training features array with bias column.
        tx_test (numpy array) - The preprocessed testing features array with bias column.
    """
    idx = select_features(x_train, names, max_missing)
    ratio = missing_ratio(x_train)[idx]
    flag_cols = (
        np.where(ratio > min_flag_ratio)[0] if add_missing_flags else np.array([], int)
    )
    d = 1 + len(idx) + len(flag_cols)
    fnames = (
        ["bias"]
        + [str(names[j]) for j in idx]
        + [f"{names[idx[j]]}_isnan" for j in flag_cols]
    )

    out = []
    for X in (x_train, x_test):
        out.append(np.ones((X.shape[0], d), dtype=np.float32))
    tx_tr, tx_te = out

    for k, j in enumerate(idx):
        ctr, cte = x_train[:, j], x_test[:, j]
        med = np.nanmedian(ctr)
        ctr = np.where(np.isnan(ctr), med, ctr)
        cte = np.where(np.isnan(cte), med, cte)
        mu, sd = ctr.mean(), ctr.std()
        sd = sd if sd > 0 else 1.0
        tx_tr[:, 1 + k] = (ctr - mu) / sd
        tx_te[:, 1 + k] = (cte - mu) / sd

    # NaN indicators for features with missing values
    for k, jj in enumerate(flag_cols):
        c = 1 + len(idx) + k
        tx_tr[:, c] = np.isnan(x_train[:, idx[jj]])
        tx_te[:, c] = np.isnan(x_test[:, idx[jj]])
    return tx_tr, tx_te, fnames
