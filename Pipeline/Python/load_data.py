import os
import numpy as np

DATA_DIR = "Data/dataset/"
CACHE = os.path.join(DATA_DIR, "cache.npz")


def _read_matrix(path, chunk=20000):
    """
    Description: Read a csv file in chunks to save memory. The empty strings are converted to NaN.
    ---
    Input: 
        path (str) - The path to the CSV file.
        chunk (int) - The number of rows to read at a time.
    ---
    Returns: 
        A numpy array containing the data from the CSV file.
    """
    blocks = []
    with open(path) as f:
        f.readline()  # header
        while True:
            lines = [l for _, l in zip(range(chunk), f)]
            if not lines:
                break
            rows = [l.rstrip("\r\n").split(",") for l in lines]
            arr = np.array(rows)
            arr[arr == ""] = "nan"
            blocks.append(arr.astype(np.float32))
    return np.vstack(blocks)


def load_csv_data(data_dir=DATA_DIR):
    """
    Description: Read x_train.csv, y_train.csv, x_test.csv. Missing values are coded as np.nan. The first column is the Id and is removed.
    ---
    Input:
        data_dir (str) - The directory containing the CSV files.
    ---
    Returns:
        x_train (N, D) float32, x_test (M, D) float,
        y_train (N,) avec labels dans {0, 1},
        train_ids (N,), test_ids (M,), feature_names (D,)
    """
    with open(os.path.join(data_dir, "x_train.csv")) as f:
        header = f.readline().strip().split(",")
    feature_names = np.array(header[1:])  # Remove ID column

    # y_train contains -1 and 1 and -1 are converted to 0
    y = np.genfromtxt(os.path.join(data_dir, "y_train.csv"),
                      delimiter=",", skip_header=1, usecols=1)
    y = (y == 1).astype(float)
    train_ids = np.genfromtxt(os.path.join(data_dir, "y_train.csv"),
                              delimiter=",", skip_header=1, usecols=0, dtype=int)

    x_tr = _read_matrix(os.path.join(data_dir, "x_train.csv"))
    x_te = _read_matrix(os.path.join(data_dir, "x_test.csv"))

    train_ids_x, test_ids = x_tr[:, 0].astype(int), x_te[:, 0].astype(int)
    assert np.array_equal(train_ids, train_ids_x), "Ids x_train / y_train différents"
    return x_tr[:, 1:], x_te[:, 1:], y, train_ids, test_ids, feature_names


def load_data(use_cache=True, data_dir=DATA_DIR):
    """
    Description: Load data from cache if available, otherwise read from CSV files and save to cache.
    ---
    Input:
        use_cache (bool) - Whether to use cached data if available.
        data_dir (str) - The directory containing the CSV files.
    ---
    Returns:
        x_train (N, D) float32, x_test (M, D) float,
        y_train (N,) avec labels dans {0, 1},
        train_ids (N,), test_ids (M,), feature_names (D,)
    """
    cache = os.path.join(data_dir, "cache.npz")
    if use_cache and os.path.exists(cache):
        d = np.load(cache, allow_pickle=False)
        return (d["x_train"], d["x_test"], d["y_train"],
                d["train_ids"], d["test_ids"], d["feature_names"])
    out = load_csv_data(data_dir)
    names = ("x_train", "x_test", "y_train", "train_ids", "test_ids", "feature_names")
    np.savez_compressed(cache, **dict(zip(names, out)))
    return out

