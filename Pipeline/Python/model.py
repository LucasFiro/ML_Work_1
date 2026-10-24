#Fichier ou mettre le modele
import implementations as imp
import numpy as np


def pred_log_regl1(X_tr, y_tr, X_te, initial_w, max_iters, gamma, lambda_):
    
    _, w = imp.logistic_regression_l1(
        y_tr, X_tr, initial_w, max_iters, gamma, lambda_
    )
    # 3. Prédictions probabilistes puis classes
    probabilities = imp.sigmoid(X_te.dot(w))
    y_pred_lr = (probabilities >= 0.5).astype(int)
    return y_pred_lr

def pred_log_reg(X_tr, y_tr, X_te, initial_w, max_iters, gamma):
    
    _, w = imp.logistic_regression(
        y_tr, X_tr, initial_w, max_iters, gamma
    )
    # 3. Prédictions probabilistes puis classes
    probabilities = imp.sigmoid(X_te.dot(w))
    y_pred_lr = (probabilities >= 0.5).astype(int)
    return y_pred_lr


def train_test_split(tx, y, test_ratio=0.2, seed=42):
    """Split aléatoire reproductible."""
    tx = np.asarray(tx)
    y = np.asarray(y).ravel()

    if len(tx) != len(y):
        raise ValueError("tx et y doivent avoir le même nombre de lignes.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y))

    n_test = int(len(y) * test_ratio)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    return tx[train_idx], tx[test_idx], y[train_idx], y[test_idx]


def evaluate_model(y_true, y_pred):
    """Calcule les métriques de classification binaire."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

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
        if precision + recall > 0 else 0.0
    )

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1-score": f1,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn
    }

def print_res(y_te, y_pred, title):
    metrics = evaluate_model(y_te, y_pred)
        
    print(title)
    for metric, value in metrics.items():
        print(f"{metric:12s}: {value:.4f}")


def compare_models_baseline( tx_train, y_train, tx_test, max_iters=1000,
    gamma=0.01,
    seed=42,
    lambda_ =0.01 ):

    # 1. Split
    X_tr, X_te, y_tr, y_te = train_test_split(
        tx_train, y_train, test_ratio=0.2, seed = seed )

    initial_w = np.zeros(tx_train.shape[1])
    
    # 2. Entraîner les models
    y_pred_lr = pred_log_reg(X_tr, y_tr, X_te, initial_w, max_iters, gamma)

    y_pred_lr1= pred_log_regl1(X_tr, y_tr, X_te, initial_w, max_iters, gamma, lambda_)

    y_pred_knn = imp.knn_predict(tx_train, y_train, tx_test, k=5)

    nb_model = imp.gaussian_naive_bayes(y_train, tx_train)
    y_pred_nb = imp.gaussian_naive_bayes_predict(nb_model, tx_test)

    # 4. Baseline : toujours prédire y = 0
    y_baseline = np.zeros(len(y_te), dtype=int)

    Y = [y_pred_lr, y_pred_lr1, y_pred_knn, y_pred_nb, y_baseline]
    Y = np.array(Y)


    print(f"Train size : {len(y_tr)}")
    print(f"Test size  : {len(y_te)}")
    print(f"Proportion positive dans le test : {np.mean(y_te):.2%}")


    titles = ["---Logistic regression---", "---Logistic regression L1---",
              "---KNN---", "--- Gaussian naive bayes---", "---Baseline---"]
    
    # 5. Évaluation

    for i in range (5):
        print_res(y_te, Y[i], titles[i])

    return Y    
    