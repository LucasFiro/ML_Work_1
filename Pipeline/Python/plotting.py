import os
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Permits to not display the plots while the code is running
import matplotlib.pyplot as plt

FIG_DIR = "figures"


def _save(fig, name):
    """
    Description: Save the given figure to the figures directory with the specified name.
    ---
    Input:
        fig (matplotlib.figure.Figure) - The figure to save.
        name (str) - The name of the file to save the figure as.
    ---
    Returns:
        None
    """
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, name), dpi=150)
    plt.close(fig)


def plot_class_balance(y):
    """
    Description: Plot the class balance of the target variable y, showing the counts and percentages of each class.
    ---
    Input:
        y (numpy array) - The target variable array containing class labels.
    ---
    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    counts = [np.sum(y == 0), np.sum(y == 1)]
    ax.bar(["Pas de MICHD (0)", "MICHD (1)"], counts, color=["tab:blue", "tab:red"])
    for i, c in enumerate(counts):
        ax.text(i, c, f"{c}\n({c / len(y):.1%})", ha="center", va="bottom")
    ax.set_ylabel("Nombre d'individus")
    ax.set_title("Déséquilibre des classes")
    _save(fig, "class_balance.png")


def plot_missing(ratios):
    """
    Description: Plot the distribution of missing values (NaN) for each feature in the input array.
    ---
    Input:
        ratios (numpy array) - The ratio of missing values for each feature.
    ---
    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(ratios, bins=40, color="tab:gray")
    ax.set_xlabel("Proportion de valeurs manquantes")
    ax.set_ylabel("Nombre de features")
    ax.set_title("Valeurs manquantes par feature")
    _save(fig, "missing_values.png")


def plot_top_correlations(corr, names, k=20):
    """
    Description: Plot the top k features with the highest absolute correlation with the target variable.
    ---
    Input:
        corr (numpy array) - The correlation coefficients between each feature and the target variable.
        names (list) - The names of the features corresponding to the correlation coefficients.
        k (int) - The number of top features to plot based on absolute correlation.
    ---
    Returns:
        None
    """
    order = np.argsort(-np.abs(corr))[:k][::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(
        [names[i] for i in order],
        corr[order],
        color=["tab:red" if corr[i] > 0 else "tab:blue" for i in order],
    )
    ax.set_xlabel("Corrélation avec _MICHD")
    ax.set_title(f"Top {k} features corrélées à la cible")
    _save(fig, "top_correlations.png")


def plot_threshold_curve(y_true, scores, f1_fn):
    """
    Description: Plot the F1 score as a function of the decision threshold for the given scores and true labels.
    ---
    Input:
        y_true (numpy array) - The true labels of the target variable.
        scores (numpy array) - The predicted scores for the target variable.
        f1_fn (function) - A function to calculate the F1 score given true labels and predicted labels.
    ---
    Returns:
        None
    """
    ths = np.quantile(scores, np.linspace(0.5, 0.999, 100))
    f1s = [f1_fn(y_true, (scores >= t).astype(float)) for t in ths]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(ths, f1s)
    ax.set_xlabel("Seuil de décision")
    ax.set_ylabel("F1 (validation)")
    ax.set_title("F1 en fonction du seuil")
    _save(fig, "f1_vs_threshold.png")


def plot_report_figure(lambdas, cv_f1, y_val, scores, threshold, path, selected=1e-3):
    """
    Description: Two-panel figure of the report. Left: cross-validated F1 of the
    L2 logistic regression for each lambda (mean +/- std over folds). Right:
    validation F1 of the final model as a function of the decision threshold.
    ---
    Input:
        lambdas (list) - The lambda values compared in cross-validation.
        cv_f1 (numpy array) - F1 scores, shape (n_folds, n_lambdas).
        y_val (numpy array) - Validation labels in {0, 1}.
        scores (numpy array) - Predicted probabilities of the final model on the validation set.
        threshold (float) - Decision threshold selected for the final model.
        path (str) - Output file (e.g. figures/fig_tuning.pdf).
        selected (float) - Lambda used by the final model (highlighted).
    ---
    Returns:
        None
    """
    blue, ink, muted, grid = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update(
        {"font.size": 7, "axes.edgecolor": muted, "axes.linewidth": 0.6}
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5, 1.65))

    # Left panel: cross-validation curve
    x = np.arange(len(lambdas))
    mean, std = cv_f1.mean(axis=0), cv_f1.std(axis=0)
    best = int(np.argmin(np.abs(np.array(lambdas) - selected)))
    ax1.errorbar(
        x,
        mean,
        yerr=std,
        color=blue,
        lw=1.2,
        marker="o",
        ms=3.5,
        capsize=2,
        elinewidth=0.8,
    )
    # Lambda used by the final model: larger marker with a dark ring
    ax1.plot(x[best], mean[best], "o", ms=6, color=blue, mec=ink, mew=0.9)
    labels = ["0"] + [f"{l:.0e}".replace("e-0", "e-") for l in lambdas[1:]]
    ax1.set_xticks(x, labels)
    ax1.set_xlabel(r"regularization $\lambda$", color=ink)
    ax1.set_ylabel("5-fold CV F1", color=ink)

    # Right panel: F1 as a function of the threshold
    ths = np.linspace(0.02, 0.6, 117)
    f1s = [f1_score_np(y_val, scores >= t) for t in ths]
    ax2.plot(ths, f1s, color=blue, lw=1.2)
    f1_sel = f1_score_np(y_val, scores >= threshold)
    f1_05 = f1_score_np(y_val, scores >= 0.5)
    ax2.plot(threshold, f1_sel, "o", ms=5.5, color=blue, mec="white", mew=1)
    ax2.annotate(
        f"t = {threshold:.2f}\nF1 = {f1_sel:.3f}",
        (threshold, f1_sel),
        xytext=(0, -28),
        ha="center",
        textcoords="offset points",
        color=ink,
        fontsize=6.5,
    )
    ax2.plot(0.5, f1_05, "o", ms=3.5, color=muted)
    ax2.annotate(
        f"t = 0.5\nF1 = {f1_05:.3f}",
        (0.5, f1_05),
        xytext=(-6, -24),
        textcoords="offset points",
        ha="right",
        color=muted,
        fontsize=6.5,
    )
    ax2.set_xlabel("decision threshold $t$", color=ink)
    ax2.set_ylabel("validation F1", color=ink)
    ax2.set_ylim(0, max(f1s) * 1.12)

    for ax in (ax1, ax2):
        ax.grid(axis="y", color=grid, lw=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=muted, length=2, width=0.6)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def f1_score_np(y_true, y_pred):
    """
    Description: F1 score of binary predictions (helper for the report figure).
    ---
    Input:
        y_true (numpy array) - True labels in {0, 1}.
        y_pred (numpy array) - Predicted labels (boolean or {0, 1}).
    ---
    Returns:
        The F1 score as a float.
    """
    y_pred = np.asarray(y_pred, dtype=bool)
    tp = np.sum(y_pred & (y_true == 1))
    if tp == 0:
        return 0.0
    precision, recall = tp / y_pred.sum(), tp / np.sum(y_true == 1)
    return float(2 * precision * recall / (precision + recall))
