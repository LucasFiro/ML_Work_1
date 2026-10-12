import os
import numpy as np
import matplotlib
matplotlib.use("Agg") #Permits to not display the plots while the code is running
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
    ax.barh([names[i] for i in order], corr[order],
            color=["tab:red" if corr[i] > 0 else "tab:blue" for i in order])
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

