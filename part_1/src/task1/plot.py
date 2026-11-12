from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas import DataFrame
from sklearn.metrics import confusion_matrix


def plot_curve(
        df: DataFrame,
        x: str,
        y: str,
        best,
        label: str,
        xlabel: str,
        ylabel: str,
        title: str,
        out_path: Path
):
    plt.figure()
    plt.plot(df[x], df[y], label=label)
    plt.plot(best[x], best[y], 'ro', markersize=8, label=f'Best point (thresh={best["thresholds"].iloc[0]:.2f})')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(out_path)


def plot_confusion_matrix(real, pred, title: str, path: Path):
    cm = confusion_matrix(real, pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=cm,
        fmt='',
        cmap='Blues',
        xticklabels=['Predicted 0', 'Predicted 1'],
        yticklabels=['Real 0', 'Real 1'],
        cbar_kws={'label': 'Count'},
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
