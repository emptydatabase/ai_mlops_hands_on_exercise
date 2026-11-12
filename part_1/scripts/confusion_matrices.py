#!/usr/bin/env python3
"""Generate confusion matrices for two best thresholds from mlp_pred.csv:
- Best ROC threshold: 0.2752
- Best PRC threshold: 0.3881

Saves JSON files and heatmap plots for each.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Paths
ROOT = Path(__file__).parent.parent
PATH_DF = ROOT / "dataset" / "census.csv"
PATH_IMPORTANCE = ROOT / "dataset" / "feature_importance.json"
PATH_ROC = ROOT / "dataset" / "mlp_roc_curve.csv"
PATH_PRC = ROOT / "dataset" / "mlp_prc_curve.csv"
PATH_PRED = ROOT / "dataset" / "mlp_pred.csv"
PATH_RF_CF = ROOT / "dataset" / "rf_confusion_matrix.json"
OUT_JSON_ROC = ROOT / "dataset" / "mlp_roc_cf.json"
OUT_PLOT_ROC = ROOT / "plots" / "mlp_roc_cf.png"
OUT_JSON_PRC = ROOT / "dataset" / "mlp_prc_cf.json"
OUT_PLOT_PRC = ROOT / "plots" / "mlp_prc_cf.png"
OUT_JSON_RF = ROOT / "dataset" / "rf_cf.json"
OUT_PLOT_RF = ROOT / "plots" / "rf_cf.png"


# Best thresholds
THRESHOLD_ROC = 0.2752
THRESHOLD_PRC = 0.3881

# Read predictions
df = pd.read_csv(PATH_PRED)
real = df['real'].values
pred_scores = df['pred'].values

def compute_confusion_matrix(real_labels, pred_scores, threshold):
    """Compute confusion matrix given a threshold."""
    binary_pred = (pred_scores >= threshold).astype(int)
    # Confusion matrix: rows = real, cols = predicted
    # [[TN, FP], [FN, TP]]
    tn = int(np.sum((real_labels == 0) & (binary_pred == 0)))
    fp = int(np.sum((real_labels == 0) & (binary_pred == 1)))
    fn = int(np.sum((real_labels == 1) & (binary_pred == 0)))
    tp = int(np.sum((real_labels == 1) & (binary_pred == 1)))
    return {
        'threshold': threshold,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp,
        'total': tn + fp + fn + tp,
    }

def save_confusion_matrix(cm, threshold_name, json_path, plot_path):
    """Save confusion matrix as JSON and heatmap plot."""
    # Save JSON
    with open(json_path, 'w') as f:
        json.dump(cm, f, indent=2)
    print(f"Saved {json_path}")

    # Plot heatmap
    cm_array = np.array([
        [cm['true_negatives'], cm['false_positives']],
        [cm['false_negatives'], cm['true_positives']],
    ])
    labels = np.array([
        ['TN', 'FP'],
        ['FN', 'TP'],
    ])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_array,
        annot=cm_array,
        fmt='',
        cmap='Blues',
        xticklabels=['Predicted 0', 'Predicted 1'],
        yticklabels=['Real 0', 'Real 1'],
        cbar_kws={'label': 'Count'},
    )
    plt.title(f'Confusion Matrix (threshold={cm["threshold"]:.4f}, {threshold_name})')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved {plot_path}")

# Compute and save for both thresholds
cm_roc = compute_confusion_matrix(real, pred_scores, THRESHOLD_ROC)
cm_prc = compute_confusion_matrix(real, pred_scores, THRESHOLD_PRC)

print(f"ROC best threshold ({THRESHOLD_ROC}): {cm_roc}")
print(f"PRC best threshold ({THRESHOLD_PRC}): {cm_prc}")

save_confusion_matrix(cm_roc, 'ROC', OUT_JSON_ROC, OUT_PLOT_ROC)
save_confusion_matrix(cm_prc, 'PRC', OUT_JSON_PRC, OUT_PLOT_PRC)

# Random forest

with open(PATH_RF_CF) as fp:
    rf_cf = json.load(fp)

plt.figure(figsize=(6, 5))
sns.heatmap(
    rf_cf,
    annot=rf_cf,
    fmt='',
    cmap='Blues',
    xticklabels=['Predicted 0', 'Predicted 1'],
    yticklabels=['Real 0', 'Real 1'],
    cbar_kws={'label': 'Count'},
)
plt.title(f'Confusion Matrix')
plt.tight_layout()
plt.savefig(OUT_PLOT_RF)
plt.close()
print(f"Saved {OUT_PLOT_RF}")
