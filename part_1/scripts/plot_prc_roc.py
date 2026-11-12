#!/usr/bin/env python3
"""Plot PRC (precision-recall) and ROC (recall vs false positive rate) curves
from mlp_prc_curve.csv and mlp_roc_curve.csv, with best threshold points marked.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Paths to the CSV files
ROOT = Path(__file__).parent.parent
PATH_IMPORTANCE = ROOT / "dataset" / "feature_importance.json"
PATH_ROC = ROOT / "dataset" / "mlp_roc_curve.csv"
PATH_PRC = ROOT / "dataset" / "mlp_prc_curve.csv"
PATH_OUT = ROOT / "plots" / "curves.png"

# Read the PRC CSV file (columns: precision, recall, threadsholds)
df_prc = pd.read_csv(PATH_PRC)
precision = df_prc['precision'].values
recall = df_prc['recall'].values
thresholds_prc = df_prc['threadsholds'].values

# Read the ROC CSV file (columns: fpr, tpr, threadsholds)
df_roc = pd.read_csv(PATH_ROC)
fpr = df_roc['fpr'].values
tpr = df_roc['tpr'].values
thresholds_roc = df_roc['threadsholds'].values

# Find best points for PRC curve (closest to precision=1, recall=1)
best_distance_prc = float('inf')
best_row_prc = None

for idx, row in enumerate(df_prc.itertuples()):
    precision_val = row.precision
    recall_val = row.recall
    # Euclidean distance to (1, 1)
    distance = np.sqrt((precision_val - 1.0) ** 2 + (recall_val - 1.0) ** 2)
    if distance < best_distance_prc:
        best_distance_prc = distance
        best_row_prc = row

# Find best points for ROC curve (closest to tpr=1, fpr=0)
best_distance_roc = float('inf')
best_row_roc = None

for idx, row in enumerate(df_roc.itertuples()):
    tpr_val = row.tpr
    fpr_val = row.fpr
    # Euclidean distance to (1, 0) - ideal point is (tpr=1, fpr=0)
    distance = np.sqrt((tpr_val - 1.0) ** 2 + (fpr_val - 0.0) ** 2)
    if distance < best_distance_roc:
        best_distance_roc = distance
        best_row_roc = row

# Sort by recall for PRC curve (recall should be increasing)
prc_sorted_indices = np.argsort(recall)
precision_sorted = precision[prc_sorted_indices]
recall_sorted = recall[prc_sorted_indices]
thresholds_prc_sorted = thresholds_prc[prc_sorted_indices]

# Sort by fpr for ROC curve
roc_sorted_indices = np.argsort(fpr)
fpr_sorted = fpr[roc_sorted_indices]
tpr_sorted = tpr[roc_sorted_indices]
thresholds_roc_sorted = thresholds_roc[roc_sorted_indices]

# Find coordinates of best points on the curves
# For PRC: find the index where threshold matches the best threshold
best_threshold_prc = best_row_prc.threadsholds
prc_best_idx = np.where(thresholds_prc_sorted == best_threshold_prc)[0]
if len(prc_best_idx) > 0:
    prc_best_idx = prc_best_idx[0]
    prc_best_precision = precision_sorted[prc_best_idx]
    prc_best_recall = recall_sorted[prc_best_idx]
else:
    # If exact match not found, find closest
    prc_best_idx = np.argmin(np.abs(thresholds_prc_sorted - best_threshold_prc))
    prc_best_precision = precision_sorted[prc_best_idx]
    prc_best_recall = recall_sorted[prc_best_idx]

# For ROC: find the index where threshold matches the best threshold
best_threshold_roc = best_row_roc.threadsholds
roc_best_idx = np.where(thresholds_roc_sorted == best_threshold_roc)[0]
if len(roc_best_idx) > 0:
    roc_best_idx = roc_best_idx[0]
    roc_best_fpr = fpr_sorted[roc_best_idx]
    roc_best_tpr = tpr_sorted[roc_best_idx]
else:
    # If exact match not found, find closest
    roc_best_idx = np.argmin(np.abs(thresholds_roc_sorted - best_threshold_roc))
    roc_best_fpr = fpr_sorted[roc_best_idx]
    roc_best_tpr = tpr_sorted[roc_best_idx]

# Plot ROC curve (tpr vs fpr) and PRC curve (precision vs recall) side by side
plt.figure(figsize=(12, 5))

# ROC curve subplot
plt.subplot(1, 2, 1)
plt.plot(fpr_sorted, tpr_sorted, label='ROC Curve')
plt.plot(roc_best_fpr, roc_best_tpr, 'ro', markersize=8, label=f'Best ROC point (thresh={best_threshold_roc:.2f})')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.grid(True)
plt.legend()

# PRC curve subplot
plt.subplot(1, 2, 2)
plt.plot(precision_sorted, recall_sorted, label='PRC Curve')
plt.plot(prc_best_precision, prc_best_recall, 'ro', markersize=8, label=f'Best PRC point (thresh={best_threshold_prc:.2f})')
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('PRC Curve')
plt.grid(True)
plt.legend()

plt.tight_layout()

# Save the plot to a file BEFORE showing
plt.savefig(PATH_OUT)
print(f"PRC best point: precision={prc_best_precision:.4f}, recall={prc_best_recall:.4f}, threshold={best_threshold_prc:.4f}")
print(f"ROC best point: fpr={roc_best_fpr:.4f}, tpr={roc_best_tpr:.4f}, threshold={best_threshold_roc:.4f}")
