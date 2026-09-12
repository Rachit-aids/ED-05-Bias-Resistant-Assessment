from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix

def macro_f1(y_true, probs):
    y_pred = np.argmax(probs, axis=1)
    return float(f1_score(y_true, y_pred, average="macro", labels=[0,1,2], zero_division=0))

def normalized_brier(y_true, probs):
    n = len(y_true)
    y = np.eye(3, dtype=float)[np.asarray(y_true)]
    return float(np.sum((probs - y) ** 2) / (2.0 * n)) if n else float("nan")

def cf_consistency(original_probs, cf_probs):
    if len(original_probs) == 0:
        return float("nan")
    a = np.argmax(original_probs, axis=1)
    b = np.argmax(cf_probs, axis=1)
    return float(np.mean(a == b))

def cf4_utility(original_probs, cf4_probs):
    if len(original_probs) == 0:
        return float("nan")
    increase = np.maximum(0.0, cf4_probs[:, 0] - original_probs[:, 0])
    return float(np.clip(1.0 - np.mean(increase), 0.0, 1.0))

def report(y_true, probs):
    pred = np.argmax(probs, axis=1)
    return classification_report(y_true, pred, target_names=["correct", "contradictory", "incorrect"], labels=[0,1,2], zero_division=0, output_dict=True)

def confusion(y_true, probs):
    return confusion_matrix(y_true, np.argmax(probs, axis=1), labels=[0,1,2]).tolist()
