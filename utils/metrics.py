import numpy as np
from scipy.special import expit
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score


def binary_metrics(y_true, logits, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    prob = expit(np.asarray(logits))
    pred = (prob >= threshold).astype(int)

    return {
        "auc": float(roc_auc_score(y_true, prob)),
        "aupr": float(average_precision_score(y_true, prob)),
        "acc": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }
