from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar

def fit_temperature(logits, labels, max_iter=100):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    def nll(log_t):
        t = float(np.exp(log_t))
        z = logits / t
        z = z - z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return float(-logp[np.arange(len(labels)), labels].mean())
    res = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded", options={"maxiter": max_iter})
    return float(np.exp(res.x))
