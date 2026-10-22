"""Thước đo đánh giá (đề bài mục 2.9) + khoảng tin cậy bootstrap theo khối thời gian (RQ5)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
                             log_loss, matthews_corrcoef, precision_score, recall_score, roc_auc_score)

DIR_CLASSES = np.array([-1, 0, 1])          # thứ tự cột xác suất: [down, flat, up]


def direction_metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    """y ∈ {−1, 0, 1}; proba (n, 3) theo thứ tự [down, flat, up]."""
    y = np.asarray(y).astype(int)
    pred = DIR_CLASSES[np.argmax(proba, axis=1)]
    onehot = (y[:, None] == DIR_CLASSES[None, :]).astype(float)
    out = {
        "n": len(y),
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", labels=DIR_CLASSES, zero_division=0),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "mcc": matthews_corrcoef(y, pred),
        "precision_macro": precision_score(y, pred, average="macro", labels=DIR_CLASSES, zero_division=0),
        "recall_macro": recall_score(y, pred, average="macro", labels=DIR_CLASSES, zero_division=0),
        "brier": float(np.mean(np.sum((proba - onehot) ** 2, axis=1))),
        "log_loss": log_loss(y, np.clip(proba, 1e-7, 1), labels=DIR_CLASSES),
    }
    f1s = f1_score(y, pred, average=None, labels=DIR_CLASSES, zero_division=0)
    out.update({"f1_down": f1s[0], "f1_flat": f1s[1], "f1_up": f1s[2]})
    return out


def stress_metrics(y: np.ndarray, p: np.ndarray, warn: float = 0.3) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    out = {"n": len(y), "base_rate": y.mean()}
    two = len(np.unique(y)) == 2
    out["roc_auc"] = roc_auc_score(y, p) if two else np.nan
    out["pr_auc"] = average_precision_score(y, p) if two else np.nan
    out["brier"] = float(np.mean((p - y) ** 2))
    out["brier_skill"] = 1 - out["brier"] / max(np.mean((y.mean() - y) ** 2), 1e-12)
    out["log_loss"] = log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])
    for thr, tag in ((0.5, "05"), (warn, "warn")):
        pred = (p > thr).astype(int)
        out[f"precision_{tag}"] = precision_score(y, pred, zero_division=0)
        out[f"recall_{tag}"] = recall_score(y, pred, zero_division=0)
        out[f"f1_{tag}"] = f1_score(y, pred, zero_division=0)
    pred = (p > 0.5).astype(int)
    out["macro_f1"] = f1_score(y, pred, average="macro", zero_division=0)
    out["balanced_acc"] = balanced_accuracy_score(y, pred)
    out["mcc"] = matthews_corrcoef(y, pred)
    return out


def block_ids(day: pd.Series | np.ndarray, ts: np.ndarray, block_s: int = 300) -> np.ndarray:
    """Khối 5 phút trong từng ngày -> đơn vị lấy mẫu lại (giữ tự tương quan trong khối)."""
    key = pd.Series(np.asarray(day).astype(str)) + "_" + pd.Series(np.asarray(ts) // (block_s * 1000)).astype(str)
    return pd.factorize(key)[0]


def block_bootstrap(fn, blocks: np.ndarray, n_boot: int = 300, seed: int = 0, *arrays) -> np.ndarray:
    """Lấy mẫu lại theo khối; fn(*arrays_subset) -> float. Trả về mảng thống kê bootstrap."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(blocks)
    order = np.argsort(blocks, kind="stable")
    sb = blocks[order]
    starts = np.searchsorted(sb, uniq)
    ends = np.searchsorted(sb, uniq, side="right")
    out = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([order[starts[j]:ends[j]] for j in pick])
        out[i] = fn(*(a[idx] for a in arrays))
    return out


def ci(x: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    a = (1 - level) / 2
    return float(np.nanquantile(x, a)), float(np.nanquantile(x, 1 - a))
