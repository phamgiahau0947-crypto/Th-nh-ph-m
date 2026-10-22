"""Huấn luyện mô hình HƯỚNG GIÁ ngắn hạn (3 lớp Up/Flat/Down, h = 10s) – đề bài 2.5–2.7.

Dùng trong evaluate_walkforward (mỗi fold) và để xuất mô hình cuối:
    python -m src.task2_models.train_direction      # = chạy evaluate_walkforward rồi lưu models/model_direction.pkl
Benchmark thống kê/cấu trúc đi kèm (checklist "econometric/statistical benchmark"):
    Majority  – luôn dự báo lớp phổ biến nhất của train (xác suất = tần suất lớp)
    OBI-rule  – luật trong kế hoạch dự án: OBI_5 > 0.3 -> Up, < −0.3 -> Down, còn lại Flat
    CST       – mô hình hàng đợi Cont–Stoikov–Talreja: P(up) cấu trúc, P(flat) = tần suất train
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .labeler import choose_delta, direction_label
from .models import DIRECTION_MODELS, fit_direction, predict_direction

BENCHMARKS = ["Majority", "OBI-rule", "CST"]


def make_labels(tr: pd.DataFrame, ev: pd.DataFrame, h: int, flat_share: float):
    """δ chọn trên TRAIN; áp dụng nguyên cho tập đánh giá."""
    col = f"ret_fwd_{h}s"
    delta = choose_delta(tr[col].to_numpy(float), flat_share)
    return delta, direction_label(tr[col].to_numpy(float), delta), direction_label(ev[col].to_numpy(float), delta)


def benchmark_proba(name: str, ev: pd.DataFrame, y_train: np.ndarray) -> np.ndarray:
    freq = np.array([(y_train == c).mean() for c in (-1, 0, 1)])
    n = len(ev)
    if name == "Majority":
        return np.tile(freq, (n, 1))
    if name == "OBI-rule":
        o = ev["obi_5"].to_numpy(float)
        p = np.full((n, 3), 0.1)
        p[o > 0.3, 2] = 0.8
        p[o < -0.3, 0] = 0.8
        p[(o >= -0.3) & (o <= 0.3), 1] = 0.8
        return p
    if name == "CST":
        pu = np.nan_to_num(ev["cst_p_up"].to_numpy(float), nan=0.5)
        f = freq[1]
        return np.column_stack([(1 - f) * (1 - pu), np.full(n, f), (1 - f) * pu])
    raise KeyError(name)


def train_eval_direction(tr: pd.DataFrame, ev: pd.DataFrame, feats: list[str], h: int, flat_share: float,
                         stride: int, seed: int, models: list[str] | None = None, n_trees: int | None = None):
    """Fit mọi mô hình trên `tr`, dự báo trên `ev`. Trả về (delta, y_ev, {tên: proba}, {tên: model}, timing)."""
    delta, ytr, yev = make_labels(tr, ev, h, flat_share)
    mtr = np.isfinite(ytr)
    mtr &= (np.arange(len(tr)) % stride) == 0
    Xtr = tr.loc[mtr, feats].to_numpy(np.float32)
    Xev = ev[feats].to_numpy(np.float32)
    probas, fitted, timing = {}, {}, {}
    for b in BENCHMARKS:
        probas[b] = benchmark_proba(b, ev, ytr[np.isfinite(ytr)])
    for name in models or DIRECTION_MODELS:
        t0 = time.perf_counter()
        m = fit_direction(name, Xtr, ytr[mtr], seed, n_trees)
        timing[name] = time.perf_counter() - t0
        probas[name] = predict_direction(m, Xev)
        fitted[name] = m
    return delta, yev, probas, fitted, timing


def class_mean_returns(tr: pd.DataFrame, h: int, delta: float) -> dict:
    """E[R_{t,h} | lớp] trên train -> lợi suất kỳ vọng = Σ P(lớp)·E[R | lớp] (dùng trong predictor)."""
    r = tr[f"ret_fwd_{h}s"].to_numpy(float)
    y = direction_label(r, delta)
    return {int(c): float(np.nanmean(r[y == c])) for c in (-1, 0, 1)}


if __name__ == "__main__":
    from .evaluate_walkforward import main
    main(["--tasks", "direction"])
