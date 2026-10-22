"""Huấn luyện mô hình XÁC SUẤT CĂNG THẲNG THANH KHOẢN P(Stress_{t+h}=1 | LOB_t, MLOFI_t, Trades_t) – (9).

Nhãn chính: stress_h{h} (chi phí tác động vượt Q0.95 của 1 giờ gần nhất, xem labeler.stress_labels).
Hiệu chỉnh xác suất isotonic (CalibratedClassifierCV) trên ngày train cuối.
Benchmark:
    Persistence – P(stress_{t+h} | stress_now_t) ước lượng trên train (trạng thái hiện tại kéo dài)
    AS-rule     – luật Avellaneda–Stoikov của đặc tả: StressRatio > 1.5 -> cảnh báo
    python -m src.task2_models.train_stress
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .models import STRESS_MODELS, fit_stress, predict_stress

BENCHMARKS = ["Persistence", "AS-rule"]


def benchmark_proba(name: str, ev: pd.DataFrame, tr: pd.DataFrame, label: str, ratio_warn: float) -> np.ndarray:
    if name == "Persistence":
        ok = tr[label].notna() & tr["stress_now"].notna()
        p1 = tr.loc[ok & (tr["stress_now"] == 1), label].mean()
        p0 = tr.loc[ok & (tr["stress_now"] == 0), label].mean()
        s = ev["stress_now"].to_numpy(float)
        return np.where(s == 1, p1, np.where(s == 0, p0, tr[label].mean()))
    if name == "AS-rule":
        r = ev["as_stress_ratio"].to_numpy(float)
        ok = tr[label].notna()
        hit = tr.loc[ok, "as_stress_ratio"] > ratio_warn
        p1 = tr.loc[ok][hit][label].mean()
        p0 = tr.loc[ok][~hit][label].mean()
        return np.where(r > ratio_warn, p1, p0)
    raise KeyError(name)


def train_eval_stress(tr: pd.DataFrame, ev: pd.DataFrame, feats: list[str], label: str, stride: int, seed: int,
                      ratio_warn: float, models: list[str] | None = None, n_trees: int | None = None):
    """Fit trên các ngày train trừ ngày cuối, hiệu chỉnh trên ngày train cuối, dự báo trên `ev`."""
    days = sorted(tr["day"].unique())
    y = tr[label].to_numpy(float)
    keep = np.isfinite(y) & ((np.arange(len(tr)) % stride) == 0)
    fit_m = keep & (tr["day"] != days[-1]).to_numpy()
    cal_m = np.isfinite(y) & (tr["day"] == days[-1]).to_numpy()
    X = tr[feats].to_numpy(np.float32)
    Xev = ev[feats].to_numpy(np.float32)
    yev = ev[label].to_numpy(float)
    probas, fitted, timing = {}, {}, {}
    for b in BENCHMARKS:
        probas[b] = benchmark_proba(b, ev, tr, label, ratio_warn)
    for name in models or STRESS_MODELS:
        t0 = time.perf_counter()
        m = fit_stress(name, X[fit_m], y[fit_m], X[cal_m], y[cal_m], seed, n_trees)
        timing[name] = time.perf_counter() - t0
        probas[name] = predict_stress(m, Xev)
        fitted[name] = m
    return yev, probas, fitted, timing


if __name__ == "__main__":
    from .evaluate_walkforward import main
    main(["--tasks", "stress"])
