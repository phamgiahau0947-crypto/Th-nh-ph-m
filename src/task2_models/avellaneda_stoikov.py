"""Avellaneda & Stoikov (2008) High-frequency trading in a limit order book – chỉ báo stress vi mô.

Bài báo (pt. 30):  δ^a + δ^b = γσ²(T−t) + (2/γ)·ln(1 + γ/k)          -> spread tối ưu của market maker
                   r(s,q,t) = s − q·γσ²(T−t)                         -> giá bảo lưu (reservation price)
                   λ(δ) = A·exp(−kδ)                                 -> cường độ lệnh thị trường chạm tới
                                                                        khoảng cách δ từ mid (pt. 12)
Quy ước ở đây: giá tính theo bps của mid, σ theo bps/√giây, T−t = chân trời dự báo h (giây).

Lưu ý so với đặc tả: công thức (10) trong đặc tả là (1/γ)ln(1+γ/k) + ½γσ²(T−t), tức NỬA spread (mỗi phía).
Vì so sánh với spread hai phía nên dùng nguyên công thức (30) của bài báo (gấp đôi).
Spread niêm yết của BTC/ETH trên Binance gần như luôn = 1 tick nên "Spread thực tế" trong (11) được thay
bằng spread hiệu dụng khứ hồi cho lệnh 10.000 USDT (impact_rt_bps) – xem README_TASK_2.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def optimal_spread_bps(sigma_bps, k: float, gamma: float, horizon_s: float):
    """δ* = γσ²(T−t) + (2/γ)·ln(1 + γ/k)   (Avellaneda–Stoikov 2008, pt. 30)."""
    sigma = np.asarray(sigma_bps, float)
    return gamma * sigma ** 2 * horizon_s + (2.0 / gamma) * np.log1p(gamma / k)


def reservation_skew_bps(ti_300s, sigma_bps, gamma: float, horizon_s: float):
    """r − s = −q·γσ²(T−t). Tồn kho market maker q ≈ −(mất cân bằng lệnh chủ động 5 phút):
    người chủ động MUA nhiều -> market maker bị bán khống (q<0) -> giá bảo lưu > mid."""
    q = -np.asarray(ti_300s, float)
    return -q * gamma * np.asarray(sigma_bps, float) ** 2 * horizon_s


def stress_ratio(effective_spread_bps, delta_star_bps, norm: float = 1.0):
    """(11) StressRatio = Spread_thực_tế / δ*, chia thêm cho trung vị trên tập train (1 = bình thường)."""
    return np.asarray(effective_spread_bps, float) / np.asarray(delta_star_bps, float) / norm


def reach_distances_bps(day: pd.DataFrame, levels: int = 20) -> np.ndarray:
    """Khoảng cách (bps từ mid ở t−1) mà khối lượng chủ động trong (t−1, t] quét tới trên sổ lệnh t−1.
    Dùng để ước lượng λ(δ) = A·exp(−kδ) từ dữ liệu 1 giây (không có từng lệnh)."""
    out = []
    for side, flow in (("asks", "buy_qty"), ("bids", "sell_qty")):
        P = day[[f"{side}_p{i}" for i in range(1, levels + 1)]].to_numpy(float)[:-1]
        Q = day[[f"{side}_q{i}" for i in range(1, levels + 1)]].to_numpy(float)[:-1]
        mid = ((day["bids_p1"] + day["asks_p1"]) / 2).to_numpy(float)[:-1]
        v = day[flow].to_numpy(float)[1:]
        ok = (v > 0) & np.isfinite(mid) & np.isfinite(v)
        P, Q, mid, v = P[ok], Q[ok], mid[ok], v[ok]
        cum = np.cumsum(Q, axis=1)
        j = np.minimum((cum < v[:, None]).sum(axis=1), levels - 1)
        px = P[np.arange(len(P)), j]
        out.append(np.abs(px - mid) / mid * 1e4)
    return np.concatenate(out) if out else np.array([])


def estimate_k(distances_bps: np.ndarray, seconds: float, n_grid: int = 30) -> tuple[float, float]:
    """Hồi quy ln λ(δ) = ln A − kδ, λ(δ) = số lần quét ≥ δ / số giây. Trả về (k [1/bps], A [1/giây])."""
    d = distances_bps[np.isfinite(distances_bps)]
    grid = np.linspace(np.quantile(d, 0.05), np.quantile(d, 0.95), n_grid)
    lam = np.array([(d >= g).sum() for g in grid]) / seconds
    ok = lam > 0
    slope, intercept = np.polyfit(grid[ok], np.log(lam[ok]), 1)
    return float(max(-slope, 1e-6)), float(np.exp(intercept))
