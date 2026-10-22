"""Gán nhãn: lợi suất ngắn hạn (12), hướng giá (7), căng thẳng thanh khoản (8).

Nhãn được phép nhìn về tương lai (t, t+h] – đó là thứ cần dự báo; đặc trưng thì không.
Ngưỡng của nhãn (δ, phân vị stress) chỉ dùng dữ liệu tính đến t hoặc dữ liệu TRAIN -> không rò rỉ.
Không gán nhãn vượt qua chỗ đứt dữ liệu: giây t+h thiếu -> nhãn NaN -> mẫu bị bỏ.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def forward_return_bps(mid: pd.Series, h: int) -> pd.Series:
    """(12) R_{t,h} = (m_{t+h} − m_t) / m_t, đơn vị bps. `mid` trên lưới 1s đều."""
    return (mid.shift(-h) - mid) / mid * 1e4


def choose_delta(r_train: np.ndarray, flat_share: float = 1 / 3) -> float:
    """Chọn δ (bps) trên TRAIN sao cho ~`flat_share` mẫu có |R| ≤ δ (3 lớp gần cân bằng).
    Nếu bản thân |R| = 0 đã chiếm hơn `flat_share` thì δ = 0."""
    a = np.abs(r_train[np.isfinite(r_train)])
    return float(np.quantile(a, flat_share)) if len(a) else 0.0


def direction_label(r_bps: np.ndarray, delta_bps: float) -> np.ndarray:
    """(7) y = +1 nếu R > δ; −1 nếu R < −δ; 0 nếu |R| ≤ δ. NaN giữ nguyên NaN."""
    r = np.asarray(r_bps, float)
    y = np.where(r > delta_bps, 1.0, np.where(r < -delta_bps, -1.0, 0.0))
    y[~np.isfinite(r)] = np.nan
    return y


def rolling_threshold(x: pd.Series, q: float, window: int, min_periods: int) -> pd.Series:
    """Phân vị q của `window` giây gần nhất, tính đến và gồm t (chỉ quá khứ)."""
    return x.rolling(window, min_periods=min_periods).quantile(q)


def stress_labels(feat: pd.DataFrame, h: int, cfg: dict) -> pd.DataFrame:
    """(8) đã điều chỉnh. Spread niêm yết gần như luôn = 1 tick (Q0.95 = min tick) nên nhãn theo spread
    luôn bằng 0. Theo đề bài mục 2.8 ("spread HOẶC price impact vượt percentile") dùng chi phí tác động.
    Chi phí đo từng giây là "nhiễu chớp" (đợt vượt ngưỡng kéo dài trung vị 1s, ACF lag10 ≈ 0.15) nên stress
    được định nghĩa là SUY GIẢM THANH KHOẢN KÉO DÀI trên cả cửa sổ h giây:

      Ī_h(t)       = trung bình ImpactRT trên (t−h, t]                                  (đo được tại t)
      thr_t        = Q0.95 của Ī_h trong 1 giờ tính đến t                              (chỉ quá khứ)
      stress_now_t = 1( Ī_h(t)   > thr_{t−1} )                                          (trạng thái hiện tại)
      stress_h     = 1( Ī_h(t+h) > thr_t )   = trung bình trên (t, t+h] vượt ngưỡng    <- NHÃN CHÍNH
      stress_multi = 1( Depth̄20 trên (t, t+h] < Q0.10  VÀ  RV trên (t, t+h] > Q0.90 )  (đa chiều, robustness)
    ImpactRT = chi phí khứ hồi (bps) của lệnh thị trường `notional_usdt` đi hết sổ lệnh 20 tầng."""
    s = cfg["stress"]
    W, mp = int(s["lookback_s"]), int(s["min_lookback_s"])
    out = pd.DataFrame(index=feat.index)

    ibar = feat["impact_rt_bps"].rolling(h, min_periods=h).mean()
    thr = rolling_threshold(ibar, s["quantile"], W, mp)
    out["stress_thr"] = thr
    out["stress_now"] = _bin(ibar > thr.shift(1), ibar, thr.shift(1))
    fut = ibar.shift(-h)
    out[f"stress_h{h}"] = _bin(fut > thr, fut, thr)

    dbar = feat["depth20_tot"].rolling(h, min_periods=h).mean()
    r1 = np.log(feat["mid"]).diff() * 1e4
    rvh = np.sqrt((r1 ** 2).rolling(h, min_periods=h).sum())
    dthr = rolling_threshold(dbar, s["multi_depth_q"], W, mp)
    vthr = rolling_threshold(rvh, s["multi_vol_q"], W, mp)
    fd, fv = dbar.shift(-h), rvh.shift(-h)
    out[f"stress_multi_h{h}"] = _bin((fd < dthr) & (fv > vthr), fd + fv, dthr + vthr)
    return out


def _bin(cond: pd.Series, *need: pd.Series) -> np.ndarray:
    y = cond.astype(float).to_numpy().copy()
    for n in need:
        y[~np.isfinite(n.to_numpy(float))] = np.nan
    return y
