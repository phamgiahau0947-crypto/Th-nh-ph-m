"""Tiện ích nhỏ dùng chung cho Task 3 (không phụ thuộc Task 1/Task 2 để Mock chạy độc lập)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

TASK3_DIR = Path(__file__).resolve().parent
ROOT = TASK3_DIR.parents[1]

LEVELS = 20
BID_Q = [f"bids_q{i}" for i in range(1, LEVELS + 1)]
ASK_Q = [f"asks_q{i}" for i in range(1, LEVELS + 1)]
BID_P = [f"bids_p{i}" for i in range(1, LEVELS + 1)]
ASK_P = [f"asks_p{i}" for i in range(1, LEVELS + 1)]

# Cột lịch sử 1 giây mà mọi nguồn (Mock / Live / Replay) đều phải cấp cho giao diện
HIST_COLS = ["mid", "spread_bps", "obi_1", "obi_5", "ofi1_10s", "mlofi_sum_10s", "impact_rt_bps",
             "p_up", "p_flat", "p_down", "p_stress", "stress_ratio", "exp_ret_bps", "pred_dir",
             # bổ sung 21/10/2026: dự báo spread (Task 2 spread_forecast.py); NaN nếu chưa có models/model_spread.pkl
             "spr_ticks", "p_spread_widen", "eff_spread_fwd_bps"]
SPREAD_PRED_KEYS = ["p_spread_widen", "eff_spread_fwd_bps"]


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(TASK3_DIR / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def impact_bps(levels: np.ndarray, notional: float, mid: float) -> tuple[float, bool]:
    """Chi phí (bps so với mid) khi khớp ngay lệnh thị trường `notional` USDT đi hết các tầng.
    levels: mảng (n, 2) [giá, KL] từ tầng 1 ra ngoài. Trả về (chi phí, có bị thiếu thanh khoản không)."""
    if levels is None or len(levels) == 0 or not np.isfinite(mid) or mid <= 0:
        return np.nan, True
    p, q = levels[:, 0].astype(float), levels[:, 1].astype(float)
    val = p * q
    cum = np.cumsum(val)
    k = int(np.searchsorted(cum, notional))
    if k >= len(p):                               # 20 tầng không đủ -> phần thiếu tính theo giá tầng cuối
        filled_qty, spent = q.sum(), val.sum()
        filled_qty += (notional - spent) / p[-1]
        unfilled = True
    else:
        spent_before = cum[k - 1] if k > 0 else 0.0
        filled_qty = (q[:k].sum() if k > 0 else 0.0) + (notional - spent_before) / p[k]
        unfilled = False
    vwap = notional / filled_qty
    return float(abs(vwap - mid) / mid * 1e4), unfilled
