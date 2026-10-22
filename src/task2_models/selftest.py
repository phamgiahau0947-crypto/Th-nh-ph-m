"""Tự kiểm tra Task 2 (không cần mạng):  python -m src.task2_models.selftest

1. OFI/MLOFI đúng ví dụ số của Xu, Gould & Howison (2019, mục 3.2): M=3 -> (7, 10, 10).
2. Không look-ahead: sửa dữ liệu SAU thời điểm t không làm đổi bất kỳ đặc trưng nào tại t.
3. Live = offline: gom buffer 100ms (giả lập streamer) về 1s cho đúng cùng đặc trưng như kho lịch sử.
4. CST: P_up đối xứng (P(a,b) + P(b,a) = 1, P(a,a) = 0.5) và tăng theo hàng đợi bid.
5. Nhãn không vượt chỗ đứt dữ liệu; δ chọn đúng tỉ lệ "đứng".
6. Walk-forward: train luôn TRƯỚC val/test, không trùng ngày.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .common import lob_days, load_config, read_day, walk_forward_split
from .cont_stoikov_talreja import CSTModel
from .labeler import choose_delta, direction_label, forward_return_bps
from .microstructure_math import build_base_features, buffer_to_1s, feature_columns, ofi_levels, regrid_1s

RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"[{'ĐẠT' if cond else 'LỖI'}] {name} {detail}")


def t_ofi_paper_example():
    # trước: b1=1.40 (10), b2=1.39 (10), b3 không có; sau lệnh mua 7 @1.41: b1=1.41(7), b2=1.40(10), b3=1.39(10)
    pb = np.array([[1.40, 1.39, np.nan], [1.41, 1.40, 1.39]])
    qb = np.array([[10, 10, np.nan], [7, 10, 10]], float)
    pa = np.array([[1.42, 1.43, 1.44], [1.42, 1.43, 1.44]])
    qa = np.array([[5, 5, 5], [5, 5, 5]], float)
    pb[0, 2], qb[0, 2] = 0.0, 0.0          # "không tồn tại" ~ giá thấp hơn mọi giá -> b3 tăng
    e = ofi_levels(pb, qb, pa, qa, 3)[1]
    check("OFI ví dụ Xu et al. 3.2", np.allclose(e, [7, 10, 10]), f"-> {e}")
    # phía bán: ask giảm (lệnh bán mới trong spread) -> e âm = áp lực bán
    pa2 = np.array([[1.42], [1.415]])
    qa2 = np.array([[5.0], [3.0]])
    e2 = ofi_levels(np.array([[1.40], [1.40]]), np.array([[10.0], [10.0]]), pa2, qa2, 1)[1, 0]
    check("OFI dấu phía bán (ask giảm -> âm)", np.isclose(e2, -3.0), f"-> {e2}")


def _sample_day(n=3600):
    sym = load_config()["symbols"][0]
    d, f = lob_days(sym)[-1]
    raw = read_day(f).iloc[20000:20000 + n].copy()
    return sym, regrid_1s(raw)


def t_no_lookahead():
    cfg = load_config()
    sym, g = _sample_day(2400)
    cols = [c for c in feature_columns(cfg) if c not in ("cst_p_up", "as_delta_star_bps", "as_stress_ratio",
                                                         "as_res_skew_bps")]
    base = build_base_features(g, cfg, sym)
    t = 1800
    g2 = g.copy()
    num = g2.select_dtypes("number").columns
    g2[num] = g2[num].astype(float)
    g2.iloc[t + 1:, g2.columns.get_indexer(num)] *= 1.37            # phá hỏng toàn bộ tương lai
    alt = build_base_features(g2, cfg, sym)
    a, b = base[cols].iloc[: t + 1].to_numpy(float), alt[cols].iloc[: t + 1].to_numpy(float)
    same = np.allclose(a, b, equal_nan=True, rtol=1e-6, atol=1e-9)
    check("Không look-ahead (đặc trưng tại t độc lập với dữ liệu > t)", same)


def t_live_equals_offline():
    cfg = load_config()
    sym, g = _sample_day(1200)
    off = build_base_features(g, cfg, sym)
    # giả lập buffer streamer: mỗi giây 1 snapshot đến lúc t − 300ms, trade của (t−1s, t] gắn vào snapshot đó
    buf = g.reset_index()
    buf["timestamp"] = buf["timestamp"] - 300
    live = build_base_features(buffer_to_1s(buf, cfg["levels"]), cfg, sym)
    cols = [c for c in feature_columns(cfg) if c in live.columns]
    a, b = off[cols].iloc[-1].to_numpy(float), live[cols].iloc[-1].to_numpy(float)
    ok = np.allclose(a, b, equal_nan=True, rtol=1e-5, atol=1e-7)
    bad = [c for c, x, y in zip(cols, a, b) if not np.isclose(x, y, equal_nan=True, rtol=1e-5, atol=1e-7)]
    check("Live (buffer 100ms -> 1s) = offline", ok, f"cột lệch: {bad[:5]}" if bad else "")


def t_cst_symmetry():
    m = CSTModel(lam_vol=1.0, mu_vol=0.8, theta=0.05, unit=1.0, n_max=40, t_max=200)
    t = m.table()[1:, 1:]
    check("CST đối xứng P(a,b)+P(b,a)=1", np.allclose(t + t.T, 1, atol=1e-6))
    check("CST P(a,a)=0.5", np.allclose(np.diag(t), 0.5, atol=1e-6))
    check("CST tăng theo hàng đợi bid b", np.all(np.diff(t[5, :]) >= -1e-9))


def t_labels():
    mid = pd.Series([100, 100.01, np.nan, 100.02, 100.03, 100.0, 100.0], dtype=float)
    r = forward_return_bps(mid, 2)
    want = mid.isna() | mid.shift(-2).isna()        # nhãn NaN khi thiếu m_t hoặc m_{t+h} (giây trên lưới đều)
    check("Nhãn NaN khi thiếu m_t / m_{t+h}; không nối qua ngày", (r.isna() == want).all() and r.iloc[-2:].isna().all())
    x = np.random.default_rng(0).normal(size=30000)
    dlt = choose_delta(x, 1 / 3)
    y = direction_label(x, dlt)
    check("δ cho ~1/3 lớp đứng", abs((y == 0).mean() - 1 / 3) < 0.01, f"-> {(y == 0).mean():.3f}")


def t_walk_forward():
    days = list(range(12))
    folds, dev, test = walk_forward_split(days, 5, 2)
    ok = all(max(tr) < min(va) for tr, va in folds) and max(dev) < min(test) and not set(dev) & set(test)
    check("Walk-forward theo thời gian, test tách biệt", ok, f"-> {len(folds)} fold, test={test}")


def main():
    for f in (t_ofi_paper_example, t_labels, t_cst_symmetry, t_walk_forward, t_no_lookahead, t_live_equals_offline):
        try:
            f()
        except Exception as e:      # noqa: BLE001
            check(f.__name__, False, f"exception: {e!r}")
    n_ok = sum(r[1] for r in RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} kiểm tra ĐẠT")
    sys.exit(0 if n_ok == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
