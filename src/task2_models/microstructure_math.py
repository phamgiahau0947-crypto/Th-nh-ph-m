"""Công thức vi cấu trúc sổ lệnh (đặc tả Task 2, công thức (1)–(6)) + đặc trưng mở rộng.

Mọi đặc trưng tại thời điểm t CHỈ dùng dữ liệu có thời gian <= t (point-in-time).
Một hàm `build_base_features` dùng chung cho cả dữ liệu lịch sử (Task 1, lưới 1s) và dữ liệu live
(`calculate_all_features` gom buffer 100ms của streamer về lưới 1s) -> train và live tính giống hệt nhau.

Tài liệu:
  - Xu, Gould & Howison (2019) Multi-Level Order-Flow Imbalance in a LOB  -> OFI/MLOFI (9)-(12)
  - Cont, Kukanov & Stoikov (2014) -> OFI tính từ snapshot đều theo thời gian (chú thích 2 của Xu et al.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import load_config

EPS = 1e-12


def _cols(side: str, kind: str, L: int) -> list[str]:
    return [f"{side}_{kind}{i}" for i in range(1, L + 1)]


# =========================================================== (1)–(4) trạng thái sổ lệnh
def mid_price(bid1, ask1):
    """(1) m_t = (a_t^1 + b_t^1) / 2"""
    return (np.asarray(bid1) + np.asarray(ask1)) / 2.0


def spread(bid1, ask1):
    """(2) Spread_t = a_t^1 - b_t^1"""
    return np.asarray(ask1) - np.asarray(bid1)


def microprice(bid1, ask1, qbid1, qask1):
    """(3) Microprice = (b·q^ask + a·q^bid) / (q^bid + q^ask)"""
    qb, qa = np.asarray(qbid1, float), np.asarray(qask1, float)
    return (np.asarray(bid1) * qa + np.asarray(ask1) * qb) / (qb + qa + EPS)


def obi(qbid: np.ndarray, qask: np.ndarray, L: int):
    """(4) OBI_{t,L} = (Σ q^bid - Σ q^ask) / (Σ q^bid + Σ q^ask) trên L tầng đầu, ∈ [-1, 1]."""
    b = qbid[:, :L].sum(axis=1)
    a = qask[:, :L].sum(axis=1)
    return (b - a) / (b + a + EPS)


# =========================================================== (5)–(6) OFI / MLOFI
def ofi_levels(pb, qb, pa, qa, M: int) -> np.ndarray:
    """(5a)-(5c) OFI tầng m giữa 2 snapshot liên tiếp t-1 -> t, m = 1..M (Xu et al. 2019, pt. 9-12).

        ΔW_m = q^bid_t            nếu P^bid_t >  P^bid_{t-1}
               q^bid_t - q^bid_{t-1}  nếu P^bid_t =  P^bid_{t-1}
               -q^bid_{t-1}       nếu P^bid_t <  P^bid_{t-1}
        ΔV_m = -q^ask_{t-1}       nếu P^ask_t >  P^ask_{t-1}
               q^ask_t - q^ask_{t-1}  nếu P^ask_t =  P^ask_{t-1}
               q^ask_t            nếu P^ask_t <  P^ask_{t-1}
        e_m  = ΔW_m - ΔV_m        (dương = áp lực MUA)

    Đầu vào là mảng (n, >=M). Dòng 0 hoặc dòng sau chỗ đứt dữ liệu (t-1 thiếu) -> NaN.
    Trả về mảng (n, M) = vector MLOFI (6) cho từng khoảng (t-1, t]."""
    pb, qb, pa, qa = (np.asarray(x, float)[:, :M] for x in (pb, qb, pa, qa))
    pb0, qb0, pa0, qa0 = (np.vstack([np.full((1, M), np.nan), x[:-1]]) for x in (pb, qb, pa, qa))
    tol = 1e-9 * np.maximum(np.abs(pb), 1.0)
    b_up, b_eq = pb > pb0 + tol, np.abs(pb - pb0) <= tol
    a_up, a_eq = pa > pa0 + tol, np.abs(pa - pa0) <= tol
    dW = np.where(b_up, qb, np.where(b_eq, qb - qb0, -qb0))
    dV = np.where(a_up, -qa0, np.where(a_eq, qa - qa0, qa))
    e = dW - dV
    e[np.isnan(pb0) | np.isnan(pa0)] = np.nan
    return e


# =========================================================== độ sâu / thanh khoản
def impact_cost_bps(prices: np.ndarray, qtys: np.ndarray, mid: np.ndarray, notional: float, side: str):
    """Chi phí tác động (bps so với mid) của lệnh thị trường `notional` USDT đi hết các tầng.
    Phần vượt quá 20 tầng giả định khớp ở giá tầng 20 (cận dưới của chi phí thật).
    Trả về (cost_bps, unfilled_flag)."""
    P = np.asarray(prices, float)
    Q = np.asarray(qtys, float)
    n = P * Q
    cum = np.cumsum(n, axis=1)
    prev = cum - n
    fill = np.clip(notional - prev, 0.0, n)
    qty = (fill / np.where(P > 0, P, np.nan)).sum(axis=1)
    rem = np.maximum(notional - cum[:, -1], 0.0)
    qty = qty + rem / P[:, -1]
    avg = notional / qty
    cost = (avg - mid) / mid * 1e4 if side == "ask" else (mid - avg) / mid * 1e4
    return cost, rem > 0


def depth_slope(prices: np.ndarray, qtys: np.ndarray, mid: np.ndarray) -> np.ndarray:
    """Độ dốc sổ lệnh: tổng KL 20 tầng / khoảng cách (bps) từ mid tới tầng 20 -> KL trên mỗi bps."""
    dist = np.abs(prices[:, -1] - mid) / mid * 1e4
    return qtys.sum(axis=1) / np.maximum(dist, 1e-6)


# =========================================================== lưới 1 giây
def regrid_1s(df: pd.DataFrame) -> pd.DataFrame:
    """Đưa 1 ngày (hoặc 1 buffer) về lưới đều 1s. Giây thiếu -> NaN (không bịa dữ liệu)."""
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    ts = df["timestamp"].astype("int64").to_numpy()
    full = np.arange(ts[0], ts[-1] + 1, 1000, dtype="int64")
    out = df.set_index("timestamp").reindex(full)
    out.index.name = "timestamp"
    return out


def buffer_to_1s(buf: pd.DataFrame, levels: int = 20, max_stale_ms: int = 5000) -> pd.DataFrame:
    """Gom buffer 100ms của streamer (Task 1) về lưới 1s giống kho nền:
    sổ lệnh tại t = snapshot cuối cùng <= t; trade gom trong (t-1s, t]."""
    if buf is None or buf.empty:
        return pd.DataFrame()
    b = buf.copy()
    b["bucket"] = (np.ceil(b["timestamp"].astype("int64") / 1000.0) * 1000).astype("int64")
    book_cols = ["mid_price", "spread"] + [c for s in ("bids", "asks") for k in "pq" for c in _cols(s, k, levels)]
    g = b.groupby("bucket")
    book = g[book_cols].last()
    book["book_ts"] = g["timestamp"].max()
    trades = g[["trade_qty", "buy_qty", "sell_qty", "trade_count"]].sum()
    out = book.join(trades)
    full = np.arange(out.index[0], out.index[-1] + 1, 1000, dtype="int64")
    out = out.reindex(full)
    out[book_cols + ["book_ts"]] = out[book_cols + ["book_ts"]].ffill()
    out[["trade_qty", "buy_qty", "sell_qty", "trade_count"]] = out[["trade_qty", "buy_qty", "sell_qty",
                                                                   "trade_count"]].fillna(0)
    stale = (out.index.to_numpy() - out["book_ts"].to_numpy()) > max_stale_ms
    out.loc[stale, book_cols] = np.nan
    out.index.name = "timestamp"
    return out


# =========================================================== bảng đặc trưng
def feature_columns(cfg: dict | None = None) -> list[str]:
    """Danh sách đặc trưng đưa vào mô hình (thứ tự cố định)."""
    cfg = cfg or load_config()
    fc = cfg["features"]
    M = cfg["mlofi_levels"]
    cols = ["spread_bps", "micro_dev_bps", "obi_1", "obi_5", "obi_10", "obi_20",
            "log_depth_b5", "log_depth_a5", "log_depth_20", "depth_depletion",
            "log_slope_b", "log_slope_a", "range20_bps", "impact_rt_bps", "impact_rel",
            "impact_ma10_rel", "impact_ma60_rel", "impact_unfilled_60s"]
    cols += [f"ofi{m}_{w}s" for w in fc["ofi_windows_s"] for m in range(1, M + 1)]
    cols += [f"mlofi_sum_{w}s" for w in fc["ofi_windows_s"]]
    cols += [f"ti_{w}s" for w in fc["ti_windows_s"]] + ["log_trades_10s", "vol_rel_10s"]
    cols += [f"ret_{w}s" for w in fc["ret_windows_s"]]
    cols += [f"rv_{w}s" for w in fc["rv_windows_s"]] + ["rv_ratio"]
    cols += ["hour_sin", "hour_cos"]
    cols += STRUCTURAL_FEATURES
    return cols


# Đặc trưng từ 2 mô hình cấu trúc (cần tham số hiệu chỉnh trên tập train) – xem add_structural_features
STRUCTURAL_FEATURES = ["cst_p_up", "as_delta_star_bps", "as_stress_ratio", "as_res_skew_bps"]

FEATURE_SETS = {
    # RQ1/RQ2: so sánh thông tin tĩnh (OBI) với dòng lệnh (OFI 1 tầng, MLOFI 10 tầng)
    "OBI": ["obi_1", "obi_5", "obi_10", "obi_20", "micro_dev_bps"],
    "OFI_L1": ["ofi1_1s", "ofi1_10s", "ofi1_30s"],
}


def feature_set(name: str, cfg: dict | None = None) -> list[str]:
    cfg = cfg or load_config()
    M = cfg["mlofi_levels"]
    W = cfg["features"]["ofi_windows_s"]
    if name in FEATURE_SETS:
        return FEATURE_SETS[name]
    if name == "MLOFI":
        return [f"ofi{m}_{w}s" for w in W for m in range(1, M + 1)]
    if name == "OBI+MLOFI":
        return feature_set("OBI", cfg) + feature_set("MLOFI", cfg)
    if name == "ALL":
        return feature_columns(cfg)
    raise KeyError(name)


def notional_for(symbol: str | None, cfg: dict) -> float:
    n = cfg["stress"]["notional_usdt"]
    if isinstance(n, dict):
        return float(n.get(str(symbol).upper(), next(iter(n.values()))))
    return float(n)


def build_base_features(df: pd.DataFrame, cfg: dict | None = None, symbol: str | None = None) -> pd.DataFrame:
    """Tính toàn bộ đặc trưng KHÔNG cần hiệu chỉnh cho 1 đoạn liên tục (1 ngày / 1 buffer).
    `df`: lưới 1s (index = timestamp ms) có cột bids_p1.., bids_q1.., asks_p1.., asks_q1..,
          trade_qty, buy_qty, sell_qty, trade_count.
    Trả về DataFrame cùng index: đặc trưng + cột phụ trợ (mid, bid1, ask1, ...) dùng cho nhãn/PnL."""
    cfg = cfg or load_config()
    L, M = cfg["levels"], cfg["mlofi_levels"]
    fc = cfg["features"]
    W = int(fc["norm_window_s"])
    if symbol is None and "symbol" in df.columns and df["symbol"].notna().any():
        symbol = df["symbol"].dropna().iloc[0]
    notional = notional_for(symbol, cfg)

    pb = df[_cols("bids", "p", L)].to_numpy(float)
    qb = df[_cols("bids", "q", L)].to_numpy(float)
    pa = df[_cols("asks", "p", L)].to_numpy(float)
    qa = df[_cols("asks", "q", L)].to_numpy(float)
    bid1, ask1 = pb[:, 0], pa[:, 0]
    m = mid_price(bid1, ask1)
    out = pd.DataFrame(index=df.index)
    out["mid"], out["bid1"], out["ask1"] = m, bid1, ask1
    out["q_bid1"], out["q_ask1"] = qb[:, 0], qa[:, 0]

    # (1)-(4)
    out["spread_bps"] = spread(bid1, ask1) / m * 1e4
    out["micro_dev_bps"] = (microprice(bid1, ask1, qb[:, 0], qa[:, 0]) - m) / m * 1e4
    for l in (1, 5, 10, 20):
        out[f"obi_{l}"] = obi(qb, qa, l)

    # độ sâu, độ dốc, chi phí tác động
    db5, da5 = qb[:, :5].sum(1), qa[:, :5].sum(1)
    d20 = qb.sum(1) + qa.sum(1)
    out["log_depth_b5"], out["log_depth_a5"] = np.log1p(db5), np.log1p(da5)
    out["log_depth_20"] = np.log1p(d20)
    out["depth20_tot"] = d20
    d20s = pd.Series(d20, index=df.index)
    out["depth_depletion"] = np.log((d20 + EPS) / (d20s.rolling(W, min_periods=60).median() + EPS))
    out["log_slope_b"] = np.log1p(depth_slope(pb, qb, m))
    out["log_slope_a"] = np.log1p(depth_slope(pa, qa, m))
    out["range20_bps"] = (pa[:, -1] - pb[:, -1]) / m * 1e4
    cb, ub = impact_cost_bps(pa, qa, m, notional, "ask")
    cs, us = impact_cost_bps(pb, qb, m, notional, "bid")
    out["impact_buy_bps"], out["impact_sell_bps"] = cb, cs
    out["impact_rt_bps"] = cb + cs
    out["impact_unfilled"] = (ub | us).astype(float)
    imp = pd.Series(out["impact_rt_bps"].to_numpy(), index=df.index)
    imp_med = imp.rolling(W, min_periods=60).median()
    out["impact_rel"] = np.log((imp + EPS) / (imp_med + EPS))
    out["impact_ma10_bps"] = imp.rolling(10, min_periods=10).mean()   # "spread thực tế" cho StressRatio (A-S)
    for w in (10, 60):     # chi phí trung bình gần đây (trạng thái thanh khoản kéo dài, không phải 1 giây)
        out[f"impact_ma{w}_rel"] = np.log((imp.rolling(w, min_periods=w).mean() + EPS) / (imp_med + EPS))
    out["impact_unfilled_60s"] = pd.Series((ub | us).astype(float), index=df.index).rolling(60, min_periods=10).mean()

    # (5)-(6) MLOFI, chuẩn hóa theo độ sâu trung bình trượt của chính tầng m (Cont et al. 2014)
    e = ofi_levels(pb, qb, pa, qa, M)
    depth_m = pd.DataFrame((qb[:, :M] + qa[:, :M]) / 2.0, index=df.index).rolling(W, min_periods=60).mean()
    e_norm = pd.DataFrame(e / (depth_m.to_numpy() + EPS), index=df.index)
    for m_ in range(M):
        out[f"ofi_raw{m_ + 1}"] = e[:, m_]
    for w in fc["ofi_windows_s"]:
        s = e_norm.rolling(w, min_periods=w).sum() if w > 1 else e_norm
        for m_ in range(M):
            out[f"ofi{m_ + 1}_{w}s"] = s[m_].to_numpy()
        out[f"mlofi_sum_{w}s"] = s.sum(axis=1, min_count=M).to_numpy()

    # dòng lệnh khớp
    buy = df["buy_qty"].astype(float)
    sell = df["sell_qty"].astype(float)
    cnt = df["trade_count"].astype(float)
    vol = df["trade_qty"].astype(float)
    for w in fc["ti_windows_s"]:
        b_, s_ = buy.rolling(w, min_periods=w).sum(), sell.rolling(w, min_periods=w).sum()
        out[f"ti_{w}s"] = ((b_ - s_) / (b_ + s_ + EPS)).to_numpy()
    out["log_trades_10s"] = np.log1p(cnt.rolling(10, min_periods=10).sum()).to_numpy()
    v10 = vol.rolling(10, min_periods=10).sum()
    out["vol_rel_10s"] = np.log((v10 + EPS) / (vol.rolling(W, min_periods=60).mean() * 10 + EPS)).to_numpy()
    b300, s300 = buy.rolling(300, min_periods=60).sum(), sell.rolling(300, min_periods=60).sum()
    out["ti_300s"] = ((b300 - s300) / (b300 + s300 + EPS)).to_numpy()   # dùng cho tồn kho market maker (A-S)

    # lợi suất quá khứ & biến động thực tế
    lm = pd.Series(np.log(m), index=df.index)
    r1 = lm.diff() * 1e4
    for w in fc["ret_windows_s"]:
        out[f"ret_{w}s"] = ((lm - lm.shift(w)) * 1e4).to_numpy()
    for w in fc["rv_windows_s"]:
        out[f"rv_{w}s"] = np.sqrt((r1 ** 2).rolling(w, min_periods=w // 2).sum()).to_numpy()
    out["rv_ratio"] = np.log((out["rv_60s"] / np.sqrt(60) + 1e-6) / (out["rv_300s"] / np.sqrt(300) + 1e-6))
    out["sigma_bps"] = out["rv_300s"] / np.sqrt(300)     # σ theo bps/√giây (Avellaneda–Stoikov)

    # thời gian trong ngày (UTC)
    sec = (df.index.to_numpy() // 1000) % 86400
    out["hour_sin"] = np.sin(2 * np.pi * sec / 86400)
    out["hour_cos"] = np.cos(2 * np.pi * sec / 86400)
    return out


def add_structural_features(feat: pd.DataFrame, params: dict | None) -> pd.DataFrame:
    """Thêm đặc trưng từ Avellaneda–Stoikov & Cont–Stoikov–Talreja với tham số đã hiệu chỉnh trên TRAIN."""
    from . import avellaneda_stoikov as AS
    from .cont_stoikov_talreja import CSTModel
    if not params:
        for c in STRUCTURAL_FEATURES:
            feat[c] = np.nan
        return feat
    cst = CSTModel.from_dict(params["cst"])
    feat["cst_p_up"] = cst.p_up(feat["q_ask1"].to_numpy(), feat["q_bid1"].to_numpy())
    a = params["as"]
    ds = AS.optimal_spread_bps(feat["sigma_bps"].to_numpy(), a["k"], a["gamma"], a["horizon_s"])
    feat["as_delta_star_bps"] = ds
    feat["as_stress_ratio"] = AS.stress_ratio(feat["impact_ma10_bps"].to_numpy(), ds, a["ratio_norm"])
    feat["as_res_skew_bps"] = AS.reservation_skew_bps(feat["ti_300s"].to_numpy(), feat["sigma_bps"].to_numpy(),
                                                      a["gamma"], a["horizon_s"])
    return feat


def calculate_all_features(data: dict, params: dict | None = None, cfg: dict | None = None) -> dict:
    """Interface Contract tích hợp (Phần 5 đặc tả): nhận dict từ BinanceLOBStreamer.get_latest_data()
    -> trả về dict đặc trưng của giây mới nhất (đã đủ cột cho ModelInference.predict_one).
    `params`: tham số cấu trúc (A-S, CST); None -> tự đọc models/model_meta.json nếu có."""
    cfg = cfg or load_config()
    symbol = data.get("symbol") or cfg["symbols"][0]
    if params is None:
        params = load_structural_params(symbol)
    df1 = buffer_to_1s(data.get("buffer_df"), cfg["levels"])
    if df1.empty or df1[["bids_p1", "asks_p1"]].iloc[-1].isna().any():
        return {}
    feat = add_structural_features(build_base_features(df1, cfg, symbol), params)
    row = feat.iloc[-1].to_dict()
    row["timestamp"] = int(feat.index[-1])
    row["symbol"] = data.get("symbol")
    row["warmup_s"] = int(len(feat))
    return row


def load_structural_params(symbol: str | None = None) -> dict | None:
    import json
    from .common import MODEL_DIR
    f = MODEL_DIR / "model_meta.json"
    if not f.exists():
        return None
    meta = json.loads(f.read_text(encoding="utf-8"))
    sym = symbol or meta.get("default_symbol", "BTCUSDT")
    return meta["symbols"].get(sym, {}).get("structural")
