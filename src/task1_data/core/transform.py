"""Biến đổi dữ liệu thô -> lưới thời gian đều (mặc định 1 giây/dòng), đảm bảo POINT-IN-TIME.

Quy ước thời gian (quan trọng để không có look-ahead):
  * Dòng có timestamp = t biểu diễn TRẠNG THÁI SỔ LỆNH TẠI ĐÚNG THỜI ĐIỂM t
    = snapshot cuối cùng có thời gian <= t (không dùng snapshot sau t).
  * Các cột trade (trade_qty, trade_count, buy_qty, sell_qty, vwap, is_buyer_maker)
    tổng hợp các lệnh khớp trong khoảng (t - Δ, t].
  * trade_price = giá khớp gần nhất tính đến t (forward-fill, chỉ dùng quá khứ).
  * Timestamp gốc được làm tròn LÊN về khung Δ  -> sự kiện không bao giờ bị kéo về quá khứ.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import level_columns, unified_columns


def _bucket(ts_ms: pd.Series | np.ndarray, interval_ms: int) -> np.ndarray:
    ts = np.asarray(ts_ms, dtype="int64")
    return -(-ts // interval_ms) * interval_ms  # ceil


# --------------------------------------------------------------------- BOOK
def reduce_book(df: pd.DataFrame, interval_ms: int, levels: int) -> pd.DataFrame:
    """Giữ snapshot CUỐI CÙNG trong mỗi khung (t-Δ, t]. Có thể gọi lặp trên từng chunk rồi
    nối lại và gọi tiếp (kết quả không đổi). Input cần cột 'ts_ms' + cột tầng giá."""
    lc = level_columns(levels)
    if df.empty:
        return pd.DataFrame(columns=["bucket", "book_ts"] + lc)
    if "bucket" in df.columns and "book_ts" in df.columns:   # đã reduce trước đó
        out = df
    else:
        out = df[["ts_ms"] + lc].rename(columns={"ts_ms": "book_ts"})
        out.insert(0, "bucket", _bucket(out["book_ts"], interval_ms))
    out = out.sort_values(["bucket", "book_ts"], kind="stable")
    out = out.drop_duplicates("bucket", keep="last")
    return out.reset_index(drop=True)


def grid_book(reduced: pd.DataFrame, interval_ms: int, max_stale_ms: int,
              start_ms: int | None = None, end_ms: int | None = None) -> pd.DataFrame:
    """Trải ra lưới đều: khung không có cập nhật mới thì mang trạng thái cũ (hợp lệ point-in-time),
    nhưng nếu trạng thái đã cũ hơn max_stale_ms (mất kết nối/tắt máy) thì bỏ dòng đó."""
    if reduced.empty:
        return reduced.assign(book_age_ms=pd.Series(dtype="int64"))
    r = reduced.set_index("bucket").sort_index()
    lo = int(r.index.min()) if start_ms is None else max(int(r.index.min()), start_ms)
    hi = int(r.index.max()) if end_ms is None else min(int(r.index.max()) + max_stale_ms, end_ms - 1)
    hi = hi - hi % interval_ms
    lo = -(-lo // interval_ms) * interval_ms
    if hi < lo:
        return reduced.iloc[0:0].assign(book_age_ms=pd.Series(dtype="int64"))
    grid = np.arange(lo, hi + 1, interval_ms, dtype="int64")
    # lấy cả snapshot trước 'lo' để có trạng thái ban đầu
    full_index = np.union1d(r.index.values.astype("int64"), grid)
    g = r.reindex(full_index).ffill().reindex(grid)
    g.index.name = "bucket"
    g = g.dropna(subset=["book_ts"])
    g["book_ts"] = g["book_ts"].astype("int64")
    g["book_age_ms"] = g.index.values - g["book_ts"].values
    g = g[g["book_age_ms"] <= max_stale_ms]
    return g.reset_index()


# --------------------------------------------------------------------- TRADES
TRADE_PARTIAL_COLS = ["bucket", "trade_count", "trade_qty", "buy_qty", "sell_qty",
                      "quote_qty", "last_ts", "trade_price", "is_buyer_maker"]


def aggregate_trades(df: pd.DataFrame, interval_ms: int) -> pd.DataFrame:
    """Gom lệnh khớp theo khung (t-Δ, t]. Input: ts_ms, price, qty, is_buyer_maker (True = bên BÁN
    chủ động, theo quy ước Binance). Kết quả là 'partial' -> gộp được nhiều chunk bằng combine_trades."""
    if df.empty:
        return pd.DataFrame(columns=TRADE_PARTIAL_COLS)
    d = pd.DataFrame({
        "bucket": _bucket(df["ts_ms"], interval_ms),
        "ts": np.asarray(df["ts_ms"], dtype="int64"),
        "price": np.asarray(df["price"], dtype="float64"),
        "qty": np.asarray(df["qty"], dtype="float64"),
        "ibm": np.asarray(df["is_buyer_maker"], dtype=bool),
    })
    d["quote"] = d["price"] * d["qty"]
    d["buy"] = np.where(~d["ibm"], d["qty"], 0.0)   # taker mua chủ động
    d["sell"] = np.where(d["ibm"], d["qty"], 0.0)   # taker bán chủ động
    d = d.sort_values(["bucket", "ts"], kind="stable")
    g = d.groupby("bucket", sort=True)
    out = pd.DataFrame({
        "trade_count": g.size(),
        "trade_qty": g["qty"].sum(),
        "buy_qty": g["buy"].sum(),
        "sell_qty": g["sell"].sum(),
        "quote_qty": g["quote"].sum(),
        "last_ts": g["ts"].last(),
        "trade_price": g["price"].last(),
        "is_buyer_maker": g["ibm"].last(),
    }).reset_index()
    return out[TRADE_PARTIAL_COLS]


def combine_trades(parts: list[pd.DataFrame]) -> pd.DataFrame:
    parts = [p for p in parts if p is not None and not p.empty]
    if not parts:
        return pd.DataFrame(columns=TRADE_PARTIAL_COLS)
    d = pd.concat(parts, ignore_index=True)
    if not d["bucket"].duplicated().any():
        return d.sort_values("bucket").reset_index(drop=True)
    d = d.sort_values(["bucket", "last_ts"], kind="stable")
    g = d.groupby("bucket", sort=True)
    out = pd.DataFrame({
        "trade_count": g["trade_count"].sum(),
        "trade_qty": g["trade_qty"].sum(),
        "buy_qty": g["buy_qty"].sum(),
        "sell_qty": g["sell_qty"].sum(),
        "quote_qty": g["quote_qty"].sum(),
        "last_ts": g["last_ts"].last(),
        "trade_price": g["trade_price"].last(),
        "is_buyer_maker": g["is_buyer_maker"].last(),
    }).reset_index()
    return out[TRADE_PARTIAL_COLS]


# --------------------------------------------------------------------- UNIFIED
def build_unified(book_grid: pd.DataFrame, trades: pd.DataFrame, *, symbol: str, levels: int,
                  interval_ms: int, book_source: str, trades_source: str) -> pd.DataFrame:
    """Ghép sổ lệnh (lưới đều) + trade đã gom -> bảng chuẩn theo Interface Contract Task 1."""
    cols = unified_columns(levels)
    if book_grid.empty:
        return conform(pd.DataFrame(columns=cols), levels)
    b = book_grid.copy()
    b = b.rename(columns={"bucket": "timestamp"})
    b["timestamp"] = b["timestamp"].astype("int64")

    if trades is not None and not trades.empty:
        t = trades.set_index("bucket").sort_index()
        b = b.merge(t.drop(columns=["trade_price", "last_ts"]), left_on="timestamp",
                    right_index=True, how="left")
        # giá khớp gần nhất tính đến t (chỉ nhìn quá khứ)
        lp = t["trade_price"]
        idx = np.union1d(lp.index.values.astype("int64"), b["timestamp"].values)
        lp_ff = lp.reindex(idx).ffill()
        b["trade_price"] = lp_ff.reindex(b["timestamp"].values).values
    else:
        for c in ["trade_count", "trade_qty", "buy_qty", "sell_qty", "quote_qty"]:
            b[c] = 0.0
        b["is_buyer_maker"] = pd.NA
        b["trade_price"] = np.nan

    for c in ["trade_count", "trade_qty", "buy_qty", "sell_qty", "quote_qty"]:
        b[c] = b[c].fillna(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        b["vwap"] = np.where(b["trade_qty"] > 0, b["quote_qty"] / b["trade_qty"], np.nan)

    b["mid_price"] = (b["bids_p1"] + b["asks_p1"]) / 2.0
    b["spread"] = b["asks_p1"] - b["bids_p1"]
    # loại sổ lệnh lỗi (rỗng hoặc bị chéo giá)
    b = b[(b["spread"] > 0) & b["bids_p1"].notna() & b["asks_p1"].notna()]

    b["symbol"] = symbol
    b["book_source"] = book_source
    b["trades_source"] = trades_source if trades is not None and not trades.empty else "none"
    b = b.sort_values("timestamp").reset_index(drop=True)
    gaps = b["timestamp"].diff()
    b["gap_flag"] = (gaps.isna() | (gaps > interval_ms)).values
    b["datetime"] = pd.to_datetime(b["timestamp"], unit="ms", utc=True)
    return conform(b, levels)


def conform(df: pd.DataFrame, levels: int) -> pd.DataFrame:
    """Ép đúng thứ tự cột + kiểu dữ liệu -> mọi file ngày có schema giống hệt nhau."""
    cols = unified_columns(levels)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols].copy()
    df["timestamp"] = df["timestamp"].astype("int64")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).astype("datetime64[ms, UTC]")
    for c in ["symbol", "book_source", "trades_source"]:
        df[c] = df[c].astype(object).where(df[c].notna(), None).astype(object)
    for side in ("bids", "asks"):
        for i in range(1, levels + 1):
            df[f"{side}_p{i}"] = df[f"{side}_p{i}"].astype("float64")
            df[f"{side}_q{i}"] = df[f"{side}_q{i}"].astype("float32")
    for c in ["mid_price", "spread", "trade_price", "trade_qty", "buy_qty", "sell_qty", "vwap"]:
        df[c] = df[c].astype("float64")
    df["trade_count"] = df["trade_count"].fillna(0).astype("int32")
    df["is_buyer_maker"] = df["is_buyer_maker"].astype("boolean")
    df["book_ts"] = df["book_ts"].astype("int64") if len(df) else df["book_ts"].astype("int64")
    df["book_age_ms"] = df["book_age_ms"].astype("int32")
    df["gap_flag"] = df["gap_flag"].astype(bool)
    return df


# --------------------------------------------------------------------- helpers
def book_lists_to_row(bids, asks, levels: int) -> list[float]:
    """[[p,q],...] (bids giảm dần, asks tăng dần) -> list giá trị theo thứ tự level_columns."""
    def side(lst):
        ps = [float(x[0]) for x in lst[:levels]]
        qs = [float(x[1]) for x in lst[:levels]]
        pad = levels - len(ps)
        return ps + [np.nan] * pad, qs + [np.nan] * pad
    bp, bq = side(bids)
    ap, aq = side(asks)
    return bp + bq + ap + aq
