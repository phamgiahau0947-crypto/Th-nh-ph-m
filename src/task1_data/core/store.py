"""Quản lý KHO NỀN của Task 1.

Cấu trúc:
  data/raw/raw_lob_trades.parquet/            <- "parquet dataset" (thư mục), mỗi ngày 1 file
        BTCUSDT_2026-10-01_seed.parquet       <- bộ nền cố định (Tardis), không bao giờ bị xóa
        BTCUSDT_2026-10-18_rolling.parquet    <- kho trượt, tự xóa khi quá retention_days
  data/raw/BTCUSDT/trades_1s/                 <- trade đã gom theo giây cho MỌI ngày trong khung
  data/raw/BTCUSDT/live_capture/YYYY-MM-DD/   <- dữ liệu tự thu bằng WS/REST, chờ chốt ngày
  data/processed/BTCUSDT_view.csv             <- 1 file CSV gọn để mở xem
  data/processed/inventory.csv                <- thống kê từng ngày trong kho
  data/processed/source_log.csv               <- nhật ký nguồn tải

Đọc toàn bộ kho (cho Task 2):
    import pandas as pd
    df = pd.read_parquet("data/raw/raw_lob_trades.parquet")      # cần pyarrow
  hoặc
    from src.task1_data import load_raw_lob_trades
    df = load_raw_lob_trades(start="2026-10-01", end="2026-10-18")
"""
from __future__ import annotations

import datetime as dt
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .common import Paths, all_symbols, get_logger, level_columns, load_config, utc_today

log = get_logger()
_DAY_RE = re.compile(r"_(\d{4}-\d{2}-\d{2})_(seed|rolling)\.parquet$")


# --------------------------------------------------------------------- IO
def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Ghi an toàn: ghi ra file tạm rồi đổi tên (không để lại file hỏng nếu bị ngắt giữa chừng)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")   # tên bắt đầu bằng "." -> pyarrow bỏ qua khi đọc thư mục
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, path)


def read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(path, columns=columns)


# --------------------------------------------------------------------- day files
def list_days(paths: Paths | None = None) -> list[tuple[dt.date, str, Path]]:
    paths = paths or Paths()
    out = []
    if not paths.raw_lob_trades.exists():
        return out
    for f in sorted(paths.raw_lob_trades.glob(f"{paths.symbol}_*.parquet")):
        m = _DAY_RE.search(f.name)
        if m:
            out.append((dt.date.fromisoformat(m.group(1)), m.group(2), f))
    return sorted(out)


def day_exists(day: dt.date, paths: Paths | None = None) -> str | None:
    for d, kind, _ in list_days(paths):
        if d == day:
            return kind
    return None


def save_day(df: pd.DataFrame, day: dt.date, kind: str, paths: Paths | None = None) -> Path:
    paths = paths or Paths()
    paths.ensure()
    # một ngày chỉ có 1 file: seed ưu tiên hơn rolling
    for d, k, f in list_days(paths):
        if d == day and k != kind:
            if k == "seed" and kind == "rolling":
                log.info(f"{day}: đã có bản seed, bỏ qua bản rolling")
                return f
            f.unlink()
    target = paths.day_file(day, kind)
    write_parquet(df, target)
    return target


def load_raw_lob_trades(start: str | dt.date | None = None, end: str | dt.date | None = None,
                        columns: list[str] | None = None, kinds: tuple[str, ...] = ("seed", "rolling"),
                        symbol: str | list[str] | None = None, paths: Paths | None = None) -> pd.DataFrame:
    """Đọc kho nền thành 1 DataFrame (sắp theo symbol, thời gian). start/end tính theo ngày UTC (bao gồm).
    symbol: "BTCUSDT" / "ETHUSDT" / list; None = mọi cặp trong config (phân biệt bằng cột 'symbol')."""
    s = dt.date.fromisoformat(str(start)) if start else dt.date.min
    e = dt.date.fromisoformat(str(end)) if end else dt.date.max
    if paths is not None:
        syms = [paths.symbol]
    elif symbol is None:
        syms = all_symbols()
    else:
        syms = [symbol] if isinstance(symbol, str) else list(symbol)
    if columns is not None and "symbol" not in columns:
        columns = ["symbol"] + list(columns)
    frames = []
    for sym in syms:
        p = paths or Paths(symbol=sym)
        frames += [read_parquet(f, columns) for d, k, f in list_days(p) if s <= d <= e and k in kinds]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "timestamp" in df.columns:
        df = df.sort_values(["symbol", "timestamp"], kind="stable").reset_index(drop=True)
    return df


# --------------------------------------------------------------------- retention
def apply_retention(paths: Paths | None = None, retention_days: int | None = None,
                    today: dt.date | None = None) -> list[str]:
    """Xóa dữ liệu ROLLING (và trade/live_capture) cũ hơn retention_days. Không đụng tới seed."""
    cfg = load_config()
    paths = paths or Paths()
    retention_days = int(retention_days or cfg["store"]["retention_days"])
    today = today or utc_today()
    cutoff = today - dt.timedelta(days=retention_days)
    seed_days = {d for d, k, _ in list_days(paths) if k == "seed"}
    removed = []
    for d, kind, f in list_days(paths):
        if kind == "rolling" and d < cutoff:
            f.unlink(); removed.append(f.name)
    for folder in (paths.trades_1s, paths.trades_tick):
        if folder.exists():
            for f in folder.glob("*.parquet"):
                m = re.search(r"(\d{4}-\d{2}-\d{2})", f.name)
                if m:
                    d = dt.date.fromisoformat(m.group(1))
                    if d < cutoff and d not in seed_days:
                        f.unlink(); removed.append(f.name)
    if paths.live_capture.exists():
        for sub in paths.live_capture.iterdir():
            try:
                d = dt.date.fromisoformat(sub.name)
            except ValueError:
                continue
            if d < cutoff:
                shutil.rmtree(sub, ignore_errors=True); removed.append(f"live_capture/{sub.name}")
    if removed:
        log.info(f"Retention {retention_days} ngày: đã xóa {len(removed)} mục cũ hơn {cutoff}")
    return removed


# --------------------------------------------------------------------- reports
def build_inventory(paths: Paths | None = None) -> pd.DataFrame:
    """Thống kê mọi ngày của MỌI cặp trong config -> data/processed/inventory.csv."""
    cfg = load_config()
    interval = cfg["store"]["sample_interval_ms"]
    rows = []
    out_path = (paths or Paths()).inventory_csv
    items = [(Paths(symbol=s), it) for s in all_symbols() for it in list_days(Paths(symbol=s))]
    for p_sym, (d, kind, f) in items:
        x = read_parquet(f, ["timestamp", "book_source", "trades_source", "trade_count", "spread", "gap_flag"])
        n = len(x)
        rows.append({
            "symbol": p_sym.symbol, "date": d.isoformat(), "kind": kind,
            "book_source": x["book_source"].iloc[0] if n else "",
            "trades_source": x["trades_source"].iloc[0] if n else "",
            "rows": n,
            "coverage_pct": round(100.0 * n / (86_400_000 / interval), 2),
            "segments": int(x["gap_flag"].sum()) if n else 0,
            "first_utc": pd.to_datetime(x["timestamp"].min(), unit="ms", utc=True).isoformat() if n else "",
            "last_utc": pd.to_datetime(x["timestamp"].max(), unit="ms", utc=True).isoformat() if n else "",
            "trades": int(x["trade_count"].sum()) if n else 0,
            "median_spread": float(x["spread"].median()) if n else np.nan,
            "file_mb": round(f.stat().st_size / 1e6, 2),
        })
    inv = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    inv.to_csv(tmp, index=False, encoding="utf-8-sig")
    if not _replace_retry(tmp, out_path):
        tmp.unlink(missing_ok=True)
        log.warning(f"KHÔNG cập nhật được {out_path.name} vì file đang mở (Excel?). Đóng file rồi chạy lại.")
    return inv


def export_csv_view(paths: Paths | None = None) -> Path | None:
    """1 file CSV gọn (mặc định 10 giây/dòng, 5 tầng) để mở bằng Excel. Ghi lại toàn bộ từ kho
    mỗi lần cập nhật -> luôn khớp với kho (kể cả sau khi xóa ngày cũ)."""
    cfg = load_config()
    paths = paths or Paths()
    every_ms = int(cfg["csv_view"]["interval_s"]) * 1000
    lv = int(cfg["csv_view"]["levels"])
    keep = (["datetime", "mid_price", "spread"] + level_columns(lv)
            + ["trade_price", "trade_count", "buy_qty", "sell_qty", "gap_flag", "book_source"])
    days = list_days(paths)
    if not days:
        return None
    tmp = paths.csv_view.with_name(paths.csv_view.name + ".tmp")
    first = True
    for d, kind, f in days:
        x = read_parquet(f, ["timestamp"] + [c for c in keep if c != "datetime"])
        # gom trade của cả khung 10s, lấy trạng thái sổ lệnh tại cuối khung
        b = -(-x["timestamp"] // every_ms) * every_ms
        agg = x.groupby(b).agg({**{c: "last" for c in x.columns if c not in
                                   ("trade_count", "buy_qty", "sell_qty", "gap_flag")},
                                "trade_count": "sum", "buy_qty": "sum", "sell_qty": "sum", "gap_flag": "max"})
        agg = agg[agg["timestamp"] == agg.index]  # chỉ giữ khung có snapshot đúng mốc
        agg.insert(0, "datetime", pd.to_datetime(agg.index, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S"))
        agg.insert(1, "kind", kind)
        agg = agg.drop(columns=["timestamp"])
        agg.to_csv(tmp, mode="w" if first else "a", header=first, index=False,
                   encoding="utf-8-sig" if first else "utf-8", float_format="%.8g")
        first = False
    if not _replace_retry(tmp, paths.csv_view):
        tmp.unlink(missing_ok=True)
        log.warning(f"KHÔNG cập nhật được {paths.csv_view.name} vì file đang mở (Excel?). "
                    f"Đóng file rồi chạy lại để có bản mới. Kho dữ liệu chính KHÔNG bị ảnh hưởng.")
        return None
    return paths.csv_view


def _replace_retry(src: Path, dst: Path, tries: int = 5) -> bool:
    """Windows khóa file khi đang mở trong Excel -> thử lại vài lần rồi bỏ qua (không làm hỏng việc khác)."""
    import time
    for i in range(tries):
        try:
            os.replace(src, dst)
            return True
        except PermissionError:
            time.sleep(1 + i)
    return False


def store_summary(paths: Paths | None = None) -> str:
    paths = paths or Paths()
    days = list_days(paths)
    total = sum(f.stat().st_size for *_, f in days)
    tr = sum(f.stat().st_size for f in paths.trades_1s.glob("*.parquet")) if paths.trades_1s.exists() else 0
    lc = sum(f.stat().st_size for f in paths.live_capture.rglob("*.parquet")) if paths.live_capture.exists() else 0
    seed = [d for d, k, _ in days if k == "seed"]
    roll = [d for d, k, _ in days if k == "rolling"]
    return (f"[{paths.symbol}] Kho nền: {len(seed)} ngày seed, {len(roll)} ngày rolling | "
            f"raw_lob_trades {total/1e6:.1f}MB, trades_1s {tr/1e6:.1f}MB, live_capture {lc/1e6:.1f}MB")
