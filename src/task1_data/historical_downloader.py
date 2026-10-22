"""Tải dữ liệu LỊCH SỬ.

1) Trades (lệnh đã khớp) – Binance Data Portal (data.binance.vision), file zip theo ngày, miễn phí.
2) Sổ lệnh lịch sử (LOB 25 tầng) – Tardis.dev, ngày mùng 1 mỗi tháng miễn phí không cần API key.
   (Binance KHÔNG cung cấp lịch sử sổ lệnh spot -> xem mục "Hướng đi khác" trong README_TASK_1.md)

Dùng độc lập:
  python -m src.task1_data.historical_downloader trades --start 2026-10-05 --end 2026-10-18
  python -m src.task1_data.historical_downloader seed --months 6
  python -m src.task1_data.historical_downloader seed --date 2026-10-01
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .core.common import (Paths, append_source_log, day_start_ms, first_days_of_months, get_logger,
                     http_session, level_columns, load_config, to_ms, DAY_MS)
from .core.store import save_day, write_parquet
from .core.transform import aggregate_trades, build_unified, combine_trades, grid_book, reduce_book

log = get_logger()

BINANCE_TRADE_COLS = ["id", "price", "qty", "quote_qty", "time", "is_buyer_maker", "is_best_match"]


# --------------------------------------------------------------------- download helper
def download(url: str, dest: Path, session=None, desc: str = "") -> int | None:
    """Tải file theo luồng (không giữ hết trong RAM). Trả số byte; None nếu không tồn tại (404/401/403)."""
    session = session or http_session()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with session.get(url, stream=True, timeout=(10, 120)) as r:
        if r.status_code in (401, 403, 404):
            log.warning(f"Không có dữ liệu ({r.status_code}): {url}")
            return None
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        done, next_report = 0, 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total and done >= next_report:
                    log.info(f"  {desc} {done/1e6:,.0f}/{total/1e6:,.0f} MB")
                    next_report += max(total // 5, 1)
    tmp.replace(dest)
    return done


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------- BINANCE TRADES
def binance_trades_url(symbol: str, day: dt.date) -> str:
    base = load_config()["sources"]["binance_vision"].rstrip("/")
    return f"{base}/data/spot/daily/trades/{symbol}/{symbol}-trades-{day.isoformat()}.zip"


def _iter_binance_trade_chunks(zip_path: Path, chunksize: int = 1_000_000):
    with zipfile.ZipFile(zip_path) as z:
        name = [n for n in z.namelist() if n.endswith(".csv")][0]
        with z.open(name) as fh:
            first = fh.readline().decode("utf-8", "ignore")
        has_header = not first.split(",")[0].strip().lstrip("-").isdigit()
        with z.open(name) as fh:
            reader = pd.read_csv(io.TextIOWrapper(fh, encoding="utf-8"), header=0 if has_header else None,
                                 names=BINANCE_TRADE_COLS, usecols=[1, 2, 4, 5], chunksize=chunksize,
                                 dtype={"price": "float64", "qty": "float64", "time": "int64"})
            for ch in reader:
                ibm = ch["is_buyer_maker"]
                if ibm.dtype != bool:
                    ibm = ibm.astype(str).str.strip().str.lower().isin(["true", "1"])
                yield pd.DataFrame({"ts_ms": to_ms(ch["time"].values), "price": ch["price"].values,
                                    "qty": ch["qty"].values, "is_buyer_maker": ibm.values})


def fetch_binance_trades_day(day: dt.date, paths: Paths | None = None, session=None,
                             force: bool = False) -> pd.DataFrame | None:
    """Tải trade 1 ngày -> gom theo khung (mặc định 1s) -> lưu trades_1s. Trả DataFrame partial."""
    cfg = load_config()
    paths = (paths or Paths()).ensure()
    out_file = paths.trades_1s_file(day)
    if out_file.exists() and not force:
        return pd.read_parquet(out_file)
    interval = cfg["store"]["sample_interval_ms"]
    url = binance_trades_url(paths.symbol, day)
    zpath = paths.tmp / Path(url).name
    session = session or http_session()
    log.info(f"[trades] {day}: tải {url}")
    nbytes = download(url, zpath, session, desc=f"trades {day}")
    if nbytes is None:
        append_source_log(paths, dataset="trades", date=day, source="binance_vision", url=url, status="missing")
        return None
    digest = sha256_of(zpath)
    status = "ok"
    try:  # kiểm tra checksum chính thức của Binance
        r = session.get(url + ".CHECKSUM", timeout=20)
        if r.ok:
            expected = r.text.split()[0].strip().lower()
            if expected != digest:
                zpath.unlink(missing_ok=True)
                raise IOError(f"Sai checksum file {zpath.name}")
            status = "ok_checksum"
    except IOError:
        raise
    except Exception:
        pass

    parts, n, ticks = [], 0, []
    keep_raw = bool(cfg["store"].get("keep_raw_trades", False))
    lo, hi = day_start_ms(day), day_start_ms(day) + DAY_MS
    for ch in _iter_binance_trade_chunks(zpath):
        n += len(ch)
        parts.append(aggregate_trades(ch, interval))
        if keep_raw:
            ticks.append(ch)
    agg = combine_trades(parts)
    agg = agg[(agg["bucket"] >= lo) & (agg["bucket"] <= hi)]
    write_parquet(agg, out_file)
    if keep_raw and ticks:
        write_parquet(pd.concat(ticks, ignore_index=True), paths.trades_tick / f"{paths.symbol}_trades_{day}.parquet")
    zpath.unlink(missing_ok=True)
    append_source_log(paths, dataset="trades", date=day, source="binance_vision", url=url, bytes=nbytes,
                      rows=n, sha256=digest, status=status)
    log.info(f"[trades] {day}: {n:,} lệnh khớp -> {len(agg):,} dòng/giây ({out_file.stat().st_size/1e6:.1f} MB)")
    return agg


# --------------------------------------------------------------------- TARDIS
def tardis_url(day: dt.date, data_type: str, symbol: str) -> str:
    cfg = load_config()
    base = cfg["sources"]["tardis_datasets"].rstrip("/")
    ex = cfg["seed"]["tardis_exchange"]
    return f"{base}/v1/{ex}/{data_type}/{day.year}/{day.month:02d}/{day.day:02d}/{symbol}.csv.gz"


def fetch_tardis_book_day(day: dt.date, paths: Paths | None = None, session=None) -> pd.DataFrame | None:
    """Tải book_snapshot_25 của Tardis (1 ngày, mọi lần sổ lệnh thay đổi) -> giữ snapshot cuối mỗi khung."""
    cfg = load_config()
    paths = (paths or Paths()).ensure()
    levels, interval = cfg["levels"], cfg["store"]["sample_interval_ms"]
    dtype_name = cfg["seed"]["tardis_data_type"]
    url = tardis_url(day, dtype_name, paths.symbol)
    gz = paths.tmp / f"tardis_{dtype_name}_{paths.symbol}_{day}.csv.gz"
    log.info(f"[seed-book] {day}: tải {url}")
    nbytes = download(url, gz, session, desc=f"LOB {day}")
    if nbytes is None:
        append_source_log(paths, dataset="lob_seed", date=day, source="tardis", url=url, status="missing")
        return None
    src_cols, rename = ["timestamp"], {}
    for i in range(levels):
        for side, short in (("bids", "bids"), ("asks", "asks")):
            src_cols += [f"{side}[{i}].price", f"{side}[{i}].amount"]
            rename[f"{side}[{i}].price"] = f"{short}_p{i+1}"
            rename[f"{side}[{i}].amount"] = f"{short}_q{i+1}"
    parts, n = [], 0
    for ch in pd.read_csv(gz, usecols=src_cols, chunksize=200_000, compression="gzip"):
        n += len(ch)
        ch = ch.rename(columns=rename)
        ch["ts_ms"] = to_ms(ch["timestamp"].values, "us")
        parts.append(reduce_book(ch, interval, levels))
    gz.unlink(missing_ok=True)
    book = reduce_book(pd.concat(parts, ignore_index=True), interval, levels)
    append_source_log(paths, dataset="lob_seed", date=day, source="tardis", url=url, bytes=nbytes, rows=n, status="ok")
    log.info(f"[seed-book] {day}: {n:,} snapshot gốc -> {len(book):,} khung")
    return book


def fetch_tardis_trades_day(day: dt.date, paths: Paths | None = None, session=None) -> pd.DataFrame | None:
    """Dự phòng khi Binance Data Portal thiếu file trade của ngày seed."""
    cfg = load_config()
    paths = (paths or Paths()).ensure()
    url = tardis_url(day, "trades", paths.symbol)
    gz = paths.tmp / f"tardis_trades_{paths.symbol}_{day}.csv.gz"
    if download(url, gz, session, desc=f"trades(tardis) {day}") is None:
        return None
    parts = []
    for ch in pd.read_csv(gz, usecols=["timestamp", "side", "price", "amount"], chunksize=1_000_000):
        parts.append(aggregate_trades(pd.DataFrame({
            "ts_ms": to_ms(ch["timestamp"].values, "us"), "price": ch["price"].values,
            "qty": ch["amount"].values, "is_buyer_maker": (ch["side"] == "sell").values}),
            cfg["store"]["sample_interval_ms"]))
    gz.unlink(missing_ok=True)
    agg = combine_trades(parts)
    write_parquet(agg, paths.trades_1s_file(day))
    append_source_log(paths, dataset="trades", date=day, source="tardis", url=url, status="ok")
    return agg


def build_seed_day(day: dt.date, paths: Paths | None = None, session=None) -> Path | None:
    cfg = load_config()
    paths = (paths or Paths()).ensure()
    session = session or http_session()
    book = fetch_tardis_book_day(day, paths, session)
    if book is None or book.empty:
        log.error(f"[seed] {paths.symbol} {day}: không lấy được sổ lệnh từ Tardis -> bỏ qua")
        return None
    trades, tsrc = fetch_binance_trades_day(day, paths, session), "binance_vision"
    if trades is None:
        trades, tsrc = fetch_tardis_trades_day(day, paths, session), "tardis"
    lo = day_start_ms(day)
    grid = grid_book(book, cfg["store"]["sample_interval_ms"], cfg["store"]["max_book_staleness_ms"],
                     lo, lo + DAY_MS)
    df = build_unified(grid, trades, symbol=paths.symbol, levels=cfg["levels"],
                       interval_ms=cfg["store"]["sample_interval_ms"],
                       book_source="tardis", trades_source=tsrc if trades is not None else "none")
    f = save_day(df, day, "seed", paths)
    log.info(f"[seed] {paths.symbol} {day}: {len(df):,} dòng -> {f.name} ({f.stat().st_size/1e6:.1f} MB)")
    return f


# --------------------------------------------------------------------- CLI
def main(argv=None):
    from .core.common import all_symbols
    ap = argparse.ArgumentParser(description="Task 1 – tải dữ liệu lịch sử")
    ap.add_argument("--symbol", action="append", default=None, help="Mặc định: mọi cặp trong config")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trades", help="Tải trades Binance theo khoảng ngày")
    t.add_argument("--start", required=True); t.add_argument("--end", required=True)
    t.add_argument("--force", action="store_true")
    s = sub.add_parser("seed", help="Dựng bộ nền cố định từ Tardis (ngày mùng 1)")
    s.add_argument("--months", type=int, default=None); s.add_argument("--date", default=None)
    a = ap.parse_args(argv)
    sess = http_session()
    for sym in (a.symbol or all_symbols()):
        paths = Paths(symbol=sym).ensure()
        if a.cmd == "trades":
            d, e = dt.date.fromisoformat(a.start), dt.date.fromisoformat(a.end)
            while d <= e:
                fetch_binance_trades_day(d, paths, sess, force=a.force)
                d += dt.timedelta(days=1)
        else:
            days = [dt.date.fromisoformat(a.date)] if a.date else \
                first_days_of_months(a.months or load_config()["seed"]["months_back"])
            for d in days:
                build_seed_day(d, paths, sess)


if __name__ == "__main__":
    main()
