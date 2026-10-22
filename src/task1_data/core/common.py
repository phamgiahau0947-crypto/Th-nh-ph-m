"""Tiện ích dùng chung cho Task 1: cấu hình, đường dẫn, schema, thời gian, HTTP."""
from __future__ import annotations

import datetime as dt
import logging
import os
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import yaml

PACKAGE_DIR = Path(__file__).resolve().parents[1]   # .../src/task1_data
PROJECT_ROOT = PACKAGE_DIR.parents[1]               # .../Đề án 5
CONFIG_PATH = PACKAGE_DIR / "config.yaml"

DAY_MS = 86_400_000


# ----------------------------------------------------------------- logging
def get_logger(name: str = "task1") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    # Windows console: tránh lỗi in tiếng Việt
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    return logger


# ----------------------------------------------------------------- config
@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict:
    p = Path(path) if path else CONFIG_PATH
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root_env = os.environ.get("TASK1_DATA_ROOT")
    cfg["_data_root"] = Path(root_env) if root_env else PROJECT_ROOT / "data"
    return cfg


class Paths:
    """Bản đồ thư mục dữ liệu của Task 1."""

    def __init__(self, cfg: dict | None = None, symbol: str | None = None):
        cfg = cfg or load_config()
        self.symbol = (symbol or cfg["symbol"]).upper()
        self.data = Path(cfg["_data_root"])
        self.raw = self.data / "raw"
        self.processed = self.data / "processed"
        # File hợp đồng bàn giao cho Task 2 (là 1 "parquet dataset" = thư mục chứa các file theo ngày)
        self.raw_lob_trades = self.raw / "raw_lob_trades.parquet"
        self.sym = self.raw / self.symbol
        self.trades_1s = self.sym / "trades_1s"          # trade đã gom theo giây (mọi ngày trong khung)
        self.trades_tick = self.sym / "trades_tick"      # trade gốc (chỉ khi keep_raw_trades=true)
        self.live_capture = self.sym / "live_capture"    # dữ liệu tự thu (WS/REST), chờ chốt ngày
        self.tmp = self.sym / "_tmp"                     # file tải tạm, xóa sau khi xử lý
        self.csv_view = self.processed / f"{self.symbol}_view.csv"
        self.inventory_csv = self.processed / "inventory.csv"
        self.source_log = self.processed / "source_log.csv"

    def ensure(self) -> "Paths":
        for p in (self.raw_lob_trades, self.trades_1s, self.live_capture, self.tmp, self.processed):
            p.mkdir(parents=True, exist_ok=True)
        return self

    def day_file(self, day: dt.date, kind: str) -> Path:
        return self.raw_lob_trades / f"{self.symbol}_{day.isoformat()}_{kind}.parquet"

    def trades_1s_file(self, day: dt.date) -> Path:
        return self.trades_1s / f"{self.symbol}_trades1s_{day.isoformat()}.parquet"

    def capture_dir(self, day: dt.date) -> Path:
        return self.live_capture / day.isoformat()


def all_symbols(cfg: dict | None = None) -> list[str]:
    cfg = cfg or load_config()
    return [s.upper() for s in (cfg.get("symbols") or [cfg["symbol"]])]


# ----------------------------------------------------------------- schema
def level_columns(levels: int) -> list[str]:
    cols: list[str] = []
    for side in ("bids", "asks"):
        cols += [f"{side}_p{i}" for i in range(1, levels + 1)]
        cols += [f"{side}_q{i}" for i in range(1, levels + 1)]
    return cols


TRADE_AGG_COLUMNS = ["trade_price", "trade_qty", "is_buyer_maker", "trade_count",
                     "buy_qty", "sell_qty", "vwap"]
META_COLUMNS = ["book_ts", "book_age_ms", "gap_flag", "book_source", "trades_source"]


def unified_columns(levels: int) -> list[str]:
    return (["timestamp", "datetime", "symbol", "mid_price", "spread"]
            + level_columns(levels) + TRADE_AGG_COLUMNS + META_COLUMNS)


# ----------------------------------------------------------------- time
def utc_today() -> dt.date:
    return dt.datetime.now(dt.timezone.utc).date()


def day_start_ms(day: dt.date) -> int:
    return int(dt.datetime(day.year, day.month, day.day, tzinfo=dt.timezone.utc).timestamp() * 1000)


def ms_to_day(ms: int) -> dt.date:
    return dt.datetime.fromtimestamp(ms / 1000, tz=dt.timezone.utc).date()


def to_ms(values, unit_hint: str | None = None):
    """Chuẩn hóa timestamp (s/ms/us/ns) về ms, làm tròn LÊN (không bao giờ đẩy sự kiện về quá khứ
    -> tránh look-ahead khi gán vào khung giây)."""
    import numpy as np
    v = np.asarray(values, dtype="int64")
    if v.size == 0:
        return v
    mx = int(v.max())
    if unit_hint == "us" or (unit_hint is None and 1e14 < mx < 1e17):
        return -(-v // 1000)
    if unit_hint == "ns" or (unit_hint is None and mx >= 1e17):
        return -(-v // 1_000_000)
    if unit_hint == "s" or (unit_hint is None and mx < 1e11):
        return v * 1000
    return v


def first_days_of_months(n: int, today: dt.date | None = None) -> list[dt.date]:
    """Ngày mùng 1 của n tháng gần nhất đã kết thúc (ngày mùng 1 phải < hôm nay)."""
    today = today or utc_today()
    out: list[dt.date] = []
    y, m = today.year, today.month
    d = dt.date(y, m, 1)
    if d >= today:  # hôm nay là mùng 1 -> chưa chốt
        m -= 1
    while len(out) < n:
        if m == 0:
            y, m = y - 1, 12
        out.append(dt.date(y, m, 1))
        m -= 1
    return sorted(out)


def daterange(a: dt.date, b: dt.date) -> Iterable[dt.date]:
    d = a
    while d <= b:
        yield d
        d += dt.timedelta(days=1)


# ----------------------------------------------------------------- HTTP
def http_session():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET", "HEAD"), respect_retry_after_header=True)
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=4))
    s.headers["User-Agent"] = "lob-task1-data-pipeline/1.0"
    return s


def server_clock_offset_ms(session=None, rest_bases: list[str] | None = None) -> tuple[float, float, str]:
    """Đo lệch đồng hồ máy so với Binance. Trả (offset_ms, rtt_ms, base_used).
    offset = server_time - local_time  ->  giờ_sàn ≈ giờ_máy + offset."""
    cfg = load_config()
    session = session or http_session()
    bases = rest_bases or cfg["sources"]["rest_bases"]
    last_err = None
    for base in bases:
        try:
            best = None
            for _ in range(5):
                t0 = time.time() * 1000
                r = session.get(f"{base}/api/v3/time", timeout=5)
                t1 = time.time() * 1000
                r.raise_for_status()
                srv = r.json()["serverTime"]
                rtt = t1 - t0
                off = srv - (t0 + t1) / 2
                if best is None or rtt < best[1]:
                    best = (off, rtt)
            return best[0], best[1], base
        except Exception as e:  # thử base kế tiếp
            last_err = e
    raise ConnectionError(f"Không kết nối được REST Binance: {last_err}")


def append_source_log(paths: Paths, **row) -> None:
    """Nhật ký nguồn dữ liệu (data source log theo checklist của thầy)."""
    import csv
    paths.processed.mkdir(parents=True, exist_ok=True)
    row = {"logged_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **row}
    fields = ["logged_at_utc", "dataset", "date", "source", "url", "bytes", "rows", "sha256", "status", "note"]
    new = not paths.source_log.exists()
    with open(paths.source_log, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})
