"""Ghi dữ liệu TỰ THU (WebSocket / REST) ra ổ đĩa theo khung 1 giây, chờ "chốt ngày" để nhập kho.

Mỗi lần flush ghi 2 file nhỏ vào data/raw/BTCUSDT/live_capture/YYYY-MM-DD/:
    book_<source>_<epoch>.parquet    (snapshot cuối mỗi khung)
    trades_<source>_<epoch>.parquet  (trade đã gom mỗi khung)
Tắt máy giữa chừng chỉ mất tối đa 1 lần flush (mặc định 60s). Chạy lại thì ghi tiếp.
"""
from __future__ import annotations

import threading
import time

import pandas as pd

from .common import Paths, get_logger, level_columns, load_config, ms_to_day
from .store import write_parquet
from .transform import aggregate_trades, reduce_book

log = get_logger()


class CaptureWriter:
    def __init__(self, source: str, paths: Paths | None = None, flush_every_s: int | None = None):
        cfg = load_config()
        self.source = source
        self.paths = (paths or Paths()).ensure()
        self.levels = cfg["levels"]
        self.interval = cfg["store"]["sample_interval_ms"]
        self.flush_every_s = flush_every_s or cfg["streamer"]["flush_every_s"]
        self._lock = threading.Lock()
        self._book_rows: dict[int, list] = {}      # bucket -> [book_ts, *levels] (giữ bản cuối)
        self._trades: list[tuple] = []            # (ts_ms, price, qty, is_buyer_maker)
        self._last_flush = time.time()
        self.rows_written = 0

    def add_book(self, ts_ms: int, level_values: list[float]) -> None:
        b = -(-int(ts_ms) // self.interval) * self.interval
        with self._lock:
            self._book_rows[b] = [int(ts_ms)] + list(level_values)
        if time.time() - self._last_flush >= self.flush_every_s:
            self.flush()

    def add_trade(self, ts_ms: int, price: float, qty: float, is_buyer_maker: bool) -> None:
        with self._lock:
            self._trades.append((int(ts_ms), float(price), float(qty), bool(is_buyer_maker)))

    def flush(self) -> None:
        with self._lock:
            book_rows, self._book_rows = self._book_rows, {}
            trades, self._trades = self._trades, []
            self._last_flush = time.time()
        if not book_rows and not trades:
            return
        stamp = int(time.time() * 1000)
        if book_rows:
            df = pd.DataFrame.from_dict(book_rows, orient="index",
                                        columns=["book_ts"] + level_columns(self.levels))
            df.index.name = "bucket"
            df = df.reset_index()
            df = reduce_book(df, self.interval, self.levels)
            for day, part in df.groupby(df["bucket"].map(lambda x: ms_to_day(int(x)))):
                write_parquet(part, self.paths.capture_dir(day) / f"book_{self.source}_{stamp}.parquet")
            self.rows_written += len(df)
        if trades:
            t = pd.DataFrame(trades, columns=["ts_ms", "price", "qty", "is_buyer_maker"])
            agg = aggregate_trades(t, self.interval)
            for day, part in agg.groupby(agg["bucket"].map(lambda x: ms_to_day(int(x)))):
                write_parquet(part, self.paths.capture_dir(day) / f"trades_{self.source}_{stamp}.parquet")


def load_capture_day(day, paths: Paths | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Đọc toàn bộ phần đã thu của 1 ngày -> (book_reduced, trades_partial, danh sách nguồn)."""
    from .transform import combine_trades
    cfg = load_config()
    paths = paths or Paths()
    folder = paths.capture_dir(day)
    if not folder.exists():
        return pd.DataFrame(), pd.DataFrame(), []
    books = [pd.read_parquet(f) for f in sorted(folder.glob("book_*.parquet"))]
    trades = [pd.read_parquet(f) for f in sorted(folder.glob("trades_*.parquet"))]
    sources = sorted({f.name.split("_")[1] for f in folder.glob("*.parquet")})
    book = reduce_book(pd.concat(books, ignore_index=True), cfg["store"]["sample_interval_ms"],
                       cfg["levels"]) if books else pd.DataFrame()
    # nếu chạy song song WS + REST có thể trùng lệnh -> chỉ lấy trade từ 1 nguồn (ưu tiên WS)
    if trades:
        ws = [t for f, t in zip(sorted(folder.glob("trades_*.parquet")), trades) if "_binance-ws_" in f.name]
        trades = ws or trades
    return book, combine_trades(trades), sources
