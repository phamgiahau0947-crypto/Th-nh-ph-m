"""TỰ KIỂM TRA TASK 1 – không cần Internet.

    python -m src.task1_data.tools.selftest

Sinh dữ liệu GIẢ đúng định dạng Tardis + Binance, dựng một máy chủ HTTP cục bộ, rồi chạy toàn bộ
pipeline (seed, trades, live capture, kho rolling, retention, CSV, streamer) trong thư mục tạm.
Kiểm tra: schema, point-in-time (không look-ahead), tổng khối lượng trade, retention, Interface Contract.
"""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import http.server
import io
import json
import os
import shutil
import socketserver
import sys
import tempfile
import threading
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

TMP = Path(tempfile.mkdtemp(prefix="task1_selftest_"))
os.environ["TASK1_DATA_ROOT"] = str(TMP / "data")

# --- nếu máy chưa có pyarrow (chỉ xảy ra trong môi trường thử), giả lập parquet bằng pickle ---
try:
    import pyarrow  # noqa: F401
    PARQUET = "pyarrow"
except ImportError:
    PARQUET = "pickle-fallback"
    _orig_read = pd.read_parquet

    def _to_parquet(self, path, index=False, compression=None, **kw):
        self.to_pickle(path, compression=None)

    def _read_parquet(path, columns=None, **kw):
        df = pd.read_pickle(path, compression=None)
        return df[columns] if columns else df

    pd.DataFrame.to_parquet = _to_parquet
    pd.read_parquet = _read_parquet

from ..core.common import Paths, load_config, level_columns, unified_columns, day_start_ms  # noqa: E402

FAILS: list[str] = []


def check(cond, msg):
    print(("  [OK]   " if cond else "  [FAIL] ") + msg)
    if not cond:
        FAILS.append(msg)


# ------------------------------------------------------------------ synthetic data
rng = np.random.default_rng(7)
LEVELS = 25


def make_book_path(start_ms, n, step_ms):
    """Chuỗi snapshot: mid đi ngẫu nhiên, tick 0.01."""
    ts = start_ms + np.cumsum(rng.integers(step_ms // 2, step_ms * 3 // 2, n))
    mid = 60000 + np.cumsum(rng.normal(0, 0.8, n))
    best_bid = np.round(mid - 0.005 - rng.integers(0, 3, n) * 0.01, 2)
    best_ask = np.round(best_bid + 0.01 + rng.integers(0, 3, n) * 0.01, 2)
    return ts, best_bid, best_ask


def make_tardis_csv_gz(day: dt.date, path: Path, hours=1.0, step_ms=300):
    start = day_start_ms(day)
    n = int(hours * 3600_000 / step_ms)
    ts, bb, ba = make_book_path(start, n, step_ms)
    cols = {"exchange": "binance", "symbol": "BTCUSDT", "timestamp": ts * 1000 + 123,
            "local_timestamp": ts * 1000 + 40_000}
    for i in range(LEVELS):
        cols[f"asks[{i}].price"] = np.round(ba + i * 0.01, 2)
        cols[f"asks[{i}].amount"] = np.round(rng.exponential(0.5, n), 5)
        cols[f"bids[{i}].price"] = np.round(bb - i * 0.01, 2)
        cols[f"bids[{i}].amount"] = np.round(rng.exponential(0.5, n), 5)
    df = pd.DataFrame(cols)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        df.to_csv(f, index=False)
    return df


def make_binance_trades_zip(day: dt.date, folder: Path, n=20000, us=True, symbol="BTCUSDT"):
    start = day_start_ms(day)
    t = np.sort(rng.integers(start, start + 86_400_000, n))
    price = np.round(60000 + rng.normal(0, 30, n), 2)
    qty = np.round(rng.exponential(0.01, n) + 0.00001, 5)
    ibm = rng.random(n) < 0.5
    tcol = t * 1000 + rng.integers(0, 999, n) if us else t
    df = pd.DataFrame({"id": np.arange(n), "price": price, "qty": qty, "quote": price * qty,
                       "time": tcol, "ibm": np.where(ibm, "True", "False"), "best": "True"})
    name = f"{symbol}-trades-{day.isoformat()}"
    zpath = folder / f"{name}.zip"
    buf = io.StringIO(); df.to_csv(buf, header=False, index=False)
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{name}.csv", buf.getvalue())
    digest = hashlib.sha256(zpath.read_bytes()).hexdigest()
    (folder / f"{name}.zip.CHECKSUM").write_text(f"{digest}  {name}.zip\n")
    return pd.DataFrame({"ts_ms": -(-tcol // 1000) if us else tcol, "qty": qty, "ibm": ibm})


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


def serve(root: Path):
    handler = lambda *a, **k: _Quiet(*a, directory=str(root), **k)  # noqa: E731
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


# ------------------------------------------------------------------ main
def main():
    print(f"Thư mục tạm: {TMP}   (parquet engine: {PARQUET})")
    cfg = load_config()
    web = TMP / "web"
    TODAY = dt.date(2026, 10, 19)
    SEED_DAY = dt.date(2026, 10, 1)
    LIVE_DAY = dt.date(2026, 10, 17)
    OLD_DAY = dt.date(2026, 9, 1)  # ngày seed cũ (vẫn trong 6 tháng)

    # --- dựng "Internet giả" ---
    tdir = web / "v1/binance/book_snapshot_25"
    for d in (SEED_DAY, OLD_DAY):
        p = tdir / f"{d.year}/{d.month:02d}/{d.day:02d}"; p.mkdir(parents=True, exist_ok=True)
        make_tardis_csv_gz(d, p / "BTCUSDT.csv.gz", hours=0.5)
    bdir = web / "data/spot/daily/trades/BTCUSDT"; bdir.mkdir(parents=True)
    truth_trades = {}
    for d in (SEED_DAY, OLD_DAY, LIVE_DAY, dt.date(2026, 10, 18), dt.date(2026, 10, 10)):
        truth_trades[d] = make_binance_trades_zip(d, bdir, us=(d != OLD_DAY))   # OLD_DAY: timestamp ms
    # ETHUSDT: có sổ lệnh cho 01/10, KHÔNG có cho 01/09 (thử xử lý thiếu dữ liệu)
    p = tdir / f"{SEED_DAY.year}/{SEED_DAY.month:02d}/{SEED_DAY.day:02d}"
    make_tardis_csv_gz(SEED_DAY, p / "ETHUSDT.csv.gz", hours=0.25)
    edir = web / "data/spot/daily/trades/ETHUSDT"; edir.mkdir(parents=True)
    make_binance_trades_zip(SEED_DAY, edir, n=5000, symbol="ETHUSDT")
    httpd, url = serve(web)
    cfg["sources"]["binance_vision"] = url
    cfg["sources"]["tardis_datasets"] = url
    cfg["sources"]["rest_bases"] = [url]
    cfg["seed"]["months_back"] = 2
    cfg["store"]["retention_days"] = 14
    cfg["symbols"] = ["BTCUSDT", "ETHUSDT"]

    from ..core.capture import CaptureWriter
    from ..core.store import list_days, load_raw_lob_trades, save_day
    from ..update_store import run
    from ..core.transform import conform
    paths = Paths().ensure()

    # --- live capture giả cho LIVE_DAY (như thể đã chạy recorder 40 phút) ---
    print("\n[1] Live capture (giả lập WebSocket recorder)")
    w = CaptureWriter(source="binance-ws", paths=paths, flush_every_s=10**9)
    ts, bb, ba = make_book_path(day_start_ms(LIVE_DAY) + 3600_000, 24000, 100)
    for i in range(len(ts)):
        bids = [[bb[i] - k * 0.01, 0.1 + k] for k in range(20)]
        asks = [[ba[i] + k * 0.01, 0.2 + k] for k in range(20)]
        from ..core.transform import book_lists_to_row
        w.add_book(int(ts[i]), book_lists_to_row(bids, asks, 20))
        if i % 3000 == 0:
            w.flush()
    w.flush()
    check(paths.capture_dir(LIVE_DAY).exists(), "live_capture ghi file theo ngày")

    # --- một ngày rolling rất cũ để thử retention ---
    old_roll = dt.date(2026, 9, 20)
    dummy = conform(pd.DataFrame({"timestamp": [day_start_ms(old_roll)], "book_ts": [day_start_ms(old_roll)],
                                  "book_age_ms": [0], "gap_flag": [True], "symbol": ["BTCUSDT"]}), 20)
    save_day(dummy, old_roll, "rolling", paths)

    print("\n[2] update_store --init (seed + trades + rolling + retention + CSV)")
    res = run(init=True, today=TODAY, symbols=["BTCUSDT"])   # lần đầu chỉ BTC (như máy thật)

    days = {d: k for d, k, _ in list_days(paths)}
    print("\n[3] Kiểm tra kết quả")
    check(days.get(SEED_DAY) == "seed" and days.get(OLD_DAY) == "seed", "Có 2 ngày seed (01/09, 01/10)")
    check(days.get(LIVE_DAY) == "rolling", "Ngày tự thu 17/10 đã nhập kho rolling")
    check(old_roll not in days, "Retention 14 ngày đã xóa rolling 20/09")
    check(paths.trades_1s_file(dt.date(2026, 10, 18)).exists(), "Trades 18/10 (không có LOB) vẫn được lưu riêng")

    df = load_raw_lob_trades(symbol="BTCUSDT")
    check(list(df.columns) == unified_columns(20), "Đúng thứ tự cột Interface Contract")
    check(df["timestamp"].is_monotonic_increasing, "timestamp tăng dần")
    check(bool((df["book_ts"] <= df["timestamp"]).all()), "Point-in-time: snapshot dùng luôn <= timestamp")
    check(bool((df["timestamp"] % 1000 == 0).all()), "Lưới đều 1 giây")
    check(bool((df["spread"] > 0).all()), "Spread > 0, không có sổ lệnh chéo")
    check(bool((df["bids_p1"] > df["bids_p2"]).all() and (df["asks_p1"] < df["asks_p2"]).all()),
          "Bids giảm dần / asks tăng dần")
    check(abs(df["mid_price"] - (df["bids_p1"] + df["asks_p1"]) / 2).max() < 1e-9, "mid_price đúng công thức")
    check(int(df["book_age_ms"].max()) <= cfg["store"]["max_book_staleness_ms"], "Không có snapshot quá cũ")

    # tổng KL trade khớp với dữ liệu gốc trong các giây có sổ lệnh
    seed = df[df["datetime"].dt.date == SEED_DAY]
    tt = truth_trades[SEED_DAY]
    bucket = -(-tt["ts_ms"] // 1000) * 1000
    expect = tt.groupby(bucket)["qty"].sum().reindex(seed["timestamp"]).fillna(0).values
    check(np.allclose(seed["trade_qty"].values, expect, atol=1e-9), "trade_qty = tổng KL gốc trong (t-1s, t]")
    buy_expect = tt[~tt["ibm"]].groupby(bucket[~tt["ibm"]])["qty"].sum().reindex(seed["timestamp"]).fillna(0).values
    check(np.allclose(seed["buy_qty"].values, buy_expect, atol=1e-9), "buy_qty = KL taker mua (is_buyer_maker=False)")
    old = df[df["datetime"].dt.date == OLD_DAY]
    check(len(old) > 0 and old["trade_count"].sum() > 0, "Đọc đúng file trades timestamp ms (định dạng cũ)")

    inv = pd.read_csv(paths.inventory_csv)
    check(len(inv) == len(days) and set(inv["symbol"]) == {"BTCUSDT"}, "inventory.csv đủ các ngày")
    v = pd.read_csv(paths.csv_view)
    check(len(v) > 0 and "bids_p5" in v.columns and "bids_p6" not in v.columns, "CSV xem nhanh 10s / 5 tầng")
    check(paths.source_log.exists(), "source_log.csv có ghi nhật ký")

    # chạy lại lần 2: không tải lại / không đổi gì (idempotent)
    before = {f.name: f.stat().st_mtime for *_, f in list_days(paths)}
    run(today=TODAY)
    after = {f.name: f.stat().st_mtime for *_, f in list_days(paths)}
    check(before == after, "Chạy lại không dựng lại ngày đã có (idempotent)")

    print("\n[3b] Thêm ETHUSDT vào kho đã có (chạy cập nhật thường, không --init)")
    eth = Paths(symbol="ETHUSDT")
    eth_days = {d: k for d, k, _ in list_days(eth)}
    check(eth_days.get(SEED_DAY) == "seed", "ETHUSDT: tự dựng seed 01/10 khi thêm vào config")
    check(OLD_DAY not in eth_days, "ETHUSDT: thiếu dữ liệu 01/09 -> bỏ qua, không lỗi")
    allx = load_raw_lob_trades()
    check(set(allx["symbol"]) == {"BTCUSDT", "ETHUSDT"}, "Đọc kho chung có cả 2 cặp (phân biệt cột symbol)")
    e_only = load_raw_lob_trades(symbol="ETHUSDT")
    check(len(e_only) > 0 and set(e_only["symbol"]) == {"ETHUSDT"}, "load_raw_lob_trades(symbol='ETHUSDT') chỉ ra ETH")
    check(list(e_only.columns) == unified_columns(20), "ETHUSDT cùng schema với BTCUSDT")
    inv = pd.read_csv(paths.inventory_csv)
    check(set(inv["symbol"]) == {"BTCUSDT", "ETHUSDT"}, "inventory.csv thống kê cả 2 cặp")
    check(eth.csv_view.exists(), "Có CSV xem nhanh riêng cho ETHUSDT")
    check(eth.trades_1s_file(SEED_DAY).exists(), "Có trades_1s riêng cho ETHUSDT")

    print("\n[4] Streamer (giả lập tin nhắn WebSocket, không cần mạng)")
    from ..websocket_streamer import BinanceLOBStreamer
    s = BinanceLOBStreamer(sync_clock=False)
    check(s.get_latest_data() is None, "get_latest_data() = None khi chưa có dữ liệu")
    now = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    s._on_message(None, json.dumps({"stream": "btcusdt@trade", "data": {
        "e": "trade", "E": now, "s": "BTCUSDT", "t": 1, "p": "60000.10", "q": "0.5", "T": now, "m": True}}))
    s._on_message(None, json.dumps({"stream": "btcusdt@depth20@100ms", "data": {
        "lastUpdateId": 1, "bids": [[f"{60000 - i*0.01:.2f}", "1.0"] for i in range(20)],
        "asks": [[f"{60000.01 + i*0.01:.2f}", "2.0"] for i in range(20)]}}))
    d = s.get_latest_data()
    keys = {"timestamp", "mid_price", "spread", "bids", "asks", "recent_trades", "buffer_df"}
    check(keys.issubset(d), "Dict trả về đủ khóa theo Interface Contract")
    check(len(d["bids"]) == 20 and len(d["asks"]) == 20, "20 tầng bid/ask")
    check(abs(d["spread"] - 0.01) < 1e-9 and abs(d["mid_price"] - 60000.005) < 1e-9, "mid/spread đúng")
    check(len(d["recent_trades"]) == 1 and d["recent_trades"][0]["is_buyer_maker"] is True, "recent_trades đúng")
    check(isinstance(d["buffer_df"], pd.DataFrame) and len(d["buffer_df"]) == 1
          and "bids_p20" in d["buffer_df"].columns, "buffer_df đúng cấu trúc")

    httpd.shutdown()
    print("\n" + ("TẤT CẢ KIỂM TRA ĐỀU ĐẠT" if not FAILS else f"CÓ {len(FAILS)} KIỂM TRA LỖI: {FAILS}"))
    shutil.rmtree(TMP, ignore_errors=True)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
