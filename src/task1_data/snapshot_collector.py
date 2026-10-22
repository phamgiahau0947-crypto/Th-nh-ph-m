"""Thu snapshot sổ lệnh qua REST: GET /api/v3/depth?symbol=BTCUSDT&limit=20 (mặc định mỗi 200ms).

Đây là phương án theo đúng đặc tả gốc. Khuyến nghị dùng WebSocket recorder (dày hơn, 100ms):
    python -m src.task1_data.websocket_streamer --record --seconds 0
REST collector hữu ích khi mạng chặn WebSocket.

Chạy:  python -m src.task1_data.snapshot_collector --seconds 0      (0 = tới khi Ctrl+C)
Dữ liệu ghi vào live_capture (1s/dòng), được nhập kho khi chạy update_store sau khi chốt ngày.
"""
from __future__ import annotations

import argparse
import time

from .core.capture import CaptureWriter
from .core.common import get_logger, http_session, load_config, server_clock_offset_ms
from .core.transform import book_lists_to_row

log = get_logger()


class SnapshotCollector:
    def __init__(self, symbol: str | None = None, poll_interval_ms: int | None = None):
        cfg = load_config()
        self.symbol = (symbol or cfg["symbol"]).upper()
        self.levels = cfg["levels"]
        self.poll_s = (poll_interval_ms or cfg["collector"]["poll_interval_ms"]) / 1000
        self.bases = cfg["sources"]["rest_bases"]
        self.session = http_session()
        from .core.common import Paths
        self.writer = CaptureWriter(source="binance-rest", paths=Paths(symbol=self.symbol))
        self.offset_ms = 0.0
        self.n_ok = self.n_err = 0

    def fetch_once(self, base: str) -> dict:
        t0 = time.time() * 1000
        r = self.session.get(f"{base}/api/v3/depth", params={"symbol": self.symbol, "limit": self.levels},
                             timeout=5)
        t1 = time.time() * 1000
        if r.status_code in (418, 429):   # bị giới hạn tần suất -> nghỉ theo Retry-After
            wait = int(r.headers.get("Retry-After", "60"))
            log.warning(f"Binance giới hạn tần suất ({r.status_code}), nghỉ {wait}s")
            time.sleep(wait)
            raise RuntimeError("rate limited")
        r.raise_for_status()
        d = r.json()
        # REST không trả thời gian -> lấy điểm giữa của request, hiệu chỉnh theo giờ sàn
        d["_ts_ms"] = int((t0 + t1) / 2 + self.offset_ms)
        return d

    def run(self, seconds: float = 0):
        try:
            self.offset_ms, rtt, _ = server_clock_offset_ms(self.session)
            log.info(f"Lệch giờ máy so với Binance: {self.offset_ms:+.1f} ms (RTT {rtt:.0f} ms)")
        except Exception as e:
            log.warning(f"Không đồng bộ được giờ: {e}")
        t_end = time.time() + seconds if seconds > 0 else float("inf")
        bi, next_t, last_log = 0, time.time(), time.time()
        try:
            while time.time() < t_end:
                try:
                    d = self.fetch_once(self.bases[bi % len(self.bases)])
                    self.writer.add_book(d["_ts_ms"], book_lists_to_row(d["bids"], d["asks"], self.levels))
                    self.n_ok += 1
                except Exception as e:
                    self.n_err += 1
                    if self.n_err % 5 == 0:
                        bi += 1
                        log.warning(f"Lỗi REST ({e}); đổi sang {self.bases[bi % len(self.bases)]}")
                if time.time() - last_log > 30:
                    log.info(f"Đã thu {self.n_ok} snapshot, lỗi {self.n_err}")
                    last_log = time.time()
                next_t += self.poll_s
                time.sleep(max(0.0, next_t - time.time()))
                if time.time() - next_t > 5:   # trễ quá nhiều (máy ngủ) -> đặt lại nhịp
                    next_t = time.time()
        except KeyboardInterrupt:
            log.info("Dừng theo yêu cầu (Ctrl+C)")
        finally:
            self.writer.flush()
            log.info(f"Kết thúc: {self.n_ok} snapshot OK, {self.n_err} lỗi")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Thu snapshot LOB qua REST API")
    ap.add_argument("--seconds", type=float, default=0)
    ap.add_argument("--interval-ms", type=int, default=None)
    ap.add_argument("--symbol", default=None)
    a = ap.parse_args(argv)
    SnapshotCollector(symbol=a.symbol, poll_interval_ms=a.interval_ms).run(a.seconds)


if __name__ == "__main__":
    main()
