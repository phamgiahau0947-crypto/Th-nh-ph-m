"""Streamer real-time Binance: sổ lệnh 20 tầng (@depth20@100ms) + lệnh khớp (@trade).

Interface Contract (Task 3 / tích hợp dùng):
    from src.task1_data.websocket_streamer import BinanceLOBStreamer
    s = BinanceLOBStreamer(symbol="BTCUSDT", buffer_minutes=15)
    s.start()
    data = s.get_latest_data()   # None cho tới khi nhận snapshot đầu tiên
    s.stop()

Test độc lập (in số liệu + đo độ trễ):
    python -m src.task1_data.websocket_streamer --seconds 30
Thu dữ liệu vào kho khi đang mở máy (Ctrl+C để dừng, dữ liệu đã ghi vẫn giữ):
    python -m src.task1_data.websocket_streamer --record --seconds 0
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from collections import deque

import numpy as np
import pandas as pd

from .core.common import get_logger, http_session, level_columns, load_config, server_clock_offset_ms
from .core.transform import book_lists_to_row

log = get_logger()


class BinanceLOBStreamer:
    def __init__(self, symbol: str = "BTCUSDT", buffer_minutes: int = 15, levels: int | None = None,
                 record: bool = False, ws_bases: list[str] | None = None, sync_clock: bool = True):
        cfg = load_config()
        self.symbol = symbol.upper()
        self.levels = int(levels or cfg["levels"])
        self.buffer_minutes = buffer_minutes
        self.ws_bases = ws_bases or cfg["sources"]["ws_bases"]
        speed = cfg["streamer"]["depth_speed"]
        s = self.symbol.lower()
        self.streams = f"{s}@depth{self.levels if self.levels in (5, 10, 20) else 20}@{speed}/{s}@trade"
        self.sync_clock = sync_clock
        self.clock_offset_ms = 0.0      # giờ_sàn ≈ giờ_máy + offset

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._ping_thread: threading.Thread | None = None
        self._ws = None
        maxlen = int(buffer_minutes * 60 * 10 * 1.2) + 100   # 100ms/snapshot, dư 20%
        self._buffer: deque = deque(maxlen=maxlen)
        self._latest: dict | None = None
        self._pending_trades: list[dict] = []
        self._buffer_df_cache: tuple[int, pd.DataFrame] | None = None
        self._version = 0

        self._recorder = None
        if record:
            from .core.capture import CaptureWriter
            from .core.common import Paths
            self._recorder = CaptureWriter(source="binance-ws", paths=Paths(symbol=self.symbol))

        # thống kê
        self.stats = {"depth_msgs": 0, "trade_msgs": 0, "reconnects": 0, "connected_url": None,
                      "started_at": None, "last_msg_at": None}
        self._event_latency = deque(maxlen=5000)   # ms: lúc nhận - event time của sàn (trade)
        self._ping_rtt = deque(maxlen=500)         # ms: ping/pong WebSocket khứ hồi
        self._depth_intervals = deque(maxlen=2000)

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> "BinanceLOBStreamer":
        if self._thread and self._thread.is_alive():
            return self
        if self.sync_clock:
            try:
                off, rtt, base = server_clock_offset_ms()
                self.clock_offset_ms = off
                log.info(f"Đồng bộ giờ với Binance ({base}): lệch {off:+.1f} ms, REST RTT {rtt:.1f} ms")
            except Exception as e:
                log.warning(f"Không đồng bộ được giờ sàn ({e}); dùng giờ máy")
        self._stop.clear()
        self.stats["started_at"] = time.time()
        self._thread = threading.Thread(target=self._run_forever, name="binance-ws", daemon=True)
        self._thread.start()
        self._ping_thread = threading.Thread(target=self._ping_loop, name="binance-ws-ping", daemon=True)
        self._ping_thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        try:
            if self._ws is not None:
                self._ws.close()
        except Exception:
            pass
        if self._thread:
            self._thread.join(timeout=5)
        if self._recorder is not None:
            self._recorder.flush()

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ------------------------------------------------------------------ public API
    def get_latest_data(self, include_buffer: bool = True) -> dict | None:
        """Snapshot mới nhất theo đúng Interface Contract của Task 1."""
        with self._lock:
            if self._latest is None:
                return None
            out = dict(self._latest)
            out["bids"] = [list(x) for x in self._latest["bids"]]
            out["asks"] = [list(x) for x in self._latest["asks"]]
            out["recent_trades"] = list(self._latest["recent_trades"])
        if include_buffer:
            out["buffer_df"] = self.get_buffer_df()
        return out

    def get_buffer_df(self) -> pd.DataFrame:
        """Dữ liệu 15 phút gần nhất (mỗi dòng = 1 snapshot 100ms), cùng tên cột với kho nền."""
        with self._lock:
            v = self._version
            if self._buffer_df_cache and self._buffer_df_cache[0] == v:
                return self._buffer_df_cache[1]
            rows = list(self._buffer)
        cols = (["timestamp", "mid_price", "spread"] + level_columns(self.levels)
                + ["trade_price", "trade_qty", "trade_count", "buy_qty", "sell_qty", "is_buyer_maker"])
        df = pd.DataFrame(rows, columns=cols)
        if not df.empty:
            df.insert(1, "datetime", pd.to_datetime(df["timestamp"], unit="ms", utc=True))
            df["trade_price"] = df["trade_price"].ffill()
            df["is_buyer_maker"] = df["is_buyer_maker"].astype("boolean")
        with self._lock:
            self._buffer_df_cache = (v, df)
        return df

    def get_stats(self) -> dict:
        def q(d, p):
            return round(float(np.percentile(list(d), p)), 2) if d else None
        st = dict(self.stats)
        st.update({
            "clock_offset_ms": round(self.clock_offset_ms, 2),
            "ws_ping_rtt_ms_p50": q(self._ping_rtt, 50), "ws_ping_rtt_ms_p95": q(self._ping_rtt, 95),
            "event_latency_ms_p50": q(self._event_latency, 50),
            "event_latency_ms_p95": q(self._event_latency, 95),
            "depth_interval_ms_p50": q(self._depth_intervals, 50),
            "buffer_rows": len(self._buffer),
        })
        return st

    # ------------------------------------------------------------------ internals
    def _now_ms(self) -> float:
        return time.time() * 1000 + self.clock_offset_ms

    def _run_forever(self):
        import websocket  # websocket-client
        attempt = 0
        while not self._stop.is_set():
            base = self.ws_bases[(attempt // 3) % len(self.ws_bases)]   # 3 lần lỗi -> đổi endpoint
            url = f"{base}/stream?streams={self.streams}"
            self._ws = websocket.WebSocketApp(
                url, on_open=self._on_open, on_message=self._on_message,
                on_error=self._on_error, on_close=self._on_close, on_pong=self._on_pong)
            self.stats["connected_url"] = url
            t0 = time.time()
            try:
                # ping_timeout=None: pong của ping đo RTT (_ping_loop, mỗi 5s) làm websocket-client tưởng
                # "pong đến muộn" -> tự ngắt mỗi ~30s. Kết nối chết đã được _ping_loop phát hiện (15s không có tin).
                self._ws.run_forever(ping_interval=20, ping_timeout=None)
            except Exception as e:
                log.warning(f"WebSocket lỗi: {e}")
            if self._stop.is_set():
                break
            attempt = 0 if time.time() - t0 > 60 else attempt + 1
            self.stats["reconnects"] += 1
            wait = min(30, 2 ** min(attempt, 5))
            log.warning(f"Mất kết nối WebSocket, thử lại sau {wait}s (lần {self.stats['reconnects']})")
            self._stop.wait(wait)

    def _ping_loop(self):
        """Gửi ping có mang thời gian -> đo RTT khứ hồi thật của kết nối WebSocket."""
        import websocket
        while not self._stop.wait(5):
            ws = self._ws
            try:
                if ws is not None and ws.sock and ws.sock.connected:
                    ws.sock.ping(str(time.perf_counter_ns()))
            except Exception:
                pass
            # tự phát hiện stream bị treo (không có tin trong 15s)
            last = self.stats.get("last_msg_at")
            if last and time.time() - last > 15 and ws is not None:
                log.warning("Không nhận dữ liệu 15s -> kết nối lại")
                try:
                    ws.close()
                except Exception:
                    pass
                self.stats["last_msg_at"] = time.time()

    def _on_open(self, ws):
        log.info(f"Đã kết nối {self.stats['connected_url']}")

    def _on_error(self, ws, err):
        if not self._stop.is_set():
            log.warning(f"WebSocket error: {err}")

    def _on_close(self, ws, code, msg):
        if not self._stop.is_set():
            log.info(f"WebSocket đóng ({code} {msg})")

    def _on_pong(self, ws, data):
        try:
            sent = int(data.decode() if isinstance(data, (bytes, bytearray)) else data)
            self._ping_rtt.append((time.perf_counter_ns() - sent) / 1e6)
        except Exception:
            pass   # pong tự động của thư viện (không mang thời gian)

    def _on_message(self, ws, message: str):
        recv_ms = self._now_ms()
        try:
            msg = json.loads(message)
        except Exception:
            return
        data = msg.get("data", msg)
        stream = msg.get("stream", "")
        self.stats["last_msg_at"] = time.time()
        if data.get("e") == "trade" or stream.endswith("@trade"):
            self._handle_trade(data, recv_ms)
        elif "bids" in data and "asks" in data:
            self._handle_depth(data, recv_ms)

    def _handle_trade(self, d: dict, recv_ms: float):
        tr = {"timestamp": int(d["T"]), "price": float(d["p"]), "qty": float(d["q"]),
              "is_buyer_maker": bool(d["m"]), "trade_id": d.get("t")}
        if "E" in d:
            self._event_latency.append(recv_ms - float(d["E"]))
        with self._lock:
            self._pending_trades.append(tr)
            self.stats["trade_msgs"] += 1
        if self._recorder is not None:
            self._recorder.add_trade(tr["timestamp"], tr["price"], tr["qty"], tr["is_buyer_maker"])

    def _handle_depth(self, d: dict, recv_ms: float):
        bids = [[float(p), float(q)] for p, q in d["bids"][: self.levels]]
        asks = [[float(p), float(q)] for p, q in d["asks"][: self.levels]]
        if not bids or not asks:
            return
        ts = int(recv_ms)   # depth20 của spot không có event time -> dùng giờ nhận đã hiệu chỉnh theo giờ sàn
        mid = (bids[0][0] + asks[0][0]) / 2.0
        spread = asks[0][0] - bids[0][0]
        levels_row = book_lists_to_row(bids, asks, self.levels)
        with self._lock:
            trades, self._pending_trades = self._pending_trades, []
            prev = self._latest["timestamp"] if self._latest else None
            if prev:
                self._depth_intervals.append(ts - prev)
            tq = sum(t["qty"] for t in trades)
            buy = sum(t["qty"] for t in trades if not t["is_buyer_maker"])
            last_px = trades[-1]["price"] if trades else np.nan
            last_ibm = trades[-1]["is_buyer_maker"] if trades else None
            self._buffer.append([ts, mid, spread, *levels_row, last_px, tq, len(trades), buy, tq - buy, last_ibm])
            self._latest = {
                "timestamp": ts / 1000.0,            # giây (float) theo đặc tả
                "timestamp_ms": ts,
                "symbol": self.symbol,
                "mid_price": mid, "spread": spread,
                "bids": bids, "asks": asks,
                "recent_trades": trades,            # lệnh khớp kể từ snapshot trước (~100ms)
                "last_update_id": d.get("lastUpdateId"),
            }
            self._version += 1
            self.stats["depth_msgs"] += 1
        if self._recorder is not None:
            self._recorder.add_book(ts, levels_row)


# ---------------------------------------------------------------------- CLI test
def main(argv=None):
    ap = argparse.ArgumentParser(description="Test BinanceLOBStreamer / thu dữ liệu live")
    ap.add_argument("--symbol", default=load_config()["symbol"])
    ap.add_argument("--seconds", type=float, default=30, help="Thời gian chạy (0 = chạy tới khi Ctrl+C)")
    ap.add_argument("--record", action="store_true", help="Ghi dữ liệu (1s/dòng) vào live_capture để nhập kho")
    a = ap.parse_args(argv)

    s = BinanceLOBStreamer(symbol=a.symbol, record=a.record).start()
    t_end = time.time() + a.seconds if a.seconds > 0 else float("inf")
    last_print = 0.0
    try:
        while time.time() < t_end:
            time.sleep(0.1)
            if time.time() - last_print >= 2:
                last_print = time.time()
                d = s.get_latest_data(include_buffer=False)
                if d is None:
                    log.info("Đang chờ snapshot đầu tiên...")
                    continue
                st = s.get_stats()
                b1, a1 = d["bids"][0], d["asks"][0]
                log.info(f"{a.symbol} mid={d['mid_price']:.2f} spread={d['spread']:.2f} "
                         f"bid1={b1[0]:.2f}x{b1[1]:.4f} ask1={a1[0]:.2f}x{a1[1]:.4f} "
                         f"trades/100ms={len(d['recent_trades'])} buffer={st['buffer_rows']} "
                         f"ping_rtt_p50={st['ws_ping_rtt_ms_p50']}ms latency_p50={st['event_latency_ms_p50']}ms")
    except KeyboardInterrupt:
        log.info("Dừng theo yêu cầu (Ctrl+C)")
    finally:
        s.stop()
    st = s.get_stats()
    log.info("===== KẾT QUẢ TEST STREAMER =====")
    for k, v in st.items():
        log.info(f"  {k}: {v}")
    df = s.get_buffer_df()
    log.info(f"  buffer_df: {df.shape[0]} dòng x {df.shape[1]} cột")
    if st["ws_ping_rtt_ms_p50"] is not None:
        ok = "ĐẠT" if st["ws_ping_rtt_ms_p50"] < 50 else "CHƯA ĐẠT (phụ thuộc vị trí mạng tới máy chủ Binance)"
        log.info(f"  Mục tiêu ping < 50ms: {ok}")
    if a.record and s._recorder is not None:
        log.info(f"  Đã ghi {s._recorder.rows_written} dòng 1s vào live_capture")
    return st


if __name__ == "__main__":
    main()
