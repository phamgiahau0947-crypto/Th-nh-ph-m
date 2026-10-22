"""Bộ sinh dữ liệu GIẢ LẬP cho Task 3 (đặc tả 4.3): test giao diện 100% độc lập với Task 1 & Task 2.

- LOB 20 tầng mỗi 100 ms + lệnh khớp, cùng cấu trúc `BinanceLOBStreamer.get_latest_data()` của Task 1.
- Xác suất giả (P(up/flat/down), P(stress), StressRatio...) cùng khóa với `ModelInference.predict_one()` của Task 2.
- Có "chế độ thị trường": bình thường <-> căng thẳng thanh khoản (spread giãn, sổ lệnh mỏng, biến động mạnh),
  P(stress) giả tăng dần ~20 s TRƯỚC khi căng thẳng xảy ra để thấy rõ đồng hồ chuyển màu.

    from src.task3_dashboard.mock_streamer import MockStreamer
    s = MockStreamer("BTCUSDT").start()
    s.get_latest_data()   # dict như Task 1
    s.get_history()       # DataFrame 1 giây: đặc trưng + dự báo giả (cột HIST_COLS + KL 20 tầng)
    s.latest_prediction() # dict như Task 2
"""
from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np
import pandas as pd

from .lob_utils import ASK_P, ASK_Q, BID_P, BID_Q, HIST_COLS, LEVELS, impact_bps, load_config

TICK_MS = 100


class MockStreamer:
    def __init__(self, symbol: str = "BTCUSDT", buffer_minutes: int = 15, seed: int | None = None,
                 prefill_minutes: float = 10.0):
        cfg = load_config()
        self.symbol = symbol.upper()
        self.buffer_minutes = buffer_minutes
        self.prefill_minutes = prefill_minutes
        self.rng = np.random.default_rng(seed)
        self.notional = float(cfg["notional_usdt"].get(self.symbol, 10000))
        self.warn = float(cfg["stress_warning_threshold"])
        self.caution = float(cfg["stress_caution_threshold"])
        self.horizon_s = int(cfg["horizon_s"])
        btc = self.symbol.startswith("BTC")
        self.tick = 0.1 if btc else 0.01
        self.base_q = 0.15 if btc else 3.0         # tầng 1 ≈ 16k / 12k USDT -> lệnh 25k / 10k ăn vài tầng
        self.mid = float(cfg["mock_start_price"].get(self.symbol, 100.0))

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._buffer: deque = deque(maxlen=buffer_minutes * 60 * 10)
        self._hist: deque = deque(maxlen=buffer_minutes * 60)
        self._latest: dict | None = None
        self._trades: list[dict] = []

        # trạng thái ẩn
        self.latent = 0.0                 # 0 = bình thường, 1 = căng thẳng
        self.obi_state = 0.0
        self.stress_until = -1.0
        self.next_stress_at = None
        self.q_bid = self.base_q * (1 + 0.15 * np.arange(LEVELS))
        self.q_ask = self.q_bid.copy()
        self.gap_b = np.ones(LEVELS - 1)
        self.gap_a = np.ones(LEVELS - 1)
        self.p_stress_s = 0.03
        self._sec_trades = {"buy": 0.0, "sell": 0.0, "n": 0}
        self._prev_q1 = None
        self._ofi1: deque = deque(maxlen=10)
        self._ofim: deque = deque(maxlen=10)
        self._depth_ma = None

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> "MockStreamer":
        if self._thread and self._thread.is_alive():
            return self
        now = time.time() * 1000
        t = now - self.prefill_minutes * 60_000
        self.next_stress_at = t / 1000 + self.rng.uniform(60, 240)
        while t < now:                     # dựng sẵn lịch sử để mở lên là có biểu đồ
            self._step(t)
            t += TICK_MS
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name=f"mock-{self.symbol}", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _loop(self):
        nxt = time.time() * 1000
        while not self._stop.is_set():
            nxt += TICK_MS
            self._step(nxt)
            delay = (nxt - time.time() * 1000) / 1000
            if delay > 0:
                time.sleep(delay)
            elif delay < -2:               # máy bị treo/ngủ -> bắt kịp đồng hồ thật
                nxt = time.time() * 1000

    # ------------------------------------------------------------------ public API (giống Task 1)
    def get_latest_data(self, include_buffer: bool = True) -> dict | None:
        with self._lock:
            if self._latest is None:
                return None
            out = dict(self._latest)
            rows = list(self._buffer) if include_buffer else None
        if include_buffer:
            out["buffer_df"] = pd.DataFrame(rows)
        return out

    def get_history(self) -> pd.DataFrame:
        with self._lock:
            rows = list(self._hist)
        if not rows:
            return pd.DataFrame(columns=HIST_COLS + BID_Q + ASK_Q)
        return pd.DataFrame(rows).set_index("timestamp")

    def latest_prediction(self) -> dict | None:
        """Dự báo giả cho giây mới nhất – cùng khóa với ModelInference.predict_one (Task 2)."""
        with self._lock:
            if not self._hist:
                return None
            r = dict(self._hist[-1])
        ps = r["p_stress"]
        return {
            "p_up": r["p_up"], "p_flat": r["p_flat"], "p_down": r["p_down"],
            "predicted_direction": int(r["pred_dir"]), "p_stress": ps,
            "is_stress_warning": bool(ps > self.warn), "stress_ratio": r["stress_ratio"],
            "expected_short_term_return_bps": r["exp_ret_bps"],
            "stress_level": "red" if ps > self.warn else ("yellow" if ps > self.caution else "green"),
            "is_stress_ratio_warning": bool(r["stress_ratio"] > 1.5), "horizon_s": self.horizon_s,
            "delta_bps": 0.0, "model_direction": "Mock", "model_stress": "Mock", "symbol": self.symbol,
            "latency_ms": 0.0,
            "spr_ticks": r["spr_ticks"], "p_spread_widen": r["p_spread_widen"],
            "eff_spread_fwd_bps": r["eff_spread_fwd_bps"], "spread_base_rate": 0.005,
            "model_spread_widen": "Mock", "model_spread_eff": "Mock",
        }

    # ------------------------------------------------------------------ mô phỏng
    def _step(self, ts_ms: float):
        rng = self.rng
        t = ts_ms / 1000
        # chế độ thị trường: P(stress) giả "báo trước" ~20 s
        if t >= self.next_stress_at - 20 and self.stress_until < self.next_stress_at:
            target = 1.0
            if t >= self.next_stress_at:
                self.stress_until = t + rng.uniform(30, 90)
        elif t < self.stress_until:
            target = 1.0
        else:
            target = 0.0
            if self.stress_until > 0 and t >= self.stress_until and self.next_stress_at < t:
                self.next_stress_at = t + rng.uniform(150, 420)
        self.latent += (target - self.latent) * 0.012
        L = self.latent

        self.obi_state = float(np.clip(0.985 * self.obi_state + 0.06 * rng.standard_normal(), -1.5, 1.5))
        sig = 0.12 * (1 + 3 * L)                                        # bps / 100 ms
        self.mid *= np.exp((0.12 * np.tanh(self.obi_state) * sig + sig * rng.standard_normal()) / 1e4)

        # sổ lệnh 20 tầng: KL thay đổi mượt, mỏng đi khi căng thẳng, lệch theo OBI ẩn
        k = np.arange(LEVELS)
        shape = self.base_q * (1 + 0.15 * k) * (1 - 0.65 * L)
        skew = np.exp(0.7 * np.tanh(self.obi_state))
        tb = shape * skew * rng.lognormal(0, 0.35, LEVELS)
        ta = shape / skew * rng.lognormal(0, 0.35, LEVELS)
        self.q_bid += (tb - self.q_bid) * 0.08
        self.q_ask += (ta - self.q_ask) * 0.08
        if rng.random() < 0.05:
            self.gap_b = 1 + rng.geometric(0.6 - 0.3 * L, LEVELS - 1) - 1
            self.gap_a = 1 + rng.geometric(0.6 - 0.3 * L, LEVELS - 1) - 1
        half = max(1, 1 + rng.poisson(3 * L)) * self.tick / 2
        bid1 = np.floor((self.mid - half) / self.tick) * self.tick
        ask1 = max(bid1 + self.tick, np.ceil((self.mid + half) / self.tick) * self.tick)
        bp = bid1 - np.concatenate([[0], np.cumsum(self.gap_b)]) * self.tick
        ap = ask1 + np.concatenate([[0], np.cumsum(self.gap_a)]) * self.tick
        mid = (bid1 + ask1) / 2

        # lệnh khớp
        n = rng.poisson(2.5 * (1 + 2 * L))
        p_buy = 0.5 + 0.3 * np.tanh(self.obi_state)
        trades = []
        for _ in range(n):
            buy = rng.random() < p_buy
            q = float(rng.exponential(self.base_q * 0.08))
            trades.append({"timestamp": ts_ms, "price": float(ask1 if buy else bid1), "qty": q,
                           "is_buyer_maker": not buy})
            self._sec_trades["buy" if buy else "sell"] += q
            self._sec_trades["n"] += 1

        row = {"timestamp": int(ts_ms), "datetime": pd.Timestamp(int(ts_ms), unit="ms", tz="UTC"),
               "mid_price": mid, "spread": ask1 - bid1}
        row.update(zip(BID_P, bp)); row.update(zip(BID_Q, self.q_bid.copy()))
        row.update(zip(ASK_P, ap)); row.update(zip(ASK_Q, self.q_ask.copy()))
        row.update({"trade_price": trades[-1]["price"] if trades else np.nan,
                    "trade_qty": sum(x["qty"] for x in trades), "trade_count": len(trades)})
        bids = np.column_stack([bp, self.q_bid])
        asks = np.column_stack([ap, self.q_ask])

        new_sec = (int(ts_ms) // 1000) != (int(ts_ms - TICK_MS) // 1000)
        hist_row = self._second_row(ts_ms, mid, bid1, ask1, bids, asks) if new_sec else None
        with self._lock:
            self._buffer.append(row)
            self._latest = {"timestamp": ts_ms / 1000, "symbol": self.symbol, "mid_price": mid,
                            "spread": ask1 - bid1, "bids": bids.tolist(), "asks": asks.tolist(),
                            "recent_trades": trades}
            if hist_row is not None:
                self._hist.append(hist_row)

    def _second_row(self, ts_ms, mid, bid1, ask1, bids, asks) -> dict:
        rng = self.rng
        qb, qa = self.q_bid, self.q_ask
        obi1 = (qb[0] - qa[0]) / (qb[0] + qa[0])
        obi5 = (qb[:5].sum() - qa[:5].sum()) / (qb[:5].sum() + qa[:5].sum())
        depth = (qb[:10] + qa[:10]) / 2
        self._depth_ma = depth if self._depth_ma is None else 0.98 * self._depth_ma + 0.02 * depth
        if self._prev_q1 is not None:
            e = (qb[:10] - self._prev_q1[0]) - (qa[:10] - self._prev_q1[1])
            self._ofi1.append(e[0] / self._depth_ma[0])
            self._ofim.append(float((e / self._depth_ma).sum()))
        self._prev_q1 = (qb[:10].copy(), qa[:10].copy())
        cb, _ = impact_bps(asks, self.notional, mid)
        cs, _ = impact_bps(bids, self.notional, mid)

        # dự báo GIẢ: hướng theo OBI ẩn, stress theo chế độ thị trường (có báo trước)
        z = 1.6 * np.tanh(self.obi_state) + 0.35 * rng.standard_normal()
        logits = np.array([-z, 0.4 - 1.2 * self.latent + 0.2 * rng.standard_normal(), z])
        p = np.exp(logits - logits.max())
        p /= p.sum()
        raw = 0.03 + 0.62 * self.latent ** 1.4 + 0.015 * rng.standard_normal()
        self.p_stress_s += (float(np.clip(raw, 0.005, 0.98)) - self.p_stress_s) * 0.35
        self._sec_trades = {"buy": 0.0, "sell": 0.0, "n": 0}
        row = {"timestamp": int(ts_ms) // 1000 * 1000, "mid": mid, "spread_bps": (ask1 - bid1) / mid * 1e4,
               "obi_1": obi1, "obi_5": obi5, "ofi1_10s": float(sum(self._ofi1)),
               "mlofi_sum_10s": float(sum(self._ofim)), "impact_rt_bps": cb + cs,
               "p_down": p[0], "p_flat": p[1], "p_up": p[2], "p_stress": self.p_stress_s,
               "stress_ratio": 0.8 + 1.4 * self.latent + 0.05 * rng.standard_normal(),
               "exp_ret_bps": 2.0 * (p[2] - p[0]), "pred_dir": int([-1, 0, 1][int(np.argmax(p))]),
               # dự báo spread GIẢ: tăng theo chế độ căng thẳng giả
               "spr_ticks": float(round((ask1 - bid1) / self.tick)),
               "p_spread_widen": float(np.clip(0.004 + 0.45 * self.latent ** 1.5 + 0.004 * rng.standard_normal(),
                                               0.001, 0.95)),
               "eff_spread_fwd_bps": float((cb + cs) * (1 + 0.8 * self.latent))}
        row.update(zip(BID_Q, qb.copy()))
        row.update(zip(ASK_Q, qa.copy()))
        return row


if __name__ == "__main__":
    s = MockStreamer("BTCUSDT", prefill_minutes=1).start()
    time.sleep(2)
    d = s.get_latest_data()
    print("mid", round(d["mid_price"], 2), "| buffer", d["buffer_df"].shape, "| hist", s.get_history().shape)
    print(s.latest_prediction())
    s.stop()
