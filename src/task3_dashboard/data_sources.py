"""3 nguồn dữ liệu của Dashboard, cùng trả về 1 'snapshot' chuẩn để giao diện chỉ việc vẽ.

    snapshot = {
      'source': 'mock' | 'live' | 'replay', 'symbol': 'BTCUSDT', 'ts_ms': int,
      'mid': float, 'bids': ndarray (20, 2) [giá, KL], 'asks': ndarray (20, 2),
      'hist': DataFrame lưới 1 giây (index = timestamp ms): HIST_COLS + KL 20 tầng (bids_q*, asks_q*),
      'pred': dict giống ModelInference.predict_one (Task 2),
      'status': {'ok': bool, 'text': str, 'details': [str, ...], 'warmup': float 0..1 | None},
      'in_sample': bool | None, 'horizon_s': int, 'delta_bps': float,
    }

- MockSource  : mock_streamer.py (không cần Task 1/Task 2).
- LiveSource  : BinanceLOBStreamer (Task 1) -> calculate đặc trưng + ModelInference (Task 2), tính ở luồng nền mỗi 1 s.
- ReplaySource: phát lại 1 ngày trong kho Task 1 với dự báo của mô hình Task 2.
"""
from __future__ import annotations

import datetime as dt
import threading
import time
import traceback

import numpy as np
import pandas as pd

from .lob_utils import ASK_P, ASK_Q, BID_P, BID_Q, HIST_COLS, LEVELS, ROOT, SPREAD_PRED_KEYS
from .mock_streamer import MockStreamer

PRED_RENAME = {"expected_short_term_return_bps": "exp_ret_bps", "predicted_direction": "pred_dir"}
ZOO_CACHE = ROOT / "data" / "processed" / "task3_zoo"       # dự báo cả ngày của bộ mô hình (Replay), tạo khi cần


def _pred_from_row(r: pd.Series, warn: float, caution: float, ratio_warn: float = 1.5, **extra) -> dict:
    ps = float(r["p_stress"])
    ratio = float(r["stress_ratio"]) if pd.notna(r["stress_ratio"]) else np.nan
    out = {"p_up": float(r["p_up"]), "p_flat": float(r["p_flat"]), "p_down": float(r["p_down"]),
           "predicted_direction": int(r["pred_dir"]), "p_stress": ps, "is_stress_warning": ps > warn,
           "stress_ratio": ratio, "expected_short_term_return_bps": float(r["exp_ret_bps"]),
           "stress_level": "red" if ps > warn else ("yellow" if ps > caution else "green"),
           "is_stress_ratio_warning": bool(np.isfinite(ratio) and ratio > ratio_warn)}
    out.update({k: float(r[k]) for k in SPREAD_PRED_KEYS + ["spr_ticks"] if k in r.index})
    out.update(extra)
    return out


def _hist_frame(feat: pd.DataFrame, pred: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Gom cột lịch sử chuẩn HIST_COLS từ đặc trưng + dự báo Task 2 (cột thiếu -> NaN)."""
    from src.task2_models.common import load_config as load_cfg2
    from src.task2_models.spread_forecast import tick_for
    hist = feat[[c for c in HIST_COLS if c in feat.columns]].join(
        pred[[c for c in HIST_COLS if c in pred.columns and c not in feat.columns]])
    hist["spr_ticks"] = ((feat["ask1"] - feat["bid1"]) / tick_for(symbol, load_cfg2())).round()
    return hist.reindex(columns=HIST_COLS)


def _spread_info(mi, symbol: str) -> dict:
    return mi.spread.info(symbol) if mi.spread is not None and mi.spread.has(symbol) else {}


def _zoo_meta(zoo) -> dict:
    return {"names": zoo.names, "dashboard_model": zoo.dashboard_model, "delta_bps": zoo.delta}


# =====================================================================  MOCK
class MockSource:
    kind = "mock"

    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        self.streamer = MockStreamer(self.symbol).start()

    def snapshot(self, window_s: int) -> dict | None:
        data = self.streamer.get_latest_data(include_buffer=False)
        hist = self.streamer.get_history()
        pred = self.streamer.latest_prediction()
        if data is None or hist.empty or pred is None:
            return None
        hist = hist.iloc[-window_s:]
        return {"source": "mock", "symbol": self.symbol, "ts_ms": int(data["timestamp"] * 1000),
                "mid": data["mid_price"], "bids": np.asarray(data["bids"]), "asks": np.asarray(data["asks"]),
                "hist": hist, "pred": pred, "in_sample": None, "horizon_s": pred["horizon_s"], "delta_bps": 0.0,
                "status": {"ok": True, "text": "Dữ liệu giả lập", "warmup": None,
                           "details": ["Sổ lệnh và xác suất đều là số GIẢ để thử giao diện",
                                       "Cứ vài phút có 1 đợt căng thẳng thanh khoản giả"]}}

    def close(self):
        self.streamer.stop()


# =====================================================================  LIVE
class LiveSource:
    kind = "live"
    WARMUP_S = 600          # cửa sổ chuẩn hóa của đặc trưng (Task 2: features.norm_window_s)

    def __init__(self, symbol: str):
        from src.task1_data.websocket_streamer import BinanceLOBStreamer
        from src.task2_models.predictor import ModelInference
        self.symbol = symbol.upper()
        self.mi = ModelInference("models/", self.symbol)
        self.params = self.mi.structural_params(self.symbol)
        self.warn, self.caution, self.ratio_warn = self.mi.warn, self.mi.caution, self.mi.ratio_warn
        d = self.mi.direction[self.symbol]
        self.horizon_s, self.delta_bps = int(d["horizon_s"]), float(d["delta_bps"])
        self.spread_info = _spread_info(self.mi, self.symbol)
        self.streamer = BinanceLOBStreamer(symbol=self.symbol).start()
        self._lock = threading.Lock()
        self._state: dict = {"hist": pd.DataFrame(), "pred": None, "feat_ms": None, "error": None, "rows": 0}
        # bảng xếp hạng: mỗi giây mới, mọi mô hình dự báo 1 lần rồi lưu lại (giữ ZOO_KEEP_S giây)
        self._zoo = None                       # None = chưa nạp, False = không có / lỗi
        self._zoo_store = pd.DataFrame()
        self._zoo_info: dict = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name=f"live-model-{self.symbol}", daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            t0 = time.time()
            try:
                self._compute()
            except Exception as e:                                  # không để luồng nền chết
                with self._lock:
                    self._state["error"] = f"{type(e).__name__}: {e}"
                traceback.print_exc()
            self._stop.wait(max(0.05, 1.0 - (time.time() - t0)))

    def _compute(self):
        from src.task2_models import microstructure_math as mm
        data = self.streamer.get_latest_data()
        if data is None or data.get("buffer_df") is None or len(data["buffer_df"]) < 20:
            return
        t0 = time.perf_counter()
        df1 = mm.buffer_to_1s(data["buffer_df"], LEVELS)
        if len(df1) < 3 or df1[["bids_p1", "asks_p1"]].iloc[-1].isna().any():
            return
        feat = mm.add_structural_features(mm.build_base_features(df1, None, self.symbol), self.params)
        feat_ms = (time.perf_counter() - t0) * 1000
        pred_df = self.mi.predict_frame(feat, self.symbol).rename(columns=PRED_RENAME)
        last = {**feat.iloc[-1].to_dict(), "symbol": self.symbol}
        pred = self.mi.predict_one(last)                            # đúng Interface Contract của Task 2
        hist = _hist_frame(feat, pred_df, self.symbol)
        pred.update({k: float(hist[k].iloc[-1]) for k in SPREAD_PRED_KEYS + ["spr_ticks"]}, **self.spread_info)
        hist = hist.join(df1[BID_Q + ASK_Q])
        with self._lock:
            self._state.update(hist=hist, pred=pred, feat_ms=feat_ms, error=None, rows=len(df1))
        self._update_zoo(feat, df1)

    ZOO_KEEP_S = 7200

    def _update_zoo(self, feat: pd.DataFrame, df1: pd.DataFrame):
        """Dự báo của mọi mô hình cho các giây MỚI (mỗi giây chỉ dự báo 1 lần, đúng lúc nó là hiện tại)."""
        if self._zoo is False:
            return
        if self._zoo is None:
            try:
                from src.task2_models.model_zoo import ModelZoo, zoo_available
                if not zoo_available():
                    self._zoo, self._zoo_info = False, {"error": "Chưa có models/model_zoo.pkl"}
                    return
                self._zoo = ModelZoo(self.symbol)
                self._zoo_info = _zoo_meta(self._zoo)
            except Exception as e:
                self._zoo, self._zoo_info = False, {"error": f"{type(e).__name__}: {e}"}
                return
        with self._lock:
            last = int(self._zoo_store.index.max()) if len(self._zoo_store) else -1
        new = feat[(feat.index > last) & feat["mid"].notna()]
        if new.empty:
            return
        z = self._zoo.predict(new, df1).astype("float32")
        z["mid"] = new["mid"].to_numpy(float)
        with self._lock:
            self._zoo_store = pd.concat([self._zoo_store, z]).iloc[-self.ZOO_KEEP_S:]

    def zoo_frame(self) -> tuple[pd.DataFrame, dict]:
        """(dự báo hướng của từng mô hình theo giây + cột mid, thông tin bộ mô hình)."""
        with self._lock:
            return self._zoo_store.copy(), dict(self._zoo_info)

    def snapshot(self, window_s: int) -> dict | None:
        data = self.streamer.get_latest_data(include_buffer=False)
        st = self.streamer.get_stats()
        with self._lock:
            s = dict(self._state)
        connected = st.get("last_msg_at") is not None and time.time() - st["last_msg_at"] < 5
        rows = s["rows"]
        details = [f"Kết nối: {st.get('connected_url') or 'đang kết nối…'}",
                   f"Ping WebSocket p50: {st.get('ws_ping_rtt_ms_p50')} ms · độ trễ sự kiện p50: "
                   f"{st.get('event_latency_ms_p50')} ms",
                   f"Đã nhận {st.get('depth_msgs', 0):,} bản tin sổ lệnh · {st.get('trade_msgs', 0):,} lệnh khớp"]
        if s["feat_ms"] is not None and s["pred"] is not None:
            details.append(f"Tính đặc trưng {s['feat_ms']:.0f} ms + dự báo {s['pred']['latency_ms']:.1f} ms mỗi giây")
        if s["error"]:
            details.append(f"Lỗi gần nhất: {s['error']}")
        status = {"ok": connected and s["error"] is None,
                  "text": "Đang nhận dữ liệu Binance" if connected else "Đang kết nối Binance…",
                  "warmup": min(1.0, rows / self.WARMUP_S), "details": details}
        if data is None or s["pred"] is None or s["hist"].empty:
            return {"source": "live", "symbol": self.symbol, "status": status, "empty": True}
        return {"source": "live", "symbol": self.symbol, "ts_ms": int(data["timestamp"] * 1000),
                "mid": data["mid_price"], "bids": np.asarray(data["bids"], float),
                "asks": np.asarray(data["asks"], float), "hist": s["hist"].iloc[-window_s:], "pred": s["pred"],
                "in_sample": False, "horizon_s": self.horizon_s, "delta_bps": self.delta_bps, "status": status}

    def close(self):
        self._stop.set()
        self.streamer.stop()


# =====================================================================  REPLAY
def replay_days(symbol: str) -> list[tuple[dt.date, str]]:
    """Các ngày phát lại được: 12 ngày Tardis (seed) + các ngày tự thu (rolling)."""
    from src.task2_models.common import lob_days
    return [(d, p.stem.split("_")[-1]) for d, p in lob_days(symbol.upper(), ("seed", "rolling"))]


class ReplaySource:
    kind = "replay"

    def __init__(self, symbol: str, day: dt.date):
        from src.task2_models.common import lob_days, load_config as load_cfg2
        from src.task2_models.dataset import cache_path, with_structural
        from src.task2_models.microstructure_math import build_base_features, regrid_1s
        from src.task2_models.predictor import ModelInference
        self.symbol, self.day = symbol.upper(), day
        mi = ModelInference("models/", self.symbol)
        self.warn, self.caution, self.ratio_warn = mi.warn, mi.caution, mi.ratio_warn
        d = mi.direction[self.symbol]
        self.horizon_s, self.delta_bps = int(d["horizon_s"]), float(d["delta_bps"])
        self.in_sample = str(day) in d["train_days"]
        self.model_names = (d["model_name"], mi.stress[self.symbol]["model_name"])

        self.spread_info = _spread_info(mi, self.symbol)
        self.path = path = dict(lob_days(self.symbol, ("seed", "rolling")))[day]
        raw = regrid_1s(pd.read_parquet(path))
        cp = cache_path(self.symbol, day)
        feat = pd.read_parquet(cp) if cp.exists() else build_base_features(raw, load_cfg2(), self.symbol)
        feat = with_structural(feat, mi.structural_params(self.symbol))
        pred = mi.predict_frame(feat, self.symbol).rename(columns=PRED_RENAME)
        hist = _hist_frame(feat, pred, self.symbol)
        book = raw[BID_P + BID_Q + ASK_P + ASK_Q]
        self.data = hist.astype("float32").assign(mid=hist["mid"].astype(float)).join(book, how="left")
        self.data = self.data[self.data["mid"].notna()]
        self.ts = self.data.index.to_numpy()
        self.t_min, self.t_max = int(self.ts[0]), int(self.ts[-1])

    def snapshot(self, window_s: int, cursor_ms: int, speed: float, playing: bool) -> dict | None:
        i = int(np.searchsorted(self.ts, cursor_ms, side="right")) - 1
        if i < 0:
            return None
        r = self.data.iloc[i]
        hist = self.data.iloc[max(0, i - window_s + 1): i + 1][HIST_COLS + BID_Q + ASK_Q]
        bids = np.column_stack([r[BID_P].to_numpy(float), r[BID_Q].to_numpy(float)])
        asks = np.column_stack([r[ASK_P].to_numpy(float), r[ASK_Q].to_numpy(float)])
        pred = _pred_from_row(r, self.warn, self.caution, self.ratio_warn, horizon_s=self.horizon_s,
                              delta_bps=self.delta_bps, model_direction=self.model_names[0],
                              model_stress=self.model_names[1], symbol=self.symbol, latency_ms=0.0,
                              **self.spread_info)
        t = pd.Timestamp(int(self.ts[i]), unit="ms", tz="UTC")
        sample = "TRONG mẫu train (chỉ minh họa)" if self.in_sample else "NGOÀI mẫu (mô hình chưa từng thấy)"
        return {"source": "replay", "symbol": self.symbol, "ts_ms": int(self.ts[i]), "mid": float(r["mid"]),
                "bids": bids, "asks": asks, "hist": hist, "pred": pred, "in_sample": self.in_sample,
                "horizon_s": self.horizon_s, "delta_bps": self.delta_bps,
                "status": {"ok": True, "warmup": None,
                           "text": f"Phát lại {self.day} · {'▶' if playing else '⏸'} ×{speed:g}",
                           "details": [f"Thời điểm: {t:%Y-%m-%d %H:%M:%S} UTC",
                                       f"Ngày này nằm {sample}",
                                       "Dự báo tính bằng đúng mô hình Task 2 dùng cho live"]}}

    def zoo_frame(self) -> tuple[pd.DataFrame, dict]:
        """Dự báo hướng của mọi mô hình cho CẢ NGÀY (tính 1 lần, lưu đĩa data/processed/task3_zoo/).
        Bảng xếp hạng chỉ chấm các giây đã qua đủ h giây tính đến con trỏ phát lại -> không nhìn trước."""
        if getattr(self, "_zoo_df", None) is not None:
            return self._zoo_df, self._zoo_info
        from src.task2_models.common import MODEL_DIR, load_config as load_cfg2
        from src.task2_models.dataset import cache_path, with_structural
        from src.task2_models.microstructure_math import build_base_features, regrid_1s
        from src.task2_models.model_zoo import ZOO_FILE, ModelZoo, zoo_available
        from src.task2_models.predictor import ModelInference
        if not zoo_available():
            self._zoo_df, self._zoo_info = pd.DataFrame(), {"error": "Chưa có models/model_zoo.pkl"}
            return self._zoo_df, self._zoo_info
        zoo = ModelZoo(self.symbol)
        self._zoo_info = _zoo_meta(zoo)
        f = ZOO_CACHE / f"{self.symbol}_{self.day}.parquet"
        newest = max((MODEL_DIR / ZOO_FILE).stat().st_mtime, self.path.stat().st_mtime)
        if f.exists() and f.stat().st_mtime > newest:
            self._zoo_df = pd.read_parquet(f)
        else:
            raw = regrid_1s(pd.read_parquet(self.path))
            cp = cache_path(self.symbol, self.day)
            feat = pd.read_parquet(cp) if cp.exists() else build_base_features(raw, load_cfg2(), self.symbol)
            feat = with_structural(feat, ModelInference("models/", self.symbol).structural_params(self.symbol))
            feat = feat[feat["mid"].notna()]
            z = zoo.predict(feat, raw).astype("float32")
            z["mid"] = feat["mid"].to_numpy(float)
            ZOO_CACHE.mkdir(parents=True, exist_ok=True)
            z.to_parquet(f, compression="zstd")
            self._zoo_df = z
        return self._zoo_df, self._zoo_info

    def close(self):
        pass
