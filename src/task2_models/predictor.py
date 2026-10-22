"""Suy luận trực tiếp cho Dashboard (Interface Contract Task 2, đặc tả 3.6).

    from src.task2_models.microstructure_math import calculate_all_features
    from src.task2_models.predictor import ModelInference
    predictor = ModelInference(model_dir="models/")
    feats = calculate_all_features(streamer.get_latest_data())     # dict đặc trưng giây mới nhất
    out = predictor.predict_one(feats)
    # {'p_up', 'p_flat', 'p_down', 'predicted_direction', 'p_stress', 'is_stress_warning',
    #  'stress_ratio', 'expected_short_term_return_bps', ...}

Chế độ Replay (Task 3):  df = load_replay("BTCUSDT", "2026-10-01")  -> đặc trưng + dự báo cho từng giây.
Test độc lập:            python -m src.task2_models.predictor            (dữ liệu test lịch sử)
                         python -m src.task2_models.predictor --live 30  (WebSocket Binance thật, 30s)
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .common import MODEL_DIR, ROOT, get_logger

log = get_logger()


class ModelInference:
    def __init__(self, model_dir: str | Path = "models/", symbol: str | None = None):
        d = Path(model_dir)
        d = d if d.is_absolute() else ROOT / d
        if not (d / "model_direction.pkl").exists():
            d = MODEL_DIR
        self.model_dir = d
        self.direction = pickle.loads((d / "model_direction.pkl").read_bytes())["symbols"]
        self.stress = pickle.loads((d / "model_stress.pkl").read_bytes())["symbols"]
        self.meta = json.loads((d / "model_meta.json").read_text(encoding="utf-8"))
        self.symbol = (symbol or self.meta.get("default_symbol", "BTCUSDT")).upper()
        self.warn = float(self.meta.get("stress_warning_threshold", 0.3))
        self.caution = float(self.meta.get("stress_caution_threshold", 0.15))
        self.ratio_warn = float(self.meta.get("stress_ratio_warning", 1.5))
        # bổ sung 21/10/2026: dự báo spread (models/model_spread.pkl, nếu đã chạy spread_forecast) – chỉ có trong
        # predict_frame (cần lịch sử spread vài chục giây); predict_one giữ nguyên Interface Contract
        from .spread_forecast import SpreadForecaster
        self.spread = SpreadForecaster.load(d)

    @property
    def symbols(self) -> list[str]:
        return sorted(set(self.direction) & set(self.stress))

    def structural_params(self, symbol: str | None = None) -> dict:
        return self.meta["symbols"][(symbol or self.symbol).upper()]["structural"]

    # ------------------------------------------------------------------ 1 dòng (live)
    def predict_one(self, features_dict: dict, symbol: str | None = None) -> dict:
        t0 = time.perf_counter()
        sym = (symbol or features_dict.get("symbol") or self.symbol).upper()
        d, s = self.direction[sym], self.stress[sym]
        x = np.array([[_num(features_dict.get(c)) for c in d["features"]]], dtype=np.float32)
        p = d["model"].predict_proba(x)[0]
        xs = x if s["features"] == d["features"] else np.array(
            [[_num(features_dict.get(c)) for c in s["features"]]], dtype=np.float32)
        ps = float(s["model"].predict_proba(xs)[0, 1])
        cm = d["class_mean_ret_bps"]
        exp_ret = float(p[0] * cm[-1] + p[1] * cm[0] + p[2] * cm[1])
        ratio = _num(features_dict.get("as_stress_ratio"))
        return {
            "p_up": float(p[2]), "p_flat": float(p[1]), "p_down": float(p[0]),
            "predicted_direction": int([-1, 0, 1][int(np.argmax(p))]),
            "p_stress": ps,
            "is_stress_warning": bool(ps > self.warn),
            "stress_ratio": ratio,
            "expected_short_term_return_bps": exp_ret,
            # bổ sung
            "stress_level": "red" if ps > self.warn else ("yellow" if ps > self.caution else "green"),
            "is_stress_ratio_warning": bool(np.isfinite(ratio) and ratio > self.ratio_warn),
            "horizon_s": int(d["horizon_s"]), "delta_bps": float(d["delta_bps"]),
            "model_direction": d["model_name"], "model_stress": s["model_name"], "symbol": sym,
            "latency_ms": (time.perf_counter() - t0) * 1000,
        }

    # ------------------------------------------------------------------ nhiều dòng (replay)
    def predict_frame(self, feat: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
        sym = (symbol or self.symbol).upper()
        d, s = self.direction[sym], self.stress[sym]
        P = d["model"].predict_proba(feat[d["features"]].to_numpy(np.float32))
        ps = s["model"].predict_proba(feat[s["features"]].to_numpy(np.float32))[:, 1]
        cm = d["class_mean_ret_bps"]
        out = pd.DataFrame({"p_down": P[:, 0], "p_flat": P[:, 1], "p_up": P[:, 2]}, index=feat.index)
        out["predicted_direction"] = np.array([-1, 0, 1])[P.argmax(1)]
        out["p_stress"] = ps
        out["is_stress_warning"] = ps > self.warn
        out["stress_level"] = np.where(ps > self.warn, "red", np.where(ps > self.caution, "yellow", "green"))
        out["stress_ratio"] = feat["as_stress_ratio"].to_numpy()
        out["expected_short_term_return_bps"] = P[:, 0] * cm[-1] + P[:, 1] * cm[0] + P[:, 2] * cm[1]
        if self.spread is not None and self.spread.has(sym):
            out = out.join(self.spread.predict_frame(feat, sym))   # p_spread_widen, eff_spread_fwd_bps
        return out


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def load_replay(symbol: str, day: str, model_dir: str | Path = "models/") -> pd.DataFrame:
    """Đặc trưng + dự báo cho từng giây của 1 ngày lịch sử (chế độ Replay của Dashboard).
    Cột: mid, bid1, ask1, spread_bps, obi_*, ofi*, impact_rt_bps, ..., p_up/p_flat/p_down, p_stress, ...
    Lưu ý: ngày thuộc tập train của mô hình thì dự báo là TRONG MẪU (chỉ để minh họa giao diện)."""
    import datetime as dt
    from .dataset import load_symbol, with_structural
    mi = ModelInference(model_dir, symbol)
    feat = load_symbol(symbol.upper(), [dt.date.fromisoformat(str(day))])
    feat = with_structural(feat, mi.structural_params(symbol))
    pred = mi.predict_frame(feat, symbol)
    out = feat.join(pred)
    out["datetime"] = pd.to_datetime(out.index, unit="ms", utc=True)
    out["in_sample"] = str(day) in mi.direction[symbol.upper()]["train_days"]
    return out


# ---------------------------------------------------------------------- CLI test
def main(argv=None):
    ap = argparse.ArgumentParser(description="Test ModelInference.predict_one")
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--live", type=float, default=0, help="Số giây chạy với WebSocket thật (0 = dùng dữ liệu test)")
    a = ap.parse_args(argv)
    mi = ModelInference("models/", a.symbol)
    sym = mi.symbol
    log.info(f"Đã nạp mô hình: {[(s, mi.direction[s]['model_name'], mi.stress[s]['model_name']) for s in mi.symbols]}")
    if a.live <= 0:
        from .dataset import load_symbol, with_structural
        test_day = max(pd.to_datetime(mi.direction[sym]["train_days"])).date()
        from .common import lob_days
        days = [d for d, _ in lob_days(sym) if d > test_day]
        feat = with_structural(load_symbol(sym, days[-1:]), mi.structural_params(sym))
        rows = feat.iloc[3600::7200]
        lat = []
        for ts, r in rows.iterrows():
            out = mi.predict_one({**r.to_dict(), "symbol": sym})
            lat.append(out["latency_ms"])
            log.info(f"{pd.to_datetime(ts, unit='ms', utc=True)} mid={r['mid']:.2f} -> "
                     f"P(up/flat/down)={out['p_up']:.2f}/{out['p_flat']:.2f}/{out['p_down']:.2f} "
                     f"dir={out['predicted_direction']:+d} P(stress)={out['p_stress']:.3f} "
                     f"ratio={out['stress_ratio']:.2f} E[R]={out['expected_short_term_return_bps']:+.3f}bps")
        log.info(f"Độ trễ predict_one: p50 = {np.median(lat):.2f} ms, max = {np.max(lat):.2f} ms (yêu cầu < 5 ms)")
        return
    from src.task1_data.websocket_streamer import BinanceLOBStreamer
    from .microstructure_math import calculate_all_features
    s = BinanceLOBStreamer(symbol=sym).start()
    params = mi.structural_params(sym)
    t_end = time.time() + a.live
    try:
        while time.time() < t_end:
            time.sleep(2)
            data = s.get_latest_data()
            if data is None:
                continue
            t0 = time.perf_counter()
            f = calculate_all_features(data, params)
            tf = (time.perf_counter() - t0) * 1000
            if not f:
                continue
            out = mi.predict_one(f)
            log.info(f"LIVE {sym} mid={f['mid']:.2f} warmup={f['warmup_s']}s | P(up/flat/down)="
                     f"{out['p_up']:.2f}/{out['p_flat']:.2f}/{out['p_down']:.2f} P(stress)={out['p_stress']:.3f} "
                     f"| features {tf:.0f}ms + predict {out['latency_ms']:.2f}ms")
    finally:
        s.stop()


if __name__ == "__main__":
    main()
