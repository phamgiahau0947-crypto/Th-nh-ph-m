"""DỰ BÁO SPREAD h giây tới – "Spread Forecast" trong Roadmap của thầy (ô số 5). Phần BỔ SUNG (21/10/2026):
không đổi mô hình hướng giá / stress và mọi số liệu đã báo cáo.

Spread niêm yết BTC/ETH gần như luôn = 1 tick nên "dự báo con số spread" vô nghĩa (luôn đoán 1 tick là đúng
~99.9%). Dự báo 2 thứ có thông tin:
  (A) P(spread giãn): xác suất trong (t, t+h] có ít nhất 1 giây spread niêm yết > 1 tick           (phân loại)
  (B) Spread hiệu dụng: chi phí khứ hồi trung bình ImpactRT trên (t, t+h] của lệnh 25k/10k USDT (bps) (hồi quy)
      = "spread thật" mà lệnh cỡ vừa phải trả; (B) là con số, còn nhãn stress của Task 2 là (B) > Q0.95.

Kiểm định giống hệt mô hình chính: walk-forward 5 fold theo ngày + 2 ngày test chưa đụng tới, mọi thứ fit trên
train; P(spread giãn) không cân bằng lớp để xác suất tự khớp tỉ lệ nền (xem fit_widen). Đặc trưng = 63 đặc trưng gốc của Task 2 (bỏ 4
đặc trưng cấu trúc để khỏi hiệu chỉnh A-S/CST lại mỗi fold) + 3 đặc trưng spread quá khứ (SPREAD_FEATURES).

    python -m src.task2_models.spread_forecast
Xuất: models/model_spread.pkl · outputs/tables/spread_forecast_walkforward.csv, spread_forecast_test.csv
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from .common import MODEL_DIR, TABLE_DIR, ensure_dirs, get_logger, load_config, walk_forward_split
from .metrics import stress_metrics
from .microstructure_math import STRUCTURAL_FEATURES, feature_columns

log = get_logger()
MODEL_FILE = "model_spread.pkl"
SPREAD_FEATURES = ["spr_ticks", "spr_widen_10s", "spr_widen_60s"]
CLS_MODELS = ["LogReg", "XGBoost"]
REG_MODELS = ["Ridge", "XGBoost"]
MAX_LATENCY_MS = 5.0


# ====================================================================== đặc trưng & nhãn
def _grid(s: pd.Series) -> pd.Series:
    """Đưa chuỗi về lưới 1s liên tục (giây thiếu = NaN) để cửa sổ trượt không nhảy qua chỗ đứt."""
    idx = s.index.to_numpy()
    return s.reindex(np.arange(idx[0], idx[-1] + 1, 1000, dtype="int64"))


def _widen(feat: pd.DataFrame, tick: float) -> pd.Series:
    t = _grid(((feat["ask1"] - feat["bid1"]) / tick).round())
    return (t > 1).astype(float).where(t.notna())


def add_spread_features(feat: pd.DataFrame, tick: float) -> pd.DataFrame:
    """3 đặc trưng spread CHỈ dùng quá khứ (≤ t). Gọi trên 1 đoạn liên tục (1 ngày / 1 buffer live)."""
    t = _grid(((feat["ask1"] - feat["bid1"]) / tick).round())
    w = (t > 1).astype(float).where(t.notna())
    feat["spr_ticks"] = t.reindex(feat.index).to_numpy(np.float32)
    feat["spr_widen_10s"] = w.rolling(10, min_periods=5).max().reindex(feat.index).to_numpy(np.float32)
    feat["spr_widen_60s"] = w.rolling(60, min_periods=30).mean().reindex(feat.index).to_numpy(np.float32)
    return feat


def spread_labels(feat: pd.DataFrame, h: int, tick: float) -> pd.DataFrame:
    """Nhãn nhìn về (t, t+h] – thứ cần dự báo. Cửa sổ có giây thiếu -> NaN (bỏ mẫu)."""
    w = _widen(feat, tick)
    imp = _grid(feat["impact_rt_bps"].astype(float))
    y_w = w.rolling(h, min_periods=h).max().shift(-h)          # rolling tại s phủ (s−h, s] -> dời −h = (t, t+h]
    y_e = imp.rolling(h, min_periods=h).mean().shift(-h)
    return pd.DataFrame({"y_widen": y_w.reindex(feat.index).to_numpy(),
                         "y_eff": y_e.reindex(feat.index).to_numpy()}, index=feat.index)


def spread_feature_columns(cfg: dict | None = None) -> list[str]:
    return [c for c in feature_columns(cfg) if c not in STRUCTURAL_FEATURES] + SPREAD_FEATURES


def tick_for(symbol: str, cfg: dict) -> float:
    return float(cfg.get("spread_forecast", {}).get("tick_size", {}).get(symbol.upper(), 0.01))


# ====================================================================== mô hình
def fit_widen(name: str, X: np.ndarray, y: np.ndarray, seed: int):
    """P(spread giãn): KHÔNG cân bằng lớp, KHÔNG hiệu chỉnh trên 1 ngày – Logistic/XGBoost tối ưu log-loss trên toàn
    bộ ngày train nên xác suất tự khớp tỉ lệ nền của train. (Hiệu chỉnh isotonic trên ngày train cuối như mô hình
    stress không dùng được: ngày 01/08/2026 spread gần như không giãn (0,06%) -> xác suất bị nén về ~0.)"""
    if name == "LogReg":
        m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    elif name == "XGBoost":
        m = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                          min_child_weight=50, reg_lambda=5.0, tree_method="hist", n_jobs=-1, random_state=seed,
                          verbosity=0, objective="binary:logistic", eval_metric="logloss")
    else:
        raise KeyError(name)
    m.fit(X, np.asarray(y).astype(int))
    return m


def make_reg(name: str, seed: int):
    if name == "Ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))
    if name == "XGBoost":
        return XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                            min_child_weight=50, reg_lambda=5.0, tree_method="hist", n_jobs=-1, random_state=seed,
                            verbosity=0)
    raise KeyError(name)


def reg_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    """y, pred theo bps. R² tính trên thang log1p (chi phí rất lệch: phần lớn ≈ 0, vài đợt rất cao)."""
    ly, lp = np.log1p(np.maximum(y, 0)), np.log1p(np.maximum(pred, 0))
    return {"n": len(y), "r2_log": r2_score(ly, lp), "mae_bps": mean_absolute_error(y, pred),
            "spearman": float(spearmanr(y, pred).statistic), "mean_y_bps": float(np.mean(y))}


def load_spread_data(symbol: str, cfg: dict) -> pd.DataFrame:
    """Cache đặc trưng Task 2 + đặc trưng spread + nhãn, từng ngày một (không nối ngày)."""
    from .dataset import load_symbol
    h, tick = int(cfg["horizon_s"]), tick_for(symbol, cfg)
    base = [c for c in feature_columns(cfg) if c not in STRUCTURAL_FEATURES]
    full = load_symbol(symbol, columns=sorted(set(base) | {"bid1", "ask1", "impact_rt_bps", "impact_ma10_bps"}))
    parts = []
    for _, d in full.groupby("day", sort=True):
        d = add_spread_features(d.copy(), tick)
        parts.append(d.join(spread_labels(d, h, tick)))
    return pd.concat(parts)


def train_eval(tr: pd.DataFrame, ev: pd.DataFrame, feats: list[str], stride: int, seed: int,
               cls_models=None, reg_models=None):
    """Fit trên `tr`, dự báo trên `ev`.
    Trả về (pred_cls {tên: p}, pred_reg {tên: bps}, fitted {('widen'|'eff', tên): model}, timing)."""
    X, Xev = tr[feats].to_numpy(np.float32), ev[feats].to_numpy(np.float32)
    every = (np.arange(len(tr)) % stride) == 0
    pc, pr, fitted, timing = {}, {}, {}, {}
    # benchmark
    yw = tr["y_widen"].to_numpy(float)
    ok = np.isfinite(yw)
    pc["Tần suất nền"] = np.full(len(ev), np.nanmean(yw))
    now = tr["spr_widen_10s"].to_numpy(float)
    p1, p0 = np.nanmean(yw[ok & (now == 1)]), np.nanmean(yw[ok & (now == 0)])
    e_now = ev["spr_widen_10s"].to_numpy(float)
    pc["Persistence"] = np.where(e_now == 1, p1, np.where(e_now == 0, p0, np.nanmean(yw)))
    pr["Persistence"] = ev["impact_ma10_bps"].to_numpy(float)
    # (A) phân loại
    fit_m = ok & every
    for name in cls_models or CLS_MODELS:
        t0 = time.perf_counter()
        m = fit_widen(name, X[fit_m], yw[fit_m], seed)
        timing[("widen", name)] = time.perf_counter() - t0
        pc[name] = m.predict_proba(Xev)[:, 1]
        fitted[("widen", name)] = m
    # (B) hồi quy trên log1p(chi phí)
    ye = tr["y_eff"].to_numpy(float)
    okr = np.isfinite(ye) & every
    for name in reg_models or REG_MODELS:
        t0 = time.perf_counter()
        m = make_reg(name, seed)
        m.fit(X[okr], np.log1p(np.maximum(ye[okr], 0)))
        timing[("eff", name)] = time.perf_counter() - t0
        pr[name] = np.expm1(m.predict(Xev))
        fitted[("eff", name)] = m
    return pc, pr, fitted, timing


def _latency(model, X: np.ndarray, proba: bool, n: int = 200) -> float:
    from .evaluate_walkforward import single_thread
    single_thread(model)
    X = X[:n]
    t = []
    for i in range(len(X)):
        t0 = time.perf_counter()
        model.predict_proba(X[i:i + 1]) if proba else model.predict(X[i:i + 1])
        t.append((time.perf_counter() - t0) * 1000)
    return float(np.median(t))


# ====================================================================== chạy toàn bộ
def run(symbols: list[str] | None = None) -> pd.DataFrame:
    cfg = load_config()
    ensure_dirs()
    h, sp = int(cfg["horizon_s"]), cfg["split"]
    stride, seed = int(sp["train_stride"]), int(cfg["random_state"])
    feats = spread_feature_columns(cfg)
    rows, bundle = [], {"symbols": {}, "horizon_s": h, "features": feats,
                        "created_utc": pd.Timestamp.now(tz="UTC").isoformat()}
    for sym in symbols or cfg["symbols"]:
        data = load_spread_data(sym, cfg)
        days = sorted(data["day"].unique())
        folds, dev, test = walk_forward_split(days, int(sp["n_folds"]), int(sp["test_days"]))
        splits = [(f"fold{k + 1}", tr, va) for k, (tr, va) in enumerate(folds)] + [("test", dev, test)]
        final = None
        for name, tr_days, ev_days in splits:
            t0 = time.time()
            tr, ev = data[data["day"].isin(tr_days)], data[data["day"].isin(ev_days)]
            pc, pr, fitted, timing = train_eval(tr, ev, feats, stride, seed)
            yw, ye = ev["y_widen"].to_numpy(float), ev["y_eff"].to_numpy(float)
            okw, oke = np.isfinite(yw), np.isfinite(ye)
            for m, p in pc.items():
                met = stress_metrics(yw[okw], p[okw])
                rows.append({"symbol": sym, "target": "widen", "split": name, "model": m,
                             **{k: met[k] for k in ("n", "base_rate", "pr_auc", "roc_auc", "brier", "brier_skill")}})
            for m, p in pr.items():
                okm = oke & np.isfinite(p)
                rows.append({"symbol": sym, "target": "eff", "split": name, "model": m,
                             **reg_metrics(ye[okm], p[okm])})
            if name == "test":
                final = (fitted, tr_days, ev[feats].to_numpy(np.float32))
            log.info(f"[spread] {sym} {name}: base {np.nanmean(yw):.2%} | "
                     f"PR-AUC " + " ".join(f"{m}={rows[-len(pc) - len(pr) + i]['pr_auc']:.3f}"
                                            for i, m in enumerate(pc)) + f" | {time.time() - t0:.0f}s")
        # chọn theo TRUNG BÌNH WALK-FORWARD (không nhìn test), chỉ mô hình dự báo 1 dòng < 5 ms
        met = pd.DataFrame(rows)
        wf = met[(met["symbol"] == sym) & met["split"].str.startswith("fold")]
        fitted, tr_days, Xte = final
        lat = {k: _latency(m, Xte, k[0] == "widen") for k, m in fitted.items()}
        log.info(f"[spread] {sym}: độ trễ dự báo 1 dòng (ms) {({f'{k[0]}:{k[1]}': round(v, 2) for k, v in lat.items()})}")

        def fast(tgt, names):
            ok = [n for n in names if lat[(tgt, n)] < MAX_LATENCY_MS]
            if not ok:          # máy đang bận lúc đo -> không loại ai, chỉ cảnh báo
                log.warning(f"[spread] {sym} {tgt}: không mô hình nào < {MAX_LATENCY_MS} ms lúc đo, chọn theo điểm")
            return ok or names
        best_w = wf[(wf["target"] == "widen") & wf["model"].isin(fast("widen", CLS_MODELS))].groupby(
            "model")["pr_auc"].mean().idxmax()
        best_e = wf[(wf["target"] == "eff") & wf["model"].isin(fast("eff", REG_MODELS))].groupby(
            "model")["r2_log"].mean().idxmax()
        tr = data[data["day"].isin(tr_days)]
        bundle["symbols"][sym] = {
            "tick": tick_for(sym, cfg), "train_days": [str(d) for d in tr_days],
            "base_rate_widen": float(np.nanmean(tr["y_widen"])),
            "widen": {"model_name": best_w, "model": fitted[("widen", best_w)]},
            "eff": {"model_name": best_e, "model": fitted[("eff", best_e)], "log_target": True},
            "latency_ms": {f"{k[0]}:{k[1]}": v for k, v in lat.items()}}
        log.info(f"[spread] {sym}: chọn P(spread giãn) = {best_w}, spread hiệu dụng = {best_e} | độ trễ {lat}")
    met = pd.DataFrame(rows)
    num = met.select_dtypes("number").columns
    met[num] = met[num].round(4)
    met[met["split"].str.startswith("fold")].to_csv(TABLE_DIR / "spread_forecast_walkforward.csv", index=False)
    met[met["split"] == "test"].to_csv(TABLE_DIR / "spread_forecast_test.csv", index=False)
    (MODEL_DIR / MODEL_FILE).write_bytes(pickle.dumps(bundle))
    log.info(f"Đã lưu {MODEL_DIR / MODEL_FILE} + bảng spread_forecast_*.csv")
    return met


# ====================================================================== suy luận (Task 3)
class SpreadForecaster:
    """Dự báo spread cho 1 đoạn liên tục (buffer live / 1 ngày replay). Cần cột bid1, ask1 + đặc trưng gốc."""

    def __init__(self, bundle: dict):
        self.b = bundle

    @classmethod
    def load(cls, model_dir: Path | None = None) -> "SpreadForecaster | None":
        f = Path(model_dir or MODEL_DIR) / MODEL_FILE
        return cls(pickle.loads(f.read_bytes())) if f.exists() else None

    def has(self, symbol: str) -> bool:
        return symbol.upper() in self.b["symbols"]

    def predict_frame(self, feat: pd.DataFrame, symbol: str) -> pd.DataFrame:
        s = self.b["symbols"][symbol.upper()]
        f = add_spread_features(feat[[c for c in self.b["features"] if c not in SPREAD_FEATURES]
                                     + ["bid1", "ask1"]].copy(), s["tick"])
        X = f[self.b["features"]].to_numpy(np.float32)
        out = pd.DataFrame(index=feat.index)
        out["p_spread_widen"] = s["widen"]["model"].predict_proba(X)[:, 1]
        e = s["eff"]["model"].predict(X)
        out["eff_spread_fwd_bps"] = np.expm1(e) if s["eff"].get("log_target") else e
        return out

    def info(self, symbol: str) -> dict:
        s = self.b["symbols"][symbol.upper()]
        return {"spread_base_rate": s["base_rate_widen"], "model_spread_widen": s["widen"]["model_name"],
                "model_spread_eff": s["eff"]["model_name"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Dự báo spread h giây tới (bổ sung)")
    ap.add_argument("--symbol", action="append")
    a = ap.parse_args(argv)
    run(a.symbol)


if __name__ == "__main__":
    main()
