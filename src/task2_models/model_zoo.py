"""BỘ MÔ HÌNH HƯỚNG GIÁ cho bảng xếp hạng chạy live/replay trên Dashboard (Task 3) – phần BỔ SUNG (21/10/2026).
Không đổi models/model_direction.pkl, model_stress.pkl và mọi số liệu đã báo cáo.

Mọi mô hình dùng CÙNG 10 ngày dev, cùng δ, cùng tham số cấu trúc (model_meta.json) với mô hình dashboard:
  - XGBoost              : lấy NGUYÊN từ models/model_direction.pkl (không train lại)
  - LogReg-OvR, MNLogit, RandomForest : train lại đúng như lượt 'test' của evaluate_walkforward (cùng seed,
                           cùng dữ liệu) -> điểm trên 2 ngày test phải khớp direction_test_metrics.csv
                           (kiểm tra tự động, ghi outputs/tables/model_zoo_check.csv)
  - DeepLOB              : models/deeplob_{SYM}.pt (cần torch; thiếu thì bỏ qua)
  - Benchmark            : Majority, OBI-rule, CST (công thức như train_direction.benchmark_proba)

    python -m src.task2_models.model_zoo          # ~2–5 phút, xuất models/model_zoo.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .common import MODEL_DIR, TABLE_DIR, ensure_dirs, get_logger, load_config, walk_forward_split
from .labeler import direction_label
from .metrics import DIR_CLASSES, direction_metrics

log = get_logger()
ZOO_FILE = "model_zoo.pkl"
TRAINED = ["LogReg-OvR", "MNLogit", "RandomForest"]
# thứ tự hiển thị + nhóm (để người xem biết mô hình thuộc loại nào)
ZOO_MODELS = {"XGBoost": "Mô hình nền (cây)", "RandomForest": "Mô hình nền (cây)",
              "MNLogit": "Mô hình nền (tuyến tính)", "LogReg-OvR": "Mô hình nền (tuyến tính)",
              "DeepLOB": "Mô hình nâng cao (học sâu)", "CST": "Mốc lý thuyết", "OBI-rule": "Mốc đơn giản",
              "Majority": "Mốc thấp nhất"}


def build(symbols: list[str] | None = None) -> pd.DataFrame:
    from .dataset import load_symbol, with_structural
    from .evaluate_walkforward import single_thread
    from .train_direction import train_eval_direction
    cfg = load_config()
    ensure_dirs()
    h, sp = int(cfg["horizon_s"]), cfg["split"]
    meta = json.loads((MODEL_DIR / "model_meta.json").read_text(encoding="utf-8"))
    direc = pickle.loads((MODEL_DIR / "model_direction.pkl").read_bytes())["symbols"]
    ref_f = TABLE_DIR / "direction_test_metrics.csv"
    ref = pd.read_csv(ref_f) if ref_f.exists() else pd.DataFrame()
    zoo, checks = {"symbols": {}, "horizon_s": h, "created_utc": pd.Timestamp.now(tz="UTC").isoformat()}, []
    for sym in symbols or cfg["symbols"]:
        t0 = time.time()
        d = direc[sym]
        full = load_symbol(sym)
        days = sorted(full["day"].unique())
        _, dev, test = walk_forward_split(days, int(sp["n_folds"]), int(sp["test_days"]))
        assert [str(x) for x in dev] == d["train_days"], "ngày dev khác với mô hình dashboard"
        full = with_structural(full, meta["symbols"][sym]["structural"])
        tr, ev = full[full["day"].isin(dev)], full[full["day"].isin(test)]
        delta, yev, probas, fitted, _ = train_eval_direction(tr, ev, d["features"], h, cfg["delta_flat_share"],
                                                            int(sp["train_stride"]), int(cfg["random_state"]),
                                                            TRAINED)
        assert abs(delta - d["delta_bps"]) < 1e-9, "δ khác với mô hình dashboard"
        probas["XGBoost"] = d["model"].predict_proba(ev[d["features"]].to_numpy(np.float32))
        ok = np.isfinite(yev)
        for m, p in probas.items():
            met = direction_metrics(yev[ok], p[ok])
            r = ref[(ref.get("symbol") == sym) & (ref.get("model") == m)] if len(ref) else ref
            checks.append({"symbol": sym, "model": m, "test_macro_f1": met["macro_f1"], "test_accuracy": met["accuracy"],
                           "report_macro_f1": float(r["macro_f1"].iloc[0]) if len(r) else np.nan})
        for m in fitted.values():
            single_thread(m)
        ytr = direction_label(tr[f"ret_fwd_{h}s"].to_numpy(float), delta)
        ytr = ytr[np.isfinite(ytr)]
        dl = MODEL_DIR / f"deeplob_{sym}.pt"
        zoo["symbols"][sym] = {"models": fitted, "features": d["features"], "delta_bps": delta,
                               "freq": [float((ytr == c).mean()) for c in (-1, 0, 1)],
                               "deeplob": dl.name if dl.exists() else None, "train_days": d["train_days"]}
        log.info(f"[zoo] {sym}: {list(fitted)} + XGBoost + benchmark | {time.time() - t0:.0f}s")
    (MODEL_DIR / ZOO_FILE).write_bytes(pickle.dumps(zoo))
    chk = pd.DataFrame(checks).round(4)
    chk["khop_bao_cao"] = (chk["test_macro_f1"] - chk["report_macro_f1"]).abs() < 1e-3
    chk.to_csv(TABLE_DIR / "model_zoo_check.csv", index=False)
    log.info(f"Đã lưu {MODEL_DIR / ZOO_FILE}\n{chk.to_string(index=False)}")
    return chk


class ModelZoo:
    """Dự báo HƯỚNG (−1/0/1) của mọi mô hình cho các dòng của 1 đoạn liên tục.
    feat: đặc trưng Task 2 (đã có đặc trưng cấu trúc); grid: lưới 1s sổ lệnh (cho DeepLOB, có thể None)."""

    def __init__(self, symbol: str, model_dir: Path | None = None, use_deeplob: bool = True):
        d = Path(model_dir or MODEL_DIR)
        self.symbol = symbol.upper()
        z = pickle.loads((d / ZOO_FILE).read_bytes())["symbols"][self.symbol]
        main = pickle.loads((d / "model_direction.pkl").read_bytes())["symbols"][self.symbol]
        self.features, self.delta, self.freq = z["features"], float(z["delta_bps"]), np.array(z["freq"])
        self.models = {"XGBoost": main["model"], **z["models"]}
        self.dashboard_model = main["model_name"]
        self.dl = None
        if use_deeplob and z.get("deeplob"):
            try:
                import torch
                from .deeplob import build_model
                ck = torch.load(d / z["deeplob"], map_location="cpu", weights_only=False)
                net = build_model()
                net.load_state_dict(ck["state_dict"])
                net.eval()
                self.dl = (net, ck["mu"], ck["sd"], int(ck["T"]))
            except Exception as e:                                   # thiếu torch -> bảng vẫn chạy, bỏ DeepLOB
                log.warning(f"Không nạp được DeepLOB: {e}")

    @property
    def names(self) -> list[str]:
        return [m for m in ZOO_MODELS if m in self.models or m in ("CST", "OBI-rule", "Majority")
                or (m == "DeepLOB" and self.dl is not None)]

    def predict(self, feat: pd.DataFrame, grid: pd.DataFrame | None = None) -> pd.DataFrame:
        out = pd.DataFrame(index=feat.index)
        X = feat[self.features].to_numpy(np.float32)
        for m, model in self.models.items():
            out[m] = DIR_CLASSES[model.predict_proba(X).argmax(1)]
        n = len(feat)
        f = self.freq[1]
        pu = np.nan_to_num(feat["cst_p_up"].to_numpy(float), nan=0.5)
        out["CST"] = DIR_CLASSES[np.column_stack([(1 - f) * (1 - pu), np.full(n, f), (1 - f) * pu]).argmax(1)]
        o = feat["obi_5"].to_numpy(float)
        out["OBI-rule"] = np.where(o > 0.3, 1, np.where(o < -0.3, -1, 0))
        out["Majority"] = int(DIR_CLASSES[self.freq.argmax()])
        if self.dl is not None and grid is not None:
            out["DeepLOB"] = self._deeplob(feat.index, grid)
        return out[[c for c in ZOO_MODELS if c in out.columns]]

    def _deeplob(self, idx: pd.Index, grid: pd.DataFrame, batch: int = 512) -> np.ndarray:
        import torch
        from .deeplob import lob_grid_tensor
        net, mu, sd, T = self.dl
        X, ts, _ = lob_grid_tensor(grid)
        good = np.isfinite(X).all(1)
        bad_win = np.convolve(~good, np.ones(T, int))[: len(good)] > 0      # cửa sổ T giây có lỗ -> không dự báo
        Xn = np.nan_to_num((X - mu) / sd).astype(np.float32)
        pos = pd.Index(ts).get_indexer(idx)
        ok = (pos >= T - 1) & ~bad_win[np.clip(pos, 0, None)] & (pos >= 0)
        res = np.full(len(idx), np.nan)
        sel = np.flatnonzero(ok)
        n_threads = torch.get_num_threads()
        if len(sel) < 64:             # live: vài cửa sổ/giây -> 1 luồng, không giành CPU của phần tính đặc trưng
            torch.set_num_threads(1)
        try:
            with torch.no_grad():
                for s in range(0, len(sel), batch):
                    ii = pos[sel[s:s + batch]]
                    xb = torch.from_numpy(np.stack([Xn[i - T + 1: i + 1] for i in ii]))
                    res[sel[s:s + batch]] = DIR_CLASSES[net(xb).argmax(1).numpy()]
        finally:
            torch.set_num_threads(n_threads)
        return res


def zoo_available(model_dir: Path | None = None) -> bool:
    return (Path(model_dir or MODEL_DIR) / ZOO_FILE).exists()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Xuất bộ mô hình cho bảng xếp hạng (bổ sung)")
    ap.add_argument("--symbol", action="append")
    a = ap.parse_args(argv)
    build(a.symbol)


if __name__ == "__main__":
    main()
