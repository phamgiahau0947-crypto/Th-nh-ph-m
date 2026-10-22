"""Kiểm định WALK-FORWARD (bắt buộc, đề bài 2.9) + kiểm tra trên tập TEST chưa đụng tới + xuất mô hình.

    python -m src.task2_models.evaluate_walkforward                 # cả 2 bài toán, cả 2 cặp
    python -m src.task2_models.evaluate_walkforward --tasks direction --symbol BTCUSDT

Chia theo NGÀY (12 ngày mỗi cặp, không xáo trộn):
    test  = 2 ngày cuối (2026-09-01, 2026-10-01)  -> chỉ dùng 1 lần ở cuối
    dev   = 10 ngày đầu; 5 fold mở rộng: train = 5..9 ngày đầu, val = ngày kế tiếp
Mỗi fold: hiệu chỉnh A-S/CST, chọn δ, fit scaler/mô hình CHỈ trên train của fold.
Mô hình xuất ra (models/*.pkl) = mô hình fit trên toàn bộ dev, được CHỌN theo trung bình walk-forward
(không nhìn test).
"""
from __future__ import annotations

import argparse
import json
import pickle
import time

import numpy as np
import pandas as pd

from .common import (FEATURE_DIR, FIG_DIR, MODEL_DIR, TABLE_DIR, ensure_dirs, get_logger, load_config,
                     walk_forward_split)
from .dataset import build_cache, calibrate_structural, load_symbol, with_structural
from .metrics import block_bootstrap, block_ids, ci, direction_metrics, stress_metrics
from .microstructure_math import feature_columns, feature_set
from .models import DIRECTION_MODELS, STRESS_MODELS, model_complexity
from .train_direction import class_mean_returns, train_eval_direction
from .train_stress import train_eval_stress

log = get_logger()
PRED_DIR = FEATURE_DIR.parent / "task2_preds"
RQ2_SETS = ["OBI", "OFI_L1", "MLOFI", "OBI+MLOFI", "ALL"]
RQ2_MODELS = ["MNLogit", "XGBoost"]


# ====================================================================== 1 cặp
def run_symbol(sym: str, tasks: list[str], cfg: dict, rq2: bool = True) -> dict:
    h = int(cfg["horizon_s"])
    feats = feature_columns(cfg)
    sp = cfg["split"]
    stride, seed = int(sp["train_stride"]), int(cfg["random_state"])
    warn = float(cfg["avellaneda_stoikov"]["ratio_warning"])
    slabel = f"stress_h{h}"

    full = load_symbol(sym)
    days = sorted(full["day"].unique())
    folds, dev, test = walk_forward_split(days, int(sp["n_folds"]), int(sp["test_days"]))
    splits = [(f"fold{k + 1}", tr, va) for k, (tr, va) in enumerate(folds)] + [("test", dev, test)]
    log.info(f"===== {sym}: {len(days)} ngày | dev {dev[0]}..{dev[-1]} | test {test} =====")

    rows, rq2_rows, base_frames, timing_rows = [], [], [], []
    final = {}
    for name, tr_days, ev_days in splits:
        t0 = time.time()
        sub = full[full["day"].isin(tr_days + ev_days)]
        tr0 = sub[sub["day"].isin(tr_days)]
        params = calibrate_structural(sym, tr_days, tr0, cfg)
        sub = with_structural(sub, params)
        tr, ev = sub[sub["day"].isin(tr_days)], sub[sub["day"].isin(ev_days)]
        base = pd.DataFrame({"split": name, "day": ev["day"].astype(str).to_numpy(), "timestamp": ev.index.to_numpy(),
                             "rv_300s": ev["rv_300s"].to_numpy(), "stress_now": ev["stress_now"].to_numpy(),
                             "ret_fwd": ev[f"ret_fwd_{h}s"].to_numpy(), "bid1": ev["bid1"].to_numpy(),
                             "ask1": ev["ask1"].to_numpy(), "bid1_fwd": ev[f"bid1_fwd_{h}s"].to_numpy(),
                             "ask1_fwd": ev[f"ask1_fwd_{h}s"].to_numpy(), "cst_p_up": ev["cst_p_up"].to_numpy(),
                             "obi_5": ev["obi_5"].to_numpy(), "ofi1_10s": ev["ofi1_10s"].to_numpy()})
        if "direction" in tasks:
            delta, yev, probas, fitted, timing = train_eval_direction(tr, ev, feats, h, cfg["delta_flat_share"],
                                                                      stride, seed)
            base["y_dir"] = yev
            base["delta_bps"] = delta
            ok = np.isfinite(yev)
            for mname, p in probas.items():
                base[[f"dir_{mname}__{c}" for c in ("down", "flat", "up")]] = p
                rows.append({"symbol": sym, "task": "direction", "split": name, "model": mname, "delta_bps": delta,
                             "train_s": timing.get(mname, np.nan), **direction_metrics(yev[ok], p[ok])})
            if rq2:
                for fs in RQ2_SETS:
                    for mname in RQ2_MODELS:
                        if fs == "ALL":
                            p = probas[mname]
                        else:
                            _, _, pr, _, _ = train_eval_direction(tr, ev, feature_set(fs, cfg), h,
                                                                  cfg["delta_flat_share"], stride * 2, seed,
                                                                  [mname], n_trees=200)
                            p = pr[mname]
                        rq2_rows.append({"symbol": sym, "split": name, "feature_set": fs, "model": mname,
                                         **direction_metrics(yev[ok], p[ok])})
            if name == "test":
                final["direction"] = {"models": fitted, "delta_bps": delta, "train_days": [str(d) for d in tr_days],
                                      "class_mean_ret_bps": class_mean_returns(tr, h, delta)}
                for mname, m in fitted.items():
                    timing_rows.append({"symbol": sym, "task": "direction", "model": mname,
                                        "train_s": timing[mname], **model_complexity(m),
                                        **_latency(m, ev[feats].to_numpy(np.float32), "direction")})
        if "stress" in tasks:
            yev, probas, fitted, timing = train_eval_stress(tr, ev, feats, slabel, stride, seed, warn)
            base["y_stress"] = yev
            ok = np.isfinite(yev)
            for mname, p in probas.items():
                base[f"st_{mname}"] = p
                rows.append({"symbol": sym, "task": "stress", "split": name, "model": mname,
                             "train_s": timing.get(mname, np.nan), **stress_metrics(yev[ok], p[ok])})
            # robustness: nhãn stress đa chiều (độ sâu + biến động), chỉ XGBoost
            ml = f"stress_multi_h{h}"
            y2, pr2, _, _ = train_eval_stress(tr, ev, feats, ml, stride, seed, warn, ["XGBoost"])
            ok2 = np.isfinite(y2)
            rows.append({"symbol": sym, "task": "stress_multi", "split": name, "model": "XGBoost",
                         **stress_metrics(y2[ok2], pr2["XGBoost"][ok2])})
            if name == "test":
                final["stress"] = {"models": fitted, "label": slabel, "train_days": [str(d) for d in tr_days]}
                for mname, m in fitted.items():
                    timing_rows.append({"symbol": sym, "task": "stress", "model": mname, "train_s": timing[mname],
                                        **model_complexity(m), **_latency(m, ev[feats].to_numpy(np.float32), "stress")})
        if name == "test":
            final["structural"] = params
            final["feature_columns"] = feats
        base_frames.append(base)
        log.info(f"[{sym}] {name}: train {len(tr):,} dòng ({len(tr_days)} ngày) | eval {len(ev):,} | "
                 f"{time.time() - t0:.0f}s")

    preds = pd.concat(base_frames, ignore_index=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(PRED_DIR / f"{sym}_preds.parquet", compression="zstd")
    return {"metrics": rows, "rq2": rq2_rows, "timing": timing_rows, "final": final, "preds": preds}


def _latency(model, X: np.ndarray, task: str, n: int = 200) -> dict:
    """Thời gian dự báo 1 dòng (ms) – yêu cầu < 5ms cho live."""
    X = X[np.isfinite(X).all(axis=1)][:n] if np.isfinite(X).all(axis=1).any() else X[:n]
    t = []
    for i in range(min(n, len(X))):
        t0 = time.perf_counter()
        model.predict_proba(X[i:i + 1])
        t.append((time.perf_counter() - t0) * 1000)
    return {"latency_ms_p50": float(np.median(t)), "latency_ms_p95": float(np.percentile(t, 95))}


# ====================================================================== chọn & lưu mô hình
MAX_LATENCY_MS = 5.0        # đặc tả 3.5: predict < 5 ms cho live
SLOW_MODELS = {"RandomForest"}   # 200 cây sklearn: ~30–40 ms/dòng -> không đạt yêu cầu live


def select_models(metrics: pd.DataFrame, timing: pd.DataFrame | None = None) -> dict:
    """Chọn mô hình cho từng (cặp, bài toán) theo TRUNG BÌNH WALK-FORWARD (không dùng test),
    chỉ trong các mô hình có độ trễ dự báo 1 dòng < 5 ms (đo được; nếu chưa đo thì loại SLOW_MODELS)."""
    wf = metrics[metrics["split"].str.startswith("fold")]
    out = {}
    for sym in wf["symbol"].unique():
        ok = lambda task, names: [m for m in names if (
            timing is None or timing.empty or
            timing[(timing["symbol"] == sym) & (timing["task"] == task) & (timing["model"] == m)]
            ["latency_ms_p50"].min() < MAX_LATENCY_MS) and m not in SLOW_MODELS]
        d = wf[(wf["symbol"] == sym) & (wf["task"] == "direction") & wf["model"].isin(ok("direction", DIRECTION_MODELS))]
        s = wf[(wf["symbol"] == sym) & (wf["task"] == "stress") & wf["model"].isin(ok("stress", STRESS_MODELS))]
        out[sym] = {}
        if len(d):
            out[sym]["direction"] = d.groupby("model")["macro_f1"].mean().idxmax()
        if len(s):
            out[sym]["stress"] = s.groupby("model")["pr_auc"].mean().idxmax()
    return out


def save_models(results: dict, selection: dict, cfg: dict) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    h = int(cfg["horizon_s"])
    meta_f = MODEL_DIR / "model_meta.json"
    meta = json.loads(meta_f.read_text(encoding="utf-8")) if meta_f.exists() else {"symbols": {}}
    for task, fname in (("direction", "model_direction.pkl"), ("stress", "model_stress.pkl")):
        path = MODEL_DIR / fname
        bundle = pickle.loads(path.read_bytes()) if path.exists() else {"symbols": {}}
        changed = False
        for sym, r in results.items():
            fin = r["final"].get(task)
            if not fin or task not in selection.get(sym, {}):
                continue
            best = selection[sym][task]
            single_thread(fin["models"][best])
            entry = {"model_name": best, "model": fin["models"][best], "features": r["final"]["feature_columns"],
                     "horizon_s": h, "train_days": fin["train_days"]}
            if task == "direction":
                entry.update({"delta_bps": fin["delta_bps"], "classes": [-1, 0, 1],
                              "class_mean_ret_bps": fin["class_mean_ret_bps"]})
            else:
                entry.update({"label": fin["label"], "warning_threshold": cfg["stress"]["warning_threshold"]})
            bundle["symbols"][sym] = entry
            changed = True
            m = meta["symbols"].setdefault(sym, {})
            m[f"{task}_model"] = best
            m["structural"] = r["final"]["structural"]
            if task == "direction":
                m["delta_bps"] = fin["delta_bps"]
        if changed:
            bundle["default_symbol"] = cfg["symbols"][0]
            path.write_bytes(pickle.dumps(bundle))
            log.info(f"Đã lưu {path}")
    meta.update({"default_symbol": cfg["symbols"][0], "horizon_s": h,
                 "stress_ratio_warning": cfg["avellaneda_stoikov"]["ratio_warning"],
                 "stress_warning_threshold": cfg["stress"]["warning_threshold"],
                 "stress_caution_threshold": cfg["stress"].get("caution_threshold", 0.15),
                 "created_utc": pd.Timestamp.now(tz="UTC").isoformat()})
    meta_f.write_text(json.dumps(meta, indent=2, default=_json_default), encoding="utf-8")


def single_thread(model) -> None:
    """Dự báo từng dòng (live): tắt song song – khởi tạo luồng của n_jobs=-1 tốn ~30–40 ms mỗi lần gọi."""
    stack = [model]
    while stack:
        m = stack.pop()
        if hasattr(m, "n_jobs"):
            try:
                m.set_params(n_jobs=1)
            except Exception:
                m.n_jobs = 1
        stack += [s[1] for s in getattr(m, "steps", [])]
        stack += [getattr(m, k) for k in ("estimator",) if getattr(m, k, None) is not None]
        stack += [c.estimator for c in getattr(m, "calibrated_classifiers_", [])]


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


# ====================================================================== bảng & hình
def test_confidence(preds: pd.DataFrame, sym: str, cfg: dict, n_boot: int = 200) -> list[dict]:
    """Khoảng tin cậy 95% (bootstrap khối 5 phút) trên TEST + kiểm định cặp XGB vs MNLogit (RQ3, RQ5)."""
    from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef
    t = preds[preds["split"] == "test"]
    out = []
    if "y_dir" in t:
        d = t[np.isfinite(t["y_dir"])]
        blocks = block_ids(d["day"], d["timestamp"].to_numpy())
        y = d["y_dir"].to_numpy().astype(int)
        cls = np.array([-1, 0, 1])
        preds_by = {m: cls[np.argmax(d[[f"dir_{m}__{c}" for c in ("down", "flat", "up")]].to_numpy(), 1)]
                    for m in DIRECTION_MODELS + ["Majority", "OBI-rule", "CST"]}
        f1 = lambda yy, pp: f1_score(yy, pp, average="macro", labels=cls, zero_division=0)
        for m, p in preds_by.items():
            b1 = block_bootstrap(f1, blocks, n_boot, 0, y, p)
            b2 = block_bootstrap(matthews_corrcoef, blocks, n_boot, 0, y, p)
            out.append({"symbol": sym, "task": "direction", "model": m, "metric": "macro_f1", "value": f1(y, p),
                        "ci_lo": ci(b1)[0], "ci_hi": ci(b1)[1]})
            out.append({"symbol": sym, "task": "direction", "model": m, "metric": "mcc",
                        "value": matthews_corrcoef(y, p), "ci_lo": ci(b2)[0], "ci_hi": ci(b2)[1]})
        for a, b in (("XGBoost", "MNLogit"), ("XGBoost", "RandomForest"), ("MNLogit", "OBI-rule"),
                     ("XGBoost", "CST")):
            diff = block_bootstrap(lambda yy, pa, pb: f1(yy, pa) - f1(yy, pb), blocks, n_boot, 1, y,
                                   preds_by[a], preds_by[b])
            out.append({"symbol": sym, "task": "direction", "model": f"{a} − {b}", "metric": "Δmacro_f1",
                        "value": f1(y, preds_by[a]) - f1(y, preds_by[b]), "ci_lo": ci(diff)[0], "ci_hi": ci(diff)[1],
                        "p_le_0": float((diff <= 0).mean())})
    if "y_stress" in t:
        s = t[np.isfinite(t["y_stress"])]
        blocks = block_ids(s["day"], s["timestamp"].to_numpy())
        y = s["y_stress"].to_numpy().astype(int)
        for m in STRESS_MODELS + ["Persistence", "AS-rule"]:
            p = s[f"st_{m}"].to_numpy()
            b = block_bootstrap(average_precision_score, blocks, n_boot, 0, y, p)
            out.append({"symbol": sym, "task": "stress", "model": m, "metric": "pr_auc",
                        "value": average_precision_score(y, p), "ci_lo": ci(b)[0], "ci_hi": ci(b)[1]})
    return out


def regime_table(preds: pd.DataFrame, sym: str) -> list[dict]:
    """RQ4: hiệu năng theo chế độ biến động. Ngưỡng tam phân vị RV_300s lấy từ các fold val (dev), áp cho test."""
    dev = preds[preds["split"] != "test"]
    q1, q2 = np.nanquantile(dev["rv_300s"], [1 / 3, 2 / 3])
    out = []
    for part, d in (("walk-forward (val)", dev), ("test", preds[preds["split"] == "test"])):
        reg = np.where(d["rv_300s"] <= q1, "low", np.where(d["rv_300s"] <= q2, "mid", "high"))
        for r in ("low", "mid", "high"):
            x = d[reg == r]
            if "y_dir" in x:
                ok = np.isfinite(x["y_dir"])
                for m in DIRECTION_MODELS + ["CST", "Majority"]:
                    cols = [f"dir_{m}__{c}" for c in ("down", "flat", "up")]
                    if ok.sum() > 100:
                        met = direction_metrics(x.loc[ok, "y_dir"].to_numpy(), x.loc[ok, cols].to_numpy())
                        out.append({"symbol": sym, "part": part, "regime": r, "task": "direction", "model": m,
                                    "n": met["n"], "macro_f1": met["macro_f1"], "mcc": met["mcc"],
                                    "balanced_acc": met["balanced_acc"]})
            if "y_stress" in x:
                ok = np.isfinite(x["y_stress"])
                for m in STRESS_MODELS:
                    if ok.sum() > 100 and x.loc[ok, "y_stress"].nunique() == 2:
                        met = stress_metrics(x.loc[ok, "y_stress"].to_numpy(), x.loc[ok, f"st_{m}"].to_numpy())
                        out.append({"symbol": sym, "part": part, "regime": r, "task": "stress", "model": m,
                                    "n": met["n"], "pr_auc": met["pr_auc"], "roc_auc": met["roc_auc"],
                                    "base_rate": met["base_rate"], "brier": met["brier"]})
    return out


def write_outputs(results: dict, cfg: dict) -> None:
    from . import figures as F
    ensure_dirs()
    met = pd.DataFrame([r for v in results.values() for r in v["metrics"]])
    if met.empty:
        return
    met.to_csv(TABLE_DIR / "walkforward_all_metrics.csv", index=False)
    wf = met[met["split"].str.startswith("fold")]
    for task, keys in (("direction", ["macro_f1", "balanced_acc", "mcc", "accuracy", "precision_macro",
                                      "recall_macro", "brier", "log_loss"]),
                       ("stress", ["pr_auc", "roc_auc", "brier", "brier_skill", "macro_f1", "balanced_acc", "mcc",
                                   "precision_warn", "recall_warn"])):
        w = wf[wf["task"] == task]
        if w.empty:
            continue
        summ = w.groupby(["symbol", "model"])[keys].agg(["mean", "std"]).round(4)
        summ.columns = [f"{a}_{b}" for a, b in summ.columns]
        summ.reset_index().to_csv(TABLE_DIR / f"{task}_walkforward_summary.csv", index=False)
        w[["symbol", "split", "model"] + keys].round(4).to_csv(TABLE_DIR / f"{task}_walkforward_by_fold.csv",
                                                                index=False)
        t = met[(met["split"] == "test") & (met["task"] == task)]
        t[["symbol", "model"] + keys + ["n"]].round(4).to_csv(TABLE_DIR / f"{task}_test_metrics.csv", index=False)
    sm = met[met["task"] == "stress_multi"]
    if len(sm):
        sm.round(4).to_csv(TABLE_DIR / "robustness_stress_multidim.csv", index=False)

    rq2 = pd.DataFrame([r for v in results.values() for r in v["rq2"]])
    if len(rq2):
        rq2.round(4).to_csv(TABLE_DIR / "rq2_feature_sets_by_split.csv", index=False)
        g = rq2[rq2["split"].str.startswith("fold")].groupby(["symbol", "model", "feature_set"])[
            ["macro_f1", "mcc", "balanced_acc"]].mean().round(4).reset_index()
        tt = rq2[rq2["split"] == "test"][["symbol", "model", "feature_set", "macro_f1", "mcc"]].rename(
            columns={"macro_f1": "test_macro_f1", "mcc": "test_mcc"}).round(4)
        g.merge(tt, on=["symbol", "model", "feature_set"]).to_csv(TABLE_DIR / "rq2_feature_sets.csv", index=False)

    tim = pd.DataFrame([r for v in results.values() for r in v["timing"]])
    if len(tim):
        tim.round(4).to_csv(TABLE_DIR / "latency_model_complexity.csv", index=False)

    conf, reg = [], []
    for sym, v in results.items():
        conf += test_confidence(v["preds"], sym, cfg)
        reg += regime_table(v["preds"], sym)
        F.prediction_figures(v["preds"], sym, cfg)
        if v["final"].get("direction"):
            F.feature_importance(v["final"]["direction"]["models"].get("XGBoost"), v["final"]["feature_columns"],
                                 sym, "direction")
        if v["final"].get("stress"):
            F.feature_importance(v["final"]["stress"]["models"].get("XGBoost"), v["final"]["feature_columns"],
                                 sym, "stress")
    pd.DataFrame(conf).round(4).to_csv(TABLE_DIR / "test_bootstrap_ci.csv", index=False)
    pd.DataFrame(reg).round(4).to_csv(TABLE_DIR / "performance_by_volatility_regime.csv", index=False)
    F.performance_through_time(met)


def refit_selected(symbols: list[str] | None = None) -> dict:
    """Chọn lại mô hình từ bảng walk-forward đã có (không chạy lại 5 fold), fit lại đúng mô hình đó trên
    toàn bộ dev (cùng seed, cùng dữ liệu như lượt 'test') và lưu models/*.pkl."""
    cfg = load_config()
    met = pd.read_csv(TABLE_DIR / "walkforward_all_metrics.csv")
    tim_f = TABLE_DIR / "latency_model_complexity.csv"
    tim = pd.read_csv(tim_f) if tim_f.exists() else None
    selection = select_models(met, tim)
    h, feats = int(cfg["horizon_s"]), feature_columns(cfg)
    sp = cfg["split"]
    results = {}
    for sym in symbols or cfg["symbols"]:
        full = load_symbol(sym)
        days = sorted(full["day"].unique())
        _, dev, test = walk_forward_split(days, int(sp["n_folds"]), int(sp["test_days"]))
        tr0 = full[full["day"].isin(dev)]
        params = calibrate_structural(sym, dev, tr0, cfg)
        tr = with_structural(tr0, params)
        ev = with_structural(full[full["day"].isin(test)], params)
        fin = {"structural": params, "feature_columns": feats}
        dm, sm_ = selection[sym]["direction"], selection[sym]["stress"]
        delta, _, _, fitted, _ = train_eval_direction(tr, ev, feats, h, cfg["delta_flat_share"],
                                                      int(sp["train_stride"]), int(cfg["random_state"]), [dm])
        fin["direction"] = {"models": fitted, "delta_bps": delta, "train_days": [str(d) for d in dev],
                            "class_mean_ret_bps": class_mean_returns(tr, h, delta)}
        _, _, fitted_s, _ = train_eval_stress(tr, ev, feats, f"stress_h{h}", int(sp["train_stride"]),
                                              int(cfg["random_state"]),
                                              float(cfg["avellaneda_stoikov"]["ratio_warning"]), [sm_])
        fin["stress"] = {"models": fitted_s, "label": f"stress_h{h}", "train_days": [str(d) for d in dev]}
        results[sym] = {"final": fin}
        log.info(f"[refit] {sym}: direction={dm}, stress={sm_}")
    save_models(results, selection, cfg)
    return selection


def main(argv=None):
    ap = argparse.ArgumentParser(description="Walk-forward + test + xuất mô hình Task 2")
    ap.add_argument("--tasks", nargs="+", default=["direction", "stress"])
    ap.add_argument("--symbol", action="append")
    ap.add_argument("--no-rq2", action="store_true", help="Bỏ so sánh tập đặc trưng (nhanh hơn)")
    a = ap.parse_args(argv)
    cfg = load_config()
    ensure_dirs()
    build_cache(a.symbol)
    results = {}
    for sym in a.symbol or cfg["symbols"]:
        results[sym] = run_symbol(sym, a.tasks, cfg, rq2=not a.no_rq2)
    met = pd.DataFrame([r for v in results.values() for r in v["metrics"]])
    selection = select_models(met, pd.DataFrame([r for v in results.values() for r in v["timing"]]))
    log.info(f"Mô hình được chọn theo walk-forward: {selection}")
    save_models(results, selection, cfg)
    write_outputs(results, cfg)
    log.info("Xong evaluate_walkforward.")
    return results


if __name__ == "__main__":
    main()
