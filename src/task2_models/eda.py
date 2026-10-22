"""Bảng mô tả & hình EDA cho báo cáo (đề bài 4.1: bảng 1–2, hình LOB heatmap, OBI vs future returns)
+ kiểm định 2 mô hình cấu trúc (Cont–Stoikov–Talreja, Avellaneda–Stoikov) trên TEST.

    python -m src.task2_models.eda      (chạy SAU evaluate_walkforward vì dùng δ & tham số đã hiệu chỉnh)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .common import FIG_DIR, MODEL_DIR, TABLE_DIR, ensure_dirs, get_logger, load_config, walk_forward_split
from .dataset import load_symbol, with_structural
from .labeler import direction_label
from . import figures as F

log = get_logger()
SUMMARY_FEATURES = ["spread_bps", "micro_dev_bps", "obi_1", "obi_5", "obi_20", "ofi1_10s", "mlofi_sum_10s",
                    "log_depth_20", "depth_depletion", "impact_rt_bps", "impact_ma10_rel", "ti_10s",
                    "log_trades_10s", "ret_10s", "rv_60s", "rv_300s", "cst_p_up", "as_delta_star_bps",
                    "as_stress_ratio", "as_res_skew_bps"]


def run(symbols: list[str] | None = None) -> None:
    cfg = load_config()
    ensure_dirs()
    h = int(cfg["horizon_s"])
    meta = json.loads((MODEL_DIR / "model_meta.json").read_text(encoding="utf-8"))
    cls_rows, summ_rows, struct_rows, day_rows = [], [], [], []
    for sym in symbols or cfg["symbols"]:
        full = load_symbol(sym)
        days = sorted(full["day"].unique())
        _, dev, test = walk_forward_split(days, cfg["split"]["n_folds"], cfg["split"]["test_days"])
        m = meta["symbols"][sym]
        full = with_structural(full, m["structural"])
        delta = m["delta_bps"]
        full["y_dir"] = direction_label(full[f"ret_fwd_{h}s"].to_numpy(float), delta)
        full["part"] = np.where(full["day"].isin(test), "test", "dev")

        # Bảng 1: phân phối lớp (theo phần dev/test và theo ngày)
        for part, g in list(full.groupby("part")) + [("all", full)]:
            y = g["y_dir"].dropna()
            s = g[f"stress_h{h}"].dropna()
            sm = g[f"stress_multi_h{h}"].dropna()
            cls_rows.append({"symbol": sym, "part": part, "n_dir": len(y), "down_pct": (y == -1).mean() * 100,
                             "flat_pct": (y == 0).mean() * 100, "up_pct": (y == 1).mean() * 100,
                             "delta_bps": delta, "n_stress": len(s), "stress_pct": s.mean() * 100,
                             "stress_multi_pct": sm.mean() * 100})
        for d, g in full.groupby("day"):
            y = g["y_dir"].dropna()
            day_rows.append({"symbol": sym, "day": str(d), "part": g["part"].iloc[0], "rows": len(g),
                             "daily_rv_pct": float(np.sqrt(np.nansum((np.diff(np.log(g["mid"])) ** 2))) * 100),
                             "down_pct": (y == -1).mean() * 100, "flat_pct": (y == 0).mean() * 100,
                             "up_pct": (y == 1).mean() * 100, "stress_pct": g[f"stress_h{h}"].mean() * 100,
                             "median_impact_bps": g["impact_rt_bps"].median()})

        # Bảng 2: tóm tắt đặc trưng (dev)
        dv = full[full["part"] == "dev"]
        for c in SUMMARY_FEATURES:
            x = dv[c].astype(float)
            summ_rows.append({"symbol": sym, "feature": c, "mean": x.mean(), "std": x.std(), "p01": x.quantile(.01),
                              "p25": x.quantile(.25), "median": x.median(), "p75": x.quantile(.75),
                              "p99": x.quantile(.99), "missing_pct": x.isna().mean() * 100})

        # Mô hình cấu trúc trên TEST
        te = full[full["part"] == "test"]
        nd = te["next_dir"]
        ok = nd.notna() & te["cst_p_up"].notna()
        acc_cst = (np.where(te.loc[ok, "cst_p_up"] > 0.5, 1, -1) == nd[ok]).mean()
        acc_obi = (np.where(te.loc[ok, "obi_1"] > 0, 1, -1) == nd[ok]).mean()
        ll = -np.mean(np.where(nd[ok] > 0, np.log(np.clip(te.loc[ok, "cst_p_up"], 1e-4, 1)),
                               np.log(np.clip(1 - te.loc[ok, "cst_p_up"], 1e-4, 1))))
        cst, asp = m["structural"]["cst"], m["structural"]["as"]
        struct_rows.append({"symbol": sym, "cst_unit_qty": cst["unit"], "cst_lambda_per_s": cst["lam_units_per_s"],
                            "cst_mu_per_s": cst["mu_units_per_s"], "cst_theta_per_s": cst["theta"],
                            "test_next_move_acc_CST": acc_cst, "test_next_move_acc_sign_OBI1": acc_obi,
                            "test_next_move_logloss_CST": ll, "as_k_per_bps": asp["k"], "as_A_per_s": asp["A"],
                            "as_gamma": asp["gamma"], "as_horizon_s": asp["horizon_s"],
                            "as_ratio_norm": asp["ratio_norm"],
                            "test_median_delta_star_bps": te["as_delta_star_bps"].median(),
                            "test_pct_stress_ratio_gt_1.5": (te["as_stress_ratio"] > 1.5).mean() * 100,
                            "test_P(stress|ratio>1.5)": te.loc[te["as_stress_ratio"] > 1.5, f"stress_h{h}"].mean(),
                            "test_P(stress|ratio<=1.5)": te.loc[te["as_stress_ratio"] <= 1.5, f"stress_h{h}"].mean()})
        _cst_calibration_figure(te, sym)

        # Hình
        F.obi_vs_future_returns(dv, sym, h)
        F.lob_heatmap(sym, day=test[-1], start_hour=14, minutes=30)
        log.info(f"[eda] {sym} xong")
    pd.DataFrame(cls_rows).round(3).to_csv(TABLE_DIR / "class_distribution.csv", index=False)
    pd.DataFrame(day_rows).round(4).to_csv(TABLE_DIR / "class_distribution_by_day.csv", index=False)
    pd.DataFrame(summ_rows).round(5).to_csv(TABLE_DIR / "feature_summary.csv", index=False)
    pd.DataFrame(struct_rows).round(5).to_csv(TABLE_DIR / "structural_models_test.csv", index=False)
    alert_threshold_table(symbols or cfg["symbols"], meta)


def alert_threshold_table(symbols: list[str], meta: dict) -> pd.DataFrame:
    """Đánh đổi của ngưỡng cảnh báo P(Stress) (mô hình đã chọn): walk-forward val (để CHỌN) và test (để báo cáo)."""
    from .evaluate_walkforward import PRED_DIR
    rows = []
    for sym in symbols:
        p = pd.read_parquet(PRED_DIR / f"{sym}_preds.parquet")
        m = meta["symbols"][sym]["stress_model"]
        for part, d in (("walk-forward (val)", p[p["split"] != "test"]), ("test", p[p["split"] == "test"])):
            d = d[np.isfinite(d["y_stress"])]
            y, s = d["y_stress"].to_numpy(), d[f"st_{m}"].to_numpy()
            for thr in (0.10, 0.15, 0.20, 0.30, 0.50, 0.70):
                a = s > thr
                rows.append({"symbol": sym, "model": m, "part": part, "threshold": thr, "base_rate": y.mean(),
                             "alert_pct": a.mean() * 100, "precision": y[a].mean() if a.any() else np.nan,
                             "recall": (a & (y == 1)).sum() / max(y.sum(), 1),
                             "lift": (y[a].mean() / y.mean()) if a.any() else np.nan})
    out = pd.DataFrame(rows).round(4)
    out.to_csv(TABLE_DIR / "stress_alert_thresholds.csv", index=False)
    return out


def _cst_calibration_figure(te: pd.DataFrame, sym: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ok = te["next_dir"].notna() & te["cst_p_up"].notna()
    d = te[ok]
    q = pd.qcut(d["cst_p_up"].rank(method="first"), 20, labels=False)
    g = d.groupby(q)
    fig, ax = plt.subplots(figsize=(4, 3.4))
    ax.plot(g["cst_p_up"].mean(), (g["next_dir"].mean() + 1) / 2, marker="o", ms=3, label="CST (Prop. 3)")
    q2 = pd.qcut(d["obi_1"].rank(method="first"), 20, labels=False)
    g2 = d.groupby(q2)
    ax.plot((g2["obi_1"].mean() + 1) / 2, (g2["next_dir"].mean() + 1) / 2, marker="s", ms=3,
            label="(OBI₁+1)/2 (thực nghiệm)")
    ax.plot([0, 1], [0, 1], "k--", lw=0.6)
    ax.set_xlabel("P(mid tăng ở lần đổi giá kế tiếp) – dự báo")
    ax.set_ylabel("Tần suất thực tế")
    ax.set_title(f"{sym}: Cont–Stoikov–Talreja trên test")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"cst_calibration_{sym}.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
