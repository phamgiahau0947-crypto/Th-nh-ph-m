"""Hình cho báo cáo APA7 (đề bài 4.1): LOB heatmap, OBI vs lợi suất tương lai, confusion matrix,
calibration, hiệu năng theo thời gian, độ quan trọng đặc trưng. Lưu PNG vào outputs/figures/."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .common import FIG_DIR, get_logger, lob_days, read_day  # noqa: E402
from .models import DIRECTION_MODELS, STRESS_MODELS  # noqa: E402

log = get_logger()
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})
COLORS = {"LogReg-OvR": "#8c8c8c", "MNLogit": "#4c72b0", "RandomForest": "#55a868", "XGBoost": "#c44e52",
          "LogReg": "#4c72b0", "CST": "#8172b2", "OBI-rule": "#ccb974", "Majority": "#bbbbbb",
          "Persistence": "#ccb974", "AS-rule": "#8172b2", "DeepLOB": "#dd8452"}


def _save(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------- dữ liệu thô
def lob_heatmap(symbol: str, day=None, start_hour: int = 12, minutes: int = 30, levels: int = 20):
    """Heatmap độ sâu sổ lệnh: trục x = thời gian, trục y = khoảng cách giá tới mid (bps), màu = KL chờ."""
    days = lob_days(symbol)
    d, f = days[-1] if day is None else next(x for x in days if str(x[0]) == str(day))
    raw = read_day(f)
    t0 = pd.Timestamp(d).tz_localize("UTC") + pd.Timedelta(hours=start_hour)
    w = raw[(raw["datetime"] >= t0) & (raw["datetime"] < t0 + pd.Timedelta(minutes=minutes))]
    mid = w["mid_price"].to_numpy()
    edges = np.linspace(-1.5, 1.5, 61)
    grid = np.zeros((len(edges) - 1, len(w)))
    for side in ("bids", "asks"):
        P = w[[f"{side}_p{i}" for i in range(1, levels + 1)]].to_numpy()
        Q = w[[f"{side}_q{i}" for i in range(1, levels + 1)]].to_numpy()
        dist = (P - mid[:, None]) / mid[:, None] * 1e4
        idx = np.digitize(dist, edges) - 1
        for j in range(levels):
            ok = (idx[:, j] >= 0) & (idx[:, j] < len(edges) - 1)
            np.add.at(grid, (idx[ok, j], np.flatnonzero(ok)), Q[ok, j])
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    x = (w["datetime"] - t0).dt.total_seconds().to_numpy() / 60
    im = ax[0].pcolormesh(x, (edges[:-1] + edges[1:]) / 2, np.log1p(grid), cmap="magma", shading="auto")
    ax[0].axhline(0, color="white", lw=0.6)
    ax[0].set_ylabel("Khoảng cách tới mid (bps)\n(− bid / + ask)")
    ax[0].set_title(f"LOB heatmap {symbol} {d} {start_hour:02d}:00 UTC (log(1+KL), 20 tầng)")
    fig.colorbar(im, ax=ax[0], pad=0.01)
    ax[1].plot(x, mid, color="#333", lw=0.8)
    ax[1].set_ylabel("Mid")
    ax[1].set_xlabel("Phút")
    _save(fig, f"lob_heatmap_{symbol}.png")


def obi_vs_future_returns(feat: pd.DataFrame, symbol: str, h: int):
    """RQ1: lợi suất tương lai trung bình theo thập phân vị OBI (L = 1, 5, 10, 20), kèm sai số chuẩn."""
    fig, ax = plt.subplots(1, 4, figsize=(11, 2.8), sharey=True)
    r = feat[f"ret_fwd_{h}s"]
    for a, L in zip(ax, (1, 5, 10, 20)):
        o = feat[f"obi_{L}"]
        ok = o.notna() & r.notna()
        q = pd.qcut(o[ok], 10, labels=False, duplicates="drop")
        g = r[ok].groupby(q)
        m, se = g.mean(), g.std() / np.sqrt(g.count())
        xc = o[ok].groupby(q).mean()
        a.errorbar(xc, m, yerr=1.96 * se, marker="o", ms=3, lw=1, color="#4c72b0")
        a.axhline(0, color="k", lw=0.5)
        a.set_title(f"OBI L={L}")
        a.set_xlabel("OBI (trung bình thập phân vị)")
    ax[0].set_ylabel(f"E[R(t,t+{h}s)] (bps)")
    fig.suptitle(f"{symbol}: OBI và lợi suất {h}s tiếp theo (dev, KTC 95%)")
    _save(fig, f"obi_vs_future_returns_{symbol}.png")


# ---------------------------------------------------------------------- kết quả dự báo
def prediction_figures(preds: pd.DataFrame, sym: str, cfg: dict):
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import confusion_matrix
    t = preds[preds["split"] == "test"]
    if "y_dir" in t:
        d = t[np.isfinite(t["y_dir"])]
        ms = ["MNLogit", "RandomForest", "XGBoost", "CST"]
        fig, ax = plt.subplots(1, len(ms), figsize=(3.1 * len(ms), 3))
        for a, m in zip(ax, ms):
            p = d[[f"dir_{m}__{c}" for c in ("down", "flat", "up")]].to_numpy()
            yp = np.array([-1, 0, 1])[p.argmax(1)]
            cm = confusion_matrix(d["y_dir"].astype(int), yp, labels=[-1, 0, 1], normalize="true")
            a.imshow(cm, cmap="Blues", vmin=0, vmax=1)
            for i in range(3):
                for j in range(3):
                    a.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="k" if cm[i, j] < .6 else "w")
            a.set_xticks(range(3), ["Down", "Flat", "Up"])
            a.set_yticks(range(3), ["Down", "Flat", "Up"])
            a.set_xlabel("Dự báo")
            a.set_title(m)
            a.grid(False)
        ax[0].set_ylabel("Thực tế")
        fig.suptitle(f"{sym}: ma trận nhầm lẫn hướng giá (test, chuẩn hóa theo hàng)")
        _save(fig, f"confusion_direction_{sym}.png")

        fig, ax = plt.subplots(1, 3, figsize=(10, 3.1))
        for a, (k, c) in zip(ax, enumerate(("down", "flat", "up"))):
            yb = (d["y_dir"].to_numpy() == [-1, 0, 1][k]).astype(int)
            for m in DIRECTION_MODELS:
                pt, pp = calibration_curve(yb, d[f"dir_{m}__{c}"], n_bins=10, strategy="quantile")
                a.plot(pp, pt, marker="o", ms=3, lw=1, label=m, color=COLORS.get(m))
            a.plot([0, 1], [0, 1], "k--", lw=0.6)
            a.set_title(f"P({c})")
            a.set_xlabel("Xác suất dự báo")
        ax[0].set_ylabel("Tần suất thực tế")
        ax[-1].legend(fontsize=7)
        fig.suptitle(f"{sym}: reliability diagram hướng giá (test)")
        _save(fig, f"calibration_direction_{sym}.png")

        d = d.assign(hour=pd.to_datetime(d["timestamp"], unit="ms", utc=True).dt.floor("h"))
        rows = []
        from sklearn.metrics import f1_score
        for hr, g in d.groupby("hour"):
            for m in ("MNLogit", "XGBoost", "CST"):
                yp = np.array([-1, 0, 1])[g[[f"dir_{m}__{c}" for c in ("down", "flat", "up")]].to_numpy().argmax(1)]
                rows.append((hr, m, f1_score(g["y_dir"].astype(int), yp, average="macro", labels=[-1, 0, 1])))
        r = pd.DataFrame(rows, columns=["hour", "model", "macro_f1"])
        fig, ax = plt.subplots(figsize=(9, 2.8))
        for m, g in r.groupby("model"):
            ax.plot(range(len(g)), g["macro_f1"], marker=".", lw=1, label=m, color=COLORS.get(m))
        ax.set_xticks(range(0, len(g), 6), [h.strftime("%m-%d %Hh") for h in g["hour"].iloc[::6]], rotation=0)
        ax.axhline(1 / 3, color="k", lw=0.5, ls="--")
        ax.set_ylabel("Macro-F1 theo giờ")
        ax.legend(fontsize=7)
        ax.set_title(f"{sym}: hiệu năng hướng giá theo từng giờ trên test")
        _save(fig, f"performance_by_hour_test_{sym}.png")

    if "y_stress" in t:
        s = t[np.isfinite(t["y_stress"])]
        fig, ax = plt.subplots(1, 2, figsize=(8, 3.2))
        for m in STRESS_MODELS + ["Persistence"]:
            pt, pp = calibration_curve(s["y_stress"].astype(int), s[f"st_{m}"], n_bins=10, strategy="quantile")
            ax[0].plot(pp, pt, marker="o", ms=3, lw=1, label=m, color=COLORS.get(m))
        ax[0].plot([0, 1], [0, 1], "k--", lw=0.6)
        ax[0].set_xlabel("P(Stress) dự báo")
        ax[0].set_ylabel("Tần suất stress thực tế")
        ax[0].legend(fontsize=7)
        ax[0].set_title("Reliability diagram (test)")
        from sklearn.metrics import precision_recall_curve
        for m in STRESS_MODELS + ["Persistence"]:
            pr, rc, _ = precision_recall_curve(s["y_stress"].astype(int), s[f"st_{m}"])
            ax[1].plot(rc, pr, lw=1, label=m, color=COLORS.get(m))
        ax[1].axhline(s["y_stress"].mean(), color="k", lw=0.5, ls="--")
        ax[1].set_xlabel("Recall")
        ax[1].set_ylabel("Precision")
        ax[1].set_title("Precision–Recall (test)")
        fig.suptitle(f"{sym}: mô hình Liquidity Stress")
        _save(fig, f"calibration_stress_{sym}.png")


def performance_through_time(met: pd.DataFrame):
    """Hiệu năng qua các cửa sổ walk-forward (mỗi fold = 1 ngày val) và test."""
    for task, key, models in (("direction", "macro_f1", DIRECTION_MODELS + ["CST", "OBI-rule"]),
                              ("stress", "pr_auc", STRESS_MODELS + ["Persistence", "AS-rule"])):
        m = met[met["task"] == task]
        if m.empty:
            continue
        syms = m["symbol"].unique()
        fig, ax = plt.subplots(1, len(syms), figsize=(5 * len(syms), 3), squeeze=False)
        for a, sym in zip(ax[0], syms):
            x = m[m["symbol"] == sym]
            order = list(dict.fromkeys(x["split"]))
            for mod in models:
                y = x[x["model"] == mod].set_index("split").reindex(order)[key]
                a.plot(range(len(order)), y, marker="o", ms=3, lw=1, label=mod, color=COLORS.get(mod))
            a.set_xticks(range(len(order)), order, rotation=30)
            a.axvline(len(order) - 1.5, color="k", lw=0.5, ls=":")
            a.set_title(sym)
            a.set_ylabel(key)
        ax[0][-1].legend(fontsize=7, loc="best")
        fig.suptitle(f"{task}: hiệu năng qua thời gian (fold walk-forward → test)")
        _save(fig, f"performance_through_time_{task}.png")


def feature_importance(model, feats: list[str], sym: str, task: str, top: int = 20):
    if model is None:
        return
    est = model
    if hasattr(est, "calibrated_classifiers_"):
        est = est.calibrated_classifiers_[0].estimator
        est = getattr(est, "estimator", est)
    imp = pd.Series(est.get_booster().get_score(importance_type="gain"))
    imp.index = [feats[int(k[1:])] if k.startswith("f") and k[1:].isdigit() else k for k in imp.index]
    imp = imp.sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(5, 0.22 * len(imp) + 0.8))
    ax.barh(imp.index, imp.values, color="#c44e52")
    ax.set_xlabel("Gain (XGBoost)")
    ax.set_title(f"{sym}: đặc trưng quan trọng – {task}")
    _save(fig, f"feature_importance_{task}_{sym}.png")
    imp.sort_values(ascending=False).to_csv(FIG_DIR.parent / "tables" / f"feature_importance_{task}_{sym}.csv",
                                            header=["gain"])
