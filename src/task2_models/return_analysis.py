"""Lợi suất ngắn hạn & tác động của căng thẳng thanh khoản (câu hỏi của Thầy; công thức (12)–(14)).

1) Hồi quy (13):  R_{t,h} = β0 + β1·OBI_t + β2·MLOFI1_t + β3·Stress_t + β4·(MLOFI1_t × Stress_t) + ε
   - mẫu KHÔNG chồng lấn (mỗi h giây 1 quan sát), sai số chuẩn Newey–West (HAC);
   - Stress_t = trạng thái stress hiện tại (đo được tại t); MLOFI1_t = OFI tầng 1 trong 10s qua (chuẩn hóa);
   - thêm (13b) cho |R| (biến động/rủi ro) và (13c) dùng P(Stress) DỰ BÁO ngoài mẫu từ walk-forward.
2) Phân phối lợi suất khi Stress vs Bình thường; theo thập phân vị P(Stress) dự báo.
3) Backtest (14) trên TEST: mua ở ask_t bán ở bid_{t+h} (Up), bán ở bid_t mua lại ở ask_{t+h} (Down), trừ 2·fee;
   so sánh khi stress / không stress và chiến lược "đứng ngoài khi P(Stress) > 0.7".

    python -m src.task2_models.return_analysis
Xuất: outputs/tables/stress_impact_on_returns.csv, return_regression_13.csv, pnl_backtest.csv + hình.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .common import FIG_DIR, MODEL_DIR, TABLE_DIR, ensure_dirs, get_logger, load_config
from .dataset import load_symbol

log = get_logger()


def _ols_hac(y: pd.Series, X: pd.DataFrame, lags: int) -> tuple[pd.DataFrame, dict]:
    ok = y.notna() & X.notna().all(axis=1)
    res = sm.OLS(y[ok], sm.add_constant(X[ok])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    tab = pd.DataFrame({"coef": res.params, "se_hac": res.bse, "t": res.tvalues, "p": res.pvalues})
    return tab, {"n": int(res.nobs), "r2": res.rsquared, "adj_r2": res.rsquared_adj}


def regressions(sym: str, cfg: dict, preds: pd.DataFrame | None) -> pd.DataFrame:
    h = int(cfg["horizon_s"])
    cols = ["obi_5", "ofi1_10s", "mlofi_sum_10s", "stress_now", f"ret_fwd_{h}s", "rv_300s", "impact_ma10_rel"]
    d = load_symbol(sym, columns=cols)
    d = d[(d.index.to_numpy() // 1000) % h == 0]            # quan sát không chồng lấn
    R = d[f"ret_fwd_{h}s"].astype(float)
    z = lambda s: (s - s.mean()) / s.std()
    X = pd.DataFrame({"OBI": d["obi_5"].astype(float), "MLOFI1": z(d["ofi1_10s"].astype(float)),
                      "Stress": d["stress_now"].astype(float)}, index=d.index)
    X["MLOFI1xStress"] = X["MLOFI1"] * X["Stress"]
    out = []
    for name, y, XX in (("(13) R", R, X), ("(13b) |R|", R.abs(), X.assign(RV300=d["rv_300s"].astype(float)))):
        tab, st = _ols_hac(y, XX, lags=5)
        tab = tab.reset_index().rename(columns={"index": "term"})
        tab.insert(0, "model", name)
        tab.insert(0, "symbol", sym)
        tab["n"], tab["r2"] = st["n"], st["r2"]
        out.append(tab)
    # (13c) P(Stress) dự báo ngoài mẫu (walk-forward val + test) thay cho Stress_t
    if preds is not None and "st_XGBoost" in preds:
        best = _best_stress_model(sym)
        p = preds[["timestamp", f"st_{best}"]].drop_duplicates("timestamp").set_index("timestamp")[f"st_{best}"]
        j = d.join(p.rename("p_stress"), how="inner")
        j = j[j["p_stress"].notna()]
        Rj = j[f"ret_fwd_{h}s"].astype(float)
        Xj = pd.DataFrame({"OBI": j["obi_5"].astype(float), "MLOFI1": z(j["ofi1_10s"].astype(float)),
                           "P_stress": j["p_stress"].astype(float)}, index=j.index)
        Xj["MLOFI1xP_stress"] = Xj["MLOFI1"] * Xj["P_stress"]
        for name, y, XX in (("(13c) R ~ P̂(Stress)", Rj, Xj),
                            ("(13d) |R| ~ P̂(Stress)", Rj.abs(), Xj.assign(RV300=j["rv_300s"].astype(float)))):
            tab, st = _ols_hac(y, XX, lags=5)
            tab = tab.reset_index().rename(columns={"index": "term"})
            tab.insert(0, "model", name)
            tab.insert(0, "symbol", sym)
            tab["n"], tab["r2"] = st["n"], st["r2"]
            out.append(tab)
    return pd.concat(out, ignore_index=True)


def _best_stress_model(sym: str) -> str:
    f = MODEL_DIR / "model_meta.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))["symbols"].get(sym, {}).get("stress_model", "XGBoost")
    return "XGBoost"


def _best_direction_model(sym: str) -> str:
    f = MODEL_DIR / "model_meta.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))["symbols"].get(sym, {}).get("direction_model", "XGBoost")
    return "XGBoost"


def return_distribution(sym: str, preds: pd.DataFrame, h: int) -> pd.DataFrame:
    """Thống kê R_{t,h} theo trạng thái stress hiện tại và theo nhóm P(Stress) dự báo (ngoài mẫu)."""
    best = _best_stress_model(sym)
    d = preds.drop_duplicates(["split", "timestamp"])
    d = d[np.isfinite(d["ret_fwd"])]
    rows = []

    def stats(g, by, level):
        r = g["ret_fwd"]
        return {"symbol": sym, "group_by": by, "group": level, "n": len(r), "mean_R_bps": r.mean(),
                "std_R_bps": r.std(), "mean_abs_R_bps": r.abs().mean(), "p95_abs_R_bps": r.abs().quantile(0.95),
                "skew": r.skew(), "kurtosis": r.kurt(), "share_neg": (r < 0).mean()}

    for lvl, g in d.groupby(d["stress_now"].map({0.0: "Bình thường", 1.0: "Stress"})):
        rows.append(stats(g, "stress_now (thực tế tại t)", lvl))
    if f"st_{best}" in d:
        q = pd.qcut(d[f"st_{best}"].rank(method="first"), 5, labels=[f"Q{i}" for i in range(1, 6)])
        for lvl, g in d.groupby(q, observed=True):
            rr = stats(g, f"P̂(Stress) {best} – ngũ phân vị", str(lvl))
            rr["mean_p_stress"] = g[f"st_{best}"].mean()
            rows.append(rr)
        warn = load_config()["stress"]["warning_threshold"]
        for lvl, g in d.groupby(d[f"st_{best}"] > warn):
            rows.append(stats(g, f"P̂(Stress) > {warn}", "Cảnh báo" if lvl else "Không cảnh báo"))
    return pd.DataFrame(rows)


def pnl_backtest(sym: str, preds: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """(14) Net PnL trên TEST, giao dịch không chồng lấn mỗi h giây theo mô hình hướng giá đã chọn."""
    h = int(cfg["horizon_s"])
    dm, sm_ = _best_direction_model(sym), _best_stress_model(sym)
    warn = cfg["stress"]["warning_threshold"]
    t = preds[(preds["split"] == "test")].copy()
    t = t[(t["timestamp"] // 1000) % h == 0]
    t = t[np.isfinite(t["bid1_fwd"]) & np.isfinite(t["ask1_fwd"])]
    rows = []
    for model in [dm, "MNLogit", "OBI-rule"] if dm != "MNLogit" else [dm, "OBI-rule"]:
        P = t[[f"dir_{model}__{c}" for c in ("down", "flat", "up")]].to_numpy()
        sig = np.array([-1, 0, 1])[P.argmax(1)]
        long_r = (t["bid1_fwd"] - t["ask1"]) / t["ask1"]
        short_r = (t["bid1"] - t["ask1_fwd"]) / t["bid1"]
        gross = np.where(sig == 1, long_r, np.where(sig == -1, short_r, 0.0))
        mid_r = np.where(sig != 0, sig * t["ret_fwd"] / 1e4, 0.0)        # trước chi phí spread
        pst = t[f"st_{sm_}"].to_numpy() if f"st_{sm_}" in t else np.zeros(len(t))
        groups = {"Tất cả": np.ones(len(t), bool), "Stress tại t": t["stress_now"].to_numpy() == 1,
                  "Không stress tại t": t["stress_now"].to_numpy() == 0,
                  f"P̂(Stress) > {warn}": pst > warn, f"Lọc: bỏ khi P̂(Stress) > {warn}": pst <= warn}
        for fee in cfg["pnl"]["fees"]:
            net = np.where(sig != 0, gross - 2 * fee, 0.0)
            for gname, gm in groups.items():
                tr = gm & (sig != 0)
                n = int(tr.sum())
                rows.append({"symbol": sym, "model": model, "fee_per_side": fee, "group": gname, "n_trades": n,
                             "hit_rate": float((mid_r[tr] > 0).mean()) if n else np.nan,
                             "mean_mid_ret_bps": float(mid_r[tr].mean() * 1e4) if n else np.nan,
                             "mean_gross_bps": float(gross[tr].mean() * 1e4) if n else np.nan,
                             "mean_net_bps": float(net[tr].mean() * 1e4) if n else np.nan,
                             "total_net_pct": float(net[tr].sum() * 100) if n else 0.0})
        if model == dm:
            _pnl_figure(sym, t, sig, gross, pst, warn, cfg)
    return pd.DataFrame(rows)


def _pnl_figure(sym, t, sig, gross, pst, warn, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = pd.to_datetime(t["timestamp"].to_numpy(), unit="ms", utc=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    for fee, ls in ((0.0, "-"), (0.0002, "--")):
        net = np.where(sig != 0, gross - 2 * fee, 0.0)
        ax[0].plot(np.arange(len(net)), np.cumsum(net) * 1e4, ls=ls, lw=1, label=f"fee {fee * 1e4:.0f}bps/chiều")
        filt = np.where(pst > warn, 0.0, net)
        ax[0].plot(np.arange(len(net)), np.cumsum(filt) * 1e4, ls=ls, lw=1, alpha=0.7,
                   label=f"fee {fee * 1e4:.0f}bps, lọc stress")
    ax[0].set_title("Net PnL tích lũy trên test (bps)")
    ax[0].set_xlabel(f"Quyết định (mỗi {cfg['horizon_s']}s)")
    ax[0].legend(fontsize=7)
    m = sig != 0
    ax[1].hist([gross[m & (pst > warn)] * 1e4, gross[m & (pst <= warn)] * 1e4], bins=60, density=True,
               label=["P̂(Stress) > 0.7", "Bình thường"], alpha=0.7)
    ax[1].set_title("Lợi suất gộp mỗi lệnh (sau spread, trước phí)")
    ax[1].set_xlabel("bps")
    ax[1].legend(fontsize=7)
    fig.suptitle(f"{sym}: backtest (14)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"pnl_backtest_{sym}.png", bbox_inches="tight")
    plt.close(fig)


def distribution_figure(sym: str, preds: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    best = _best_stress_model(sym)
    d = preds.drop_duplicates(["split", "timestamp"])
    d = d[np.isfinite(d["ret_fwd"])]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    for lvl, lab in ((0.0, "Bình thường"), (1.0, "Stress tại t")):
        r = d.loc[d["stress_now"] == lvl, "ret_fwd"].clip(-15, 15)
        ax[0].hist(r, bins=120, density=True, histtype="step", lw=1.2, label=f"{lab} (n={len(r):,})")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("R(t, t+h) (bps)")
    ax[0].set_title("Phân phối lợi suất ngắn hạn")
    ax[0].legend(fontsize=7)
    if f"st_{best}" in d:
        q = pd.qcut(d[f"st_{best}"].rank(method="first"), 10, labels=False)
        g = d.groupby(q)
        ax[1].plot(g[f"st_{best}"].mean(), g["ret_fwd"].apply(lambda r: r.abs().mean()), marker="o", label="E|R|")
        ax[1].plot(g[f"st_{best}"].mean(), g["ret_fwd"].std(), marker="s", label="Độ lệch chuẩn R")
        ax[1].set_xlabel(f"P̂(Stress) ({best}, ngoài mẫu) – thập phân vị")
        ax[1].set_ylabel("bps")
        ax[1].set_title("Rủi ro lợi suất theo xác suất stress dự báo")
        ax[1].legend(fontsize=7)
    fig.suptitle(f"{sym}: căng thẳng thanh khoản và lợi suất {load_config()['horizon_s']}s")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"stress_return_distribution_{sym}.png", bbox_inches="tight")
    plt.close(fig)


def run(symbols: list[str] | None = None) -> None:
    from .evaluate_walkforward import PRED_DIR
    cfg = load_config()
    ensure_dirs()
    regs, dist, pnl = [], [], []
    for sym in symbols or cfg["symbols"]:
        pf = PRED_DIR / f"{sym}_preds.parquet"
        preds = pd.read_parquet(pf) if pf.exists() else None
        regs.append(regressions(sym, cfg, preds))
        if preds is not None:
            dist.append(return_distribution(sym, preds, cfg["horizon_s"]))
            pnl.append(pnl_backtest(sym, preds, cfg))
            distribution_figure(sym, preds)
        log.info(f"[returns] {sym} xong")
    pd.concat(regs).round(5).to_csv(TABLE_DIR / "return_regression_13.csv", index=False)
    if dist:
        pd.concat(dist).round(4).to_csv(TABLE_DIR / "stress_impact_on_returns.csv", index=False)
    if pnl:
        pd.concat(pnl).round(4).to_csv(TABLE_DIR / "pnl_backtest.csv", index=False)
    log.info("Xong return_analysis.")


if __name__ == "__main__":
    run()
