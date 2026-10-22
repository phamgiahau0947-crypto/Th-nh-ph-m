"""Tái lập Xu, Gould & Howison (2019) "Multi-Level Order-Flow Imbalance in a LOB" trên BTC/ETH Binance
và mở rộng sang DỰ BÁO (RQ1, RQ2).

Thiết kế theo bài báo (mục 4.2, 5):
  - Chia mỗi ngày thành cửa sổ ΔT = 30 phút, mỗi cửa sổ gồm K = 180 khoảng Δt = 10 giây không chồng lấn.
  - Mỗi khoảng: ΔP = thay đổi mid (bps), MLOFI_m = Σ e_m (pt. 11) của các giây trong khoảng.
  - Hồi quy ĐỒNG THỜI  ΔP = α + Σ_{m≤M} β_m·MLOFI_m + ε  (pt. 16), M = 1..10, OLS và Ridge,
    ước lượng RIÊNG cho từng cửa sổ 30 phút -> báo cáo trung bình qua các cửa sổ.
  - λ của Ridge: 50 giá trị log-đều trong [1e-5, 1e5], chọn bằng 5-fold CV (fold LIỀN KHỐI theo thời gian,
    không xáo trộn) trên các ngày dev, X chuẩn hóa trong từng cửa sổ.
  - Ngoài mẫu: bài báo dùng 5-fold CV; ở đây dùng cửa sổ KẾ TIẾP (fit cửa sổ i, đánh giá i+1) để tôn trọng
    thứ tự thời gian (đề bài: không shuffle).
Mở rộng dự báo: ΔP_(t, t+10s] theo MLOFI_(t−10s, t] (quá khứ) so với OBI_L tại t (tĩnh).

    python -m src.task2_models.mlofi_regression
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import FIG_DIR, TABLE_DIR, ensure_dirs, get_logger, load_config, walk_forward_split
from .dataset import load_symbol

log = get_logger()
DT, WIN = 10, 1800


def interval_panel(feat: pd.DataFrame, M: int) -> pd.DataFrame:
    """Gom 1 ngày (lưới 1s) thành các khoảng 10s: ΔP, MLOFI_1..M, OBI đầu khoảng, ΔP khoảng kế tiếp."""
    ts = feat.index.to_numpy()
    k = (ts // 1000) // DT
    lm = np.log(feat["mid"].to_numpy())
    g = pd.DataFrame({"k": k, "lm": lm})
    for m in range(1, M + 1):
        g[f"e{m}"] = feat[f"ofi_raw{m}"].to_numpy()
    for L in (1, 5, 10, 20):
        g[f"obi_{L}"] = feat[f"obi_{L}"].to_numpy()
    agg = {"lm": ["first", "last"], **{f"e{m}": "sum" for m in range(1, M + 1)},
           **{f"obi_{L}": "last" for L in (1, 5, 10, 20)}}     # OBI tại giây cuối khoảng = thời điểm dự báo
    cnt = g.groupby("k")["lm"].count()
    p = g.groupby("k").agg(agg)
    p.columns = ["lm_first", "lm_last"] + [f"MLOFI{m}" for m in range(1, M + 1)] + [f"OBI{L}" for L in (1, 5, 10, 20)]
    p = p[cnt == DT]                                       # chỉ khoảng đủ 10 giây (không đứt dữ liệu)
    # ΔP trong khoảng: từ mid cuối khoảng trước tới mid cuối khoảng này (khớp với Σ e của các giây trong khoảng)
    prev_last = p["lm_last"].shift(1).where(p.index.to_series().diff() == 1)
    p["dP"] = (p["lm_last"] - prev_last) * 1e4
    p["dP_next"] = p["dP"].shift(-1).where(p.index.to_series().diff(-1) == -1)
    p["window"] = (p.index.to_numpy() * DT) // WIN
    return p.dropna(subset=["dP"])


def _std(X: np.ndarray) -> np.ndarray:
    s = X.std(0)
    return (X - X.mean(0)) / np.where(s > 0, s, 1)


def _fit(X, y, lam):
    """OLS (lam=0) / Ridge trên X đã chuẩn hóa, có hệ số chặn. Trả về (α, β)."""
    xm, ym = X.mean(0), y.mean()
    Xc, yc = X - xm, y - ym
    A = Xc.T @ Xc + lam * np.eye(X.shape[1])
    b = np.linalg.lstsq(A, Xc.T @ yc, rcond=None)[0]
    return ym - xm @ b, b


def choose_lambda(panels: list[pd.DataFrame], M: int, grid=np.logspace(-5, 5, 50)) -> float:
    """5-fold CV liền khối theo thời gian, trên các cửa sổ dev (X, y chuẩn hóa trong từng cửa sổ)."""
    Xs, ys = [], []
    for p in panels:
        for _, w in p.groupby("window"):
            if len(w) < 60:
                continue
            Xs.append(_std(w[[f"MLOFI{m}" for m in range(1, M + 1)]].to_numpy(float)))
            ys.append(w["dP"].to_numpy(float) - w["dP"].mean())
    X, y = np.vstack(Xs), np.concatenate(ys)
    folds = np.array_split(np.arange(len(y)), 5)
    n_tr = len(y) * 4 / 5
    err = []
    for lam in grid:
        e = 0.0
        for f in folds:
            tr = np.setdiff1d(np.arange(len(y)), f)
            a, b = _fit(X[tr], y[tr], lam * n_tr / 180)     # lưới λ tính cho cửa sổ 180 khoảng, quy đổi theo n
            e += np.mean((y[f] - a - X[f] @ b) ** 2)
        err.append(e / 5)
    return float(grid[int(np.argmin(err))])                # λ áp cho 1 cửa sổ 180 khoảng (×n/180 nếu khác)


def windowed_fits(panels: list[pd.DataFrame], M_max: int, lam: dict, target: str = "dP",
                  extra: list[str] | None = None) -> pd.DataFrame:
    """Fit từng cửa sổ 30 phút; trả về các dòng (method, M, adjR2, oos_mse, oos_var, coef...).
    `lam`: {M: λ Ridge} (chọn riêng cho từng M)."""
    rows = []
    for p in panels:
        wins = [w for _, w in p.groupby("window") if len(w) >= 60 and w[target].notna().sum() >= 60]
        for i, w in enumerate(wins):
            w = w.dropna(subset=[target])
            y = w[target].to_numpy(float)
            nxt = wins[i + 1].dropna(subset=[target]) if i + 1 < len(wins) else None
            specs = [(f"MLOFI", M, [f"MLOFI{m}" for m in range(1, M + 1)]) for M in range(1, M_max + 1)]
            for nm in extra or []:
                specs.append((nm, 1, [nm]))
            for name, M, cols in specs:
                X = w[cols].to_numpy(float)
                mu, sd = X.mean(0), X.std(0)
                sd = np.where(sd > 0, sd, 1)
                Z = (X - mu) / sd
                for method, l in (("OLS", 0.0), ("Ridge", lam.get(M, 0.0) * len(y) / 180)):
                    if name != "MLOFI" and method == "Ridge":
                        continue
                    a, b = _fit(Z, y, l)
                    res = y - a - Z @ b
                    n, k = len(y), len(cols)
                    r2 = 1 - res.var() / y.var() if y.var() > 0 else np.nan
                    adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)
                    row = {"spec": name, "method": method, "M": M, "adj_r2": adj, "n": n}
                    if nxt is not None:
                        Xn = (nxt[cols].to_numpy(float) - mu) / sd
                        yn = nxt[target].to_numpy(float)
                        row["oos_mse"] = np.mean((yn - a - Xn @ b) ** 2)
                        row["oos_var"] = np.mean((yn - yn.mean()) ** 2)
                        row["oos_mse_zero"] = np.mean(yn ** 2)
                    if name == "MLOFI" and M in (1, M_max):
                        # t-stat OLS cho bảng hệ số (sai số chuẩn cổ điển trong từng cửa sổ)
                        if method == "OLS":
                            s2 = res @ res / max(n - k - 1, 1)
                            Zc = Z - Z.mean(0)
                            cov = s2 * np.linalg.pinv(Zc.T @ Zc)
                            tval = b / np.sqrt(np.maximum(np.diag(cov), 1e-30))
                            row.update({f"t{j + 1}": tval[j] for j in range(k)})
                        row.update({f"b{j + 1}": b[j] for j in range(k)})
                    rows.append(row)
    return pd.DataFrame(rows)


def summarize(fits: pd.DataFrame) -> pd.DataFrame:
    fits = fits.assign(oos_r2_w=1 - fits["oos_mse"] / fits["oos_mse_zero"])
    g = fits.groupby(["spec", "method", "M"])
    out = g.agg(n_windows=("adj_r2", "size"), adj_r2_mean=("adj_r2", "mean"), adj_r2_std=("adj_r2", "std"),
                oos_mse=("oos_mse", "mean"), oos_var=("oos_var", "mean"), oos_mse_zero=("oos_mse_zero", "mean"),
                oos_r2_median_window=("oos_r2_w", "median"))   # bền với vài cửa sổ cực đoan
    out["oos_rmse_bps"] = np.sqrt(out["oos_mse"])
    out["oos_r2"] = 1 - out["oos_mse"] / out["oos_mse_zero"]   # so với dự báo ΔP = 0 (trung lập)
    return out.reset_index()


def coef_table(fits: pd.DataFrame, M_max: int) -> pd.DataFrame:
    rows = []
    for (method, M), g in fits[(fits["spec"] == "MLOFI") & fits["M"].isin([1, M_max])].groupby(["method", "M"]):
        for j in range(1, M + 1):
            b = g[f"b{j}"]
            r = {"method": method, "M": M, "coef": f"β{j}", "mean_std_beta": b.mean(), "se_across_windows": b.std() / np.sqrt(len(b))}
            if method == "OLS":
                t = g[f"t{j}"]
                r.update({"mean_t": t.mean(), "pct_significant_95": (t.abs() > 1.96).mean() * 100})
            rows.append(r)
    return pd.DataFrame(rows)


def run(symbols: list[str] | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg = load_config()
    ensure_dirs()
    M = int(cfg["mlofi_levels"])
    summ_all, coef_all, pred_all = [], [], []
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for si, sym in enumerate(symbols or cfg["symbols"]):
        cols = ["mid"] + [f"ofi_raw{m}" for m in range(1, M + 1)] + [f"obi_{L}" for L in (1, 5, 10, 20)]
        full = load_symbol(sym, columns=cols)
        days = sorted(full["day"].unique())
        _, dev, _ = walk_forward_split(days, cfg["split"]["n_folds"], cfg["split"]["test_days"])
        panels = {d: interval_panel(full[full["day"] == d], M) for d in days}
        lam = {m: choose_lambda([panels[d] for d in dev], m) for m in range(1, M + 1)}
        log.info(f"[MLOFI] {sym}: λ Ridge theo M (CV trên dev) = { {m: round(v, 3) for m, v in lam.items()} }")
        fits = windowed_fits(list(panels.values()), M, lam)
        s = summarize(fits)
        s.insert(0, "symbol", sym)
        s["ridge_lambda"] = s["M"].map(lam).where(s["method"] == "Ridge")
        summ_all.append(s)
        c = coef_table(fits, M)
        c.insert(0, "symbol", sym)
        coef_all.append(c)
        # dự báo: ΔP khoảng kế tiếp theo MLOFI khoảng hiện tại / OBI đầu khoảng kế tiếp
        pp = [p.dropna(subset=["dP_next"]) for p in panels.values()]
        pf = windowed_fits(pp, M, lam, target="dP_next", extra=["OBI1", "OBI5", "OBI10", "OBI20"])
        ps = summarize(pf)
        ps.insert(0, "symbol", sym)
        pred_all.append(ps)
        for ax, (title, d) in zip(axes, (("Đồng thời (giải thích ΔP)", s), ("Dự báo ΔP 10s kế tiếp", ps))):
            for method, ls in (("OLS", "-"), ("Ridge", "--")):
                x = d[(d["spec"] == "MLOFI") & (d["method"] == method)]
                ax.plot(x["M"], x["oos_r2"], ls=ls, marker="o", ms=3, color=["#c44e52", "#4c72b0"][si],
                        label=f"{sym} {method} (ngoài mẫu)")
                if title.startswith("Đồng"):
                    ax.plot(x["M"], x["adj_r2_mean"], ls=ls, lw=0.8, color=["#c44e52", "#4c72b0"][si], alpha=0.4,
                            label=f"{sym} {method} (adj R² trong mẫu)")
            ax.set_title(title)
            ax.set_xlabel("M (số tầng MLOFI)")
            ax.set_ylabel("R²")
    axes[0].legend(fontsize=6)
    axes[1].legend(fontsize=6)
    fig.suptitle("MLOFI: R² theo số tầng M (cửa sổ 30 phút, Δt = 10s)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "mlofi_r2_vs_M.png", bbox_inches="tight")
    plt.close(fig)
    pd.concat(summ_all).round(5).to_csv(TABLE_DIR / "mlofi_regression_contemporaneous.csv", index=False)
    pd.concat(coef_all).round(4).to_csv(TABLE_DIR / "mlofi_coefficients.csv", index=False)
    pd.concat(pred_all).round(5).to_csv(TABLE_DIR / "mlofi_predictive_vs_obi.csv", index=False)
    log.info("Xong mlofi_regression.")


if __name__ == "__main__":
    run()
