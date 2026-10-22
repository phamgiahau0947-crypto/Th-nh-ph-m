"""Bảng xếp hạng mô hình hướng giá chạy song song trên cùng luồng dữ liệu (bổ sung 21/10/2026, việc D).

Mỗi giây t, mọi mô hình dự báo hướng h giây tới bằng dữ liệu ≤ t; đáp án có sau h giây. Chỉ chấm các giây đã
có đáp án (không nhìn trước). Xếp theo TỈ LỆ ĐOÁN ĐÚNG: trong cửa sổ vài phút, 1 lớp (thường là "Đứng") có thể
gần như vắng mặt -> macro-F1 bị kéo về ~0.37 cho MỌI mô hình dù đúng 55% (gây hiểu lầm); trên cả ngày 3 lớp gần
cân bằng (δ chọn trên train) nên tỉ lệ đúng ≈ macro-F1. Mô hình "luôn đoán 1 đáp án" (Majority) vẫn bị lộ vì tỉ lệ
đúng thấp. Macro-F1 giữ làm cột phụ. Sai số ước lượng: các giây liền nhau gần như trùng thông tin nên số mẫu độc
lập ≈ số giây / h.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import f1_score

from .theme import C, base_layout, rgba

GROUP = {"XGBoost": "Nền · cây", "RandomForest": "Nền · cây", "MNLogit": "Nền · tuyến tính",
         "LogReg-OvR": "Nền · tuyến tính", "DeepLOB": "Nâng cao · học sâu", "CST": "Mốc lý thuyết",
         "OBI-rule": "Mốc đơn giản", "Majority": "Mốc thấp nhất"}
LINE_COLORS = {"XGBoost": "#4C8DF0", "RandomForest": "#1DB584", "MNLogit": "#E0773A", "LogReg-OvR": "#B4BBC9",
               "DeepLOB": "#7C6CF2", "CST": "#33C2E8", "OBI-rule": "#F5B021", "Majority": "#7A8396"}
RANDOM_ACC = 0.36         # đoán bừa ≈ 0.33 + biên nhiễu
MIN_SECONDS = 300         # < 5 phút có đáp án -> chưa đủ để xếp hạng


def realized_labels(z: pd.DataFrame, h: int, delta_bps: float) -> np.ndarray:
    mid = z["mid"].astype(float)
    fwd = mid.reindex(z.index + h * 1000).to_numpy()
    r = (fwd - mid.to_numpy()) / mid.to_numpy() * 1e4
    y = np.where(r > delta_bps, 1.0, np.where(r < -delta_bps, -1.0, 0.0))
    y[~np.isfinite(r)] = np.nan
    return y


def score(z: pd.DataFrame, names: list[str], h: int, delta_bps: float, t_end: int,
          window_s: int | None) -> tuple[pd.DataFrame, pd.Series]:
    """Chấm các dự báo đưa ra trong [t_end − window − h, t_end − h] (đáp án đã có tại t_end)."""
    if z.empty:
        return pd.DataFrame(), pd.Series(dtype=float)
    y = realized_labels(z, h, delta_bps)
    ts = z.index.to_numpy()
    sel = (ts + h * 1000 <= t_end) & np.isfinite(y)
    if window_s:
        sel &= ts > t_end - (window_s + h) * 1000
    rows = []
    for m in names:
        if m not in z.columns:
            continue
        p = z[m].to_numpy(float)
        ok = sel & np.isfinite(p)
        n = int(ok.sum())
        if n == 0:
            rows.append({"model": m, "n": 0})
            continue
        yy, pp = y[ok].astype(int), p[ok].astype(int)
        acc = float((yy == pp).mean())
        mv = pp != 0
        rows.append({"model": m, "n": n, "acc": acc,
                     "macro_f1": f1_score(yy, pp, average="macro", labels=[-1, 0, 1], zero_division=0),
                     "acc_move": float((yy[mv] == pp[mv]).mean()) if mv.any() else np.nan,
                     "share_move": float(mv.mean()),
                     "err": 1.96 * np.sqrt(max(acc * (1 - acc), 1e-6) / max(n / h, 1))})
    df = pd.DataFrame(rows).sort_values("acc", ascending=False, na_position="last").reset_index(drop=True)
    return df, pd.Series(sel, index=z.index)


def table_html(df: pd.DataFrame, dashboard_model: str, h: int = 10) -> str:
    if df.empty or df["n"].max() < MIN_SECONDS:
        n = int(df["n"].max()) if len(df) else 0
        return (f"<div class='panel'><div class='hint'>Chưa đủ dữ liệu để xếp hạng: mới có {n:,} giây có đáp án, "
                f"cần ít nhất {MIN_SECONDS // 60} phút. Live: chờ thêm; Replay: bấm ⏯ cho chạy.</div></div>")
    best = df.iloc[0]
    head = ("<div class='lb-row lb-head'><div>#</div><div>Mô hình</div><div>Tỉ lệ đoán đúng</div>"
            "<div>Macro-F1</div><div>Khi dám đoán Tăng/Giảm</div><div>Đánh giá</div></div>")
    out = [f"<div class='panel lb'>{head}"]
    for i, r in df.iterrows():
        if not r["n"]:
            continue
        if r["acc"] >= best["acc"] - best["err"]:
            tone, verdict = C["good"], "Nhóm dẫn đầu"
        elif r["acc"] < RANDOM_ACC:
            tone, verdict = C["critical"], "Không hơn đoán bừa"
        else:
            tone, verdict = C["warning"], "Kém nhóm đầu"
        badge = " <span class='lb-badge'>đang dùng trên dashboard</span>" if r["model"] == dashboard_model else ""
        w = max(0.0, min(1.0, r["acc"] / 0.8)) * 100
        mv = (f"{r['acc_move']:.0%} <small>({r['share_move']:.0%} số giây)</small>"
              if np.isfinite(r.get("acc_move", np.nan)) else "<small>không bao giờ đoán</small>")
        out.append(
            f"<div class='lb-row' style='--tone:{tone}'><div class='lb-rank'>{i + 1}</div>"
            f"<div><b>{r['model']}</b>{badge}<div class='lb-grp'>{GROUP.get(r['model'], '')}</div></div>"
            f"<div><div class='lb-bar'><i style='width:{w:.0f}%'></i></div><span class='lb-num'>{r['acc']:.1%}</span>"
            f" <small class='lb-num'>± {r['err']:.1%}</small></div><div class='lb-num'>{r['macro_f1']:.3f}</div>"
            f"<div class='lb-num'>{mv}</div><div><span class='lb-tag'>{verdict}</span></div></div>")
    out.append(f"<div class='hint' style='margin-top:8px'>Chấm trên {int(df['n'].max()):,} giây có đáp án · "
               f"đoán bừa ≈ 33% · sai số ± ước lượng với ~{int(df['n'].max()) // h:,} mẫu độc lập (cửa sổ {h} giây "
               f"không chồng nhau). Hai mô hình chênh nhau ít hơn sai số thì coi như <b>ngang nhau</b>.</div></div>")
    return "".join(out)


def rolling_figure(z: pd.DataFrame, sel: pd.Series, names: list[str], h: int, delta_bps: float,
                   dashboard_model: str, roll_s: int = 300, height: int = 300) -> go.Figure:
    """Tỉ lệ đoán đúng trượt `roll_s` giây của từng mô hình theo thời gian."""
    y = realized_labels(z, h, delta_bps)
    zz = z[sel.to_numpy()]
    yy = y[sel.to_numpy()]
    x = pd.to_datetime(zz.index, unit="ms", utc=True)
    fig = go.Figure()
    fig.add_hline(y=33.3, line=dict(color=C["muted"], width=1, dash="dot"))
    for m in names:
        if m not in zz.columns:
            continue
        p = zz[m].to_numpy(float)
        hit = pd.Series(np.where(np.isfinite(p), (p == yy).astype(float), np.nan))
        r = hit.rolling(roll_s, min_periods=roll_s // 3).mean().to_numpy() * 100
        main = m == dashboard_model
        fig.add_scatter(x=x, y=r, mode="lines", name=m,
                        line=dict(color=LINE_COLORS.get(m, C["text2"]), width=3 if main else 1.4),
                        opacity=1 if main else 0.75,
                        hovertemplate=f"{m}<br>%{{x|%H:%M:%S}} UTC<br>%{{y:.1f}}%<extra></extra>")
    base_layout(fig, height, title=f"Tỉ lệ đoán đúng {roll_s // 60} phút gần nhất – theo thời gian (%)", legend=True)
    fig.update_layout(legend=dict(y=1.18))
    fig.update_xaxes(tickformat="%H:%M", showgrid=False)
    fig.update_yaxes(ticksuffix="%")
    return fig
