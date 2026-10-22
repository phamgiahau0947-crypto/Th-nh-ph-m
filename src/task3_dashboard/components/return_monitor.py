"""Theo dõi lợi suất ngắn hạn: lợi suất kỳ vọng của mô hình vs lợi suất thực tế h giây sau, và tỉ lệ đoán đúng."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .theme import C, base_layout, rgba


def realized_forward_bps(hist: pd.DataFrame, h: int) -> pd.Series:
    """Lợi suất thực tế (t, t+h] theo bps – chỉ có cho các giây đã 'qua' đủ h giây (không nhìn trước)."""
    lm = np.log(hist["mid"].astype(float))
    return (lm.shift(-h) - lm) * 1e4


def hit_rate(hist: pd.DataFrame, h: int, delta_bps: float) -> dict:
    r = realized_forward_bps(hist, h)
    ok = r.notna() & hist["pred_dir"].notna()
    if ok.sum() == 0:
        return {"n": 0, "acc": np.nan, "acc_move": np.nan}
    real = np.where(r[ok] > delta_bps, 1, np.where(r[ok] < -delta_bps, -1, 0))
    pred = hist.loc[ok, "pred_dir"].to_numpy(int)
    moved = pred != 0
    return {"n": int(ok.sum()), "acc": float((real == pred).mean()),
            "acc_move": float((real[moved] == pred[moved]).mean()) if moved.any() else np.nan}


def hit_figure(hist: pd.DataFrame, h: int, delta_bps: float, height: int = 200, roll_s: int = 60) -> go.Figure:
    """Bổ sung 21/10/2026: mỗi giây 1 chấm ĐÚNG (xanh) / SAI (đỏ) theo hướng, + tỉ lệ đúng trượt `roll_s` giây,
    so với đoán bừa 33%. Chỉ có cho các giây đã qua đủ h giây."""
    x = pd.to_datetime(hist.index, unit="ms", utc=True)
    r = realized_forward_bps(hist, h)
    ok = (r.notna() & hist["pred_dir"].notna()).to_numpy()
    real = np.where(r > delta_bps, 1, np.where(r < -delta_bps, -1, 0))
    hit = pd.Series(np.where(ok, (real == hist["pred_dir"].to_numpy()).astype(float), np.nan), index=hist.index)
    roll = hit.rolling(roll_s, min_periods=max(10, roll_s // 3)).mean() * 100
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=33.3, fillcolor=rgba(C["critical"], 0.05), line_width=0, layer="below")
    fig.add_hline(y=33.3, line=dict(color=C["muted"], width=1, dash="dot"))
    for val, name, col, y in ((1.0, "Đúng", C["bid"], -5), (0.0, "Sai", C["ask"], -11)):
        m = (hit == val).to_numpy()
        fig.add_scatter(x=x[m], y=np.full(m.sum(), y), mode="markers", name=name,
                        marker=dict(color=col, size=4, symbol="square"),
                        hovertemplate=f"%{{x|%H:%M:%S}} UTC<br>{name}<extra></extra>")
    fig.add_scatter(x=x, y=roll, mode="lines", name=f"Tỉ lệ đúng {roll_s}s", line=dict(color=C["accent"], width=2),
                    hovertemplate="%{x|%H:%M:%S} UTC<br>Tỉ lệ đúng: %{y:.0f}%<extra></extra>")
    base_layout(fig, height, title=f"Đoán hướng {h} giây: đúng / sai từng giây", legend=True)
    fig.update_xaxes(tickformat="%H:%M:%S", showgrid=False)
    fig.update_yaxes(range=[-15, 102], ticksuffix="%", tickvals=[0, 33.3, 50, 75, 100],
                     ticktext=["0", "33 (bừa)", "50", "75", "100"], zeroline=False)
    return fig


def return_figure(hist: pd.DataFrame, h: int, height: int = 230) -> go.Figure:
    x = pd.to_datetime(hist.index, unit="ms", utc=True)
    r = realized_forward_bps(hist, h)
    fig = go.Figure()
    fig.add_scatter(x=x, y=r, mode="lines", name=f"Thực tế {h}s sau", line=dict(color=C["series2"], width=1),
                    fill="tozeroy", fillcolor=rgba(C["series2"], 0.12),
                    hovertemplate="%{x|%H:%M:%S} UTC<br>Thực tế: %{y:+.2f} bps<extra></extra>")
    fig.add_scatter(x=x, y=hist["exp_ret_bps"], mode="lines", name="Kỳ vọng của mô hình",
                    line=dict(color=C["series1"], width=2),
                    hovertemplate="%{x|%H:%M:%S} UTC<br>Kỳ vọng: %{y:+.2f} bps<extra></extra>")
    base_layout(fig, height, title=f"Lợi suất {h} giây tới (bps)", legend=True)
    fig.update_xaxes(tickformat="%H:%M:%S", showgrid=False)
    fig.update_yaxes(zeroline=True, zerolinecolor=C["axis"])
    return fig
