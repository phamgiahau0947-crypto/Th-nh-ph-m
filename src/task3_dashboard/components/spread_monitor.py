"""Dự báo spread h giây tới (bổ sung 21/10/2026 – "Spread Forecast" trong Roadmap của thầy):
khối số liệu (cột phải) + biểu đồ spread hiệu dụng dự báo vs thực tế (cột trái)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .theme import C, LEVEL_STYLE, base_layout, rgba


def _level(mult: float) -> str:
    return "red" if mult >= 5 else ("yellow" if mult >= 2 else "green")


def spread_forecast_html(pred: dict, hist: pd.DataFrame, notional: float, h: int = 10) -> str:
    p, base = pred.get("p_spread_widen", np.nan), pred.get("spread_base_rate", np.nan)
    head = f"Dự báo spread {h} giây tới"
    if p is None or not np.isfinite(p):
        return (f"<div class='panel'><div class='ph'>{head}</div><div class='hint'>Chưa có mô hình dự báo spread – "
                f"bấm <b>src\\task2_models\\CHAY_BO_SUNG.bat</b> một lần rồi mở lại dashboard.</div></div>")
    mult = p / base if base and np.isfinite(base) and base > 0 else np.nan
    s = LEVEL_STYLE[_level(mult if np.isfinite(mult) else 0)]
    ticks = pred.get("spr_ticks", np.nan)
    eff = pred.get("eff_spread_fwd_bps", np.nan)
    now = float(hist["impact_rt_bps"].iloc[-10:].mean()) if "impact_rt_bps" in hist else np.nan
    arrow = "" if not (np.isfinite(eff) and np.isfinite(now)) else ("↑" if eff > now * 1.1 else ("↓" if eff < now * 0.9
                                                                                           else "≈"))
    tick_txt = f"{ticks:.0f} tick" if np.isfinite(ticks) else "–"
    mult_txt = f"gấp {mult:.1f}× mức nền {base:.2%}" if np.isfinite(mult) else ""
    usd = notional * eff / 1e4 if np.isfinite(eff) else np.nan
    return (f"<div class='alert' style='--tone:{s['color']};--tone-bg:{rgba(s['color'], 0.10)}'>"
            f"<div class='head'><span class='badge'>{s['icon']} {s['label']}</span>{head}</div>"
            f"<div class='grid'><div><div class='k'>Spread hiện tại</div><div class='v'>{tick_txt}</div></div>"
            f"<div><div class='k'>P(giãn &gt; 1 tick)</div><div class='v'>{p:.1%}</div></div>"
            f"<div><div class='k'>Spread hiệu dụng</div><div class='v'>{_bps(eff)} {arrow}</div></div></div>"
            f"<div class='msg'>Khả năng spread niêm yết giãn rộng hơn 1 tick trong {h} giây tới: <b>{p:.1%}</b> "
            f"({mult_txt}). Spread hiệu dụng dự báo (mua + bán ngay lệnh <b>{notional:,.0f} USDT</b>): "
            f"<b>{_bps(eff)}</b> ≈ {usd:,.2f} USDT · 10 giây qua: {_bps(now)}.</div></div>")


def _bps(v: float) -> str:
    if v is None or not np.isfinite(v):
        return "–"
    return f"{v:.3f} bps" if abs(v) < 1 else f"{v:.2f} bps"


def spread_figure(hist: pd.DataFrame, h: int, height: int = 210) -> go.Figure:
    """Spread hiệu dụng (bps): dự báo tại t cho (t, t+h] so với thực tế trung bình (t, t+h] (chỉ vẽ phần đã qua)
    + dấu ▲ ở những giây spread niêm yết thật sự > 1 tick."""
    x = pd.to_datetime(hist.index, unit="ms", utc=True)
    imp = hist["impact_rt_bps"].astype(float)
    real = imp.rolling(h, min_periods=h).mean().shift(-h)
    fig = go.Figure()
    fig.add_scatter(x=x, y=real, mode="lines", name=f"Thực tế {h}s sau", line=dict(color=C["series2"], width=1),
                    fill="tozeroy", fillcolor=rgba(C["series2"], 0.10),
                    hovertemplate="%{x|%H:%M:%S} UTC<br>Thực tế: %{y:.3f} bps<extra></extra>")
    fig.add_scatter(x=x, y=hist["eff_spread_fwd_bps"], mode="lines", name="Dự báo",
                    line=dict(color=C["series1"], width=2),
                    hovertemplate="%{x|%H:%M:%S} UTC<br>Dự báo: %{y:.3f} bps<extra></extra>")
    wide = hist["spr_ticks"].to_numpy(float) > 1
    if wide.any():
        top = np.nanmax([np.nanmax(real.to_numpy()) if real.notna().any() else 0,
                         np.nanmax(hist["eff_spread_fwd_bps"].to_numpy(float))
                         if hist["eff_spread_fwd_bps"].notna().any() else 0])
        fig.add_scatter(x=x[wide], y=np.full(wide.sum(), top * 1.05 if np.isfinite(top) and top > 0 else 1),
                        mode="markers", name="Spread > 1 tick",
                        marker=dict(color=C["warning"], size=7, symbol="triangle-down"),
                        hovertemplate="%{x|%H:%M:%S} UTC<br>Spread niêm yết > 1 tick<extra></extra>")
    base_layout(fig, height, title=f"Spread hiệu dụng {h} giây tới (bps)", legend=True)
    fig.update_xaxes(tickformat="%H:%M:%S", showgrid=False)
    fig.update_yaxes(rangemode="tozero")
    return fig
