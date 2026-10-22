"""Cột phải: đồng hồ Liquidity Stress Meter, 3 thanh xác suất hướng giá, hộp cảnh báo trượt giá, diễn biến P(stress)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..lob_utils import impact_bps
from .theme import C, LEVEL_STYLE, MONO, base_layout, rgba


def gauge_html(p_stress: float, warn: float, caution: float,
               title: str = "Liquidity Stress Meter · P(stress)") -> str:
    """Đồng hồ bán nguyệt 0–100% (SVG thuần – vẽ lại mỗi 0.5 s không nhấp nháy):
    vùng xanh (< caution) / vàng / đỏ (> warn), cung màu = giá trị hiện tại, vạch trắng = ngưỡng cảnh báo."""
    v = float(np.clip(p_stress, 0, 1))
    level = "red" if p_stress > warn else ("yellow" if p_stress > caution else "green")
    col = LEVEL_STYLE[level]["color"]
    cx, cy, r = 160, 168, 122

    def pt(f, rad):
        a = np.pi * (1 - f)
        return cx + rad * np.cos(a), cy - rad * np.sin(a)

    def arc(f0, f1, color, w):
        (x0, y0), (x1, y1) = pt(f0, r), pt(max(f1, f0 + 1e-4), r)
        return (f"<path d='M{x0:.1f},{y0:.1f} A{r},{r} 0 0 1 {x1:.1f},{y1:.1f}' stroke='{color}' "
                f"stroke-width='{w}' fill='none' stroke-linecap='butt'/>")

    zones = (arc(0, caution, rgba(C["good"], 0.28), 24) + arc(caution, warn, rgba(C["warning"], 0.28), 24)
             + arc(warn, 1, rgba(C["critical"], 0.20), 24))
    ticks = ""
    for f in (0, caution, warn, 0.5, 0.75, 1):
        x, y = pt(f, r + 24)
        ticks += (f"<text x='{x:.1f}' y='{y + 4:.1f}' fill='{C['muted']}' font-size='11' text-anchor='middle' "
                  f"font-family='{MONO}'>{f * 100:.0f}</text>")
    (tx0, ty0), (tx1, ty1) = pt(warn, r - 17), pt(warn, r + 17)
    svg = (f"<svg viewBox='0 14 320 172' width='100%' style='max-height:210px;display:block;margin:auto'>"
           f"{zones}{arc(0, v, col, 24)}{ticks}"
           f"<line x1='{tx0:.1f}' y1='{ty0:.1f}' x2='{tx1:.1f}' y2='{ty1:.1f}' stroke='{C['text']}' stroke-width='2'/>"
           f"<text x='{cx}' y='{cy - 22}' fill='{C['text']}' font-size='44' font-weight='600' text-anchor='middle' "
           f"font-family='{MONO}'>{v * 100:.1f}%</text>"
           f"<text x='{cx}' y='{cy}' fill='{C['muted']}' font-size='12' text-anchor='middle'>"
           f"xác suất căng thẳng thanh khoản</text></svg>")
    return f"<div class='panel' style='padding-bottom:6px'><div class='ph'>{title}</div>{svg}</div>"


def level_html(p_stress: float, warn: float, caution: float, base_rate: float, ratio: float,
               ratio_warn: float) -> str:
    level = "red" if p_stress > warn else ("yellow" if p_stress > caution else "green")
    s = LEVEL_STYLE[level]
    mult = p_stress / base_rate if base_rate > 0 else np.nan
    text = {"green": "Thanh khoản bình thường trong 10 giây tới",
            "yellow": "Rủi ro căng thẳng đang tăng – theo dõi sát",
            "red": "Nguy cơ cạn thanh khoản – tránh lệnh thị trường lớn"}[level]
    ratio_txt = (f" · StressRatio A–S <b style='color:{C['critical'] if ratio > ratio_warn else C['text']}'>"
                 f"{ratio:.2f}</b> (ngưỡng {ratio_warn:g})") if np.isfinite(ratio) else ""
    return (f"<div class='level' style='--tone:{s['color']};--tone-bg:{rgba(s['color'], 0.10)}'>"
            f"<div class='ic'>{s['icon']}</div><div><div class='t1'>{s['label']} · {text}</div>"
            f"<div class='t2'>Gấp <b style='color:{C['text']}'>{mult:.1f}×</b> mức nền {base_rate:.0%}"
            f"{ratio_txt}</div></div></div>")


def prob_bars_html(pred: dict, horizon_s: int) -> str:
    rows = [("▲ P(Tăng)", pred["p_up"], C["bid"]), ("■ P(Đứng)", pred["p_flat"], C["flat"]),
            ("▼ P(Giảm)", pred["p_down"], C["ask"])]
    html = [f"<div class='panel'><div class='ph'>Dự báo hướng giá {horizon_s} giây tới</div>"]
    for name, p, col in rows:
        html.append(f"<div class='prob-row'><div class='name'>{name}</div><div class='track'>"
                    f"<div class='fill' style='width:{p * 100:.1f}%;background:{col}'></div></div>"
                    f"<div class='pct'>{p * 100:.1f}%</div></div>")
    d = pred["predicted_direction"]
    word, col = {1: ("TĂNG", C["bid"]), 0: ("ĐỨNG", C["text2"]), -1: ("GIẢM", C["ask"])}[d]
    er = pred["expected_short_term_return_bps"]
    html.append(f"<div class='verdict'>Mô hình nghiêng về <b style='color:{col}'>{word}</b> · lợi suất kỳ vọng "
                f"<b>{er:+.2f} bps</b> <span class='hint'>(1 bps = 0,01%)</span></div></div>")
    return "".join(html)


def slippage_alert_html(snap: dict, notional: float) -> str:
    """Live Slippage Alert: chi phí khớp ngay lệnh thị trường `notional` USDT so với mức bình thường."""
    pred, hist = snap["pred"], snap["hist"]
    mid = snap["mid"]
    cb, ub = impact_bps(snap["asks"], notional, mid)
    cs, us = impact_bps(snap["bids"], notional, mid)
    past = hist["impact_rt_bps"].dropna().to_numpy(float)
    normal = float(np.median(past)) if len(past) else np.nan
    # vị trí của chi phí hiện tại trong cửa sổ (phân vị) – bền hơn tỉ số khi chi phí bình thường gần 0
    pct = float((past < cb + cs).mean() * 100) if len(past) else np.nan
    lvl_model = pred["stress_level"]
    lvl_now = "red" if pct >= 95 else ("yellow" if pct >= 80 else "green")
    order = ["green", "yellow", "red"]
    level = max(lvl_model, lvl_now, key=order.index)
    s = LEVEL_STYLE[level]
    head = {"green": "Trượt giá ở mức bình thường", "yellow": "Thanh khoản đang mỏng đi",
            "red": "Cảnh báo trượt giá"}[level]
    usd = notional * cb / 1e4
    unfilled = " · ⚠ 20 tầng không đủ khớp hết lệnh" if (ub or us) else ""
    pct_txt = f"cao hơn {pct:.0f}%" if np.isfinite(pct) else "–"
    return (f"<div class='alert' style='--tone:{s['color']};--tone-bg:{rgba(s['color'], 0.14)}'>"
            f"<div class='head'><span class='badge'>{s['icon']} {s['label']}</span>Live Slippage Alert · {head}</div>"
            f"<div class='grid'><div><div class='k'>Mua ngay</div><div class='v'>{_bps(cb)}</div></div>"
            f"<div><div class='k'>Bán ngay</div><div class='v'>{_bps(cs)}</div></div>"
            f"<div><div class='k'>So với cửa sổ</div><div class='v'>{pct_txt}</div></div></div>"
            f"<div class='msg'>Lệnh <b>{notional:,.0f} USDT</b> khớp ngay sẽ trượt khoảng <b>{usd:,.2f} USDT</b> "
            f"so với giá mid. Chi phí khứ hồi hiện tại cao hơn {pct_txt.replace('cao hơn ', '')} số giây trong cửa sổ "
            f"(trung vị {_bps(normal)}){unfilled}.</div></div>")


def _bps(v: float) -> str:
    if not np.isfinite(v):
        return "–"
    return f"{v:.3f} bps" if abs(v) < 1 else f"{v:.2f} bps"


def stress_timeline_figure(hist: pd.DataFrame, warn: float, caution: float, height: int = 190) -> go.Figure:
    x = pd.to_datetime(hist.index, unit="ms", utc=True)
    y = hist["p_stress"].to_numpy(float) * 100
    fig = go.Figure()
    top = max(np.nanmax(y) * 1.15 if len(y) else 0, warn * 100 * 1.6)
    for lo, hi, col in ((0, caution, C["good"]), (caution, warn, C["warning"]), (warn, top / 100, C["critical"])):
        fig.add_hrect(y0=lo * 100, y1=hi * 100, fillcolor=rgba(col, 0.07), line_width=0, layer="below")
    fig.add_scatter(x=x, y=y, mode="lines", line=dict(color=rgba(C["accent"], 0.35), width=1), name="Từng giây",
                    fill="tozeroy", fillcolor=rgba(C["accent"], 0.08),
                    hovertemplate="%{x|%H:%M:%S} UTC<br>P(stress) %{y:.1f}%<extra></extra>")
    ma = pd.Series(y).rolling(10, min_periods=1).mean().to_numpy()
    fig.add_scatter(x=x, y=ma, mode="lines", line=dict(color=C["accent"], width=2), name="TB 10 giây",
                    hovertemplate="%{x|%H:%M:%S} UTC<br>TB 10s %{y:.1f}%<extra></extra>")
    for v, col in ((caution, C["warning"]), (warn, C["critical"])):
        fig.add_hline(y=v * 100, line=dict(color=col, width=1, dash="dot"))
    base_layout(fig, height, title="Diễn biến P(stress) – %", legend=True)
    fig.update_yaxes(range=[0, top], ticksuffix="%")
    fig.update_xaxes(tickformat="%H:%M:%S", showgrid=False)
    return fig
