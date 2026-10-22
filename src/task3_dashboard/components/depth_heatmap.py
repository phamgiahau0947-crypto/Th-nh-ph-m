"""Cột trái: bậc thang 20 tầng (Bids xanh / Asks đỏ), depth curve, LOB heatmap theo thời gian, đường giá mid."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..lob_utils import ASK_Q, BID_Q, LEVELS
from .theme import C, base_layout, rgba


def _fmt_price(p: float) -> str:
    return f"{p:,.2f}"


def ladder_html(bids: np.ndarray, asks: np.ndarray, title: str = "Sổ lệnh 20 tầng – khối lượng chờ") -> str:
    """Bậc thang 20 tầng mỗi phía (HTML thuần, nhảy số mượt mỗi 0.5 s): Asks đỏ ở trên, Bids xanh ở dưới.
    Thanh đậm = KL tầng đó; nền nhạt = KL cộng dồn từ giá tốt nhất ra ngoài.
    Độ dài thanh theo CĂN BẬC HAI của KL: sổ lệnh thật rất lệch (1 tầng có thể gấp vài nghìn lần tầng khác),
    thang tuyến tính sẽ làm các tầng nhỏ biến mất."""
    qmax = np.sqrt(max(float(np.nanmax(bids[:, 1])), float(np.nanmax(asks[:, 1])), 1e-12))
    cmax = np.sqrt(max(float(np.nansum(bids[:, 1])), float(np.nansum(asks[:, 1])), 1e-12))

    def rows(side, levels, order):
        cum = np.cumsum(levels[:, 1])
        col = C["ask"] if side == "ask" else C["bid"]
        out = []
        for i in order:
            p, q = levels[i]
            out.append(f"<div class='lrow' title='Tầng {i + 1} · KL cộng dồn {cum[i]:,.4f}'>"
                       f"<span class='lv'>{'A' if side == 'ask' else 'B'}{i + 1}</span>"
                       f"<span class='lp' style='color:{col}'>{_fmt_price(p)}</span>"
                       f"<span class='lt'><i class='lc' style='width:{np.sqrt(cum[i]) / cmax * 100:.1f}%;"
                       f"background:{rgba(col, 0.13)}'></i>"
                       f"<i class='lf' style='width:{np.sqrt(q) / qmax * 100:.1f}%;background:{rgba(col, 0.85)}'></i></span>"
                       f"<span class='lq'>{q:,.4f}</span></div>")
        return "".join(out)

    mid = (bids[0, 0] + asks[0, 0]) / 2
    spr = asks[0, 0] - bids[0, 0]
    mid_row = (f"<div class='lmid'><span>Mid <b>{_fmt_price(mid)}</b></span>"
               f"<span>Spread <b>{spr:,.2f}</b> ({spr / mid * 1e4:.3f} bps)</span></div>")
    legend = (f"<span class='lleg'><i style='background:{C['ask']}'></i>Ask (bán)"
              f"<i style='background:{C['bid']}'></i>Bid (mua)</span>")
    return (f"<div class='panel'><div class='ph' style='display:flex;justify-content:space-between'>{title}{legend}</div>"
            f"<div class='ladder'>{rows('ask', asks, range(LEVELS - 1, -1, -1))}{mid_row}"
            f"{rows('bid', bids, range(LEVELS))}</div></div>")


def depth_curve_figure(bids: np.ndarray, asks: np.ndarray, height: int = 440) -> go.Figure:
    """Depth curve: khối lượng cộng dồn theo giá – 'vách' càng dốc thì thanh khoản càng dày."""
    fig = go.Figure()
    fig.add_scatter(x=bids[:, 0], y=np.cumsum(bids[:, 1]), mode="lines", name="Bid cộng dồn", line_shape="hv",
                    line=dict(color=C["bid"], width=2), fill="tozeroy", fillcolor=rgba(C["bid"], 0.16),
                    hovertemplate="Giá %{x:,.2f}<br>Cộng dồn %{y:,.4f}<extra>Bid</extra>")
    fig.add_scatter(x=asks[:, 0], y=np.cumsum(asks[:, 1]), mode="lines", name="Ask cộng dồn", line_shape="hv",
                    line=dict(color=C["ask"], width=2), fill="tozeroy", fillcolor=rgba(C["ask"], 0.16),
                    hovertemplate="Giá %{x:,.2f}<br>Cộng dồn %{y:,.4f}<extra>Ask</extra>")
    mid = (bids[0, 0] + asks[0, 0]) / 2
    fig.add_vline(x=mid, line=dict(color=C["muted"], width=1, dash="dot"))
    base_layout(fig, height, legend=True, left=64)
    fig.update_xaxes(title=dict(text="Giá", font=dict(size=11, color=C["muted"])), tickformat=",.2f")
    fig.update_yaxes(title=dict(text="KL cộng dồn", font=dict(size=11, color=C["muted"])))
    return fig


def _scale(q: np.ndarray) -> np.ndarray:
    """log KL, kéo giãn giữa phân vị 5% và 99% của cửa sổ -> 0..1 (tương phản tốt cả khi KL lệch rất mạnh)."""
    lq = np.log10(np.where(q > 0, q, np.nan))
    if not np.isfinite(lq).any():
        return np.zeros_like(q)
    lo, hi = np.nanpercentile(lq, [5, 99])
    return np.nan_to_num(np.clip((lq - lo) / max(hi - lo, 1e-9), 0, 1) * 0.92 + 0.08, nan=0.0)


def lob_heatmap_figure(hist: pd.DataFrame, max_cols: int = 300, height: int = 330) -> go.Figure:
    """LOB heatmap: trục ngang = thời gian, trục dọc = 20 tầng Ask (trên) / 20 tầng Bid (dưới),
    độ đậm = khối lượng chờ (log, chuẩn hóa theo từng phía trong cửa sổ)."""
    h = hist[BID_Q + ASK_Q].dropna(how="all")
    if len(h) > max_cols:
        h = h.iloc[np.linspace(0, len(h) - 1, max_cols).round().astype(int)]
    qb = h[BID_Q].to_numpy(float).T                         # (20, T) tầng 1..20
    qa = h[ASK_Q].to_numpy(float).T
    nb, na = _scale(qb), _scale(qa)
    # thứ tự hàng từ dưới lên: Bid20 .. Bid1, Ask1 .. Ask20
    z = np.vstack([nb[::-1], -na])
    q = np.vstack([qb[::-1], qa])
    y = [f"Bid {i}" for i in range(LEVELS, 0, -1)] + [f"Ask {i}" for i in range(1, LEVELS + 1)]
    x = pd.to_datetime(h.index, unit="ms", utc=True)
    scale = [[0.0, C["ask"]], [0.25, rgba(C["ask"], 0.55)], [0.5, C["panel2"]],
             [0.75, rgba(C["bid"], 0.55)], [1.0, C["bid"]]]
    fig = go.Figure(go.Heatmap(
        z=z, x=x, y=y, customdata=q, zmin=-1, zmax=1, colorscale=scale, showscale=False, xgap=0, ygap=0,
        hovertemplate="%{x|%H:%M:%S} UTC<br>%{y}<br>KL chờ: %{customdata:,.4f}<extra></extra>"))
    base_layout(fig, height, title="LOB Heatmap – độ dày 20 tầng theo thời gian", left=60)
    fig.update_yaxes(tickvals=["Bid 20", "Bid 10", "Bid 1", "Ask 10", "Ask 20"], showgrid=False)
    fig.update_xaxes(showgrid=False, tickformat="%H:%M:%S")
    fig.add_hline(y=LEVELS - 0.5, line=dict(color=C["text2"], width=1, dash="dot"))
    return fig


def mid_price_figure(hist: pd.DataFrame, height: int = 150) -> go.Figure:
    x = pd.to_datetime(hist.index, unit="ms", utc=True)
    fig = go.Figure(go.Scatter(x=x, y=hist["mid"], mode="lines", line=dict(color=C["series1"], width=2),
                               fill="tozeroy", fillcolor=rgba(C["series1"], 0.08), name="Giá mid",
                               hovertemplate="%{x|%H:%M:%S} UTC<br>Mid %{y:,.2f}<extra></extra>"))
    base_layout(fig, height, title="Giá mid", left=78)
    lo, hi = np.nanmin(hist["mid"]), np.nanmax(hist["mid"])
    pad = max((hi - lo) * 0.15, hi * 1e-5)
    fig.update_yaxes(range=[lo - pad, hi + pad], tickformat=",.2f")
    fig.update_xaxes(tickformat="%H:%M:%S", showgrid=False)
    return fig
