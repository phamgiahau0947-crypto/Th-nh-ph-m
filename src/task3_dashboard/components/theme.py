"""Bảng màu + CSS + khung Plotly chung cho Dashboard (giao diện tối kiểu 'financial terminal')."""
from __future__ import annotations

import plotly.graph_objects as go

C = {
    # mặt nền
    "page": "#0A0D14", "panel": "#11151F", "panel2": "#161B27", "border": "rgba(255,255,255,0.07)",
    "grid": "#1E2432", "axis": "#2A3142",
    # chữ
    "text": "#ECEEF3", "text2": "#B4BBC9", "muted": "#7A8396",
    # nhận diện (bid/ask = mua/bán; up/down = hướng giá)
    "bid": "#1DB584", "ask": "#E8606A", "flat": "#7A8396",
    "series1": "#4C8DF0", "series2": "#E0773A",
    "accent": "#7C6CF2", "accent2": "#33C2E8",
    # trạng thái (chỉ dùng cho mức rủi ro, luôn kèm biểu tượng + chữ)
    "good": "#18B25C", "warning": "#F5B021", "critical": "#E5484D",
}

LEVEL_STYLE = {
    "green": {"color": C["good"], "icon": "✓", "label": "AN TOÀN"},
    "yellow": {"color": C["warning"], "icon": "!", "label": "CHÚ Ý"},
    "red": {"color": C["critical"], "icon": "▲", "label": "CẢNH BÁO"},
}

FONT = "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif"
MONO = "'JetBrains Mono', 'Cascadia Mono', Consolas, monospace"


def rgba(hex_color: str, a: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def base_layout(fig: go.Figure, height: int, title: str | None = None, legend: bool = False,
                left: int = 52) -> go.Figure:
    fig.update_layout(
        template="none",                        # bỏ template mặc định của Plotly -> JSON nhẹ hơn, vẽ nhanh hơn
        height=height, margin=dict(l=left, r=12, t=40 if (title or legend) else 10, b=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=C["text2"]),
        title=dict(text=title, x=0.01, y=0.97, font=dict(size=13, color=C["text"])) if title else None,
        showlegend=legend,
        legend=dict(orientation="h", x=1, xanchor="right", y=1.12, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11, color=C["text2"])),
        hoverlabel=dict(bgcolor=C["panel2"], bordercolor=C["axis"], font=dict(family=FONT, color=C["text"])),
    )
    fig.update_xaxes(gridcolor=C["grid"], zerolinecolor=C["axis"], linecolor=C["axis"],
                     tickfont=dict(color=C["muted"], size=11), showspikes=False, automargin=True)
    fig.update_yaxes(gridcolor=C["grid"], zerolinecolor=C["axis"], linecolor=C["axis"],
                     tickfont=dict(color=C["muted"], size=11), automargin=True)
    return fig


PLOTLY_CONFIG = {"displayModeBar": False}


CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');
html, body, [class*="css"], .stApp {{ font-family: {FONT}; }}
.stApp {{
  background: radial-gradient(1200px 600px at 10% -10%, rgba(124,108,242,0.10), transparent 60%),
              radial-gradient(900px 500px at 100% 0%, rgba(51,194,232,0.07), transparent 60%), {C['page']};
}}
header[data-testid="stHeader"] {{ background: transparent; }}
/* tắt hiệu ứng làm mờ khi Streamlit vẽ lại mỗi 0.5 s -> không nhấp nháy */
[data-stale="true"], .stale-element, [data-testid="stElementContainer"][data-stale="true"] {{
  opacity: 1 !important; filter: none !important; transition: none !important; }}
div[data-testid="stStatusWidget"] {{ visibility: hidden; }}
.block-container {{ padding-top: 1.2rem; padding-bottom: 1rem; max-width: 1680px; }}
section[data-testid="stSidebar"] {{ background: {C['panel']}; border-right: 1px solid {C['border']}; }}
div[data-testid="stVerticalBlockBorderWrapper"] > div {{ }}
div[data-testid="stPlotlyChart"] {{ background: {C['panel']}; border: 1px solid {C['border']};
  border-radius: 14px; padding: 6px 6px 2px; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
.stTabs [data-baseweb="tab"] {{ background: {C['panel']}; border-radius: 8px 8px 0 0; padding: 6px 14px;
  color: {C['text2']}; }}
.stTabs [aria-selected="true"] {{ color: {C['text']}; }}

.lsm-header {{ display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap;
  padding: 14px 18px; border-radius: 16px; border: 1px solid {C['border']};
  background: linear-gradient(120deg, rgba(124,108,242,0.16), rgba(51,194,232,0.06) 55%, rgba(17,21,31,0.6));
  margin-bottom: 14px; }}
.lsm-title {{ font-size: 22px; font-weight: 700; color: {C['text']}; letter-spacing: -0.01em; }}
.lsm-title span {{ color: {C['text2']}; font-weight: 500; }}
.lsm-sub {{ color: {C['muted']}; font-size: 12.5px; margin-top: 2px; }}
.lsm-right {{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; }}
.pill {{ display:inline-flex; align-items:center; gap:7px; padding: 5px 11px; border-radius: 999px;
  font-size: 12px; font-weight: 600; border: 1px solid {C['border']}; background: {C['panel2']}; color: {C['text']}; }}
.dot {{ width:8px; height:8px; border-radius:50%; display:inline-block; }}
.dot.pulse {{ animation: pulse 1.6s infinite; }}
@keyframes pulse {{ 0% {{ box-shadow: 0 0 0 0 rgba(24,178,92,0.6); }} 70% {{ box-shadow: 0 0 0 8px rgba(24,178,92,0); }}
  100% {{ box-shadow: 0 0 0 0 rgba(24,178,92,0); }} }}
.clock {{ font-family: {MONO}; color: {C['text2']}; font-size: 12.5px; }}

.cards {{ display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 10px; margin-bottom: 10px; }}
@media (max-width: 1100px) {{ .cards {{ grid-template-columns: repeat(2, minmax(0,1fr)); }} }}
.card {{ background: {C['panel']}; border: 1px solid {C['border']}; border-radius: 14px; padding: 12px 14px;
  position: relative; overflow: hidden; }}
.card::before {{ content:""; position:absolute; inset:0 auto 0 0; width:3px; background: var(--tone); opacity:.9; }}
.card .lbl {{ color: {C['muted']}; font-size: 11.5px; text-transform: uppercase; letter-spacing: .06em; font-weight: 600; }}
.card .val {{ color: {C['text']}; font-size: 22px; font-weight: 700; margin-top: 4px; font-family: {MONO};
  letter-spacing: -0.02em; white-space: nowrap; }}
.card .val small {{ font-size: 13px; color: {C['text2']}; font-weight: 500; }}
.card .sub {{ color: {C['text2']}; font-size: 12px; margin-top: 2px; white-space: nowrap; overflow:hidden; text-overflow: ellipsis; }}
.card .sub b {{ color: var(--tone); font-weight: 600; }}

.panel {{ background: {C['panel']}; border: 1px solid {C['border']}; border-radius: 14px; padding: 14px 16px;
  margin-bottom: 10px; }}
.panel .ph {{ margin: 0 0 10px; font-size: 13.5px; color: {C['text']}; font-weight: 600; align-items: center; }}
.panel .hint {{ color: {C['muted']}; font-size: 11.5px; }}

.prob-row {{ display:grid; grid-template-columns: 92px 1fr 58px; align-items:center; gap: 10px; margin: 8px 0; }}
.prob-row .name {{ color: {C['text2']}; font-size: 13px; font-weight: 500; }}
.prob-row .track {{ height: 12px; background: {C['panel2']}; border-radius: 999px; overflow: hidden;
  border: 1px solid {C['border']}; }}
.prob-row .fill {{ height: 100%; border-radius: 999px; transition: width .35s ease; }}
.prob-row .pct {{ text-align:right; font-family: {MONO}; color: {C['text']}; font-size: 13px; }}
.verdict {{ margin-top: 10px; padding: 9px 12px; border-radius: 10px; background: {C['panel2']};
  color: {C['text2']}; font-size: 12.5px; border: 1px solid {C['border']}; }}
.verdict b {{ color: {C['text']}; }}

.level {{ display:flex; align-items:center; gap:10px; padding: 10px 12px; border-radius: 12px;
  border: 1px solid var(--tone); background: var(--tone-bg); margin-top: -6px; }}
.level .ic {{ width: 26px; height: 26px; border-radius: 50%; display:flex; align-items:center; justify-content:center;
  background: var(--tone); color: #0A0D14; font-weight: 800; font-size: 14px; flex: none; }}
.level .t1 {{ color: {C['text']}; font-weight: 700; font-size: 14px; }}
.level .t2 {{ color: {C['text2']}; font-size: 12px; }}

.alert {{ border-radius: 14px; padding: 12px 14px; margin-bottom: 10px; border: 1px solid var(--tone);
  background: linear-gradient(120deg, var(--tone-bg), rgba(17,21,31,0.85)); }}
.alert .head {{ display:flex; align-items:center; gap: 8px; color: {C['text']}; font-weight: 700; font-size: 13.5px; }}
.alert .badge {{ background: var(--tone); color: #0A0D14; border-radius: 6px; padding: 1px 7px; font-size: 11px; font-weight: 800; }}
.alert .grid {{ display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 8px; margin-top: 10px; }}
.alert .k {{ color: {C['muted']}; font-size: 11px; text-transform: uppercase; letter-spacing: .05em; }}
.alert .v {{ color: {C['text']}; font-family: {MONO}; font-size: 16px; font-weight: 600; }}
.alert .msg {{ color: {C['text2']}; font-size: 12.5px; margin-top: 8px; }}

.ladder {{ font-family: {MONO}; font-size: 11px; }}
.lrow {{ display:grid; grid-template-columns: 30px 92px 1fr 74px; align-items:center; gap: 8px; height: 12.5px; }}
.lrow .lv {{ color: {C['muted']}; font-size: 10px; }}
.lrow .lp {{ text-align: right; }}
.lrow .lt {{ position: relative; height: 9px; }}
.lrow .lt i {{ position:absolute; left:0; top:0; bottom:0; border-radius: 0 3px 3px 0; transition: width .3s ease; }}
.lrow .lq {{ text-align:right; color: {C['text2']}; }}
.lmid {{ display:flex; justify-content:space-between; margin: 5px 0; padding: 5px 10px; border-radius: 8px;
  background: {C['panel2']}; border: 1px solid {C['border']}; color: {C['text2']}; font-size: 11.5px; }}
.lmid b {{ color: {C['text']}; }}
.lleg {{ font-weight: 400; color: {C['text2']}; font-size: 11.5px; display:flex; align-items:center; gap:6px; }}
.lleg i {{ width: 10px; height: 10px; border-radius: 2px; display:inline-block; margin-left: 8px; }}
.score {{ display:flex; gap: 14px; flex-wrap: wrap; color: {C['text2']}; font-size: 12.5px; margin: 2px 2px 8px; }}
.score b {{ color: {C['text']}; font-family: {MONO}; }}
.warmup {{ margin: 0 0 12px; }}

.note {{ color: {C['muted']}; font-size: 12px; line-height: 1.5; margin: -2px 4px 12px; padding: 6px 10px;
  border-left: 2px solid {C['axis']}; background: rgba(255,255,255,0.015); border-radius: 0 8px 8px 0; }}
.note b {{ color: {C['text2']}; font-weight: 600; }}
.gtag {{ font-size: 11.5px; font-weight: 700; letter-spacing: .08em; margin: 4px 0 6px; }}

.lb-row {{ display:grid; grid-template-columns: 34px 1.4fr 1.9fr 0.7fr 1.25fr 1fr; align-items:center; gap: 12px;
  padding: 10px 8px; border-bottom: 1px solid {C['border']}; position: relative; }}
.lb-row:not(.lb-head)::before {{ content:""; position:absolute; left:0; top:8px; bottom:8px; width:3px;
  border-radius: 2px; background: var(--tone); }}
.lb-head {{ color: {C['muted']}; font-size: 11px; text-transform: uppercase; letter-spacing: .05em; font-weight: 600;
  padding-top: 0; }}
.lb-rank {{ font-family: {MONO}; color: {C['text2']}; font-size: 15px; text-align: center; }}
.lb-row b {{ color: {C['text']}; font-size: 14px; }}
.lb-grp {{ color: {C['muted']}; font-size: 11.5px; }}
.lb-badge {{ display:inline-block; margin-left: 6px; padding: 1px 7px; border-radius: 999px; font-size: 10.5px;
  font-weight: 700; color: {C['page']}; background: {C['series1']}; vertical-align: 2px; }}
.lb-bar {{ display:inline-block; width: calc(100% - 130px); height: 9px; background: {C['panel2']};
  border-radius: 999px; overflow: hidden; vertical-align: middle; border: 1px solid {C['border']}; }}
.lb-bar i {{ display:block; height:100%; background: var(--tone); border-radius: 999px; }}
.lb-num {{ font-family: {MONO}; color: {C['text']}; font-size: 13px; margin-left: 6px; }}
.lb-num small {{ color: {C['muted']}; font-size: 11px; }}
.lb-tag {{ color: var(--tone); font-size: 12.5px; font-weight: 600; }}
.stTabs [data-baseweb="tab"] p {{ font-size: 14.5px; font-weight: 600; }}
</style>
"""
