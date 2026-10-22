"""Thanh tiêu đề + 4 thẻ chỉ số vi cấu trúc (Mid, Spread, OBI, OFI)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .theme import C

SOURCE_PILL = {"mock": ("GIẢ LẬP (MOCK)", C["warning"]), "live": ("LIVE · BINANCE", C["good"]),
               "replay": ("REPLAY LỊCH SỬ", C["accent"])}


def header_html(snap: dict) -> str:
    sym = snap["symbol"]
    pair = f"{sym[:-4]}/{sym[-4:]}" if sym.endswith("USDT") else sym
    label, col = SOURCE_PILL[snap["source"]]
    st = snap["status"]
    pulse = " pulse" if snap["source"] == "live" and st["ok"] else ""
    st_col = C["good"] if st["ok"] else C["warning"]
    ts = snap.get("ts_ms")
    clock = pd.Timestamp(ts, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC") if ts else "--"
    sample = ""
    if snap.get("in_sample") is True:
        sample = f"<span class='pill' style='color:{C['warning']}'>Ngày TRONG mẫu train</span>"
    elif snap["source"] == "replay":
        sample = f"<span class='pill' style='color:{C['good']}'>Ngày NGOÀI mẫu</span>"
    return (f"<div class='lsm-header'><div><div class='lsm-title'>Liquidity Stress Meter <span>· {pair}</span></div>"
            f"<div class='lsm-sub'>Vi cấu trúc sổ lệnh 20 tầng · dự báo hướng giá & rủi ro thanh khoản "
            f"{snap.get('horizon_s', 10)} giây tới</div></div><div class='lsm-right'>"
            f"<span class='pill'><span class='dot' style='background:{col}'></span>{label}</span>"
            f"<span class='pill'><span class='dot{pulse}' style='background:{st_col}'></span>{st['text']}</span>"
            f"{sample}<span class='clock'>{clock}</span></div></div>")


def _card(label: str, value: str, sub: str, tone: str) -> str:
    return (f"<div class='card' style='--tone:{tone}'><div class='lbl'>{label}</div>"
            f"<div class='val'>{value}</div><div class='sub'>{sub}</div></div>")


def _side(v: float, pos: str, neg: str, eps: float = 0.05) -> tuple[str, str]:
    if not np.isfinite(v) or abs(v) < eps:
        return "cân bằng", C["text2"]
    return (pos, C["bid"]) if v > 0 else (neg, C["ask"])


def cards_html(snap: dict) -> str:
    hist, mid = snap["hist"], snap["mid"]
    last = hist.iloc[-1]
    first_mid = hist["mid"].dropna().iloc[0] if hist["mid"].notna().any() else mid
    chg = (mid / first_mid - 1) * 1e4
    win_min = max(1, round(len(hist) / 60))
    chg_col = C["bid"] if chg > 0 else (C["ask"] if chg < 0 else C["text2"])
    spread = (snap["asks"][0, 0] - snap["bids"][0, 0])
    spread_bps = spread / mid * 1e4
    med = float(np.nanmedian(hist["spread_bps"]))
    sp_col = C["critical"] if spread_bps > 2 * med else (C["warning"] if spread_bps > 1.3 * med else C["series1"])
    obi1, obi5 = float(last["obi_1"]), float(last["obi_5"])
    obi_w, obi_c = _side(obi5, "nghiêng MUA", "nghiêng BÁN")
    ofi, ofi1 = float(last["mlofi_sum_10s"]), float(last["ofi1_10s"])
    ofi_w, ofi_c = _side(ofi, "áp lực MUA", "áp lực BÁN", 0.1)
    if not np.isfinite(ofi):
        ofi_w = "đang khởi động (cần ~1 phút)"
    return "<div class='cards'>" + "".join([
        _card("Giá mid", f"{mid:,.2f}", f"<b style='color:{chg_col}'>{chg:+.2f} bps</b> / {win_min} phút",
              C["series1"]),
        _card("Spread", f"{spread_bps:.3f}<small> bps</small>", f"{spread:,.2f} USDT · TV {_f(med, '.3f')}", sp_col),
        _card("OBI · 5 tầng", _f(obi5, "+.3f"), f"<b>{obi_w}</b> · L1 {_f(obi1, '+.2f')}", obi_c),
        _card("MLOFI · 10 giây", _f(ofi, "+.2f"), f"<b>{ofi_w}</b> · L1 {_f(ofi1, '+.2f')}", ofi_c),
    ]) + "</div>"


def _f(v: float, spec: str) -> str:
    return format(v, spec) if np.isfinite(v) else "–"
