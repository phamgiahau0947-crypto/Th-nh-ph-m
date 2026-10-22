"""TASK 3 – Streamlit Dashboard 'Liquidity Stress Meter'.

Chạy:  streamlit run src/task3_dashboard/app.py      (hoặc bấm src/task3_dashboard/CHAY_TASK3.bat)

3 nguồn dữ liệu (nút chuyển ở thanh bên):
  - Giả lập (Mock)          : mock_streamer.py, không cần Task 1/Task 2.
  - Live WebSocket Binance  : BinanceLOBStreamer (Task 1) + ModelInference (Task 2), dự báo mỗi giây.
  - Replay lịch sử          : phát lại 1 ngày trong kho Task 1 với dự báo của mô hình Task 2.
"""
from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.task3_dashboard.components import cards, depth_heatmap as dh, guide, return_monitor as rm, stress_meter as sm  # noqa: E402
from src.task3_dashboard.components import leaderboard as lb, spread_monitor as spm  # noqa: E402
from src.task3_dashboard.components.theme import C, CSS, PLOTLY_CONFIG  # noqa: E402
from src.task3_dashboard.data_sources import LiveSource, MockSource, ReplaySource, replay_days  # noqa: E402
from src.task3_dashboard.lob_utils import load_config  # noqa: E402

st.set_page_config(page_title="Liquidity Stress Meter", page_icon="📊", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)
CFG = load_config()

SOURCES = {"Giả lập (Mock)": "mock", "Live WebSocket Binance": "live", "Replay lịch sử": "replay"}


# ----------------------------------------------------------------------------- nguồn dữ liệu (dùng chung)
@st.cache_resource(max_entries=4, show_spinner="Đang khởi động nguồn dữ liệu…")
def get_stream_source(kind: str, symbol: str):
    return MockSource(symbol) if kind == "mock" else LiveSource(symbol)


@st.cache_resource(max_entries=2, show_spinner="Đang nạp ngày lịch sử và chạy mô hình cho cả ngày (~10–30 giây)…")
def get_replay_source(symbol: str, day: str):
    return ReplaySource(symbol, dt.date.fromisoformat(day))


@st.cache_data(ttl=600)
def list_days(symbol: str):
    return [(str(d), k) for d, k in replay_days(symbol)]


def plot(fig, key: str):
    st.plotly_chart(fig, key=key, config=PLOTLY_CONFIG, theme=None, width="stretch")


# ----------------------------------------------------------------------------- thanh bên
with st.sidebar:
    st.markdown(f"<div style='font-size:18px;font-weight:700;color:{C['text']};margin-bottom:2px'>⚙️ Điều khiển</div>"
                f"<div style='color:{C['muted']};font-size:12px;margin-bottom:10px'>Liquidity Stress Meter · Task 3</div>",
                unsafe_allow_html=True)
    # cho phép mở thẳng 1 chế độ qua link, vd: http://localhost:8501/?source=live
    default_kind = st.query_params.get("source", CFG["default_source"])
    default_label = next((k for k, v in SOURCES.items() if v == default_kind),
                         next(k for k, v in SOURCES.items() if v == CFG["default_source"]))
    src_label = st.radio("Nguồn dữ liệu", list(SOURCES), index=list(SOURCES).index(default_label))
    kind = SOURCES[src_label]
    symbol = st.selectbox("Cặp giao dịch", CFG["symbols"], format_func=lambda s: f"{s[:-4]}/{s[-4:]}")
    refresh = st.select_slider("Tốc độ làm mới", CFG["refresh_options_s"], value=CFG["refresh_s"],
                               format_func=lambda v: f"{v:g} s")
    window_min = st.select_slider("Cửa sổ biểu đồ", CFG["window_options_min"], value=CFG["window_min"],
                                  format_func=lambda v: f"{v} phút")

    replay = None
    if kind == "replay":
        st.divider()
        days = list_days(symbol)
        if not days:
            st.error("Chưa có ngày nào trong kho Task 1 (data/raw/raw_lob_trades.parquet).")
        else:
            from src.task2_models.predictor import ModelInference
            train_days = set(ModelInference("models/", symbol).direction[symbol]["train_days"])
            labels = {d: f"{d} · {'tự thu' if k == 'rolling' else 'Tardis'} · "
                         f"{'trong mẫu' if d in train_days else 'NGOÀI mẫu'}" for d, k in days}
            # mặc định: ngày Tardis ngoài mẫu mới nhất (đủ 24h); ngày tự thu thường chỉ vài giờ
            outs = ([d for d, k in days if d not in train_days and k == "seed"]
                    or [d for d, _ in days if d not in train_days])
            day = st.selectbox("Ngày phát lại", [d for d, _ in days][::-1], format_func=labels.get,
                               index=[d for d, _ in days][::-1].index(outs[-1]) if outs else 0,
                               help="Ngày NGOÀI mẫu = mô hình chưa từng thấy khi train → dự báo trung thực.")
            start = st.slider("Bắt đầu từ (giờ UTC)", value=dt.time(14, 0), min_value=dt.time(0, 0),
                              max_value=dt.time(23, 45), step=dt.timedelta(minutes=15), format="HH:mm")
            speed = st.select_slider("Tốc độ phát", CFG["replay_speeds"], value=5, format_func=lambda v: f"×{v}")
            c1, c2 = st.columns(2)
            play_clicked = c1.button("⏯ Phát / Dừng", width="stretch")
            reset_clicked = c2.button("⏮ Về đầu", width="stretch")
            replay = {"day": day, "start": start, "speed": speed, "play": play_clicked, "reset": reset_clicked}

    st.divider()
    show_notes = st.toggle("Hiện chú thích dưới từng khối", value=True,
                           help="Mỗi khối có 1 dòng 'Cách đọc' ngắn. Hướng dẫn đầy đủ ở nút 📖 phía trên.")
    st.caption("📖 Hướng dẫn sử dụng & cách đọc kết quả: nút **Hướng dẫn** ở đầu trang.")


def note(key: str, **kw):
    if show_notes:
        st.markdown(guide.note_html(key, **kw), unsafe_allow_html=True)


# ----------------------------------------------------------------------------- đồng hồ phát lại
def replay_cursor(src: ReplaySource, rp: dict) -> tuple[int, bool]:
    key = (symbol, rp["day"], rp["start"])
    s = st.session_state.get("rp")
    now = time.time()
    if s is None or s["key"] != key or rp["reset"]:
        start_ms = int(dt.datetime.combine(dt.date.fromisoformat(rp["day"]), rp["start"],
                                           tzinfo=dt.timezone.utc).timestamp() * 1000)
        start_ms = min(max(start_ms, src.t_min + window_min * 60_000), src.t_max)
        s = {"key": key, "anchor_ms": start_ms, "anchor_wall": now, "playing": True, "speed": rp["speed"]}
        rp["reset"] = False                                 # các lần làm mới sau (fragment) không reset lại
    cur = s["anchor_ms"] + ((now - s["anchor_wall"]) * 1000 * s["speed"] if s["playing"] else 0)
    if rp["play"] or s["speed"] != rp["speed"]:            # đổi trạng thái -> neo lại tại vị trí hiện tại
        s.update(anchor_ms=cur, anchor_wall=now, speed=rp["speed"], playing=s["playing"] ^ bool(rp["play"]))
        rp["play"] = False
    if cur >= src.t_max:
        cur, s["playing"] = src.t_max, False
        s.update(anchor_ms=cur, anchor_wall=now)
    st.session_state["rp"] = s
    return int(cur), s["playing"]


# ----------------------------------------------------------------------------- giao diện chính
def _thresholds(snap: dict) -> tuple[float, float, float]:
    src_obj = snap.get("_src")
    return (float(getattr(src_obj, "warn", CFG["stress_warning_threshold"])),
            float(getattr(src_obj, "caution", CFG["stress_caution_threshold"])),
            float(getattr(src_obj, "ratio_warn", 1.5)))


def render_fast(snap: dict):
    """Phần nhảy số theo chu kỳ làm mới (0.25–2 s): tiêu đề, thẻ chỉ số, sổ lệnh, đồng hồ, xác suất, cảnh báo."""
    st.markdown(cards.header_html(snap), unsafe_allow_html=True)
    stt = snap["status"]
    if stt.get("warmup") is not None and stt["warmup"] < 1:
        n = int(stt["warmup"] * LiveSource.WARMUP_S)
        st.progress(stt["warmup"], text=f"Đang tích lũy dữ liệu: {n}/{LiveSource.WARMUP_S} giây – dự báo ổn định "
                                        f"sau ~10 phút (các chỉ số chuẩn hóa theo 10 phút gần nhất)")
    if snap.get("empty"):
        st.info("Đang chờ dữ liệu đầu tiên… " + " · ".join(stt["details"][:1]))
        return

    pred = snap["pred"]
    h = int(snap.get("horizon_s", 10))
    warn, caution, ratio_warn = _thresholds(snap)
    left, right = st.columns([1.18, 1], gap="medium")
    with left:
        st.markdown(cards.cards_html(snap), unsafe_allow_html=True)
        note("cards")
        view = st.segmented_control("Sổ lệnh", ["Bậc thang 20 tầng", "Depth curve"], default="Bậc thang 20 tầng",
                                    key="book_view", label_visibility="collapsed")
        if view == "Depth curve":
            plot(dh.depth_curve_figure(snap["bids"], snap["asks"], height=560), "depth_curve")
            note("depth")
        else:
            st.markdown(dh.ladder_html(snap["bids"], snap["asks"]), unsafe_allow_html=True)
            note("ladder")
        notional = float(CFG["notional_usdt"].get(snap["symbol"], 10000))
        st.markdown(spm.spread_forecast_html(pred, snap["hist"], notional, h), unsafe_allow_html=True)
        note("spread", h=h)
    with right:
        st.markdown(sm.gauge_html(pred["p_stress"], warn, caution,
                                  title=f"Liquidity Stress Meter · P(stress) {h} giây tới"), unsafe_allow_html=True)
        st.markdown(sm.level_html(pred["p_stress"], warn, caution, CFG["stress_base_rate"],
                                  pred.get("stress_ratio", float("nan")), ratio_warn), unsafe_allow_html=True)
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        note("gauge", warn=warn, caution=caution, h=h)
        st.markdown(sm.prob_bars_html(pred, h), unsafe_allow_html=True)
        note("probs", h=h)
        st.markdown(sm.slippage_alert_html(snap, notional), unsafe_allow_html=True)
        note("slippage")


def render_slow(snap: dict):
    """Biểu đồ lịch sử (nặng) – vẽ lại thưa hơn để giao diện mượt."""
    if snap.get("empty"):
        return
    pred, hist = snap["pred"], snap["hist"]
    h = int(snap.get("horizon_s", 10))
    warn, caution, _ = _thresholds(snap)
    stt = snap["status"]
    left, right = st.columns([1.18, 1], gap="medium")
    with left:
        plot(dh.mid_price_figure(hist, height=160), "mid")
        note("mid")
        plot(dh.lob_heatmap_figure(hist, CFG["heatmap_max_cols"], height=330), "heat")
        note("heat")
        plot(spm.spread_figure(hist, h, height=210), "spread")
        note("spread_chart", h=h)
    with right:
        plot(rm.return_figure(hist, h, height=220), "returns")
        hr = rm.hit_rate(hist, h, snap.get("delta_bps", 0.0))
        if hr["n"]:
            st.markdown(f"<div class='score'>Trong cửa sổ: đoán đúng hướng <b>{hr['acc']:.0%}</b> "
                        f"({hr['n']:,} giây) · khi đoán Tăng/Giảm: <b>{hr['acc_move']:.0%}</b> · "
                        f"đoán bừa ≈ 33%</div>", unsafe_allow_html=True)
        note("returns", h=h)
        plot(rm.hit_figure(hist, h, snap.get("delta_bps", 0.0), height=200), "hits")
        note("hits", h=h)
        plot(sm.stress_timeline_figure(hist, warn, caution, height=230), "stress_tl")
        note("stress_tl")

    with st.expander("🔌 Chi tiết nguồn dữ liệu & mô hình"):
        st.markdown("\n".join(f"- {d}" for d in stt["details"]))
        if snap["source"] != "mock":
            st.markdown(f"- Mô hình hướng giá: **{pred.get('model_direction')}** · mô hình stress: "
                        f"**{pred.get('model_stress')}** · δ = {snap.get('delta_bps', 0):.3f} bps")


def get_snapshot() -> dict | None:
    window_s = int(window_min * 60)
    if kind in ("mock", "live"):
        try:
            src = get_stream_source(kind, symbol)
        except Exception as e:
            st.error(f"Không khởi động được nguồn {src_label}: {e}")
            return None
        snap = src.snapshot(window_s)
    else:
        if replay is None:
            return None
        src = get_replay_source(symbol, replay["day"])
        cur, playing = replay_cursor(src, replay)
        snap = src.snapshot(window_s, cur, replay["speed"], playing)
    if snap is not None:
        snap["_src"] = src
    return snap


@st.fragment(run_every=refresh)
def fast_panel():
    snap = get_snapshot()
    if snap is None:
        st.info("Đang chuẩn bị dữ liệu…")
        return
    render_fast(snap)


@st.fragment(run_every=max(2.0, refresh))
def slow_panel():
    snap = get_snapshot()
    if snap is not None:
        render_slow(snap)


LB_WINDOWS = {"5 phút": 300, "15 phút": 900, "30 phút": 1800, "60 phút": 3600, "Toàn bộ": None}


@st.fragment(run_every=max(2.0, refresh))
def leaderboard_panel(window_s: int | None):
    """Trang 🏆: mọi mô hình hướng giá chạy song song trên cùng luồng dữ liệu, chấm khi đã có đáp án."""
    if kind == "mock":
        st.info("Chế độ Giả lập không có mô hình thật. Chọn **Live WebSocket Binance** hoặc **Replay lịch sử** "
                "ở thanh bên để xem bảng xếp hạng.")
        return
    if kind == "live":
        src = get_stream_source(kind, symbol)
        t_end = int(time.time() * 1000)
    else:
        if replay is None:
            return
        src = get_replay_source(symbol, replay["day"])
        t_end, _ = replay_cursor(src, replay)
        if getattr(src, "_zoo_df", None) is None:
            with st.spinner("Lần đầu mở ngày này: đang chạy cả 8 mô hình cho cả ngày (~1–3 phút, sau đó lưu lại)…"):
                src.zoo_frame()
    z, info = src.zoo_frame()
    if info.get("error"):
        st.warning(f"Chưa chạy được bảng xếp hạng: {info['error']}. Bấm **src\\task2_models\\CHAY_BO_SUNG.bat** "
                   f"một lần rồi mở lại dashboard.")
        return
    if z.empty:
        st.info("Đang chờ dữ liệu đầu tiên…")
        return
    if kind == "live":
        t_end = int(z.index.max())                     # đồng hồ theo dữ liệu đã nhận (không theo máy)
    h = int(CFG["horizon_s"])
    df, sel = lb.score(z, info["names"], h, info["delta_bps"], t_end, window_s)
    t = pd.Timestamp(t_end, unit="ms", tz="UTC")
    st.markdown(f"<div class='lsm-header'><div><div class='lsm-title'>So sánh mô hình <span>· "
                f"{symbol[:-4]}/{symbol[-4:]} · {src_label}</span></div><div class='lsm-sub'>{len(info['names'])} mô "
                f"hình cùng đoán hướng giá {h} giây tới trên cùng dữ liệu · chấm đến {t:%Y-%m-%d %H:%M:%S} UTC</div>"
                f"</div></div>", unsafe_allow_html=True)
    st.markdown(lb.table_html(df, info["dashboard_model"], h), unsafe_allow_html=True)
    note("leaderboard")
    if len(df) and df["n"].max() >= lb.MIN_SECONDS:
        plot(lb.rolling_figure(z, sel, info["names"], h, info["delta_bps"], info["dashboard_model"]), "lb_roll")
        note("lb_chart")


# Chọn trang ở phía server (không dùng st.tabs): trang ẩn không chạy fragment làm mới -> nhẹ máy khi đọc hướng dẫn
PAGES = {"📊 Dashboard": "dash", "🏆 So sánh mô hình": "models", "📖 Hướng dẫn & cách đọc kết quả": "guide"}
default_page = next((k for k, v in PAGES.items() if v == st.query_params.get("page")), "📊 Dashboard")
page = st.segmented_control("Trang", list(PAGES), default=default_page, key="page", label_visibility="collapsed")
if PAGES.get(page or "📊 Dashboard") == "guide":
    guide.guide_page(float(CFG["stress_warning_threshold"]), float(CFG["stress_caution_threshold"]),
                     CFG["notional_usdt"], int(CFG["horizon_s"]))
elif PAGES.get(page or "📊 Dashboard") == "models":
    win = st.segmented_control("Chấm trên", list(LB_WINDOWS), default="15 phút", key="lb_window")
    leaderboard_panel(LB_WINDOWS[win or "15 phút"])
else:
    fast_panel()
    slow_panel()
