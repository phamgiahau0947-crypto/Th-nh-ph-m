"""Tự kiểm tra Task 3:  python -m src.task3_dashboard.selftest [--live]

1. Cấu hình đọc được.
2. Hàm chi phí khớp lệnh (impact_bps) đúng trên sổ lệnh tự tính tay.
3. MockStreamer trả dữ liệu đúng Interface Contract Task 1 (get_latest_data).
4. MockStreamer trả dự báo giả đúng khóa của Task 2 (predict_one).
5. Lịch sử Mock đủ cột giao diện cần (HIST_COLS + KL 20 tầng).
6. Mọi khối giao diện (HTML + Plotly) dựng được từ snapshot Mock.
7. Replay: nạp 1 ngày ngoài mẫu, dự báo trùng ModelInference.predict_one, không nhìn trước tương lai.
8. Giao diện Streamlit chạy hết 1 lượt không lỗi (AppTest, nguồn Mock).
9. Bảng xếp hạng: chấm đúng (mô hình biết trước = 100%), chỉ chấm giây đã có đáp án.
10. Bộ mô hình + dự báo spread trên Replay: cột XGBoost của bảng xếp hạng trùng dự báo trên dashboard.
11. (--live) Live: kết nối Binance thật, có dự báo trong ~40 s.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.task3_dashboard.lob_utils import ASK_Q, BID_Q, HIST_COLS, impact_bps, load_config  # noqa: E402

TASK1_KEYS = {"timestamp", "mid_price", "spread", "bids", "asks", "recent_trades", "buffer_df"}
TASK2_KEYS = {"p_up", "p_flat", "p_down", "predicted_direction", "p_stress", "is_stress_warning", "stress_ratio",
              "expected_short_term_return_bps"}


def t_config():
    cfg = load_config()
    assert cfg["symbols"] and 0.1 <= cfg["refresh_s"] <= 0.5, cfg["refresh_s"]


def t_impact():
    mid = 100.0
    asks = np.array([[100.5, 10.0], [101.0, 10.0], [102.0, 100.0]])
    # 1500 USDT: 10 @100.5 (1005) + 495/101 @101  -> vwap = 1500 / (10 + 4.90099) = 100.6644
    cost, unfilled = impact_bps(asks, 1500, mid)
    assert abs(cost - (1500 / (10 + 495 / 101) - mid) / mid * 1e4) < 1e-9 and not unfilled
    cost2, unfilled2 = impact_bps(asks, 1e6, mid)
    assert unfilled2 and cost2 > cost


def t_mock_task1():
    from src.task3_dashboard.mock_streamer import MockStreamer
    s = MockStreamer("BTCUSDT", prefill_minutes=1, seed=1).start()
    time.sleep(0.5)
    d = s.get_latest_data()
    s.stop()
    assert TASK1_KEYS <= set(d), TASK1_KEYS - set(d)
    assert len(d["bids"]) == 20 and len(d["asks"]) == 20
    assert d["bids"][0][0] < d["asks"][0][0], "bid1 phải < ask1"
    assert all(d["bids"][i][0] > d["bids"][i + 1][0] for i in range(19)), "giá bid phải giảm dần"
    assert len(d["buffer_df"]) >= 500


def t_mock_task2():
    from src.task3_dashboard.mock_streamer import MockStreamer
    s = MockStreamer("ETHUSDT", prefill_minutes=1, seed=2).start()
    p = s.latest_prediction()
    s.stop()
    assert TASK2_KEYS <= set(p), TASK2_KEYS - set(p)
    assert abs(p["p_up"] + p["p_flat"] + p["p_down"] - 1) < 1e-6 and 0 <= p["p_stress"] <= 1


def t_mock_hist():
    from src.task3_dashboard.data_sources import MockSource
    m = MockSource("BTCUSDT")
    snap = m.snapshot(300)
    m.close()
    miss = set(HIST_COLS + BID_Q + ASK_Q) - set(snap["hist"].columns)
    assert not miss, miss
    assert len(snap["hist"]) == 300
    return snap


def t_components():
    from src.task3_dashboard.components import cards, depth_heatmap as dh, return_monitor as rm, stress_meter as sm
    from src.task3_dashboard.components import spread_monitor as spm
    snap = t_mock_hist()
    h = snap["hist"]
    for html in (cards.header_html(snap), cards.cards_html(snap), dh.ladder_html(snap["bids"], snap["asks"]),
                 sm.gauge_html(0.42, 0.3, 0.15), sm.prob_bars_html(snap["pred"], 10),
                 sm.level_html(0.42, 0.3, 0.15, 0.05, 1.7, 1.5), sm.slippage_alert_html(snap, 25000),
                 spm.spread_forecast_html(snap["pred"], h, 25000, 10)):
        assert isinstance(html, str) and html.startswith("<div") and "nan" not in html.lower(), html[:200]
    for fig in (dh.depth_curve_figure(snap["bids"], snap["asks"]), dh.lob_heatmap_figure(h),
                dh.mid_price_figure(h), rm.return_figure(h, 10), sm.stress_timeline_figure(h, 0.3, 0.15),
                rm.hit_figure(h, 10, 0.0), spm.spread_figure(h, 10)):
        fig.to_json()
    hr = rm.hit_rate(h, 10, 0.0)
    assert hr["n"] == len(h) - 10


def t_leaderboard():
    """Chấm điểm bảng xếp hạng: mô hình 'biết trước' đúng 100%, chỉ chấm giây đã có đáp án (không nhìn trước)."""
    import pandas as pd
    from src.task3_dashboard.components import leaderboard as lb
    rng = np.random.default_rng(0)
    ts = np.arange(1_000_000, 1_000_000 + 1200 * 1000, 1000)
    mid = 100 * np.exp(np.cumsum(rng.normal(0, 1e-4, len(ts))))
    z = pd.DataFrame({"mid": mid}, index=ts)
    y = lb.realized_labels(z, 10, 0.0)
    z["Oracle"], z["Random"] = y, rng.choice([-1, 0, 1], len(ts))
    t_end = int(ts[-1]) - 100 * 1000
    df, sel = lb.score(z, ["Oracle", "Random"], 10, 0.0, t_end, None)
    assert df.iloc[0]["model"] == "Oracle" and abs(df.iloc[0]["acc"] - 1) < 1e-9, df
    assert 0.2 < df.iloc[1]["acc"] < 0.45
    assert (z.index[sel.to_numpy()] + 10_000 <= t_end).all(), "chấm cả giây chưa có đáp án!"
    assert "Oracle" in lb.table_html(df, "Oracle")
    lb.rolling_figure(z, sel, ["Oracle", "Random"], 10, 0.0, "Oracle").to_json()


def t_zoo_spread_replay():
    """Bộ mô hình (bảng xếp hạng) + dự báo spread trên ngày Tardis ngoài mẫu:
    cột XGBoost của bảng xếp hạng phải trùng dự báo hướng trên dashboard; spread có số."""
    from src.task2_models.model_zoo import zoo_available
    from src.task2_models.predictor import ModelInference
    from src.task3_dashboard.data_sources import ReplaySource, replay_days
    sym = "BTCUSDT"
    mi = ModelInference("models/", sym)
    assert mi.spread is not None and zoo_available(), "chưa chạy src/task2_models/CHAY_BO_SUNG.bat"
    train = set(mi.direction[sym]["train_days"])
    day = [d for d, k in replay_days(sym) if str(d) not in train and k == "seed"][-1]
    src = ReplaySource(sym, day)
    z, info = src.zoo_frame()
    assert not info.get("error") and len(z) > 80_000, info
    both = src.data["pred_dir"].reindex(z.index)
    ok = both.notna()
    assert (z.loc[ok, "XGBoost"].to_numpy() == both[ok].to_numpy()).mean() > 0.999, "XGBoost ≠ dashboard"
    snap = src.snapshot(600, src.t_min + 6 * 3600 * 1000, 5, True)
    p = snap["pred"]
    assert 0 <= p["p_spread_widen"] <= 1 and np.isfinite(p["eff_spread_fwd_bps"]), p


def t_replay():
    from src.task2_models.predictor import ModelInference
    from src.task3_dashboard.data_sources import ReplaySource, replay_days
    sym = "BTCUSDT"
    mi = ModelInference("models/", sym)
    train = set(mi.direction[sym]["train_days"])
    # chỉ ngày Tardis (seed): cache đặc trưng Task 2 (dùng để đối chiếu bên dưới) không có ngày tự thu (rolling)
    outs = [d for d, k in replay_days(sym) if str(d) not in train and k == "seed"]
    assert outs, "không có ngày ngoài mẫu để phát lại"
    src = ReplaySource(sym, outs[-1])
    assert src.in_sample is False
    cursor = src.t_min + 6 * 3600 * 1000 + 500
    snap = src.snapshot(600, cursor, 5, True)
    assert snap["hist"].index.max() <= cursor, "replay nhìn trước tương lai!"
    assert len(snap["hist"]) == 600
    # dự báo replay (tính cả ngày 1 lần) phải trùng predict_one của Task 2 trên đúng dòng đó
    from src.task2_models.dataset import cache_path, load_symbol, with_structural
    if cache_path(sym, outs[-1]).exists():
        base = load_symbol(sym, [outs[-1]])
    else:       # bản bàn giao không kèm cache đặc trưng Task 2 -> tính lại từ kho Task 1 (giống ReplaySource)
        import pandas as pd
        from src.task2_models.common import load_config as load_cfg2
        from src.task2_models.microstructure_math import build_base_features, regrid_1s
        base = build_base_features(regrid_1s(pd.read_parquet(src.path)), load_cfg2(), sym)
    feat = with_structural(base, mi.structural_params(sym))
    one = mi.predict_one({**feat.loc[snap["ts_ms"]].to_dict(), "symbol": sym})
    for k in ("p_up", "p_down", "p_stress"):
        assert abs(one[k] - snap["pred"][k]) < 1e-4, (k, one[k], snap["pred"][k])


def t_app():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "src" / "task3_dashboard" / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    html = " ".join(m.value for m in at.markdown)
    assert "Liquidity Stress Meter" in html and "P(Tăng)" in html and "Live Slippage Alert" in html


def t_live():
    from src.task3_dashboard.data_sources import LiveSource
    src = LiveSource("BTCUSDT")
    try:
        t_end = time.time() + 45
        snap = None
        while time.time() < t_end:
            time.sleep(3)
            snap = src.snapshot(600)
            if snap and not snap.get("empty"):
                break
        assert snap and not snap.get("empty"), snap["status"] if snap else "không có dữ liệu"
        assert TASK2_KEYS <= set(snap["pred"])
        print("      ", " | ".join(snap["status"]["details"][1:4]))
        from src.task2_models.model_zoo import zoo_available
        if zoo_available():
            for _ in range(10):
                z, info = src.zoo_frame()
                if len(z):
                    break
                time.sleep(2)
            assert len(z) and not info.get("error"), info
            print(f"       Bảng xếp hạng: {len(info['names'])} mô hình, {len(z)} giây đã dự báo")
    finally:
        src.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="thử thêm kết nối Binance thật")
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    tests = [("Cấu hình", t_config), ("Chi phí khớp lệnh impact_bps", t_impact),
             ("Mock đúng Interface Contract Task 1", t_mock_task1), ("Mock đúng khóa dự báo Task 2", t_mock_task2),
             ("Lịch sử Mock đủ cột", t_mock_hist), ("Dựng mọi khối giao diện", t_components),
             ("Replay ngày ngoài mẫu (không nhìn trước, khớp Task 2)", t_replay),
             ("Streamlit AppTest chạy 1 lượt", t_app),
             ("Bảng xếp hạng chấm đúng, không nhìn trước", t_leaderboard),
             ("Bộ mô hình + dự báo spread trên Replay (XGBoost khớp dashboard)", t_zoo_spread_replay)]
    if a.live:
        tests.append(("Live WebSocket Binance", t_live))
    ok = 0
    for i, (name, fn) in enumerate(tests, 1):
        t0 = time.time()
        try:
            fn()
            ok += 1
            print(f"[PASS] {i}. {name} ({time.time() - t0:.1f}s)")
        except Exception as e:
            print(f"[FAIL] {i}. {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{ok}/{len(tests)} mục đạt")
    return 0 if ok == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
