"""Dựng bảng đặc trưng + nhãn cho từng (cặp, ngày) từ kho Task 1 và lưu cache.

    python -m src.task2_models.dataset            # dựng cache cho mọi ngày còn thiếu
    python -m src.task2_models.dataset --rebuild  # dựng lại toàn bộ

Mỗi ngày xử lý như 1 đoạn độc lập (các ngày seed không liền nhau): cửa sổ trượt khởi động lại từ đầu ngày,
nhãn không vượt sang ngày khác.
Cache: data/processed/task2_features/{SYMBOL}_{YYYY-MM-DD}.parquet (+ _stats.npz cho hiệu chỉnh mô hình cấu trúc).
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import pandas as pd

from . import avellaneda_stoikov as AS
from .common import FEATURE_DIR, ensure_dirs, get_logger, load_config, lob_days, read_day
from .cont_stoikov_talreja import CSTModel, next_move_direction
from .labeler import forward_return_bps, stress_labels
from .microstructure_math import add_structural_features, build_base_features, regrid_1s

log = get_logger()


def horizons(cfg: dict) -> list[int]:
    return sorted({int(cfg["horizon_s"]), *map(int, cfg["robust_horizons_s"])})


def build_day(raw: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    g = regrid_1s(raw)
    feat = build_base_features(g, cfg)
    feat["next_dir"] = next_move_direction(feat["mid"])     # hướng đổi giá kế tiếp (hiệu chỉnh CST)
    for h in horizons(cfg):
        feat[f"ret_fwd_{h}s"] = forward_return_bps(feat["mid"], h).to_numpy()
        lab = stress_labels(feat, h, cfg)
        for c in lab.columns:
            if c not in feat.columns:
                feat[c] = lab[c].to_numpy()
    # giá tốt nhất tại t+h cho mô phỏng PnL (14)
    h = int(cfg["horizon_s"])
    feat[f"bid1_fwd_{h}s"] = feat["bid1"].shift(-h)
    feat[f"ask1_fwd_{h}s"] = feat["ask1"].shift(-h)
    feat = feat[feat["mid"].notna()]
    num = feat.select_dtypes("float64").columns
    keep64 = ["mid", "bid1", "ask1", f"bid1_fwd_{h}s", f"ask1_fwd_{h}s"]
    feat[num.difference(keep64)] = feat[num.difference(keep64)].astype("float32")
    return feat


def day_stats(raw: pd.DataFrame) -> dict:
    """Thống kê đủ để hiệu chỉnh A-S (khoảng cách quét) và CST (tổng KL thêm/hủy/khớp) theo từng ngày."""
    g = regrid_1s(raw).reset_index()
    return {"reach": AS.reach_distances_bps(g).astype("float32"), "seconds": float(len(g))}


def cache_path(symbol: str, day: dt.date):
    return FEATURE_DIR / f"{symbol}_{day.isoformat()}.parquet"


def build_cache(symbols: list[str] | None = None, rebuild: bool = False) -> None:
    cfg = load_config()
    ensure_dirs()
    for sym in symbols or cfg["symbols"]:
        for day, f in lob_days(sym):
            out = cache_path(sym, day)
            stats = out.with_name(out.stem + "_stats.npz")
            if out.exists() and stats.exists() and not rebuild:
                continue
            raw = read_day(f)
            feat = build_day(raw, cfg)
            feat.to_parquet(out, compression="zstd")
            st = day_stats(raw)
            np.savez_compressed(stats, reach=st["reach"], seconds=st["seconds"])
            log.info(f"[dataset] {sym} {day}: {len(feat):,} dòng, {feat.shape[1]} cột -> {out.name}")


def load_symbol(symbol: str, days: list[dt.date] | None = None, columns: list[str] | None = None) -> pd.DataFrame:
    """Đọc cache của 1 cặp (các ngày chỉ định), thêm cột 'day'."""
    frames = []
    for day, _ in lob_days(symbol):
        if days is not None and day not in days:
            continue
        p = cache_path(symbol, day)
        if not p.exists():
            raise FileNotFoundError(f"{p} chưa có – chạy: python -m src.task2_models.dataset")
        d = pd.read_parquet(p, columns=columns)
        d["day"] = day
        frames.append(d)
    return pd.concat(frames) if frames else pd.DataFrame()


def calibrate_structural(symbol: str, train_days: list[dt.date], train_feat: pd.DataFrame, cfg: dict) -> dict:
    """Hiệu chỉnh A-S và CST CHỈ trên các ngày train (đọc lại kho Task 1 cho CST)."""
    reach, secs = [], 0.0
    for d in train_days:
        z = np.load(cache_path(symbol, d).with_name(f"{symbol}_{d.isoformat()}_stats.npz"))
        reach.append(z["reach"])
        secs += float(z["seconds"])
    k, A = AS.estimate_k(np.concatenate(reach), secs)
    a = cfg["avellaneda_stoikov"]
    ds = AS.optimal_spread_bps(train_feat["sigma_bps"].to_numpy(), k, a["gamma"], cfg["horizon_s"])
    norm = float(np.nanmedian(train_feat["impact_ma10_bps"].to_numpy() / ds))
    raws = [regrid_1s(read_day(f)).reset_index() for d, f in lob_days(symbol) if d in train_days]
    cst = CSTModel.estimate(raws, train_feat["q_ask1"], train_feat["q_bid1"], train_feat["next_dir"],
                            n_max=int(cfg["cst"]["max_queue_units"]), seed=cfg["random_state"])
    return {"as": {"k": k, "A": A, "gamma": a["gamma"], "horizon_s": cfg["horizon_s"], "ratio_norm": norm},
            "cst": cst.to_dict()}


def with_structural(feat: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = add_structural_features(feat.copy(), params)
    for c in ("cst_p_up", "as_delta_star_bps", "as_stress_ratio", "as_res_skew_bps"):
        out[c] = out[c].astype("float32")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Dựng cache đặc trưng Task 2")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--symbol", action="append")
    a = ap.parse_args(argv)
    build_cache(a.symbol, a.rebuild)


if __name__ == "__main__":
    main()
