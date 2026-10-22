"""CẬP NHẬT KHO NỀN – chạy bất cứ lúc nào, tự bù mọi ngày đã chốt còn thiếu.

    python -m src.task1_data.update_store --init     # lần đầu: dựng bộ nền seed + trades
    python -m src.task1_data.update_store            # các lần sau (hằng ngày / khi nhớ ra)

Mỗi lần chạy:
  1. Bộ nền seed (Tardis, ngày mùng 1): bổ sung tháng mới nếu thiếu, giữ N tháng gần nhất.
  2. Trades (Binance Data Portal): tải mọi ngày đã chốt trong khung retention còn thiếu.
  3. Ngày đã chốt có dữ liệu tự thu (live_capture) -> ghép với trades -> lưu vào kho (rolling).
  4. Xóa dữ liệu rolling cũ hơn retention_days (14/30 ngày, sửa trong config.yaml).
  5. Xuất lại inventory.csv và file CSV xem nhanh.
"""
from __future__ import annotations

import argparse
import datetime as dt

from .core.capture import load_capture_day
from .core.common import Paths, all_symbols, day_start_ms, first_days_of_months, get_logger, http_session, load_config, utc_today, DAY_MS
from .historical_downloader import build_seed_day, fetch_binance_trades_day
from .core.store import (apply_retention, build_inventory, export_csv_view, list_days, read_parquet,
                    save_day, store_summary)
from .core.transform import build_unified, grid_book

log = get_logger()


def build_rolling_day(day: dt.date, paths: Paths, trades_df) -> bool:
    cfg = load_config()
    book, cap_trades, sources = load_capture_day(day, paths)
    if book is None or book.empty:
        return False
    if trades_df is not None and not trades_df.empty:
        trades, tsrc = trades_df, "binance_vision"
    else:
        trades, tsrc = cap_trades, "binance-ws-live"
    interval = cfg["store"]["sample_interval_ms"]
    lo = day_start_ms(day)
    grid = grid_book(book, interval, cfg["store"]["max_book_staleness_ms"], lo, lo + DAY_MS)
    bsrc = "+".join(sources) or "live"
    df = build_unified(grid, trades, symbol=paths.symbol, levels=cfg["levels"], interval_ms=interval,
                       book_source=bsrc, trades_source=tsrc)
    if df.empty:
        return False
    f = save_day(df, day, "rolling", paths)
    log.info(f"[rolling] {day}: {len(df):,} dòng ({100*len(df)/(DAY_MS/interval):.1f}% ngày) -> {f.name}")
    return True


def _needs_rebuild(day: dt.date, paths: Paths, trades_available: bool) -> bool:
    existing = {d: (k, f) for d, k, f in list_days(paths)}
    if day not in existing:
        return True
    kind, f = existing[day]
    if kind == "seed":
        return False
    cap = paths.capture_dir(day)
    if cap.exists() and any(p.stat().st_mtime > f.stat().st_mtime for p in cap.glob("*.parquet")):
        return True
    if trades_available:
        src = read_parquet(f, ["trades_source"])["trades_source"]
        return len(src) > 0 and src.iloc[0] != "binance_vision"
    return False


def run(init: bool = False, seed_months: int | None = None, no_seed: bool = False,
        retention_days: int | None = None, today: dt.date | None = None, skip_trades: bool = False,
        symbols: list[str] | None = None) -> dict:
    cfg = load_config()
    syms = [s.upper() for s in (symbols or all_symbols(cfg))]
    sess = http_session()
    today = today or utc_today()
    retention = int(retention_days or cfg["store"]["retention_days"])
    # kho đã từng được dựng (bất kỳ cặp nào) -> cặp mới thêm vào config sẽ tự được dựng seed
    initialized = any(k == "seed" for s in all_symbols(cfg) for _, k, _ in list_days(Paths(symbol=s)))
    results = {}
    for sym in syms:
        paths = Paths(symbol=sym).ensure()
        results[sym] = _run_symbol(paths, sess, today, retention, init or initialized, seed_months,
                                   no_seed, skip_trades)
    try:
        inv = build_inventory()
    except Exception as e:
        inv = []
        log.warning(f"Không xuất được inventory.csv ({e})")
    log.info("===== TỔNG KẾT =====")
    for sym, r in results.items():
        log.info(r["summary"])
        if r["csv"]:
            log.info(f"  CSV xem nhanh {sym}: {r['csv']}")
    log.info(f"Thống kê từng ngày (mọi cặp): {Paths().inventory_csv}")
    return {"symbols": results, "inventory_rows": len(inv)}


def _run_symbol(paths: Paths, sess, today: dt.date, retention: int, allow_seed: bool,
                seed_months: int | None, no_seed: bool, skip_trades: bool) -> dict:
    cfg = load_config()
    yesterday = today - dt.timedelta(days=1)
    window_start = today - dt.timedelta(days=retention)
    log.info(f"===== CẬP NHẬT KHO NỀN {paths.symbol} | hôm nay (UTC) {today} | khung {retention} ngày =====")

    # 1) seed
    if not no_seed:
        months = seed_months or cfg["seed"]["months_back"]
        wanted = first_days_of_months(months, today)
        have = {d for d, k, _ in list_days(paths) if k == "seed"}
        missing = [d for d in wanted if d not in have]
        if missing and allow_seed:
            log.info(f"[seed] {paths.symbol} cần dựng: {[d.isoformat() for d in missing]}")
            for d in missing:
                try:
                    build_seed_day(d, paths, sess)
                except Exception as e:
                    log.error(f"[seed] {paths.symbol} {d}: lỗi {e}")
        elif missing:
            log.info("[seed] chưa có bộ nền -> chạy với --init để tải")
        # giữ đúng N tháng gần nhất
        for d, k, f in list_days(paths):
            if k == "seed" and d < wanted[0]:
                f.unlink(); log.info(f"[seed] {paths.symbol} bỏ ngày cũ {d}")

    # 2) + 3) từng ngày đã chốt trong khung
    built = 0
    for i in range((yesterday - window_start).days + 1):
        day = window_start + dt.timedelta(days=i)
        trades = None
        if not skip_trades:
            try:
                trades = fetch_binance_trades_day(day, paths, sess)
            except Exception as e:
                log.error(f"[trades] {paths.symbol} {day}: lỗi {e}")
        if paths.capture_dir(day).exists() and _needs_rebuild(day, paths, trades is not None):
            built += int(build_rolling_day(day, paths, trades))

    # 4) retention
    apply_retention(paths, retention, today)

    # 5) báo cáo
    try:
        csv_path = export_csv_view(paths)
    except Exception as e:   # file xem nhanh lỗi không được làm dừng cập nhật kho
        csv_path = None
        log.warning(f"{paths.symbol}: không xuất được CSV xem nhanh ({e})")
    if paths.capture_dir(today).exists():
        log.info(f"{paths.symbol}: hôm nay ({today}) có dữ liệu tự thu, sẽ nhập kho sau khi chốt ngày (07:00 giờ VN).")
    return {"built_rolling": built, "summary": store_summary(paths), "csv": csv_path}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cập nhật kho nền Task 1")
    ap.add_argument("--init", action="store_true", help="Lần đầu: tải bộ nền seed từ Tardis")
    ap.add_argument("--seed-months", type=int, default=None)
    ap.add_argument("--no-seed", action="store_true")
    ap.add_argument("--skip-trades", action="store_true", help="Không tải trades (chỉ nhập live_capture)")
    ap.add_argument("--retention", type=int, default=None, help="Ghi đè retention_days (14/30)")
    ap.add_argument("--today", default=None, help="(test) giả lập ngày hôm nay UTC, YYYY-MM-DD")
    ap.add_argument("--symbol", action="append", default=None, help="Chỉ cập nhật cặp này (lặp lại được)")
    a = ap.parse_args(argv)
    run(init=a.init, seed_months=a.seed_months, no_seed=a.no_seed, retention_days=a.retention,
        today=dt.date.fromisoformat(a.today) if a.today else None, skip_trades=a.skip_trades,
        symbols=a.symbol)


if __name__ == "__main__":
    main()
