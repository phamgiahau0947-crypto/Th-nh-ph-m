"""Tiện ích dùng chung cho Task 2: đường dẫn, cấu hình, đọc dữ liệu Task 1, chia walk-forward."""
from __future__ import annotations

import datetime as dt
import logging
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
TASK2_DIR = Path(__file__).resolve().parent
RAW_STORE = ROOT / "data" / "raw" / "raw_lob_trades.parquet"
FEATURE_DIR = ROOT / "data" / "processed" / "task2_features"
MODEL_DIR = ROOT / "models"
TABLE_DIR = ROOT / "outputs" / "tables"
FIG_DIR = ROOT / "outputs" / "figures"


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(TASK2_DIR / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_logger(name: str = "task2") -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
    return log


def ensure_dirs() -> None:
    for d in (FEATURE_DIR, MODEL_DIR, TABLE_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def lob_days(symbol: str, kinds: tuple[str, ...] | None = None) -> list[tuple[dt.date, Path]]:
    """Các ngày có sổ lệnh trong kho Task 1, sắp theo thời gian.
    kinds: ("seed",) = 12 ngày Tardis cố định (mặc định – số liệu báo cáo không đổi khi có ngày tự thu);
           ("seed", "rolling") = thêm các ngày tự thu bằng WebSocket. Đổi trong config.yaml: data_kinds."""
    kinds = tuple(kinds or load_config().get("data_kinds", ["seed"]))
    out = []
    for f in RAW_STORE.glob(f"{symbol}_*.parquet"):
        parts = f.stem.split("_")          # SYMBOL_YYYY-MM-DD_kind
        if parts[-1] in kinds:
            out.append((dt.date.fromisoformat(parts[1]), f))
    return sorted(out)


def read_day(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def walk_forward_split(days: list, n_folds: int, test_days: int) -> tuple[list[tuple[list, list]], list, list]:
    """Chia theo NGÀY, không xáo trộn.
    - test = `test_days` ngày cuối (untouched).
    - dev = phần còn lại; fold k: train = dev[:n0+k], val = dev[n0+k] (cửa sổ mở rộng dần).
    Trả về (folds, dev, test)."""
    days = list(days)
    test = days[-test_days:] if test_days else []
    dev = days[: len(days) - len(test)]
    n0 = len(dev) - n_folds
    if n0 < 1:
        raise ValueError(f"Không đủ ngày cho {n_folds} fold (dev có {len(dev)} ngày)")
    folds = [(dev[: n0 + k], [dev[n0 + k]]) for k in range(n_folds)]
    return folds, dev, test
