"""Task 1 – Data Streamer Pipeline (LOB & Trades, Binance BTCUSDT).

Lối vào chính:
    from src.task1_data.websocket_streamer import BinanceLOBStreamer   # real-time
    from src.task1_data import load_raw_lob_trades                      # đọc kho nền lịch sử
"""


def load_raw_lob_trades(*args, **kwargs):
    """Đọc kho nền thành 1 DataFrame. Xem core/store.py để biết tham số."""
    from .core.store import load_raw_lob_trades as _f
    return _f(*args, **kwargs)
