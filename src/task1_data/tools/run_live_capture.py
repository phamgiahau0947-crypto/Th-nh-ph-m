"""Thu sổ lệnh live (WebSocket) cho mọi cặp trong config, chạy ngầm – dùng cho lịch "tự thu khi mở máy".

- 1 tiến trình cho cả BTCUSDT + ETHUSDT; chỉ cho phép 1 bản chạy (khóa cổng localhost).
- Máy ngủ/mất mạng: streamer tự kết nối lại; khoảng đứt được đánh dấu gap khi nhập kho (không bịa dữ liệu).
- Dữ liệu ghi vào data/raw/<SYMBOL>/live_capture/YYYY-MM-DD/ mỗi 60s; lịch cập nhật 07:30 sẽ nhập ngày đã chốt vào kho.
- Nhật ký: data/processed/live_capture_log.txt
"""
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
LOCK_PORT = 47231


def main():
    if not sys.stdout or not sys.stdout.isatty():
        log_dir = ROOT / "data" / "processed"
        log_dir.mkdir(parents=True, exist_ok=True)
        f = open(log_dir / "live_capture_log.txt", "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = f
    lock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        lock.bind(("127.0.0.1", LOCK_PORT))
    except OSError:
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "Đã có 1 bản thu live đang chạy -> thoát.")
        return
    from src.task1_data.core.common import all_symbols, get_logger
    from src.task1_data.websocket_streamer import BinanceLOBStreamer
    log = get_logger()
    streamers = [BinanceLOBStreamer(symbol=s, record=True).start() for s in all_symbols()]
    log.info(f"Bắt đầu thu live: {[s.symbol for s in streamers]}")
    try:
        last = time.time()
        while True:
            time.sleep(5)
            for s in streamers:
                if not s.is_running():          # luồng chết bất thường -> khởi động lại
                    log.warning(f"{s.symbol}: streamer dừng, khởi động lại")
                    s.start()
            if time.time() - last >= 1800:      # mỗi 30 phút ghi 1 dòng tình trạng
                last = time.time()
                for s in streamers:
                    st = s.get_stats()
                    log.info(f"{s.symbol}: depth_msgs={st['depth_msgs']} reconnects={st['reconnects']} "
                             f"rows_written={s._recorder.rows_written if s._recorder else 0}")
    except KeyboardInterrupt:
        pass
    finally:
        for s in streamers:
            s.stop()
        log.info("Dừng thu live (đã ghi phần còn lại ra ổ đĩa).")


if __name__ == "__main__":
    main()
