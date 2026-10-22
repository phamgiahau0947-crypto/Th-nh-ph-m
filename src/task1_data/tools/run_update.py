"""Chạy update_store không phụ thuộc thư mục hiện hành (dùng cho lịch tự động của Windows).
Ghi nhật ký vào data/processed/update_log.txt khi chạy ngầm."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    if not sys.stdout or not sys.stdout.isatty():
        log_dir = ROOT / "data" / "processed"
        log_dir.mkdir(parents=True, exist_ok=True)
        f = open(log_dir / "update_log.txt", "a", encoding="utf-8")
        sys.stdout = sys.stderr = f
    from src.task1_data.update_store import main
    main(sys.argv[1:])
