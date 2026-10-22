# README_TASK_1 – Biên bản bàn giao Task 1: Data Streamer Pipeline

**Đề tài:** Limit Order Book & Liquidity Stress Prediction (Research Blueprint 05)
**Phạm vi Task 1:** thu thập, chuẩn hóa, lưu trữ và streaming dữ liệu sổ lệnh (LOB) + lệnh khớp (trades) của **BTC/USDT và ETH/USDT – Binance spot** (đề bài mục 3.1: ưu tiên BTC/USDT, ETH/USDT). Task 1 không làm toán, không làm ML, không làm UI.
**Thư mục:** `src/task1_data/` (code) · `data/raw/`, `data/processed/` (dữ liệu)
**Ngày bàn giao:** 20/10/2026 · **Phiên bản:** 1.2 (thêm ETH/USDT; bộ nền seed tăng lên 12 tháng)

---

## 0. Tóm tắt nhanh cho người tích hợp (AI thứ 4 / Task 2 / Task 3)

| Cần gì | Dùng cái này |
|---|---|
| Dữ liệu lịch sử để train (Task 2) | `from src.task1_data import load_raw_lob_trades` → `load_raw_lob_trades(symbol="BTCUSDT")` / `"ETHUSDT"` / `None` (cả hai). Hoặc `pd.read_parquet("data/raw/raw_lob_trades.parquet")` rồi **lọc theo cột `symbol`** |
| Dữ liệu real-time (Task 3 / tích hợp) | `from src.task1_data.websocket_streamer import BinanceLOBStreamer` → `.start()` → `.get_latest_data()` |
| Xem bằng mắt / Excel | `data/processed/BTCUSDT_view.csv`, `data/processed/ETHUSDT_view.csv` |
| Biết kho đang có những ngày nào | `data/processed/inventory.csv` |
| Nguồn gốc từng file đã tải | `data/processed/source_log.csv` |

Interface Contract trong file đặc tả **được giữ nguyên tên cột và tên hàm**. Những chỗ khác/bổ sung đều ghi ở **mục 6 – Hướng đi khác**.

---

## 1. Cài đặt & chạy (Windows, bấm đúp file .bat trong `src/task1_data/`)

**Chỉ cần bấm đúp `CHAY_TASK1.bat` một lần.** File này tự làm hết các bước:
1. Chưa cài thì tạo `.venv` ở gốc dự án, cài thư viện (`requirements.txt`) và chạy tự kiểm tra.
2. Chưa có kho thì dựng kho lần đầu (seed Tardis + trades 30 ngày, mất khoảng 30–90 phút). Có rồi thì chỉ cập nhật.
3. Đăng ký lịch Windows `Task1_CapNhatKhoLOB`: **tự cập nhật mỗi ngày lúc 07:30**, chạy ngầm. Nếu lúc đó máy tắt thì lần bật máy sau sẽ tự chạy bù. Nhật ký ghi ở `data/processed/update_log.txt`. Muốn gỡ: `schtasks /Delete /TN Task1_CapNhatKhoLOB /F`.

Bấm lại file này bất cứ lúc nào cũng an toàn (chỉ bổ sung phần thiếu).

| File phụ | Việc nó làm | Khi nào |
|---|---|---|
| `tools/THU_LIVE_khong_bat_buoc.bat` | Thu sổ lệnh live vào kho khi máy đang mở (đóng cửa sổ để dừng) | Không bắt buộc |
| `tools/BAT_TU_THU_LIVE_KHI_MO_MAY.bat` | **Tự thu live mỗi khi đăng nhập Windows**, chạy ngầm không cửa sổ, BTC + ETH trong 1 tiến trình. Máy tắt/ngủ thì tạm ngừng, mở lại tự thu tiếp. Lịch Windows `Task1_ThuLiveKhiMoMay`; nhật ký `data/processed/live_capture_log.txt` | Không bắt buộc (đang BẬT từ 20/10/2026) |
| `tools/TAT_TU_THU_LIVE.bat` | Tắt tự thu live (dữ liệu đã thu vẫn giữ) | Khi không muốn thu nữa |
| `tools/TEST_STREAMER.bat` | Chạy streamer 60s, in số liệu và **đo độ trễ** | Kiểm thử / trước khi demo |

Lệnh tương đương (chạy ở thư mục gốc dự án, sau khi đã kích hoạt `.venv`):
```bash
python -m pip install -r src/task1_data/requirements.txt
python -m src.task1_data.tools.selftest                       # tự kiểm tra, không cần Internet
python -m src.task1_data.update_store --init            # dựng kho lần đầu
python -m src.task1_data.update_store                   # cập nhật
python -m src.task1_data.websocket_streamer --seconds 30          # test streamer (lệnh test trong Phần 5 của đặc tả)
python -m src.task1_data.websocket_streamer --record --seconds 0  # thu live vào kho
python -m src.task1_data.snapshot_collector --seconds 0           # thu snapshot qua REST 200ms (phương án gốc)
python -m src.task1_data.historical_downloader trades --start 2026-10-05 --end 2026-10-18
python -m src.task1_data.historical_downloader seed --date 2026-10-01
```

**Không cần API key, không cần file `.env`.** Mọi tùy chỉnh nằm trong `src/task1_data/config.yaml`:
`retention_days` (14/30), `sample_interval_ms` (1000 = 1 giây/dòng), `seed.months_back`, `csv_view.interval_s`, …

> Mọi thời gian theo **UTC**. "Chốt ngày" = 00:00 UTC = **07:00 giờ Việt Nam**.

---

## 2. Cấu trúc file

```
src/task1_data/
├── historical_downloader.py  [đặc tả] Trades từ data.binance.vision + LOB lịch sử từ Tardis (ngày mùng 1)
├── snapshot_collector.py     [đặc tả] Poll REST /api/v3/depth?limit=20 mỗi 200ms
├── websocket_streamer.py     [đặc tả] class BinanceLOBStreamer (depth20@100ms + trade), buffer 15 phút
├── README_TASK_1.md          [đặc tả] Biên bản bàn giao (file này)
├── update_store.py           [thêm]   Lệnh "cập nhật kho" (seed → trades → rolling → retention → báo cáo)
├── CHAY_TASK1.bat            [thêm]   Bấm đúp: cài đặt + dựng/cập nhật kho + đăng ký lịch tự động 07:30
├── config.yaml                        Cấu hình (14/30 ngày, 1 giây/dòng, ...)
├── requirements.txt                   Thư viện cần cài
├── core/                              Lõi dùng chung (người khác không cần mở)
│   ├── common.py                      Đường dẫn, schema, thời gian, HTTP, đồng bộ giờ với sàn
│   ├── transform.py                   Lưới thời gian đều, gom trade, ghép bảng chuẩn (point-in-time)
│   ├── store.py                       Kho nền: ghi/đọc theo ngày, retention, inventory, CSV xem nhanh
│   └── capture.py                     Ghi dữ liệu tự thu ra ổ đĩa (1s/dòng), chờ chốt ngày
└── tools/                             Công cụ phụ
    ├── selftest.py                    Tự kiểm tra 34 hạng mục với dữ liệu giả, không cần mạng
    ├── run_update.py                  Chạy cập nhật không phụ thuộc thư mục (dùng cho lịch tự động)
    ├── THU_LIVE_khong_bat_buoc.bat    Thu sổ lệnh live vào kho khi máy đang mở
    ├── TEST_STREAMER.bat              Test streamer 60s + đo độ trễ
    └── DON_CACHE_O_C.bat              (1 lần) xóa cache pip ở ổ C

data/
├── raw/
│   ├── raw_lob_trades.parquet/            ← FILE BÀN GIAO (parquet dataset = thư mục, mỗi ngày 1 file)
│   │   ├── BTCUSDT_2026-05-01_seed.parquet      bộ nền cố định (Tardis), không bị retention xóa
│   │   ├── BTCUSDT_2026-10-18_rolling.parquet   kho trượt, tự xóa khi quá retention_days
│   │   └── ETHUSDT_2026-05-01_seed.parquet      ETH cùng thư mục, phân biệt bằng tên file + cột symbol
│   ├── ETHUSDT/                           (cấu trúc giống BTCUSDT/)
│   └── BTCUSDT/
│       ├── trades_1s/                     trade đã gom theo giây cho mọi ngày trong khung (cả ngày không có LOB)
│       ├── live_capture/YYYY-MM-DD/       dữ liệu tự thu bằng WS/REST, chờ chốt ngày để nhập kho
│       └── _tmp/                          file tải tạm (tự xóa)
└── processed/
    ├── BTCUSDT_view.csv, ETHUSDT_view.csv CSV gọn để mở xem (10s/dòng, 5 tầng), mỗi cặp 1 file
    ├── inventory.csv                      thống kê từng ngày của mọi cặp: số dòng, % phủ, số đoạn, dung lượng
    └── source_log.csv                     nhật ký nguồn: URL, ngày, byte, số dòng, sha256
```

---

## 3. Data dictionary – `raw_lob_trades.parquet`

Mỗi dòng = trạng thái thị trường tại **một mốc thời gian t** trên lưới đều (mặc định 1 giây).

| Cột | Kiểu | Ý nghĩa |
|---|---|---|
| `timestamp` | int64 | Mốc t, **mili-giây** epoch UTC (chia hết cho 1000) |
| `datetime` | datetime64[ms, UTC] | Như trên, dạng ngày giờ |
| `symbol` | str | `BTCUSDT` hoặc `ETHUSDT` |
| `mid_price` | float64 | (bids_p1 + asks_p1) / 2 |
| `spread` | float64 | asks_p1 − bids_p1 (luôn > 0; dòng sổ lệnh lỗi/chéo đã bị loại) |
| `bids_p1..p20`, `bids_q1..q20` | float64 / float32 | Giá/khối lượng chờ MUA tầng 1..20 (giá giảm dần, tầng 1 = best bid) |
| `asks_p1..p20`, `asks_q1..q20` | float64 / float32 | Giá/khối lượng chờ BÁN tầng 1..20 (giá tăng dần, tầng 1 = best ask) |
| `trade_price` | float64 | Giá khớp **gần nhất tính đến t** (forward-fill, chỉ dùng quá khứ) |
| `trade_qty` | float64 | Tổng khối lượng khớp trong (t−1s, t] |
| `is_buyer_maker` | boolean | Của lệnh khớp **cuối cùng** trong (t−1s, t]. **True = bên BÁN chủ động** (người mua là maker); NA nếu không có lệnh khớp |
| `trade_count` | int32 | Số lệnh khớp trong (t−1s, t] |
| `buy_qty` / `sell_qty` | float64 | KL **taker mua** / **taker bán** chủ động trong (t−1s, t] → dùng cho Trade Imbalance |
| `vwap` | float64 | Giá trung bình theo KL trong (t−1s, t]; NaN nếu không có lệnh |
| `book_ts` | int64 | Thời điểm thật của snapshot sổ lệnh được dùng (≤ timestamp) |
| `book_age_ms` | int32 | timestamp − book_ts (độ "cũ" của sổ lệnh, ≤ 5000ms) |
| `gap_flag` | bool | True nếu dòng trước cách hơn 1 khung (bắt đầu đoạn dữ liệu mới / sau mất kết nối) |
| `book_source` | str | `tardis` / `binance-ws` / `binance-rest` |
| `trades_source` | str | `binance_vision` / `tardis` / `binance-ws-live` / `none` |

### Quy tắc point-in-time (chống look-ahead)
- Sổ lệnh tại t = **snapshot cuối cùng có thời gian ≤ t**. Không bao giờ dùng snapshot sau t.
- Trade được gán vào khung (t−1s, t]. Timestamp gốc được **làm tròn lên**, nên sự kiện không bị kéo về quá khứ.
- Sổ lệnh không đổi trong một khung thì giữ trạng thái cũ (hợp lệ). Nếu cũ hơn 5 giây (mất mạng, tắt máy) thì **bỏ dòng**, không bịa dữ liệu. Chỗ đứt được đánh dấu `gap_flag=True`.

### Lưu ý cho Task 2
- **Không gán nhãn vượt qua chỗ đứt.** Ví dụ với h = 10s: nếu trong (t, t+h] có dòng `gap_flag=True` hoặc thiếu dòng thì bỏ mẫu đó.
- Các ngày seed **không liền nhau** (mỗi tháng 1 ngày). Walk-forward nên tịnh tiến theo ngày, xử lý từng ngày như một đoạn độc lập.
- Công thức OFI (5a/5b) cần snapshot t và t−1: dùng hai dòng liên tiếp **cùng đoạn** (`gap_flag` của dòng t phải False).
- ⚠️ **Phát hiện quan trọng từ dữ liệu thật: spread BTCUSDT/ETHUSDT gần như luôn bằng 1 tick (0.01 USDT).** Spread > 1 tick chỉ chiếm **0.006–0.1%** thời gian mỗi ngày; Q0.95 = Q0.99 = 1 tick. Vì vậy nhãn `Stress = 1(Spread > Q0.95)` (công thức 8) sẽ **toàn số 0**. Task 2 đã xử lý bằng nhãn **price impact** (đề bài mục 2.8 cho phép) – xem `src/task2_models/README_TASK_2.md`.
- ⚠️ Sổ lệnh `depth20` với tick 0.01 chỉ phủ **~1 bps quanh giá** (BTC: 20 tầng ≈ ±3–4 USDT). Tổng giá trị 20 tầng mỗi phía trung vị chỉ 30–200 nghìn USDT tùy ngày → không đo được tác động giá của lệnh rất lớn (> ~100k USDT).
- **Kho có 2 cặp:** luôn lọc/nhóm theo `symbol` trước khi tính đặc trưng hay gán nhãn, để không nối dòng BTC với ETH. ETH dùng để kiểm tra độ bền (cùng model, cặp khác) hoặc so sánh 2 thị trường.
- `trades_1s/` có trade của mọi ngày trong khung, kể cả ngày không có sổ lệnh. Cột: `bucket` (= timestamp ms), `trade_count, trade_qty, buy_qty, sell_qty, quote_qty, last_ts, trade_price, is_buyer_maker`. Dùng được cho realized volatility / trade imbalance dài ngày.

---

## 4. Interface Contract real-time – `BinanceLOBStreamer`

```python
from src.task1_data.websocket_streamer import BinanceLOBStreamer
s = BinanceLOBStreamer(symbol="BTCUSDT", buffer_minutes=15)   # record=True để ghi vào kho
s.start()                        # chạy ngầm (thread), tự kết nối lại, tự đổi endpoint dự phòng
d = s.get_latest_data()          # None cho tới khi có snapshot đầu tiên
s.stop()
```
`get_latest_data()` trả về:
```python
{
  'timestamp': float,            # GIÂY epoch (theo đặc tả); có thêm 'timestamp_ms': int
  'mid_price': float, 'spread': float,
  'bids': [[price, qty], ... 20], 'asks': [[price, qty], ... 20],
  'recent_trades': [ {'timestamp': ms, 'price', 'qty', 'is_buyer_maker', 'trade_id'}, ... ],  # từ snapshot trước (~100ms)
  'buffer_df': pd.DataFrame,     # 15 phút gần nhất, mỗi dòng 1 snapshot 100ms, cùng tên cột với kho nền
  'symbol': 'BTCUSDT', 'last_update_id': int
}
```
Thêm: `get_buffer_df()`, `get_stats()` (số tin, reconnect, ping RTT, độ trễ sự kiện), `get_latest_data(include_buffer=False)` để nhẹ hơn khi chỉ cần snapshot.

Endpoint: `wss://stream.binance.com:9443/stream?streams=btcusdt@depth20@100ms/btcusdt@trade`, dự phòng `wss://data-stream.binance.vision:9443`.

---

## 5. Kích thước dữ liệu & độ trễ

### Dung lượng – **số đo thật** (cập nhật 20/10/2026, sau khi tăng `seed.months_back` 6 → 12)
| Thành phần | Mỗi ngày (đo thật) | Hiện tại |
|---|---|---|
| raw_lob_trades (1s, 20 tầng, zstd) | **8.5–22.3 MB** (TB ~15 MB) | 12 ngày seed × 2 cặp = **361 MB** |
| trades_1s | **1.2–2.8 MB** | ~169 MB (BTC 92 + ETH 77) |
| CSV xem nhanh (10s, 5 tầng) + inventory/source_log | | ~53 MB |
| **Tổng `data/`** (chưa tính cache của Task 2) | | **~582 MB** |

**Bộ nền seed hiện có:** ngày mùng 1 của **12 tháng 11/2025 → 10/2026**, mỗi cặp 12 ngày (≈1,04 triệu dòng/cặp). Mỗi ngày **86.386–86.399 dòng/86.400 (≈100% phủ)**, liền mạch (ETH 01/08/2026 có 10 đoạn ngắn), **0.8–11 triệu lệnh khớp/ngày** – trải nhiều chế độ thị trường (01/12/2025 và 01/02/2026 rất sôi động; 01/01/2026 và 01/08/2026 rất trầm). Trades kiểm tra checksum SHA-256 chính thức của Binance (`ok_checksum` trong `source_log.csv`). Kho không tự phình thêm: seed cố định, trades rolling tự xóa sau `retention_days`.

### Độ trễ WebSocket (yêu cầu đặc tả: ping < 50ms)
- Đo bằng `TEST_STREAMER.bat`: `ws_ping_rtt_ms_p50/p95` (ping/pong khứ hồi thật trên kết nối WS) và `event_latency_ms_p50/p95` (giờ nhận − event time của sàn, đã hiệu chỉnh lệch đồng hồ).
- **Kết quả đo** (mạng gia đình tại Việt Nam, `python -m src.task1_data.websocket_streamer --seconds 60`, 22/10/2026): `ws_ping_rtt_ms_p50 = 136 ms`, `event_latency_ms_p50 = 96 ms`; 590 bản tin sổ lệnh + 4.116 lệnh khớp trong 60 s, **0 lần nối lại**. Các lần đo trước (20–21/10, lúc selftest Task 3): ping p50 dao động **~150–220 ms**, độ trễ sự kiện p50 ~70 ms.
- Số p95 (ping ~3,9 s, sự kiện ~5 s) bị kéo lên bởi vài giây đầu khi mới kết nối và đang đồng bộ giờ với sàn. Sau đó độ trễ ổn định quanh mức p50.
- **Kết luận:** mục tiêu ping < 50 ms của đặc tả **chưa đạt** từ mạng Việt Nam. Với lưới dữ liệu 1 giây và horizon 10 giây thì độ trễ ~0,1–0,2 s không ảnh hưởng tới dự báo.
- Lưu ý: máy chủ Binance đặt tại AWS Tokyo. Từ Việt Nam RTT thường **50–100ms**, nên mục tiêu < 50ms phụ thuộc đường mạng, không phụ thuộc code. Số đo được ghi trung thực, không làm đẹp.

### Trạng thái kiểm thử
| Hạng mục | Trạng thái |
|---|---|
| `tools/selftest.py` (34 kiểm tra: schema, point-in-time, tổng KL trade, retention, idempotent, nhiều cặp BTC+ETH, contract streamer) | **ĐẠT** trên Python 3.10 (pandas 2.3) và 3.13 (pandas 3.0) |
| Tải dữ liệu thật Binance (trades) + Tardis (LOB) qua `CHAY_TASK1.bat` | **ĐẠT** (20/10/2026): 12 ngày seed × 2 cặp, ≈100% phủ (ETH 01/08/2026: 99,98%), 34 ngày trades, 0 sổ lệnh chéo, 0 ô trống |
| Kết nối WebSocket thật + đo độ trễ | **ĐẠT** (22/10/2026): chạy 60 s liên tục, 0 lần nối lại; ping p50 136 ms, độ trễ sự kiện p50 96 ms (số liệu ở trên) |

---

## 6. Hướng đi khác (so với file Đặc tả nhiệm vụ v2.0)

> Theo lưu ý của file đặc tả: được đổi cách làm miễn đạt yêu cầu. Dưới đây là mọi điểm khác biệt, lý do, và ảnh hưởng tới việc kết nối.

| # | Đặc tả gốc | Cách làm thực tế | Lý do | Ảnh hưởng tích hợp |
|---|---|---|---|---|
| 1 | LOB lịch sử: poll REST 200ms trong 2–3 ngày liên tục | **Bộ nền seed từ Tardis.dev**: ngày mùng 1 của **12 tháng gần nhất** (`seed.months_back: 12`; miễn phí, không cần key, 25 tầng, ghi mọi lần sổ lệnh thay đổi). REST collector vẫn có, chạy tùy ý | Binance không cung cấp lịch sử sổ lệnh spot. Treo máy 2–3 ngày không khả thi. Tardis cho 12 ngày × 24h dữ liệu thật/cặp trải nhiều trạng thái thị trường (đủ cho walk-forward 5 fold + 2 ngày test và RQ4). Ban đầu 6 tháng, tăng lên 12 ngày 20/10/2026 vì 6 ngày quá ít cho walk-forward | Không ảnh hưởng: cùng schema. Các ngày không liền nhau (xem lưu ý Task 2) |
| 2 | Tần suất lưu: theo snapshot (100–200ms) | Kho nền lưu **1 giây/dòng** (sửa được: `sample_interval_ms`). Live 100ms vẫn có trong `buffer_df` | Giảm dung lượng khoảng 10 lần. Horizon dự báo 10s–1 phút nên 1s là đủ | Task 2 dùng horizon theo giây (10 dòng = 10s) |
| 3 | `trade_price, trade_qty, is_buyer_maker` theo từng lệnh | Trade **gom theo khung** (t−1s, t] + cột bổ sung `trade_count, buy_qty, sell_qty, vwap` | Trade đến hàng chục lệnh/ms, không ghép 1-1 với snapshot được. Gom theo khung đúng point-in-time và không mất thông tin KL mua/bán chủ động | Tên cột gốc giữ nguyên, ý nghĩa ghi ở mục 3 |
| 4 | `raw_lob_trades.parquet` là 1 file | Là **parquet dataset** (thư mục cùng tên, mỗi ngày 1 file) | Cho phép cộng dồn theo ngày và xóa ngày cũ mà không phải ghi lại cả kho, cũng không nhân đôi dung lượng | `pd.read_parquet("data/raw/raw_lob_trades.parquet")` vẫn đọc được cả thư mục (pyarrow). Polars: `pl.read_parquet("data/raw/raw_lob_trades.parquet/*.parquet")` |
| 5 | Tải trades 7–14 ngày một lần | **Kho trượt** `retention_days` (14 hoặc 30) + lệnh `update_store` tự bù ngày thiếu, tự xóa ngày quá hạn | Theo yêu cầu nhóm: dữ liệu ngắn hạn, cộng dồn và xóa định kỳ | Không ảnh hưởng |
| 6 | Không có | Thêm `BTCUSDT_view.csv`, `inventory.csv`, `source_log.csv` | Để mở xem bằng Excel. `source_log` đáp ứng mục "Data dictionary và source log" trong checklist của thầy | Không bắt buộc dùng |
| 7 | Timestamp đồng nhất | Book dùng **giờ nhận đã hiệu chỉnh theo giờ sàn** (`/api/v3/time`); trade dùng **giờ khớp của sàn**. Tardis dùng timestamp của sàn | Stream depth20 và REST depth của spot không trả về thời gian | Sai lệch cỡ độ trễ mạng (~vài chục ms), nhỏ so với khung 1s |
| 8 | Không có | Dữ liệu đứt (tắt máy, mất mạng) không nối liền mà đánh dấu `gap_flag`; snapshot cũ hơn 5s bị bỏ | Chống bịa dữ liệu và look-ahead | Task 2 cần bỏ mẫu vượt qua chỗ đứt |
| 9 | Ghi `config/` ở gốc | Cấu hình đặt trong `src/task1_data/config.yaml`; `.venv` ở gốc dự án | Giữ đúng phạm vi thư mục của Task 1. `.venv` dùng chung được cho cả 3 task | Task 2/3 có thể cài thêm thư viện vào cùng `.venv` |
| 10 | Chỉ BTC/USDT | Thêm **ETH/USDT** (cùng 12 ngày seed, cùng khung trades, cùng schema). Danh sách cặp ở `symbols:` trong `config.yaml` | Đề bài của thầy mục 3.1 ưu tiên cả BTC/USDT và ETH/USDT | Task 2 lọc theo cột `symbol`. Streamer real-time mặc định vẫn là BTCUSDT (`symbol=` để đổi) |

| 11 | Không có | Tự thu live khi mở máy (`BAT_TU_THU_LIVE_KHI_MO_MAY.bat`) | Đề bài 3.2 gợi ý nguồn "Binance websocket"; lịch sử sổ lệnh spot không tải được nên cách miễn phí duy nhất để có ngày liên tục là tự ghi. Không cần 24/24: ngày thiếu giờ vẫn nhập kho, chỗ đứt được đánh dấu `gap_flag` | Ngày tự thu vào kho dạng `*_rolling.parquet`. **Task 2 mặc định chỉ dùng seed** (`data_kinds: [seed]` trong config Task 2) để số liệu báo cáo không đổi |

**Sửa lỗi 20/10/2026 – `websocket_streamer.py`:** trước đây kết nối bị ngắt **mỗi ~30 giây** (log "ping/pong timed out"): pong của ping đo RTT (`_ping_loop`, mỗi 5s) làm websocket-client hiểu nhầm là "pong đến muộn". Đã đặt `ping_timeout=None`; kết nối chết vẫn được phát hiện bởi `_ping_loop` (15s không có tin → nối lại). Sau khi sửa: chạy liên tục không ngắt. Dữ liệu seed Tardis **không bị ảnh hưởng** (không đi qua streamer).

### Giới hạn đã biết
- Ngày trong khung rolling **chỉ có sổ lệnh nếu máy có chạy `THU_LIVE_khong_bat_buoc.bat`** (hoặc REST collector) trong ngày đó. Ngày máy tắt thì chỉ có trade (`trades_1s`).
- Tardis chỉ miễn phí ngày mùng 1. Muốn thêm ngày liên tục thì cần tự thu hoặc mua gói Tardis (đổi `tardis_datasets` và thêm API key).
- Nếu máy bị chặn `stream.binance.com`, streamer tự đổi sang `data-stream.binance.vision`. Nếu chặn hết thì dùng `snapshot_collector.py` với `data-api.binance.vision`.

---

## 7. Checklist bàn giao Task 1

- [x] Pipeline thu thập/đọc LOB (Blueprint 2.1): `historical_downloader.py`, `snapshot_collector.py`, `websocket_streamer.py`
- [x] BTC/USDT + ETH/USDT Binance (3.1), kết hợp WebSocket + historical trades (3.2)
- [x] Đúng raw fields: 20 tầng bid/ask + trades có aggressor side (3.3)
- [x] Parquet, chia partition theo ngày/symbol (3.5)
- [x] Script tải/chuẩn hóa chạy lại được, idempotent
- [x] Không look-ahead: quy tắc point-in-time + tự kiểm tra
- [x] Data dictionary (mục 3) + source log (`source_log.csv`)
- [x] `BinanceLOBStreamer` đúng Interface Contract, buffer 15 phút, tự kết nối lại
- [x] Dung lượng thật (mục 5)
- [x] Độ trễ đo thật (mục 5): ping p50 136 ms, sự kiện p50 96 ms (22/10/2026). Mục tiêu < 50 ms chưa đạt do khoảng cách mạng VN → Tokyo
