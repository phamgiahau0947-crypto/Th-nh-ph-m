# README_TASK_2 – Biên bản bàn giao Task 2: Toán học vi cấu trúc, ML, Liquidity Stress & Lợi suất

**Đề tài:** Limit Order Book & Liquidity Stress Prediction (Research Blueprint 05)
**Phạm vi:** `src/task2_models/` (code) · `models/` (mô hình) · `outputs/tables/`, `outputs/figures/` (số liệu & hình cho báo cáo APA7) · `data/processed/task2_features/` (cache đặc trưng)
**Dữ liệu đầu vào:** kho Task 1 `data/raw/raw_lob_trades.parquet` – BTCUSDT & ETHUSDT, 12 ngày × 24h mỗi cặp (ngày mùng 1, 11/2025 → 10/2026), lưới 1 giây, sổ lệnh 20 tầng + trade gom theo giây.
**Ngày bàn giao:** 20/10/2026 · **Phiên bản:** 1.0

## 0. Tóm tắt nhanh

| Cần gì | Ở đâu |
|---|---|
| Mô hình cho Dashboard | `models/model_direction.pkl`, `models/model_stress.pkl`, `models/model_meta.json` – dùng qua `ModelInference` (mục 7) |
| Bảng số liệu cho báo cáo APA7 | `outputs/tables/*.csv` (danh sách ở mục 5) |
| Hình cho báo cáo | `outputs/figures/*.png` |
| Chạy lại toàn bộ | bấm đúp `src/task2_models/CHAY_TASK2.bat` |

**Mô hình đang dùng cho live** (chọn theo walk-forward, độ trễ < 5 ms): hướng giá = **XGBoost** (BTC, ETH); stress = **Logistic Regression** (BTC, ETH). Train trên 10 ngày dev, h = 10 giây. `predict_one` p50 ≈ 3 ms; tính đặc trưng live từ buffer ≈ 80 ms. Đã chạy thử nối WebSocket Binance thật → đặc trưng → dự báo: **hoạt động**.

**Kết quả chính trên TEST chưa đụng tới (01/09 & 01/10/2026):**

| | BTCUSDT | ETHUSDT | Benchmark tốt nhất (BTC / ETH) |
|---|---|---|---|
| Hướng giá – macro-F1 (KTC 95%) | **0.544** (0.536–0.550) | **0.492** (0.485–0.499) | OBI-rule 0.467 / 0.436; ngẫu nhiên ≈ 0.33 |
| Hướng giá – MCC | **0.317** | **0.238** | OBI-rule 0.230 / 0.161 |
| Stress – PR-AUC (tỉ lệ nền ≈ 5.2%) | **0.15–0.17** | **0.13–0.14** | Persistence 0.094 / 0.080; AS-rule 0.084 / 0.073 |
| Stress – ROC-AUC | 0.73 | 0.69–0.71 | 0.59–0.65 |

---

## 1. Cài đặt & chạy

```bash
# (đã có .venv từ Task 1)  – bấm đúp src\task2_models\CHAY_TASK2.bat, hoặc:
python -m pip install -r src/task2_models/requirements.txt
python -m src.task2_models.selftest            # 10 kiểm tra: công thức OFI, no-lookahead, live = offline, CST...
python -m src.task2_models.run_all             # toàn bộ pipeline (~1–2 giờ CPU)
python -m src.task2_models.run_all --deeplob   # + DeepLOB (cần torch CPU, thêm ~1 giờ)
python -m src.task2_models.predictor           # test predict_one trên dữ liệu test
python -m src.task2_models.predictor --live 30 # test với WebSocket Binance thật 30 giây
```

Từng bước riêng (chạy lại an toàn): `dataset` → `evaluate_walkforward` → `mlofi_regression` → `return_analysis` → `eda` → `deeplob`.
Mọi tham số nằm ở `src/task2_models/config.yaml` (h, δ, ngưỡng stress, quy mô lệnh, γ của A-S, số fold…).

## 2. Cấu trúc file

```
src/task2_models/
├── config.yaml                 Tham số nghiên cứu
├── common.py                   Đường dẫn, đọc kho Task 1, chia walk-forward theo ngày
├── microstructure_math.py      [đặc tả] Công thức (1)–(6): mid, spread, microprice, OBI, OFI/MLOFI
│                               + độ sâu, độ dốc, chi phí tác động, trade imbalance, RV;
│                               calculate_all_features(data) cho live (cùng code với offline)
├── avellaneda_stoikov.py       [thêm] Mô hình A-S 2008: δ*, giá bảo lưu, StressRatio (10)–(11), ước lượng k
├── cont_stoikov_talreja.py     [thêm] Mô hình hàng đợi CST 2010 (tài liệu/orderbook.pdf): P(mid tăng | hàng đợi)
├── labeler.py                  [đặc tả] (12) lợi suất, (7) nhãn hướng giá, (8) nhãn stress
├── dataset.py                  [thêm] Dựng & cache đặc trưng + nhãn theo (cặp, ngày); hiệu chỉnh A-S/CST trên train
├── models.py                   [thêm] Logistic OvR, Multinomial Logit, Random Forest, XGBoost (+ hiệu chỉnh isotonic)
├── train_direction.py          [đặc tả] Huấn luyện hướng giá + benchmark (Majority, OBI-rule, CST)
├── train_stress.py             [đặc tả] Huấn luyện P(Stress) + benchmark (Persistence, AS-rule)
├── evaluate_walkforward.py     [đặc tả] Walk-forward 5 fold + test chưa đụng tới, chọn & lưu mô hình, bảng/hình
├── metrics.py                  [thêm] macro-F1, balanced acc, MCC, precision/recall, Brier, PR-AUC; bootstrap khối
├── mlofi_regression.py         [thêm] Tái lập Xu–Gould–Howison 2019 (OLS vs Ridge, M = 1..10) + biến thể dự báo
├── return_analysis.py          [đặc tả] Hồi quy (13), stress vs lợi suất, backtest Net PnL (14)
├── deeplob.py                  [thêm] DeepLOB CNN–Inception–LSTM (mô hình nâng cao, RQ3)
├── eda.py                      [thêm] Bảng phân phối lớp, tóm tắt đặc trưng, LOB heatmap, OBI vs lợi suất, kiểm định CST/A-S
├── figures.py                  [thêm] Hàm vẽ hình
├── predictor.py                [đặc tả] class ModelInference (predict_one < 5 ms) + load_replay cho Dashboard
├── selftest.py                 [thêm] Tự kiểm tra
├── run_all.py, CHAY_TASK2.bat  [thêm] Chạy toàn bộ
├── spread_forecast.py          [bổ sung 21/10] Dự báo spread 10s tới – "Spread Forecast" trong Roadmap (mục 7b)
├── model_zoo.py                [bổ sung 21/10] Bộ 8 mô hình cho bảng xếp hạng Dashboard (mục 7b)
├── CHAY_BO_SUNG.bat            [bổ sung 21/10] Chạy riêng 2 phần trên (~15 phút)
└── README_TASK_2.md            [đặc tả] Biên bản này
models/   model_direction.pkl, model_stress.pkl, model_meta.json (tham số A-S/CST, δ), deeplob_*.pt
          model_spread.pkl, model_zoo.pkl (bổ sung – không thay file nào ở trên)
```

## 3. Phương pháp

### 3.1. Dữ liệu & chống rò rỉ (leakage controls)
- Mỗi **ngày là một đoạn độc lập** (các ngày seed cách nhau 1 tháng): mọi cửa sổ trượt khởi động lại đầu ngày, nhãn không nối sang ngày khác.
- Đặc trưng tại t **chỉ dùng dữ liệu ≤ t** (kiểm chứng tự động: sửa toàn bộ dữ liệu sau t → đặc trưng tại t không đổi – `selftest`).
- Giây thiếu dữ liệu giữ NaN (không nội suy); OFI cần cả t−1 và t; nhãn cần m_t và m_{t+h}.
- **Walk-forward theo NGÀY, không xáo trộn**: 2 ngày cuối (01/09/2026, 01/10/2026) là **test chưa đụng tới**; 10 ngày dev chia 5 fold mở rộng (train 5→9 ngày đầu, val = ngày kế tiếp).
- Trong mỗi fold, **mọi thứ học từ dữ liệu chỉ fit trên train của fold**: δ, tham số A-S (k, chuẩn hóa StressRatio), tham số CST (λ, μ, θ, đơn vị lệnh), scaler/imputer (trong Pipeline), mô hình, hiệu chỉnh xác suất (ngày train cuối).
- Mô hình xuất ra được **chọn theo trung bình walk-forward** (macro-F1 cho hướng giá, PR-AUC cho stress), không nhìn test.
- Ngưỡng stress là phân vị **trượt 1 giờ tính đến t** (chỉ quá khứ).

### 3.2. Công thức (đặc tả 3.4) – ánh xạ vào code
| Đặc tả | Code | Ghi chú |
|---|---|---|
| (1) Mid, (2) Spread, (3) Microprice | `microstructure_math.mid_price/spread/microprice` | spread & microprice tính theo bps của mid |
| (4) OBI_L, L = 1, 5, 10, 20 | `obi()` | |
| (5a–c) OFI tầng m, (6) MLOFI, M = 10 | `ofi_levels()` | **Theo đúng bài báo Xu et al. (pt. 9–12)**: e_m = ΔW_m − ΔV_m (xem 6.1). Tính từ snapshot 1s như Cont et al. 2014; chuẩn hóa theo độ sâu trung bình 10 phút của tầng m; cộng dồn 1s/10s/30s |
| (7) Nhãn hướng giá | `labeler.direction_label` | h = 10s; δ chọn trên train để ~1/3 mẫu "đứng" |
| (8) Nhãn stress | `labeler.stress_labels` | **Thay spread bằng price impact** (xem 6.2) |
| (9) P(Stress) | `train_stress.py` | LogReg / RF / XGBoost + isotonic |
| (10) δ* Avellaneda–Stoikov | `avellaneda_stoikov.optimal_spread_bps` | **Spread 2 phía theo pt. (30) của bài báo** (xem 6.3) |
| (11) StressRatio | `avellaneda_stoikov.stress_ratio` | "Spread thực tế" = spread hiệu dụng của lệnh giả định (xem 6.3) |
| (12) Lợi suất ngắn hạn | `labeler.forward_return_bps` | bps |
| (13) Hồi quy tác động lợi suất | `return_analysis.regressions` | HAC Newey–West, mẫu không chồng lấn |
| (14) Net PnL | `return_analysis.pnl_backtest` | mua ở ask/bán ở bid, 4 mức phí |

### 3.3. Ba mô hình trong `tài liệu/`
1. **MLOFI – Xu, Gould & Howison (2019)**: đặc trưng chính (OFI 10 tầng × 3 cửa sổ) + tái lập nguyên bài báo trong `mlofi_regression.py` (cửa sổ 30 phút, Δt = 10s, OLS vs Ridge với λ chọn bằng CV, M = 1..10, R² trong mẫu & ngoài mẫu) + mở rộng sang dự báo.
2. **Avellaneda & Stoikov (2008)**: λ(δ) = A·e^{−kδ} ước lượng từ khoảng cách quét sổ lệnh của khối lượng chủ động mỗi giây; δ* (pt. 30) với σ = RV 5 phút, T−t = h; giá bảo lưu với tồn kho market maker ≈ −(mất cân bằng lệnh chủ động 5 phút); StressRatio > 1.5 = cảnh báo sớm (benchmark "AS-rule" + đặc trưng).
3. **Cont, Stoikov & Talreja (2010)** (file `orderbook.pdf` – *không phải* DeepLOB như kế hoạch dự án đoán): hàng đợi tốt nhất là quá trình sinh–tử; P(mid tăng | q_ask, q_bid) theo Mệnh đề 3 (spread = 1 tick). Benchmark cấu trúc cho hướng giá ("CST") + đặc trưng `cst_p_up`.

### 3.4. Mô hình học máy
- Hướng giá (3 lớp, trọng số cân bằng): **LogReg-OvR, Multinomial Logit, Random Forest, XGBoost** + benchmark Majority, OBI-rule (luật của kế hoạch dự án: OBI > 0.3 → Up), CST.
- Stress (nhị phân, ~5% dương): **LogReg, Random Forest, XGBoost** (hiệu chỉnh isotonic trên ngày train cuối) + benchmark Persistence, AS-rule.
- Nâng cao: **DeepLOB** (Zhang et al. 2019; 143k tham số – bằng bài báo) trên cửa sổ 50 giây × 10 tầng.
- Đặc trưng: 67 cột (`microstructure_math.feature_columns`): trạng thái sổ lệnh (spread, microprice, OBI ×4, độ sâu, độ dốc, chi phí tác động), MLOFI (10 tầng × 1/10/30s + tổng), trade imbalance & cường độ, lợi suất quá khứ, RV, giờ trong ngày, 4 đặc trưng cấu trúc (CST, A-S).
- Train lấy 1/5 số giây (giây liền kề gần như trùng thông tin); **đánh giá trên mọi giây** của ngày val/test.

### 3.5. Đánh giá (đề bài 2.9)
macro-F1, balanced accuracy, MCC, precision/recall, Brier, log-loss, calibration curve; với stress thêm PR-AUC, ROC-AUC, Brier skill, precision/recall tại ngưỡng cảnh báo 0.7. Khoảng tin cậy 95% bằng **bootstrap khối 5 phút** trên test (giữ tự tương quan) và kiểm định chênh lệch cặp giữa mô hình (RQ3, RQ5). Hiệu năng theo chế độ biến động (tam phân vị RV 5 phút, ngưỡng lấy từ dev) cho RQ4.

### 3.6. Cách chọn δ (đặc tả 3.7 yêu cầu giải thích)
δ chọn trên TRAIN sao cho |R_{t,10s}| ≤ δ chiếm ~1/3 mẫu → 3 lớp gần cân bằng (tránh mô hình "luôn đoán đứng"), và chọn lại trong từng fold.
- **BTCUSDT: δ = 0** – riêng các giây mid **không đổi** sau 10s đã chiếm ~34% (tick 0.01 USDT ≈ 0.0012 bps rất nhỏ, nhưng sổ lệnh tầng 1 rất dày nên giá hay "đứng yên"). Lớp "Đứng" = giá không đổi.
- **ETHUSDT: δ = 0.335 bps**.
- Phân phối lớp (dev / test): BTC 32.9/34.2/33.0% ↔ 34.8/29.6/35.6% (Down/Flat/Up); ETH 33.1/33.3/33.5% ↔ 34.0/31.4/34.7%. Chi tiết theo ngày: `class_distribution_by_day.csv`.

## 4. Kết quả

Fold walk-forward ↔ ngày val: fold1 = 01/04/2026, fold2 = 01/05, fold3 = 01/06, fold4 = 01/07, fold5 = 01/08; test = 01/09 + 01/10/2026.

### 4.1. Dự báo hướng giá (3 lớp, h = 10s) – `direction_walkforward_summary.csv`, `direction_test_metrics.csv`, `test_bootstrap_ci.csv`

| Mô hình | BTC WF macro-F1 (TB ± SD) | BTC test macro-F1 / MCC | ETH WF macro-F1 | ETH test macro-F1 / MCC |
|---|---|---|---|---|
| Majority | 0.135 | 0.152 / 0.000 | 0.157 | 0.172 / 0.000 |
| CST (cấu trúc) | 0.321 | 0.407 / 0.233 | 0.305 | 0.331 / 0.140 |
| OBI-rule | 0.441 | 0.467 / 0.230 | 0.413 | 0.436 / 0.161 |
| LogReg-OvR | 0.504 ± 0.021 | 0.542 / 0.316 | 0.446 ± 0.004 | 0.473 / 0.230 |
| Multinomial Logit | 0.505 ± 0.022 | 0.543 / 0.317 | 0.448 ± 0.011 | 0.476 / 0.231 |
| Random Forest | **0.516** ± 0.053 | **0.544** / 0.317 | 0.464 ± 0.036 | **0.492** / 0.239 |
| **XGBoost** | 0.513 ± 0.049 | **0.544** / 0.317 | **0.465** ± 0.040 | **0.492** / 0.238 |
| DeepLOB (chỉ dev → test) | – | 0.536 / 0.310 | – | 0.479 / 0.226 |

Kiểm định chênh lệch (bootstrap khối 5 phút trên test): XGBoost − MNLogit = +0.001 (KTC −0.002…+0.004) với BTC – **không khác biệt**; = +0.016 (+0.012…+0.021) với ETH – **có ý nghĩa**. MNLogit − OBI-rule = +0.076 (BTC) / +0.040 (ETH), XGBoost − CST = +0.14 / +0.16 – đều có ý nghĩa (p < 0.005).

### 4.2. Liquidity stress (nhị phân, ~5.2%) – `stress_*_metrics.csv`, `stress_alert_thresholds.csv`

| Mô hình | BTC WF PR-AUC | BTC test PR-AUC (KTC) | ETH WF PR-AUC | ETH test PR-AUC (KTC) |
|---|---|---|---|---|
| AS-rule (StressRatio > 1.5) | 0.094 | 0.084 | 0.077 | 0.073 |
| Persistence | 0.117 | 0.094 | 0.092 | 0.080 |
| **LogReg** | **0.174** | 0.153 (0.134–0.172) | 0.142 | 0.128 (0.112–0.144) |
| Random Forest | 0.173 | 0.167 (0.146–0.192) | **0.149** | 0.142 (0.123–0.166) |
| XGBoost | 0.155 | 0.164 (0.143–0.190) | 0.135 | 0.142 (0.123–0.162) |

Ngưỡng cảnh báo trên test (mô hình đã chọn): P(Stress) > 0.30 → cảnh báo 3.3% (BTC) / 4.9% (ETH) thời gian, precision 24% / 21% (**gấp 4.7 / 4.0 lần** tỉ lệ nền), recall 15% / 19%. P > 0.70 gần như không bao giờ xảy ra (xem 6.13).
Nhãn đa chiều (robustness, `robustness_stress_multidim.csv`): khó dự báo hơn (PR-AUC 0.03 / 0.07 với tỉ lệ nền 1.8% / 2.6%).

### 4.3. RQ1–RQ5

- **RQ1 – OBI có dự báo mid-price không? CÓ.** Lợi suất 10s tăng đơn điệu theo thập phân vị OBI, từ −0.85 lên +0.82 bps (BTC; `obi_vs_future_returns_*.png`). Hồi quy (13): hệ số OBI t ≈ 45 (BTC), 35 (ETH). OBI-rule đơn giản đã đạt MCC 0.23 / 0.16 trên test. L = 1, 5, 10, 20 cho kết quả gần như nhau (R² ngoài mẫu 2.6–2.7% BTC) – thông tin tập trung ở tầng 1.
- **RQ2 – OFI có nhiều thông tin hơn OBI tĩnh không? Tùy mục đích.**
  - *Giải thích đồng thời* (tái lập Xu et al., `mlofi_regression_contemporaneous.csv`): OFI tầng 1 giải thích **~51% (BTC) / 48% (ETH)** biến động mid trong cùng 10s (adj R²; trung vị R² ngoài mẫu 0.49 / 0.45) – cùng bậc với Cont et al. (2014). Thêm tầng sâu (M = 10) tăng R² trong mẫu (0.55) nhưng **giảm** R² ngoài mẫu (0.42 / 0.39): khác Xu et al. (cổ phiếu Nasdaq), với crypto tầng sâu ít giá trị; Ridge (λ chọn CV) gần như trùng OLS.
  - *Dự báo 10s kế tiếp* (`mlofi_predictive_vs_obi.csv`): MLOFI quá khứ **gần như vô dụng** (R² ngoài mẫu ≈ 0, M = 10 âm), OBI tĩnh tốt hơn (trung vị R² ngoài mẫu 5.1% BTC / 3.7% ETH).
  - Trong mô hình phân loại (`rq2_feature_sets.csv`, test macro-F1 XGBoost): OBI 0.486 · OFI tầng 1 0.493 · MLOFI 0.502 · OBI+MLOFI 0.525 · tất cả 0.544 (BTC); ETH tương tự. → OFI và OBI **bổ sung cho nhau**; trạng thái thanh khoản & biến động đóng góp thêm.
- **RQ3 – DL/ML hơn logistic bao nhiêu?** XGBoost/RF hơn Logistic **+0.001 macro-F1 (BTC, không ý nghĩa)**, **+0.016 (ETH, có ý nghĩa)**. **DeepLOB không cải thiện** trên test chưa đụng tới (`deep_model_metrics.csv`, `deep_vs_baseline_test.csv`): macro-F1 0.536 (BTC) / 0.479 (ETH), **kém XGBoost 0.009 / 0.013** (KTC 95% đều < 0); ngang MNLogit ở ETH (+0.003, không ý nghĩa), kém MNLogit 0.007 ở BTC. DeepLOB có 143k tham số (bằng bài báo gốc), train 6–15 phút CPU, dừng sớm sau 3–5 epoch; chỉ nhìn 10 tầng sổ lệnh 50 giây gần nhất, không có trade/biến động/MLOFI chuẩn hóa như mô hình bảng. → Theo nguyên tắc của đề bài ("không chạy DL chỉ vì hiện đại… phải chứng minh incremental performance"), **không dùng DeepLOB cho sản phẩm**; phần lớn thông tin là tuyến tính, mô hình phức tạp chỉ thêm ít.
- **RQ4 – Ổn định theo chế độ biến động?** (`performance_by_volatility_regime.csv`, tam phân vị RV 5 phút lấy từ dev) Hướng giá **kém đi khi biến động cao**: XGBoost test macro-F1 BTC 0.54 (thấp) → 0.51 (vừa) → 0.43 (cao); ETH 0.44 → 0.47 → 0.39. Stress cũng vậy (LogReg PR-AUC BTC 0.21 → 0.14 → 0.12; ETH 0.18 → 0.13 → 0.11) dù tỉ lệ nền gần như không đổi (5–6%). Qua các fold (`performance_through_time_*.png`) macro-F1 dao động 0.45–0.56 (BTC): ổn định về thứ hạng mô hình, không ổn định về mức.
- **RQ5 – Accuracy có còn ý nghĩa khi tính bất định & mất cân bằng lớp?** Có, nhưng phải báo cáo đúng cách: lớp gần cân bằng (δ chọn theo train) nên accuracy ≈ macro-F1 ≈ balanced accuracy (0.54 BTC); Majority chỉ đạt macro-F1 0.15, MCC 0. KTC 95% bootstrap khối của macro-F1 hẹp (±0.007) nhưng **chênh giữa các mô hình tốt nhất nằm trong KTC**. Với stress (5% dương) accuracy vô nghĩa (luôn đoán 0 → 95%) → dùng PR-AUC, Brier skill, precision/recall tại ngưỡng.

### 4.4. Căng thẳng thanh khoản tác động thế nào tới lợi suất ngắn hạn? (câu hỏi của Thầy) – `return_regression_13.csv`, `stress_impact_on_returns.csv`, `pnl_backtest.csv`

1. **Stress làm tăng RỦI RO, hầu như không đổi lợi suất KỲ VỌNG.** Hồi quy (13) (HAC, mẫu không chồng lấn mỗi 10s, ~101k quan sát): hệ số Stress_t lên R **không có ý nghĩa** (BTC −0.05 bps, p = 0.21; ETH +0.05, p = 0.31). Lên |R| (13b, đã kiểm soát RV 5 phút) **dương, có ý nghĩa**: +0.21 bps (t = 6.9, BTC; ≈ +19% so với E|R| = 1.11), +0.18 bps (t = 4.1, ETH; ≈ +12%). Dùng P(Stress) **dự báo ngoài mẫu** (13d): hệ số 2.32 (t = 17, BTC) / 1.90 (t = 12.6, ETH) → ngũ phân vị P̂(Stress) cao nhất có E|R| 1.62 bps so với 0.75 bps ở nhóm thấp nhất (BTC; ETH 1.93 vs 1.17). Với ETH, P̂(Stress) có tác động dương nhỏ lên R (+0.57 bps, t = 2.6) – yếu, không thấy ở BTC.
2. **Khi thanh khoản căng, dòng lệnh truyền vào giá MẠNH hơn (BTC).** Hệ số tương tác β4 (MLOFI × Stress) **dương**: +0.06 (t = 2.1) với Stress_t, +0.46 (t = 4.1) với P̂(Stress). Ý nghĩa kinh tế: sổ lệnh mỏng → cùng một lượng lệnh mất cân bằng làm giá dịch nhiều hơn (price impact lớn hơn). Với ETH β4 không có ý nghĩa (+0.20, t = 1.0). Đặc tả kỳ vọng β4 âm ("lợi suất giảm do trượt giá") – **dữ liệu không ủng hộ**; trượt giá thể hiện ở rủi ro & chi phí giao dịch chứ không ở lợi suất mid.
3. **Backtest (14) trên test** (mỗi 10s, mua ask/bán bid): tín hiệu XGBoost có lợi suất gộp **+0.53 bps/lệnh** (BTC, tỉ lệ đúng 55%) / +0.50 bps (ETH) – dương nhưng **nhỏ hơn rất nhiều phí sàn**: chỉ cần 2 bps/chiều đã thành −3.5 bps/lệnh; phí taker thường 10 bps/chiều → −19.5 bps/lệnh. → Khả năng dự báo có ý nghĩa thống kê **không đủ để kiếm lời** sau chi phí cho nhà giao dịch thường (phù hợp thị trường hiệu quả). Lúc P̂(Stress) > 0.3, tín hiệu **đúng hơn** (71% BTC, 68% ETH) và lãi gộp lớn hơn (+0.91 / +0.99 bps) vì giá biến động mạnh hơn – "lọc bỏ lúc stress" làm **giảm** lãi gộp, đổi lại tránh rủi ro: độ lệch chuẩn lợi suất lúc cảnh báo cao hơn 59% (BTC) / 25% (ETH).
4. **Mô hình cấu trúc** (`structural_models_test.csv`): CST dự báo đúng hướng đổi giá kế tiếp 68.6% (BTC) – **ngang** dấu OBI tầng 1 (68.6%) – và 63.3% (ETH, kém OBI 65.9%). StressRatio A-S > 1.5 xảy ra ~23% thời gian, P(stress | ratio > 1.5) = 11% vs 3% khi ≤ 1.5 → có thông tin (gấp ~3 lần) nhưng yếu hơn mô hình ML.

## 5. Danh sách bảng & hình (cho báo cáo APA7)

| Đề bài 4.1 | File |
|---|---|
| Bảng 1 class distribution | `class_distribution.csv`, `class_distribution_by_day.csv` |
| Bảng 2 feature summary | `feature_summary.csv` |
| Bảng 3 baseline metrics | `direction_walkforward_summary.csv`, `direction_walkforward_by_fold.csv`, `direction_test_metrics.csv`, `stress_*` (cùng mẫu), `test_bootstrap_ci.csv` |
| Bảng 4 deep model metrics | `deep_model_metrics.csv`, `deep_vs_baseline_test.csv` |
| Bảng 5 performance by volatility regime | `performance_by_volatility_regime.csv` |
| Bảng 6 latency / model complexity | `latency_model_complexity.csv` |
| Thêm | `rq2_feature_sets.csv`, `mlofi_regression_contemporaneous.csv`, `mlofi_coefficients.csv`, `mlofi_predictive_vs_obi.csv`, `return_regression_13.csv`, `stress_impact_on_returns.csv`, `pnl_backtest.csv`, `stress_alert_thresholds.csv`, `structural_models_test.csv`, `robustness_stress_multidim.csv`, `feature_importance_*.csv`, `walkforward_all_metrics.csv` |
| Hình LOB heatmap | `lob_heatmap_{BTC,ETH}USDT.png` |
| Hình OBI vs future returns | `obi_vs_future_returns_*.png` |
| Hình confusion matrices | `confusion_direction_*.png` |
| Hình calibration | `calibration_direction_*.png`, `calibration_stress_*.png`, `cst_calibration_*.png` |
| Hình performance through time | `performance_through_time_{direction,stress}.png`, `performance_by_hour_test_*.png` |
| Thêm | `mlofi_r2_vs_M.png`, `stress_return_distribution_*.png`, `pnl_backtest_*.png`, `feature_importance_*.png` |

### Giới hạn (nên nêu trong báo cáo)
- 12 ngày/cặp, cách nhau 1 tháng (Tardis miễn phí ngày mùng 1) – đủ cho walk-forward theo ngày nhưng chưa phải chuỗi liên tục; 2 ngày test.
- Snapshot 1 giây, 20 tầng (~±1 bps quanh giá): OFI ước lượng từ snapshot (như Cont et al. 2014) chứ không từ từng sự kiện; không đo được tác động của lệnh > ~100k USDT.
- Dữ liệu train lấy từ Tardis (Binance spot), live lấy từ Binance WebSocket – cùng sàn, cùng định dạng (selftest kiểm chứng code live = offline), nhưng chưa có ngày live đủ dài để so phân phối.
- Mô hình chọn theo walk-forward, test chỉ dùng 1 lần; một số quyết định thiết kế (nhãn stress trung bình cửa sổ, ngưỡng 0.3, Q20/Q80 nhãn đa chiều) được đưa ra sau khi xem các fold dev (không xem test).

## 6. Hướng đi khác (so với file Đặc tả nhiệm vụ v2.0)

| # | Đặc tả gốc | Cách làm thực tế | Lý do |
|---|---|---|---|
| 6.1 | (5b) ΔOFI^ask = −q_t (ask giảm) / −(q_t − q_{t−1}) / +q_{t−1}; OFI = ΔOFI^bid − ΔOFI^ask | e_m = ΔW_m − ΔV_m theo **Xu et al. pt. (9)–(12)**: ΔV = q_t (ask giảm) / q_t − q_{t−1} / −q_{t−1} (ask tăng) | Công thức (5b) trong đặc tả bị **ngược dấu**: thực chất cho ΔW + ΔV, tức lệnh bán mới làm OFI **tăng**. Kiểm chứng bằng ví dụ số mục 3.2 của bài báo (`selftest`) |
| 6.2 | (8) Stress = 1(Spread_{t+h} > Q0.95(Spread)) | Stress = 1(chi phí tác động khứ hồi **trung bình trên (t, t+h]** của lệnh 25k USDT (BTC) / 10k USDT (ETH) > Q0.95 trượt 1 giờ tính đến t) | Spread niêm yết = 1 tick gần như 100% thời gian → nhãn theo spread **toàn 0**. Đề bài mục 2.8 cho phép "spread **hoặc price impact** vượt percentile". Lấy trung bình cửa sổ vì chi phí từng giây là "nhiễu chớp" (đợt vượt ngưỡng kéo dài trung vị 1s, ACF lag-10 ≈ 0.15) → nhãn 1 giây gần như không dự báo được và ít ý nghĩa kinh tế. Thêm nhãn **đa chiều** (độ sâu < Q20 và RV tương lai > Q80) làm robustness |
| 6.3 | (10) δ* = (1/γ)ln(1+γ/k) + ½γσ²(T−t); (11) StressRatio = Spread thực tế / δ* | δ* = γσ²(T−t) + (2/γ)ln(1+γ/k) (pt. 30, **spread 2 phía**); "Spread thực tế" = **spread hiệu dụng** (chi phí tác động khứ hồi trung bình 10s), StressRatio chia thêm trung vị train (1 = bình thường) | Công thức (10) của đặc tả là **nửa** spread nhưng được so với spread 2 phía. Spread niêm yết = 1 tick nên Spread/δ* gần như hằng số. γ = 0.1 (ví dụ số của bài báo, giá theo bps) |
| 6.4 | `orderbook.pdf` "có thể là DeepLOB hoặc survey" (kế hoạch dự án) | Đó là **Cont–Stoikov–Talreja (2010)** → cài đặt thành mô hình cấu trúc (benchmark + đặc trưng). DeepLOB vẫn được làm riêng (`deeplob.py`) theo đề bài 2.7 | |
| 6.5 | CST: nghịch đảo Laplace (pt. 10–11) | Tính **tương đương trong miền thời gian** P[σ_A < σ_B] = ∫ f_a S_b dt qua exp(QΔt) của xích sinh–tử; khối lượng chưa chạm 0 sau 300s chia đôi | Ổn định số học; giải hệ tuyến tính 2 chiều trực tiếp bị suy biến khi hàng đợi lớn (đã thử) |
| 6.6 | CST: đơn vị lệnh = KL trung bình lệnh giới hạn | Đơn vị chọn trên **train** theo log-loss của hướng đổi giá kế tiếp (lưới bội số độ sâu tầng 1) | Dữ liệu 1s không có từng lệnh. λ, μ, θ ước lượng từ thay đổi ròng mỗi giây (thêm = max(Δq + KL khớp, 0)) |
| 6.7 | Walk-forward "chia 5 cửa sổ thời gian tịnh tiến" | 5 fold **mở rộng theo ngày** + 2 ngày test riêng | Các ngày seed không liền nhau; dùng ngày làm đơn vị tránh rò rỉ qua tự tương quan trong ngày |
| 6.8 | `models/scaler.pkl` riêng | Scaler/imputer nằm **trong Pipeline** của từng mô hình (`model_direction.pkl`, `model_stress.pkl`) | Tránh lệch scaler giữa train và live. XGBoost không cần scaler |
| 6.9 | 1 mô hình/bài toán | `model_*.pkl` chứa **mỗi cặp (BTC, ETH) 1 mô hình** + `model_meta.json` (δ, tham số A-S/CST đã hiệu chỉnh) | Thị trường khác nhau về tick tương đối & độ sâu |
| 6.10 | Không có | `calculate_all_features(data)` gom buffer 100ms của streamer về lưới 1s rồi dùng **chung hàm đặc trưng với offline** | Live = offline (kiểm chứng trong `selftest`). Cần ~10 phút buffer để mọi cửa sổ chuẩn hóa đầy (streamer giữ 15 phút); trước đó đặc trưng dùng cửa sổ ngắn hơn |
| 6.11 | Mô hình xuất ra train trên toàn bộ dữ liệu | Mô hình xuất ra train trên **10 ngày dev** (đúng mô hình đã báo cáo trên test) | Số liệu test trong báo cáo khớp với mô hình chạy trên dashboard |
| 6.12 | DeepLOB walk-forward | DeepLOB chỉ train 1 lần (dev → test), cửa sổ 50s, 10 tầng | Chi phí CPU (không có GPU) |
| 6.13 | `is_stress_warning` khi P(Stress) > 0.7 | Cảnh báo khi P(Stress) > **0.30** (đỏ), 0.15–0.30 vàng, < 0.15 xanh; thêm khóa `stress_level` | Tỉ lệ nền stress ~5% và xác suất đã hiệu chỉnh tốt (P̂ > 0.5 → 47% đúng) nên P > 0.7 xảy ra 0% thời gian → đồng hồ không bao giờ đỏ. 0.30 chọn trên **các fold walk-forward** (precision gấp 4.5–6 lần nền), không dùng test |
| 6.14 | Chọn mô hình tốt nhất | Chọn theo walk-forward **trong các mô hình có độ trễ < 5 ms** | Random Forest (200 cây) mất 30–40 ms/dòng → loại khỏi live dù điểm ngang XGBoost (chênh test −0.0002, không ý nghĩa). Mô hình dùng chung: n_jobs = 1 khi dự báo từng dòng |
| 6.15 | Hồi quy (13) dùng Stress_t | Thêm (13b) cho \|R\| và (13c–d) với P(Stress) dự báo ngoài mẫu | Trả lời trực tiếp "dự báo căng thẳng thanh khoản tác động thế nào tới lợi suất" |
| 6.17 | Không có | `config.yaml: data_kinds: [seed]` – chỉ dùng 12 ngày Tardis; thêm `rolling` để đưa ngày tự thu live (Task 1) vào | Từ 20/10/2026 máy tự thu live khi mở; nếu không khóa lại thì lần train sau sẽ tự đổi ngày test và số liệu báo cáo |
| 6.16 | MLOFI: λ Ridge 1 giá trị | λ chọn riêng cho từng M (CV liền khối trên dev), quy về cửa sổ 180 khoảng; thêm trung vị R² ngoài mẫu theo cửa sổ | Trung bình MSE bị vài cửa sổ cực đoan chi phối (ETH) |

## 7. Hướng dẫn test `predict_one` & kết nối Task 3

```python
from src.task1_data.websocket_streamer import BinanceLOBStreamer
from src.task2_models.microstructure_math import calculate_all_features
from src.task2_models.predictor import ModelInference, load_replay

streamer = BinanceLOBStreamer(symbol="BTCUSDT"); streamer.start()
predictor = ModelInference(model_dir="models/")          # symbol mặc định BTCUSDT; ModelInference(..., symbol="ETHUSDT")
params = predictor.structural_params("BTCUSDT")         # tham số A-S/CST đã hiệu chỉnh (đọc 1 lần)
data = streamer.get_latest_data()
features = calculate_all_features(data, params)          # {} nếu chưa có dữ liệu
preds = predictor.predict_one(features)
# preds = {'p_up','p_flat','p_down','predicted_direction','p_stress','is_stress_warning','stress_ratio',
#          'expected_short_term_return_bps', 'is_stress_ratio_warning','horizon_s','delta_bps',
#          'model_direction','model_stress','symbol','latency_ms'}

# Chế độ Replay (1 ngày lịch sử, mỗi giây 1 dòng: đặc trưng + dự báo + cột 'in_sample'):
df = load_replay("BTCUSDT", "2026-10-01")
```
Gợi ý cho Task 3: tính `calculate_all_features` mỗi 0.5–1s (mất ~50–150 ms do gom 15 phút buffer), `predict_one` < 5 ms. Hiển thị `features['spread_bps']`, `obi_5`, `ofi1_10s`, `impact_rt_bps`, `as_stress_ratio`, `cst_p_up` cùng các xác suất.

## 7b. Bổ sung 21/10/2026 – Dự báo spread & bộ mô hình cho bảng xếp hạng

> **Không thay đổi** `model_direction.pkl`, `model_stress.pkl`, `model_meta.json` hay bất kỳ số liệu nào ở mục 4. Hai phần dưới đây là file **mới**. Chạy lại: bấm `src/task2_models/CHAY_BO_SUNG.bat` (~15 phút), hoặc `run_all` (đã thêm 2 bước cuối).

### 7b.1. Dự báo spread 10 giây tới (`spread_forecast.py` → `models/model_spread.pkl`)
Lý do: ảnh Roadmap của thầy (ô số 5) ghi "Spread Forecast". Spread niêm yết BTC/ETH gần như luôn = 1 tick, nên "đoán con số spread" vô nghĩa (luôn đoán 1 tick là đúng ~99,9%). Vì vậy dự báo 2 thứ:

| | Nhãn | Loại | Mô hình | Benchmark |
|---|---|---|---|---|
| (A) **P(spread giãn)** | trong (t, t+10s] có ≥ 1 giây spread niêm yết > 1 tick | phân loại, **không** cân bằng lớp, **không** hiệu chỉnh isotonic (*) | LogReg, XGBoost | Tần suất nền, Persistence (giãn trong 10s qua) |
| (B) **Spread hiệu dụng** | chi phí khứ hồi trung bình ImpactRT trên (t, t+10s] (bps), lệnh 25k/10k USDT | hồi quy trên log(1+chi phí) | Ridge, XGBoost | Persistence (trung bình 10s qua) |

- Kiểm định giống hệt mục 3.1: walk-forward 5 fold theo ngày + 2 ngày test chưa đụng tới; chọn mô hình theo trung bình walk-forward (PR-AUC cho A, R² thang log cho B), không nhìn test.
- Đặc trưng: 63 đặc trưng gốc (bỏ 4 đặc trưng cấu trúc A-S/CST để khỏi hiệu chỉnh lại mỗi fold) + 3 đặc trưng spread quá khứ (`spr_ticks`, `spr_widen_10s`, `spr_widen_60s`).
- Bảng: `outputs/tables/spread_forecast_walkforward.csv`, `spread_forecast_test.csv`.
- (*) Lần chạy đầu dùng hiệu chỉnh isotonic trên ngày train cuối (giống mô hình stress). Kết quả: xác suất bị **nén về ~0**, vì ngày 01/08/2026 spread gần như không giãn (0,06%). Đã bỏ: Logistic/XGBoost tối ưu log-loss trên toàn bộ ngày train, nên xác suất tự khớp tỉ lệ nền của train. PR-AUC trên test cũng tăng (0,008 → 0,015 với BTC).
- Mô hình được chọn (theo walk-forward): (A) XGBoost cho cả 2 cặp; (B) Ridge cho cả 2. Độ trễ 1 dòng 0,2–1 ms.

**Kết quả (TB walk-forward → test 01/09 + 01/10/2026):**

| | BTCUSDT | ETHUSDT | Benchmark tốt nhất (BTC / ETH) |
|---|---|---|---|
| (A) Tỉ lệ nền spread giãn | 1,18% (10 ngày dev) → 0,50% (test) | 0,89% → 0,26% | – |
| (A) PR-AUC | 0,027 → **0,015** (3,0× nền) | 0,026 → **0,010** (3,9× nền) | Persistence 0,0053 / 0,0031 |
| (A) ROC-AUC | 0,67 → **0,65** | 0,77 → **0,70** | Persistence 0,51 / 0,51 |
| (B) R² (thang log) | 0,18 → **0,02** | 0,11 → **0,10** | Persistence −0,30 / −0,38 |
| (B) Tương quan hạng Spearman | 0,47 → **0,44** | 0,42 → **0,39** | Persistence 0,39 / 0,32 |
| (B) Sai số tuyệt đối TB (bps) | 0,067 → 0,050 | 0,101 → 0,083 | Persistence 0,048 / 0,092 |

Theo màu trên Dashboard (gấp mấy lần tỉ lệ nền train), trên **test**:

| Màu | BTC: % thời gian → spread giãn thật | ETH: % thời gian → spread giãn thật |
|---|---|---|
| Xanh (< 2×) | 94,2% → 0,43% | 95,7% → 0,22% |
| Vàng (2–5×) | 5,5% → 1,45% | 4,0% → 0,89% |
| Đỏ (≥ 5×) | 0,36% → **3,9%** | 0,35% → **2,7%** |

**Diễn giải (nói thật):**
- (A) Spread niêm yết giãn **rất hiếm** (0,3–1% số cửa sổ 10s) và **chớp nhoáng**. Mô hình **xếp hạng rủi ro tốt**: lúc "đỏ" spread giãn thật nhiều gấp ~9–12 lần lúc "xanh". Nhưng chỉ bắt được tốt hơn đoán mò **3–4 lần** (PR-AUC).
- Con số % tuyệt đối **cao hơn thực tế khoảng 2 lần vào ngày êm** như 2 ngày test, vì tỉ lệ nền thay đổi mạnh theo ngày (0,06–3%). Brier skill ≈ 0. → Nên đọc **màu / thứ hạng**, đừng đọc con số %.
- (B) Mô hình xếp hạng "lúc nào chi phí sẽ cao hơn" tốt hơn việc lấy chi phí 10 giây vừa qua (Spearman 0,44 so với 0,39; R² dương so với âm). Nhưng mức tuyệt đối khó đoán vì chi phí rất "gai". Riêng BTC, sai số tuyệt đối của Persistence nhỏ hơn một chút (0,048 so với 0,050 bps): phần lớn thời gian chi phí BTC ≈ 0 nên "giữ nguyên" đã gần đúng.
- Ý nghĩa với đề tài: tín hiệu sổ lệnh dự báo **spread / thanh khoản** được, nhưng yếu hơn nhiều so với dự báo **hướng giá** (mục 4.1) và **stress kéo dài** (mục 4.2). Lý do: spread niêm yết là sự kiện chớp nhoáng 1–2 giây.
- Dùng trong Dashboard qua `ModelInference.predict_frame` (thêm cột `p_spread_widen`, `eff_spread_fwd_bps`). `predict_one` **giữ nguyên** Interface Contract và độ trễ.

### 7b.2. Bộ mô hình cho bảng xếp hạng (`model_zoo.py` → `models/model_zoo.pkl`)
- Dùng cho trang **🏆 So sánh mô hình** của Dashboard (việc D): 8 mô hình đoán hướng cùng lúc trên cùng luồng dữ liệu live/replay:
  - XGBoost, Random Forest, MNLogit, LogReg-OvR, DeepLOB,
  - 3 mốc so sánh: CST, OBI-rule, Majority.
- XGBoost lấy nguyên từ `model_direction.pkl`. LogReg-OvR, MNLogit, RandomForest train lại **đúng như lượt test** của `evaluate_walkforward` (cùng 10 ngày dev, cùng seed, cùng δ, cùng tham số cấu trúc). DeepLOB lấy từ `deeplob_{SYM}.pt`.
- **Kiểm chứng tự động** (`outputs/tables/model_zoo_check.csv`): điểm macro-F1 trên 2 ngày test của cả 8 mô hình × 2 cặp **trùng khớp** bảng `direction_test_metrics.csv` (cột `khop_bao_cao` = True cho cả 14 dòng; DeepLOB đã chấm riêng ở `deep_model_metrics.csv`).
- `deeplob.py`: tách hàm `lob_grid_tensor(g)` từ `lob_tensor` (nhận sẵn lưới 1s) để dùng cho live. Kết quả của `lob_tensor` không đổi.

## 8. Checklist bàn giao Task 2

| Yêu cầu (đề bài / đặc tả) | Trạng thái |
|---|---|
| RQ1–RQ5 được trả lời có số liệu | ✅ mục 4.3 |
| Công thức (1)–(14) | ✅ `microstructure_math`, `labeler`, `avellaneda_stoikov`, `return_analysis` (sửa 2 lỗi đặc tả, mục 6) |
| 3 mô hình trong `tài liệu/` (MLOFI, A-S, CST) | ✅ |
| Baseline Logistic / Multinomial / RF / XGBoost | ✅ |
| Benchmark thống kê / kinh tế lượng | ✅ Majority, OBI-rule, CST, Persistence, AS-rule; hồi quy MLOFI OLS/Ridge; hồi quy (13) HAC |
| Ít nhất 1 mô hình nâng cao (DeepLOB) | ✅ `deeplob.py` – đã train & đánh giá trên test (không hơn XGBoost) |
| Liquidity-stress model P(Stress) có hiệu chỉnh xác suất | ✅ isotonic, Brier, reliability diagram |
| Walk-forward, không shuffle; test chưa đụng tới | ✅ 5 fold theo ngày + 2 ngày test |
| Không look-ahead / leakage | ✅ quy tắc mục 3.1 + `selftest` |
| Metrics: macro-F1, balanced acc, MCC, precision/recall, Brier, calibration | ✅ |
| Robustness / sensitivity | ✅ chế độ biến động, nhãn stress đa chiều, tập đặc trưng (RQ2), cặp thứ 2 (ETH), KTC bootstrap |
| Lợi suất ngắn hạn & tác động stress, Net PnL | ✅ mục 4.4 |
| `models/model_direction.pkl`, `model_stress.pkl` | ✅ (+ `model_meta.json`) |
| `ModelInference.predict_one` < 5 ms | ✅ p50 ≈ 3 ms (đo khi máy đang bận train DeepLOB) |
| Bảng & hình cho APA7 | ✅ mục 5 |
| Script chạy lại được, README | ✅ `run_all`, `CHAY_TASK2.bat`, file này |
